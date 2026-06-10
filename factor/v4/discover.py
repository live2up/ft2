"""
factor/v3/discover.py — 因子发现引擎（v3 核心创新）
=============================================================================

将 v2 的 gp_miner.py + gp_evaluator.py 重构为可插拔适应度 + 迭代发现流水线。

三层架构：
  Layer 1: FitnessMode + FitnessCalculator  — 可插拔适应度
  Layer 2: GPEngine                          — GP 符号回归（适应度可切换）
  Layer 3: FactorDiscoveryEngine             — 迭代发现流水线 + FactorLibrary 闭环

关键创新：
  - 适应度策略可运行时切换 (icir / sharpe / multi_freq)
  - 多轮迭代：discover → validate → register → 扩大种子池 → 下一轮
  - FactorLibrary 自增长，发现越多种子越多

[新增] 2026-06-01 v3 核心模块
=============================================================================
"""

import random
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from signals.v4 import Expression
from .base import FactorCategory, LibraryEntry, FactorLibrary

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# GP 配置常量
# ═══════════════════════════════════════════════════════════════════════════

TERMINALS = ['close', 'volume', 'open', 'high', 'low', 'amount']
# [新增] 2026-06-05 白名单：仅允许已知安全的终端变量参与 GP 随机树生成
# 排除 change/changeRatio/preclose/pe_ttm_index 等前视偏差源或噪声源
SAFE_GP_TERMINALS = {
    'open', 'high', 'low', 'close', 'volume', 'amount',
    'rel_close', 'share', 'downside_vol', 'rel_volume', 'rel_amount',
}
UNARY_OPS = ['abs', 'sqrt', 'log', 'neg', 'exp', 'tanh', 'not']  # [新增] 2026-06-07 not
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'max', 'min',
              'gt', 'lt', 'and', 'or']  # [新增] 2026-06-07 比较/逻辑算子
CONSTANTS = [0.0, -1.0, 1.0, 0.5, 2.0]

PRIMITIVE_WITH_PARAMS = [
    ('ts_rank',       'window', [5, 10, 15, 20, 30, 60]),
    ('ts_zscore',     'window', [10, 20, 30, 60]),
    ('delay',         'window', [1, 5, 10, 20, 60]),
    ('delta',         'window', [1, 5, 10, 20, 60]),  # [新增] v3 语法糖
    ('decay_linear',  'window', [5, 10, 20, 30]),
    ('cs_zscore',     'window', [20]),
    ('cs_rank',       None, []),
    ('cs_mean',       None, []),            # [新增] 2026-06-01 截面均值
    ('signed_power',  'exponent', [2.0]),
    ('winsorize',     'n', [2.0, 3.0, 5.0]),  # [新增] v3 截尾
    ('ts_sum',        'window', [5, 10, 15, 20, 30, 60]),
    ('ts_mean',       'window', [5, 10, 15, 20, 30, 60]),
    ('ts_std',        'window', [10, 20, 30, 60]),
    ('ts_max',        'window', [5, 10, 20, 30, 60]),
    ('ts_min',        'window', [5, 10, 20, 30, 60]),
    ('sma',           'window', [5, 10, 20, 30]),
    ('ts_argmin',     'window', [5, 10, 20]),
    ('ts_argmax',     'window', [5, 10, 20]),
    ('ts_skew',       'window', [10, 20, 30, 60]),  # [新增] 2026-06-01
    ('ret',           'window', [1, 5, 10, 20, 60]),  # [新增] v3 零参数语法糖
    ('adv',           'window', [5, 10, 15, 20, 30, 60]),  # [新增] v3 零参数语法糖
    ('intra_ret',     None, []),  # [新增] v3 日内收益语法糖
    ('ts_regression_residual', 'window', [5, 10, 20, 30]),  # [新增] 2026-06-07 线性回归残差
    ('correlation',   'window', [10, 20, 30]),  # [新增] 2026-06-07 滚动相关系数
]

DEFAULT_GP_CONFIG = {
    'population_size': 300,
    'generations': 30,
    'max_depth': 8,
    'tournament_size': 5,
    'crossover_prob': 0.75,
    'mutation_prob': 0.15,
    'elite_ratio': 0.05,
    'random_inject_ratio': 0.05,
    'parsimony_penalty': 0.0005,
}


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: 可插拔适应度
# ═══════════════════════════════════════════════════════════════════════════

class FitnessMode(Enum):
    """适应度模式"""
    ICIR = 'icir'            # RankIC × ICIR（快速，适合大规模种群）
    SHARPE = 'sharpe'        # Pipeline 回测 Sharpe（真实绩效）
    MULTI_FREQ = 'multi_freq'  # ME/W/5D 取最优 Sharpe
    WQB = 'wqb'              # WQB 五维漏斗：ICIR→HR→QuickBacktest
    INDUSTRY = 'industry'    # [新增] 行业轮动自适应：N<50跳过IC, N≥50逐步启用IC门控


class FitnessCalculator:
    """适应度计算器基类

    所有配置参数通过构造注入，避免子类硬编码。

    [重构] 2026-06-05 统一参数规范:
      scheduler:  调度器对象(FixedScheduler/IntervalScheduler)，所有模式统一
      allocator:  分配器对象(TopNEqualWeight等)，所有模式统一
      不再使用 freq_list/quick_freq 等字符串简写
    """

    def __init__(self, data: Dict[str, np.ndarray], future_returns: pd.DataFrame,
                 returns: pd.DataFrame = None, cost_rate: float = 0.0,
                 scheduler=None, allocator=None):
        self.data = data
        self.future_returns = future_returns
        self.returns = returns
        self.cost_rate = cost_rate
        self.scheduler = scheduler
        self.allocator = allocator
        self._shape = self._infer_shape(data)

    @staticmethod
    def _infer_shape(data) -> Tuple[int, int]:
        for arr in data.values():
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                return arr.shape
        return (1, 1)

    def compute(self, factor_values: np.ndarray) -> float:
        """计算适应度，子类实现"""
        raise NotImplementedError

    def _validate(self, factor_values: np.ndarray) -> bool:
        """检查因子值是否有效"""
        if np.all(np.isnan(factor_values)):
            return False
        if np.nanstd(factor_values) < 1e-10:
            return False
        return True


class ICIRFitness(FitnessCalculator):
    """RankIC 适应度：|IC_mean| × ICIR × 100

    适合 GP 初期快速筛选。每日一个 IC 值，统计量更稳定。

    注意：纯 IC 方向容易被高 IC 低 Sharpe 的因子欺骗（如 streak²、body_mean）。
    推荐优先使用 WQBFitness 替代。
    """

    def compute(self, factor_values: np.ndarray) -> float:
        if not self._validate(factor_values):
            return -999.0
        daily_ics, _ = _compute_daily_ics(factor_values, self.future_returns)
        if daily_ics is None:
            return -999.0
        ic_mean = float(np.mean(daily_ics))
        ic_std = float(np.std(daily_ics, ddof=1))
        if ic_std < 1e-10:
            return -999.0
        icir = ic_mean / ic_std
        return abs(ic_mean) * icir * 100.0


def _compute_daily_ics(factor_values: np.ndarray, future_returns: pd.DataFrame
                       ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """计算日频 IC 序列 + 有效日索引（供各 FitnessCalculator 复用）

    Args:
        factor_values: (T, N) 因子值数组
        future_returns: T 行收益 DataFrame

    Returns:
        (daily_ics, valid_indices):
          daily_ics:    长度为 n 的 IC 数组，n < 30 时返回 (None, None)
          valid_indices: 长度为 n 的整数数组，记录每个 IC 对应的原始 t 索引
    """
    T = factor_values.shape[0]
    daily_ics = []
    valid_indices = []
    for t in range(T):
        fv = factor_values[t, :]
        rv = future_returns.iloc[t].values
        mask = ~np.isnan(fv) & ~np.isnan(rv)
        if mask.sum() < 5:
            continue
        corr = np.corrcoef(fv[mask], rv[mask])[0, 1]
        if not np.isnan(corr):
            daily_ics.append(corr)
            valid_indices.append(t)
    if len(daily_ics) < 30:
        return None, None
    return np.array(daily_ics), np.array(valid_indices, dtype=int)


class SharpeFitness(FitnessCalculator):
    """Pipeline Sharpe 适应度：真实回测绩效

    使用 self.scheduler / self.allocator（构造注入），不为空时优先于默认值。

    [重构] 2026-06-01 用注入参数替代硬编码 FixedScheduler('ME') + TopNEqualWeight(3)
    """

    def compute(self, factor_values: np.ndarray) -> float:
        if not self._validate(factor_values):
            return -999.0
        if self.returns is None:
            return -999.0
        try:
            from .backtest import FactorPipeline, FixedScheduler, TopNEqualWeight
            T, N = self._shape
            fv_df = pd.DataFrame(factor_values, index=self.returns.index[:T],
                                 columns=self.returns.columns[:N])
            # [重构] 2026-06-01 使用注入参数，否则用安全默认值
            scheduler = self.scheduler if self.scheduler is not None else FixedScheduler('ME')
            allocator = self.allocator if self.allocator is not None else TopNEqualWeight(3)
            pipeline = FactorPipeline(self.returns, scheduler, allocator, self.cost_rate)
            result = pipeline.evaluate(fv_df)
            return float(result.sharpe_ratio)
        except Exception as e:
            logger.debug(f"SharpeFitness 计算失败: {e}")
            return -999.0


class MultiFreqFitness(FitnessCalculator):
    """多频率适应度：取多调度器中最优 Sharpe

    兼顾多周期稳健性，适合最终筛选阶段。

    参数:
      scheduler_list: 调度器列表, 默认 [FixedScheduler('ME'), FixedScheduler('W'), IntervalScheduler(5)]
      如果只传了 scheduler(单个), 则退化为单频率模式

    [重构] 2026-06-05 freq_list → scheduler_list, 统一用调度器对象
    """

    def __init__(self, data: Dict[str, np.ndarray], future_returns: pd.DataFrame,
                 returns: pd.DataFrame = None, cost_rate: float = 0.0,
                 scheduler=None, allocator=None,
                 scheduler_list: list = None):
        super().__init__(data, future_returns, returns, cost_rate,
                         scheduler, allocator)
        # 默认三频率: 月末 / 周末 / 5天
        if scheduler_list is not None:
            self.scheduler_list = scheduler_list
        elif scheduler is not None:
            self.scheduler_list = [scheduler]
        else:
            self.scheduler_list = [FixedScheduler('ME'), FixedScheduler('W'), IntervalScheduler(5)]

    def compute(self, factor_values: np.ndarray) -> float:
        if not self._validate(factor_values):
            return -999.0
        if self.returns is None:
            return -999.0
        try:
            from .backtest import FactorPipeline, TopNEqualWeight
            T, N = self._shape
            fv_df = pd.DataFrame(factor_values, index=self.returns.index[:T],
                                 columns=self.returns.columns[:N])
            allocator = self.allocator if self.allocator is not None else TopNEqualWeight(3)
            sharpes = []
            for sched in self.scheduler_list:
                pipeline = FactorPipeline(self.returns, sched, allocator, self.cost_rate)
                result = pipeline.evaluate(fv_df)
                sharpes.append(result.sharpe_ratio)
            return float(max(sharpes)) if sharpes else -999.0
        except Exception as e:
            logger.debug(f"MultiFreqFitness 计算失败: {e}")
            return -999.0


class WQBFitness(FitnessCalculator):
    """WQB 漏斗适应度 — 防高IC低Sharpe陷阱

    两阶段级联（GP每代内嵌）:
      Stage 1: WQB版IC过滤器 ← ICIR + HR(胜率) + 分年度稳定性，秒级
      Stage 2: 快速回测       ← 单频率Pipeline Sharpe，~0.05秒

    与 ICIRFitness 相比:
      HR<50% 直接淘汰（方向不稳定，无策略价值 → 淘汰 STR_20d 型陷阱）
      分年度IC方向不一致直接淘汰（只有特定年份有效的因子不要）
      最终 fitness 是 Sharpe 而非 IC 复合值

    参数:
      scheduler:       Stage2回测调度器, 默认FixedScheduler('ME')
      min_hr:          Stage1 HR阈值，默认0.5
      min_yearly_stability: Stage1 分年度稳定性阈值，默认0.6

    [新增] 2026-06-05 WQB方法论
    [重构] 2026-06-05 quick_freq/freq_list → scheduler, 统一参数规范
    """

    def __init__(self, data: Dict[str, np.ndarray], future_returns: pd.DataFrame,
                 returns: pd.DataFrame = None, cost_rate: float = 0.0,
                 scheduler=None, allocator=None,
                 min_hr: float = 0.5, min_yearly_stability: float = 0.6):
        super().__init__(data, future_returns, returns, cost_rate,
                         scheduler, allocator)
        self.min_hr = min_hr
        self.min_yearly_stability = min_yearly_stability

    def _compute_wqb_ic(self, factor_values: np.ndarray) -> Dict[str, float]:
        """Stage 1: ICIR + HR + 分年度方向一致性

        [修复] 2026-06-05 使用 _compute_daily_ics 公共函数 + valid_indices 修正年对齐
        """
        daily_ics, valid_indices = _compute_daily_ics(factor_values, self.future_returns)
        if daily_ics is None:
            return {'ic_mean': 0.0, 'icir': 0.0, 'hr': 0.0,
                    'yearly_stability': 0.0, 'pass': False}

        ic_mean = float(np.mean(daily_ics))
        ic_std = float(np.std(daily_ics, ddof=1))
        icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
        hr = float(np.mean(daily_ics > 0))

        # 分年度: 用 valid_indices 取对应年份，消除索引不对齐问题
        yearly_stability = 0.5
        if (self.future_returns is not None and
                hasattr(self.future_returns, 'index') and
                hasattr(self.future_returns.index, 'year')):
            # 只取有 IC 值那几天的年份
            valid_years = np.asarray(self.future_returns.index[valid_indices].year)
            yearly_means = []
            for yr in sorted(set(valid_years)):
                yr_ics = daily_ics[valid_years == yr]
                if len(yr_ics) > 0:
                    yearly_means.append(float(np.mean(yr_ics)))
            if yearly_means:
                main_dir = np.sign(ic_mean)
                yearly_stability = float(np.mean(
                    [1.0 for ym in yearly_means if np.sign(ym) == main_dir]))

        pass_stage1 = (hr >= self.min_hr and
                       yearly_stability >= self.min_yearly_stability)

        return {'ic_mean': ic_mean, 'icir': icir, 'hr': hr,
                'yearly_stability': yearly_stability, 'pass': pass_stage1}

    def _quick_backtest(self, factor_values: np.ndarray) -> float:
        """Stage 2: 单频率Pipeline回测"""
        from .backtest import FactorPipeline, FixedScheduler, TopNEqualWeight
        T, N = self._shape
        fv_df = pd.DataFrame(factor_values,
                             index=self.returns.index[:T],
                             columns=self.returns.columns[:N])
        scheduler = self.scheduler if self.scheduler is not None else FixedScheduler('ME')
        allocator = self.allocator if self.allocator is not None else TopNEqualWeight(3)
        pipeline = FactorPipeline(self.returns, scheduler, allocator, self.cost_rate)
        result = pipeline.evaluate(fv_df)
        return float(result.sharpe_ratio)

    def compute(self, factor_values: np.ndarray) -> float:
        """主入口: IC过滤器 → 快速回测 → 返回Sharpe"""
        if not self._validate(factor_values):
            return -999.0

        stats = self._compute_wqb_ic(factor_values)
        if not stats['pass']:
            return -999.0

        if self.returns is None:
            return -999.0
        try:
            return self._quick_backtest(factor_values)
        except Exception as e:
            logger.debug(f"WQBFitness 计算失败: {e}")
            return -999.0


def make_fitness_calculator(mode: FitnessMode, data: Dict[str, np.ndarray],
                            future_returns: pd.DataFrame, returns: pd.DataFrame = None,
                            cost_rate: float = 0.0,
                            scheduler=None, allocator=None,
                            # WQB 特有参数
                            min_hr: float = 0.5,
                            min_yearly_stability: float = 0.6) -> FitnessCalculator:
    """工厂函数：根据模式创建适应度计算器

    统一参数规范 (2026-06-05):
      scheduler:  调度器对象, 所有模式通用, 默认FixedScheduler('ME')
      allocator:  分配器对象, 所有模式通用, 默认TopNEqualWeight(3)
    """
    if mode == FitnessMode.ICIR:
        return ICIRFitness(data, future_returns)
    elif mode == FitnessMode.SHARPE:
        return SharpeFitness(data, future_returns, returns, cost_rate,
                             scheduler=scheduler, allocator=allocator)
    elif mode == FitnessMode.MULTI_FREQ:
        return MultiFreqFitness(data, future_returns, returns, cost_rate,
                                scheduler=scheduler, allocator=allocator)
    elif mode == FitnessMode.WQB:
        return WQBFitness(data, future_returns, returns, cost_rate,
                          scheduler=scheduler, allocator=allocator,
                          min_hr=min_hr, min_yearly_stability=min_yearly_stability)
    elif mode == FitnessMode.INDUSTRY:
        from .industry_fitness import IndustryFitness
        return IndustryFitness(data, future_returns, returns, cost_rate,
                               scheduler=scheduler, allocator=allocator)
    else:
        raise ValueError(f"未知适应度模式: {mode}")


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: GPEngine — GP 符号回归引擎
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Individual:
    """GP 个体"""
    tree: ASTNode
    fitness: float = -999.0
    expression_str: str = ''
    ic_mean: float = 0.0
    ic_std: float = 0.0
    icir: float = 0.0
    sharpe: float = 0.0
    generation: int = 0
    depth: int = 0
    node_count: int = 0
    is_seed: bool = False

    def __repr__(self):
        return (f"Individual(fitness={self.fitness:.3f}, "
                f"depth={self.depth}, expr={self.expression_str[:50]}...)")


class GPEngine:
    """GP 符号回归引擎

    核心创新：适应度通过 fitness_calculator 注入，支持运行时切换。

    [重构] 2026-06-01 从 v2/gp_miner.py 移植树操作，适应度委托给 FitnessCalculator
    """

    def __init__(self, data: Dict[str, np.ndarray],
                 future_returns: pd.DataFrame,
                 fitness_calculator: FitnessCalculator = None,
                 fitness_mode: FitnessMode = None,
                 returns: pd.DataFrame = None,
                 config: Dict = None,
                 custom_terminals: List[str] = None,
                 custom_primitives: List[tuple] = None,
                 seed_expressions: List[str] = None,
                 random_seed: int = None,
                 # [重构] 2026-06-05 统一调度参数: 去掉freq_list, 只用scheduler
                 scheduler=None, allocator=None,
                 cost_rate: float = 0.0):
        self.data = data
        self.future_returns = future_returns
        self.returns = returns

        # Config
        cfg = dict(DEFAULT_GP_CONFIG)
        if config:
            cfg.update(config)
        self.population_size = cfg['population_size']
        self.generations = cfg['generations']
        self.max_depth = cfg['max_depth']
        self.tournament_size = cfg['tournament_size']
        self.crossover_prob = cfg['crossover_prob']
        self.mutation_prob = cfg['mutation_prob']
        self.elite_count = max(1, int(self.population_size * cfg['elite_ratio']))
        self.random_inject_count = max(1, int(self.population_size * cfg['random_inject_ratio']))
        self.parsimony_penalty = cfg['parsimony_penalty']

        # Customizable primitives
        # [新增] 2026-06-05 自动检测 data 中的自定义终端
        # 如果用户传了 custom_terminals，以它为准；否则自动从 data 字典中
        # 提取白名单字段（如 rel_close/share/downside_vol）合并到 TERMINALS
        # [修复] 2026-06-05 黑名单→白名单：防止 change/pe_ttm 等前视偏差源渗入 GP
        if custom_terminals is not None:
            self.terminals = custom_terminals
        else:
            auto_terminals = list(TERMINALS)
            for key in sorted(data.keys()):
                key_lower = key.lower()
                if key_lower not in auto_terminals and key_lower in SAFE_GP_TERMINALS:
                    auto_terminals.append(key_lower)
            self.terminals = auto_terminals
        self.primitives = custom_primitives or PRIMITIVE_WITH_PARAMS
        self.seed_expressions = seed_expressions or []

        # Fitness
        self.cost_rate = cost_rate
        if fitness_calculator is not None:
            self.fitness_calc = fitness_calculator
        elif fitness_mode is not None:
            self.fitness_calc = make_fitness_calculator(
                fitness_mode, data, future_returns, returns, self.cost_rate,
                scheduler=scheduler, allocator=allocator)
        else:
            self.fitness_calc = ICIRFitness(data, future_returns)

        # Shape
        self._shape = None
        for arr in data.values():
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                self._shape = arr.shape
                break
        if self._shape is None:
            self._shape = (1, 1)

        # State
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict] = []

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    # ── 随机树生成 ──

    def _random_terminal(self) -> ASTNode:
        if random.random() < 0.85:
            return ASTNode(NodeType.VARIABLE, random.choice(self.terminals))
        else:
            return ASTNode(NodeType.CONSTANT, random.choice(CONSTANTS))

    def _random_function(self) -> Tuple[str, Dict, str]:
        r = random.random()
        if r < 0.25:
            return random.choice(UNARY_OPS), {}, 'unary'
        elif r < 0.55:
            return random.choice(BINARY_OPS), {}, 'binary'
        else:
            prim_name, param_key, param_vals = random.choice(self.primitives)
            params = {}
            if param_key and param_vals:
                params[param_key] = random.choice(param_vals)
            return prim_name, params, 'primitive'

    def _grow_random_tree(self, depth: int) -> ASTNode:
        if depth <= 1 or random.random() < 0.25:
            return self._random_terminal()
        fn_name, params, fn_type = self._random_function()
        if fn_type == 'unary':
            return ASTNode(NodeType.UNARY, fn_name, [self._grow_random_tree(depth - 1)])
        elif fn_type == 'binary':
            return ASTNode(NodeType.FUNCTION, fn_name,
                           [self._grow_random_tree(depth - 1), self._grow_random_tree(depth - 1)])
        else:
            if fn_name == 'correlation':
                return ASTNode(NodeType.FUNCTION, fn_name,
                               [self._grow_random_tree(depth - 1), self._grow_random_tree(depth - 1)], params)
            return ASTNode(NodeType.FUNCTION, fn_name,
                           [self._grow_random_tree(depth - 1)], params)

    # ── 适应度评估 ──

    def _evaluate_individual(self, ind: Individual) -> float:
        """委托给 FitnessCalculator"""
        try:
            factor_values = evaluate_node(ind.tree, self.data)
        except Exception as e:
            if ind.is_seed:
                logger.warning(f"[GP Eval] 种子求值异常: {e}")
            return -999.0
        # [修复] 2026-06-01 拦截变异产生的 NaN/Inf，防止适应度计算错误
        if not np.isfinite(factor_values).all():
            return -999.0
        if np.allclose(factor_values, factor_values.flat[0], atol=1e-10):
            return -999.0  # 常数因子无区分度
        fitness = self.fitness_calc.compute(factor_values)
        # [修复] 2026-06-01 用 > -998.0 替代 <= -999，避免 float vs int 比较
        if fitness < -998.0:
            return -999.0
        ind.depth = FactorExpression._calc_depth(ind.tree)
        ind.node_count = FactorExpression._count_nodes(ind.tree)
        ind.expression_str = ind.tree.to_str()
        # Complexity penalty
        penalty = self.parsimony_penalty * ind.node_count
        ind.fitness = fitness * (1.0 - penalty)
        return ind.fitness

    def _evaluate_population(self):
        # [修复] 2026-06-01 只评估未初始化的个体 (fitness==-999.0)
        # 去掉 ==0.0 检查，因为有效的个体可能 fitness=0 (如 Sharpe=0)
        for ind in self.population:
            if ind.fitness == -999.0:
                self._evaluate_individual(ind)

    # ── 遗传操作 ──

    def _tournament_select(self) -> Individual:
        candidates = random.sample(self.population,
                                   min(self.tournament_size, len(self.population)))
        return max(candidates, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        tree1 = _clone_tree(parent1.tree)
        nodes1 = _collect_nodes(tree1)
        nodes2 = _collect_nodes(parent2.tree)
        if not nodes1 or not nodes2:
            return Individual(tree1, generation=parent1.generation + 1,
                              expression_str=tree1.to_str())
        n1 = random.choice(nodes1)
        n2 = random.choice(nodes2)
        new_depth = _calc_depth_after_swap(tree1, n1, n2)
        if new_depth > self.max_depth:
            return Individual(tree1, generation=parent1.generation + 1,
                              expression_str=tree1.to_str())
        _replace_node(tree1, n1, _clone_tree(n2))
        # [修复] 2026-06-01 交叉后立即记录表达式，避免空字符串
        return Individual(tree1, generation=parent1.generation + 1,
                          expression_str=tree1.to_str())

    def _mutate(self, individual: Individual) -> Individual:
        tree = _clone_tree(individual.tree)
        nodes = _collect_nodes(tree)
        if not nodes:
            return Individual(tree, generation=individual.generation + 1,
                              expression_str=tree.to_str())
        target = random.choice(nodes)
        replacement = self._grow_random_tree(random.randint(1, min(self.max_depth, 4)))
        _replace_node(tree, target, replacement)
        # [修复] 2026-06-01 变异后立即记录表达式
        return Individual(tree, generation=individual.generation + 1,
                          expression_str=tree.to_str())

    # ── 主循环 ──

    def run(self, verbose: bool = True) -> 'GPEngine':
        self._initialize_population()
        for gen in range(self.generations):
            self._evaluate_population()
            gen_best = max(self.population, key=lambda x: x.fitness)
            gen_stats = {
                'generation': gen,
                'best_fitness': gen_best.fitness,
                'best_depth': gen_best.depth,
                'best_expression': gen_best.expression_str,
                'avg_fitness': np.mean([i.fitness for i in self.population if i.fitness > -999]),
                'valid_count': sum(1 for i in self.population if i.fitness > -999),
            }
            self.history.append(gen_stats)
            if verbose and gen % 5 == 0:
                logger.info(f"Gen {gen:3d} | best_f={gen_best.fitness:.3f} "
                            f"depth={gen_best.depth} valid={gen_stats['valid_count']}")
            if self.best_individual is None or gen_best.fitness > self.best_individual.fitness:
                self.best_individual = gen_best
            # [修复] 2026-06-01 最后一代不繁殖，避免循环后重复评估
            if gen < self.generations - 1:
                self.population = self._next_generation(gen)
        return self

    def best(self) -> Optional[Individual]:
        return self.best_individual

    def top(self, n: int = 10) -> List[Individual]:
        return sorted([i for i in self.population if i.fitness > -999],
                      key=lambda x: x.fitness, reverse=True)[:n]

    def _initialize_population(self):
        self.population = []
        random_count = self.population_size - len(self.seed_expressions)
        for _ in range(max(random_count, self.population_size // 2)):
            tree = self._grow_random_tree(random.randint(2, self.max_depth))
            self.population.append(Individual(tree, generation=0))
        for expr_str in self.seed_expressions:
            try:
                fe = FactorExpression(expr_str)
                ind = Individual(fe.ast, generation=0, is_seed=True)
                ind.depth = fe.depth
                ind.node_count = fe.node_count
                ind.expression_str = expr_str
                self.population.append(ind)
            except Exception as e:
                logger.warning(f"种子表达式解析失败: {expr_str} ({e})")
        if len(self.population) > self.population_size:
            self.population = self.population[:self.population_size]

    def _next_generation(self, gen: int) -> List[Individual]:
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_pop = []
        for i in range(self.elite_count):
            elite = sorted_pop[i]
            new_pop.append(Individual(
                _clone_tree(elite.tree), fitness=elite.fitness,
                expression_str=elite.expression_str, generation=gen + 1,
                depth=elite.depth, node_count=elite.node_count))
        while len(new_pop) < self.population_size - self.random_inject_count:
            r = random.random()
            if r < self.crossover_prob:
                child = self._crossover(self._tournament_select(), self._tournament_select())
            elif r < self.crossover_prob + self.mutation_prob:
                child = self._mutate(self._tournament_select())
            else:
                parent = self._tournament_select()
                child = Individual(_clone_tree(parent.tree), generation=gen + 1)
            new_pop.append(child)
        for _ in range(self.random_inject_count):
            new_pop.append(Individual(
                self._grow_random_tree(random.randint(2, self.max_depth)),
                generation=gen + 1))
        return new_pop[:self.population_size]

    def report(self) -> str:
        if not self.history:
            return "尚未运行 GP 搜索"
        lines = ["=" * 60, "GP 符号回归搜索报告", "=" * 60,
                 f"种群: {self.population_size}  代数: {self.generations}  最大深度: {self.max_depth}", ""]
        if self.best_individual:
            b = self.best_individual
            lines += [f"最优: fitness={b.fitness:.3f}, depth={b.depth}, nodes={b.node_count}",
                      f"表达式: {b.expression_str}", ""]
        lines += [f"{'Gen':>4}  {'Best_F':>8}  {'Depth':>6}  {'Valid':>6}",
                  "-" * 35]
        for s in self.history:
            if s['generation'] % 5 == 0 or s['generation'] == self.history[-1]['generation']:
                lines.append(f"{s['generation']:4d}  {s['best_fitness']:8.3f}  "
                             f"{s['best_depth']:6d}  {s['valid_count']:6d}")
        return '\n'.join(lines)


# ── GPEngine 树操作辅助 ──

def _clone_tree(node: ASTNode) -> ASTNode:
    children = [_clone_tree(c) for c in node.children]
    return ASTNode(node.node_type, node.value, children, dict(node.params))


def _collect_nodes(node: ASTNode) -> List[ASTNode]:
    result = []
    if node.children:
        result.append(node)
    for child in node.children:
        result += _collect_nodes(child)
    return result


def _calc_depth_after_swap(root: ASTNode, old_node: ASTNode, new_node: ASTNode) -> int:
    new_root = _clone_tree(root)
    _replace_node(new_root, old_node, _clone_tree(new_node))
    return FactorExpression._calc_depth(new_root)


def _replace_node(root: ASTNode, target: ASTNode, replacement: ASTNode) -> bool:
    if root is target:
        root.node_type = replacement.node_type
        root.value = replacement.value
        root.children = replacement.children
        root.params = dict(replacement.params)
        return True
    for child in root.children:
        if _replace_node(child, target, replacement):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: FactorDiscoveryEngine — 迭代发现流水线
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DiscoveryReport:
    """发现报告"""
    rounds: List[Dict] = field(default_factory=list)
    total_discovered: int = 0
    library_size: int = 0
    start_time: str = ''
    end_time: str = ''

    def to_dataframe(self) -> pd.DataFrame:
        if not self.rounds:
            return pd.DataFrame()
        return pd.DataFrame(self.rounds)

    def __repr__(self):
        return f"DiscoveryReport(rounds={len(self.rounds)}, total_discovered={self.total_discovered})"


class FactorDiscoveryEngine:
    """因子发现引擎 — 迭代发现流水线

    自动执行多轮 GP 探索，每轮：
    1. 从 formulas.py 和 FactorLibrary 收集种子
    2. 用选定的适应度模式运行 GP
    3. 对 Top N 候选做 Pipeline 验证
    4. 通过验证的因子注册到 FactorLibrary
    5. 下一轮用更大的种子池继续探索

    Example:
        >>> engine = FactorDiscoveryEngine(data, returns, seed_formulas=ALPHA101)
        >>> report = engine.run_pipeline(
        ...     rounds=[
        ...         {'mode': 'icir', 'generations': 30, 'top_n': 50},
        ...         {'mode': 'sharpe', 'generations': 50, 'top_n': 30,
        ...          'freq': 'ME', 'val_top_n': 5},
        ...         {'mode': 'multi_freq', 'generations': 30, 'top_n': 20,
        ...          'freq': 'W', 'val_top_n': 3},
        ...     ]
        ... )
        >>> print(f"Found {engine.library.size()} factors")

    [新增] 2026-06-01 v3 核心
    """

    def __init__(self, data: Dict[str, np.ndarray],
                 returns: pd.DataFrame,
                 future_returns: pd.DataFrame = None,
                 seed_formulas: Dict[str, str] = None,
                 cost_rate: float = 0.0,
                 seed_top_n: int = 100,
                 random_seed: int = None,
                 save_dir: str = None,
                 custom_terminals: List[str] = None):  # [新增] 2026-06-05
        self.data = data
        self.returns = returns
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns = self.returns.copy()
            self.returns.index = pd.to_datetime(self.returns.index)
        self.future_returns = future_returns or self.returns.shift(-1)
        self.cost_rate = cost_rate
        self.seed_top_n = seed_top_n
        self.random_seed = random_seed
        self.save_dir = save_dir  # [新增] 2026-06-04 GP 结果自动持久化目录
        self.custom_terminals = custom_terminals  # [新增] 2026-06-05

        # Seed formulas
        self.seed_formulas = seed_formulas or {}

        # Factor library
        self.library = FactorLibrary()
        self._register_seed_formulas()

        # Report
        self.report = DiscoveryReport()
        self.report.start_time = datetime.now().isoformat()

    def _register_seed_formulas(self):
        """将种子公式注册到 FactorLibrary"""
        if not self.seed_formulas:
            return
        entries = []
        for aid, expr in self.seed_formulas.items():
            entries.append(LibraryEntry(
                alpha_id=aid, expression=expr,
                category=FactorCategory.CUSTOM,
                source='formula', discovered_at=0))
        n = self.library.register_batch(entries)
        logger.info(f"种子公式入库: {n} 个")

    def run_pipeline(self, rounds: List[Dict], verbose: bool = True) -> DiscoveryReport:
        """执行多轮因子发现

        Args:
            rounds: 每轮配置列表，每项包含：
                - mode: 'icir' | 'sharpe' | 'multi_freq'
                - generations: GP 代数
                - population_size: 种群大小（可选）
                - top_n: 入库的 Top N 数量
                - max_depth: 最大树深（可选）
                - freq: 验证阶段调仓频率（可选，默认 'ME'）
                - val_top_n: 验证阶段持仓数（可选，默认 3）
            verbose: 是否打印进度

        Returns:
            DiscoveryReport
        """
        self.report.rounds = []
        self.report.total_discovered = 0

        from .backtest import FactorPipeline, TopNEqualWeight, parse_scheduler

        for round_idx, config in enumerate(rounds):
            mode_str = config.get('mode', 'icir')
            mode = FitnessMode(mode_str)
            generations = config.get('generations', 30)
            pop_size = config.get('population_size', 300)
            max_depth = config.get('max_depth', 8)
            top_n = config.get('top_n', 50)
            # [重构] 2026-06-01 调度参数可配置，不再硬编码
            val_freq = config.get('freq', 'ME')
            val_top_n = config.get('val_top_n', 3)
            val_scheduler = parse_scheduler(val_freq)
            val_allocator = TopNEqualWeight(val_top_n)

            if verbose:
                print(f"\n{'='*50}")
                print(f"Round {round_idx+1}/{len(rounds)}: mode={mode_str}, "
                      f"gen={generations}, pop={pop_size}, max_depth={max_depth}, "
                      f"val_freq={val_freq}, val_top_n={val_top_n}")
                print(f"{'='*50}")

            # 1. Collect seeds from library
            seeds = self.library.seed_expressions(self.seed_top_n, sort_by='fitness')
            if verbose:
                print(f"种子数: {len(seeds)} (from library size={self.library.size()})")

            # 2. Build fitness calculator (透传调度参数)
            fitness_calc = make_fitness_calculator(
                mode, self.data, self.future_returns, self.returns, self.cost_rate,
                scheduler=val_scheduler, allocator=val_allocator)

            # 3. Run GP
            # [新增] 2026-06-05 透传 custom_terminals，使 GP 能探索自定义变量
            gp = GPEngine(
                self.data, self.future_returns,
                fitness_calculator=fitness_calc,
                returns=self.returns,
                config={'population_size': pop_size, 'generations': generations,
                        'max_depth': max_depth},
                seed_expressions=seeds,
                random_seed=self.random_seed,
                custom_terminals=self.custom_terminals,
            )
            gp.run(verbose=verbose)

            # 4. Take top candidates
            candidates = gp.top(top_n)
            if verbose:
                print(f"\nGP 完成，Top {top_n} 候选: "
                      f"best_fitness={candidates[0].fitness:.3f}")

            # 5. Pipeline validation for each candidate
            validated = []
            for i, ind in enumerate(candidates[:top_n]):
                try:
                    expr_str = ind.expression_str
                    FactorExpression(expr_str)  # Re-parse to verify

                    factor_values = evaluate_node(ind.tree, self.data)
                    T, N = factor_values.shape[:2]
                    fv_df = pd.DataFrame(factor_values, index=self.returns.index[:T],
                                         columns=self.returns.columns[:N])
                    # [重构] 2026-06-01 使用可配置的调度器和分配器
                    pipeline = FactorPipeline(self.returns, val_scheduler,
                                              val_allocator, self.cost_rate)
                    bt = pipeline.evaluate(fv_df)
                    validated.append((ind, bt.sharpe_ratio, expr_str))
                except Exception as e:
                    logger.debug(f"候选 {i} 验证失败: {e}")

            if verbose:
                print(f"Pipeline 验证通过: {len(validated)}/{len(candidates[:top_n])}")


            # 6. Register to library
            new_count = 0
            for ind, sharpe, expr_str in validated[:top_n]:
                entry = LibraryEntry(
                    alpha_id=f'gp_r{round_idx+1}_{new_count:03d}',
                    expression=expr_str,
                    category=FactorCategory.CUSTOM,
                    fitness=ind.fitness,
                    sharpe=sharpe,
                    discovered_at=round_idx + 1,
                    source='gp',
                )
                if self.library.register(entry):
                    new_count += 1

            # 7. Record round stats
            round_stats = {
                'round': round_idx + 1,
                'mode': mode_str,
                'generations': generations,
                'val_freq': val_freq,
                'val_top_n': val_top_n,
                'best_fitness': round(candidates[0].fitness, 3) if candidates else -999,
                'best_expression': candidates[0].expression_str if candidates else '',
                'validated': len(validated),
                'registered': new_count,
                'library_size': self.library.size(),
            }
            self.report.rounds.append(round_stats)
            self.report.total_discovered += new_count

            # [新增] 2026-06-04 每轮结束后自动持久化到 discovered/ 目录
            if self.save_dir and new_count > 0:
                import os as _os
                _os.makedirs(self.save_dir, exist_ok=True)
                save_path = _os.path.join(self.save_dir, f'gp_round_{round_idx+1:03d}.json')
                self.library.save(save_path)
                logger.info(f"因子库自动保存: {save_path} ({self.library.size()} 条)")
                if verbose:
                    print(f"已保存到: {save_path}")

            if verbose:
                print(f"入库: {new_count} 个新因子, 因子库总量: {self.library.size()}")
                if candidates:
                    print(f"最优表达式: {candidates[0].expression_str[:100]}")

        self.report.end_time = datetime.now().isoformat()
        self.report.library_size = self.library.size()

        if verbose:
            print(f"\n{'='*50}")
            print(f"发现完成！总计: {self.report.total_discovered} 个新因子")
            print(f"因子库总量: {self.library.size()}")
            print(f"{'='*50}")

        return self.report

    def run_single(self, mode: str = 'icir', generations: int = 30,
                   population_size: int = 300, max_depth: int = 8,
                   top_n: int = 50, freq: str = 'ME', val_top_n: int = 3,
                   verbose: bool = True) -> DiscoveryReport:
        """单轮快速发现（便捷接口）
        
        Args:
            freq: 验证阶段调仓频率，默认 'ME'
            val_top_n: 验证阶段持仓数，默认 3
        """
        return self.run_pipeline([{
            'mode': mode, 'generations': generations,
            'population_size': population_size, 'max_depth': max_depth,
            'top_n': top_n, 'freq': freq, 'val_top_n': val_top_n,
        }], verbose=verbose)


__all__ = [
    'FitnessMode', 'FitnessCalculator',
    'ICIRFitness', 'SharpeFitness', 'MultiFreqFitness', 'WQBFitness',
    'make_fitness_calculator',
    'Individual', 'GPEngine',
    'DiscoveryReport', 'FactorDiscoveryEngine',
    'DEFAULT_GP_CONFIG', 'TERMINALS', 'SAFE_GP_TERMINALS', 'PRIMITIVE_WITH_PARAMS',
]
