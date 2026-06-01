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
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .engine import (
    FactorExpression, ASTNode, NodeType, evaluate_node,
    UNARY_FUNCTIONS, BINARY_FUNCTIONS, PRIMITIVE_FUNCTIONS,
)
from .base import FactorCategory, LibraryEntry, FactorLibrary

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# GP 配置常量
# ═══════════════════════════════════════════════════════════════════════════

TERMINALS = ['close', 'volume', 'open', 'high', 'low', 'amount']
UNARY_OPS = ['abs', 'sqrt', 'log', 'neg']
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'max', 'min']
CONSTANTS = [0.0, -1.0, 1.0, 0.5, 2.0]

PRIMITIVE_WITH_PARAMS = [
    ('ts_rank',       'period', [5, 10, 15, 20, 30, 60]),
    ('ts_zscore',     'period', [10, 20, 30, 60]),
    ('delay',         'period', [1, 5, 10, 20, 60]),
    ('decay_linear',  'period', [5, 10, 20, 30]),
    ('cs_zscore',     'period', [20]),
    ('cs_rank',       None, []),
    ('signed_power',  'exponent', [2.0]),
    ('ts_sum',        'period', [5, 10, 15, 20, 30, 60]),
    ('ts_mean',       'period', [5, 10, 15, 20, 30, 60]),
    ('ts_std',        'period', [10, 20, 30, 60]),
    ('ts_max',        'period', [5, 10, 20, 30, 60]),
    ('ts_min',        'period', [5, 10, 20, 30, 60]),
    ('sma',           'period', [5, 10, 20, 30]),
    ('ts_argmin',     'period', [5, 10, 20]),
    ('ts_argmax',     'period', [5, 10, 20]),
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


class FitnessCalculator:
    """适应度计算器基类

    所有配置参数通过构造注入，避免子类硬编码。
    scheduler / allocator 仅 SHARPE / MULTI_FREQ 模式使用，ICIR 模式忽略。

    [重构] 2026-06-01 支持 scheduler/allocator/freq_list 构造注入
    """

    def __init__(self, data: Dict[str, np.ndarray], future_returns: pd.DataFrame,
                 returns: pd.DataFrame = None, cost_rate: float = 0.0,
                 scheduler=None, allocator=None, freq_list: List[str] = None):
        self.data = data
        self.future_returns = future_returns
        self.returns = returns
        self.cost_rate = cost_rate
        # [重构] 2026-06-01 构造注入，外部可自定义调度器和分配器
        self.scheduler = scheduler
        self.allocator = allocator
        self.freq_list = freq_list or ['ME', 'W', '5D']
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
    """

    def compute(self, factor_values: np.ndarray) -> float:
        if not self._validate(factor_values):
            return -999.0
        T, N = self._shape
        daily_ics = []
        for t in range(T):
            fv = factor_values[t, :]
            rv = self.future_returns.iloc[t].values
            mask = ~np.isnan(fv) & ~np.isnan(rv)
            if mask.sum() < 5:
                continue
            corr = np.corrcoef(fv[mask], rv[mask])[0, 1]
            if not np.isnan(corr):
                daily_ics.append(corr)
        if len(daily_ics) < 30:
            return -999.0
        ic_mean = float(np.mean(daily_ics))
        ic_std = float(np.std(daily_ics, ddof=1))
        if ic_std < 1e-10:
            return -999.0
        icir = ic_mean / ic_std
        return abs(ic_mean) * icir * 100.0


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
    """多频率适应度：取 self.freq_list 中最优 Sharpe

    兼顾多周期稳健性，适合最终筛选阶段。
    self.freq_list 可通过构造注入自定义。

    [重构] 2026-06-01 用注入参数替代硬编码
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
            results = pipeline.compare_frequencies(fv_df, self.freq_list)
            sharpes = [r.sharpe_ratio for r in results.values()]
            return float(max(sharpes)) if sharpes else -999.0
        except Exception as e:
            logger.debug(f"MultiFreqFitness 计算失败: {e}")
            return -999.0


def make_fitness_calculator(mode: FitnessMode, data: Dict[str, np.ndarray],
                            future_returns: pd.DataFrame, returns: pd.DataFrame = None,
                            cost_rate: float = 0.0,
                            scheduler=None, allocator=None,
                            freq_list: List[str] = None) -> FitnessCalculator:
    """工厂函数：根据模式创建适应度计算器

    [重构] 2026-06-01 支持 scheduler/allocator/freq_list 透传
    """
    if mode == FitnessMode.ICIR:
        return ICIRFitness(data, future_returns)
    elif mode == FitnessMode.SHARPE:
        return SharpeFitness(data, future_returns, returns, cost_rate,
                             scheduler=scheduler, allocator=allocator)
    elif mode == FitnessMode.MULTI_FREQ:
        return MultiFreqFitness(data, future_returns, returns, cost_rate,
                                scheduler=scheduler, allocator=allocator,
                                freq_list=freq_list)
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
                 # [重构] 2026-06-01 调度参数，仅 fitness_mode 模式使用
                 scheduler=None, allocator=None, freq_list: List[str] = None,
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
        self.terminals = custom_terminals or TERMINALS
        self.primitives = custom_primitives or PRIMITIVE_WITH_PARAMS
        self.seed_expressions = seed_expressions or []

        # Fitness
        self.cost_rate = cost_rate
        if fitness_calculator is not None:
            self.fitness_calc = fitness_calculator
        elif fitness_mode is not None:
            self.fitness_calc = make_fitness_calculator(
                fitness_mode, data, future_returns, returns, self.cost_rate,
                scheduler=scheduler, allocator=allocator, freq_list=freq_list)
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
                 seed_n: int = 100,
                 random_seed: int = None):
        self.data = data
        self.returns = returns
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns = self.returns.copy()
            self.returns.index = pd.to_datetime(self.returns.index)
        self.future_returns = future_returns or self.returns.shift(-1)
        self.cost_rate = cost_rate
        self.seed_n = seed_n
        self.random_seed = random_seed

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
            verbose: 是否打印进度

        Returns:
            DiscoveryReport
        """
        self.report.rounds = []
        self.report.total_discovered = 0

        from .backtest import FactorPipeline, FixedScheduler, IntervalScheduler, TopNEqualWeight

        def _make_scheduler(freq_str: str):
            """将频率字符串转为 Scheduler 对象"""
            f = freq_str.upper()
            if f in ('ME', 'W', 'M'):
                return FixedScheduler(f)
            if freq_str.upper().endswith('D'):
                return IntervalScheduler(int(freq_str.upper().replace('D', '')))
            return IntervalScheduler(int(freq_str))

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
            val_scheduler = _make_scheduler(val_freq)
            val_allocator = TopNEqualWeight(val_top_n)

            if verbose:
                print(f"\n{'='*50}")
                print(f"Round {round_idx+1}/{len(rounds)}: mode={mode_str}, "
                      f"gen={generations}, pop={pop_size}, max_depth={max_depth}, "
                      f"val_freq={val_freq}, val_top_n={val_top_n}")
                print(f"{'='*50}")

            # 1. Collect seeds from library
            seeds = self.library.seed_expressions(self.seed_n, sort_by='fitness')
            if verbose:
                print(f"种子数: {len(seeds)} (from library size={self.library.size()})")

            # 2. Build fitness calculator (透传调度参数)
            fitness_calc = make_fitness_calculator(
                mode, self.data, self.future_returns, self.returns, self.cost_rate,
                scheduler=val_scheduler, allocator=val_allocator)

            # 3. Run GP
            gp = GPEngine(
                self.data, self.future_returns,
                fitness_calculator=fitness_calc,
                returns=self.returns,
                config={'population_size': pop_size, 'generations': generations,
                        'max_depth': max_depth},
                seed_expressions=seeds,
                random_seed=self.random_seed,
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
    'FitnessMode', 'FitnessCalculator', 'ICIRFitness', 'SharpeFitness', 'MultiFreqFitness',
    'make_fitness_calculator',
    'Individual', 'GPEngine',
    'DiscoveryReport', 'FactorDiscoveryEngine',
    'DEFAULT_GP_CONFIG', 'TERMINALS', 'PRIMITIVE_WITH_PARAMS',
]
