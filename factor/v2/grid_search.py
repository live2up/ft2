"""
因子参数网格搜索——Phase A 核心工具

设计思路：
--------
1. 保留 V1 网格搜索的完整功能，作为因子探索的基线工具
2. 接入 V2 新基建：Pipeline（回测评估替代纯 IC）+ Scheduler（频率维度）+ Allocator（权重策略）
3. 搜索空间：{因子参数} × {调仓频率} × {TopN} → 三维网格
4. 评判标准从 IC 升级为 Sharpe（但保留 IC 作为辅助指标）

使用方式：
--------
>>> from factor.v2.grid_search import FactorGridSearch, FactorGridConfig
>>> config = FactorGridConfig(
...     factor_class=MomentumFactor,
...     param_grid={'lookback': [20, 40, 60, 80, 120]},
...     freq_grid=['ME', 'W', '5D'],
...     topn_grid=[5, 10, 15],
...     allocator_type='equal',
... )
>>> gs = FactorGridSearch(config, data, returns)
>>> results = gs.run()         # 返回 GridSearchResult 列表
>>> best = gs.best()           # 最优组合
>>> gs.to_dataframe()          # 汇总为 DataFrame

与 V1 的关系：
------------
V1 中不存在独立的网格搜索模块，Phase A 逻辑散布在脚本中。
V2 将其形式化为独立模块，直接输出可排序、可对比的结果。

依赖说明：
--------
依赖 .scheduler、.allocator、.pipeline、factor.base（Factor 基类）
"""

import itertools
import time
import warnings
from typing import Dict, List, Optional, Any, Type, Union, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .base import Factor, FactorFrequency
from .validator import FactorValidator

from .scheduler import (
    RebalanceScheduler,
    FixedScheduler,
    IntervalScheduler,
    recommend_scheduler_from_decay,
)
from .allocator import WeightAllocator, TopNEqualWeight
from .pipeline import FactorPipeline, BacktestResult
from .cache_store import FactorCacheStore

# [修复] 2026-05-20 移除全局 warnings.filterwarnings('ignore')
# 旧实现在模块级全局抑制警告，会隐藏 NaN/数据对齐等重要诊断信息
# 新实现改为在 run() 内部用 catch_warnings 局部抑制


@dataclass
class FactorGridConfig:
    """因子网格搜索配置
    
    定义一次网格搜索的全部参数空间。
    """
    # 因子定义（必填）
    factor_class: Type[Factor] = None     # 因子类（继承自 Factor）
    
    # 参数网格
    param_grid: Dict[str, List[Any]] = field(default_factory=dict)
    # 例: {'lookback': [20, 40, 60, 80, 120]}
    
    # 频率网格
    freq_grid: List[str] = field(default_factory=lambda: ['ME'])
    # 例: ['ME', 'W', '5D']
    
    # TopN 网格（用于 TopNEqualWeight）
    topn_grid: List[int] = field(default_factory=lambda: [5])
    
    # 分配器类型
    allocator_type: str = 'equal'   # 'equal' | 'proportional'
    
    # Pipeline 参数
    cost_rate: float = 0.001       # 单边交易成本
    rf_annual: float = 0.0         # 无风险利率
    
    # 因子计算参数
    factor_data_fields: List[str] = field(default_factory=lambda: ['close'])
    # 传递给 Factor.calculate() 的 data 中需要的字段
    
    # 是否计算 IC 辅助指标
    calc_ic: bool = True
    
    # [新增] 2026-05-20 OOS 样本外验证比例
    # 0.0 = 全量 in-sample（无样本外），0.3 = 后30%数据留作验证
    # 设为 0.0 可恢复旧行为（纯探索模式）
    oos_ratio: float = 0.3
    
    # 缓存配置
    cache_dir: Optional[str] = None   # 缓存目录，None 表示不使用缓存
    
    # 因子名称前缀（用于缓存和结果标识）
    name_prefix: str = ''


@dataclass
class GridSearchResult:
    """单次网格搜索的评估结果
    
    包含因子参数、调度器、分配器配置和对应的回测绩效。
    """
    # 因子信息
    factor_name: str = ''
    factor_params: Dict[str, Any] = field(default_factory=dict)
    
    # 搜索维度
    frequency: str = ''           # 调仓频率标识
    top_n: int = 5               # 持仓数量
    
    # 回测绩效
    sharpe_ratio: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    total_return: float = 0.0
    annual_volatility: float = 0.0
    
    # IC 指标（辅助）
    ic_mean: float = 0.0
    ir: float = 0.0
    half_life: Optional[float] = None  # IC 半衰期（交易日）
    
    # 交易信息
    n_rebalances: int = 0
    avg_turnover: float = 0.0
    win_rate: float = 0.0
    
    # [新增] 2026-05-20 OOS 样本外绩效
    oos_sharpe_ratio: float = 0.0
    oos_annual_return: float = 0.0
    oos_max_drawdown: float = 0.0
    is_sharpe_ratio: float = 0.0     # IS Sharpe（与 sharpe_ratio 重复时使用）
    
    # 排序键（默认用 OOS Sharpe，无 OOS 时回退 IS Sharpe）
    rank_score: float = 0.0
    
    def __repr__(self) -> str:
        return (f"GridSearchResult({self.factor_name}, "
                f"freq={self.frequency}, topn={self.top_n}, "
                f"Sharpe={self.sharpe_ratio:.2f})")


class FactorGridSearch:
    """因子参数网格搜索编排器
    
    遍历 {参数 × 频率 × TopN} 三维网格，对每个组合执行：
    1. 创建因子实例 → 计算因子值
    2. Pipeline.evaluate() → 回测绩效
    3. FactorValidator → IC/IR 辅助指标
    4. 汇总排序输出
    
    这是 V1 Phase A 的 V2 升级版，是用户当前的主力探索工具。
    """

    def __init__(self,
                 config: FactorGridConfig,
                 data: Dict[str, pd.DataFrame],
                 returns: pd.DataFrame,
                 future_returns: Optional[pd.DataFrame] = None):
        """初始化网格搜索
        
        Args:
            config: 网格搜索配置
            data: OHLCV 数据字典，key=字段名，value=DataFrame（index=日期，columns=标的）
            returns: 各标的日收益率 DataFrame，用于 Pipeline 回测
            future_returns: 未来收益率 DataFrame，用于 IC 计算。
                           None 时自动从 returns.shift(-1) 生成。
        """
        self.config = config
        self.data = data
        self.returns = returns

        # 确保 index 是 DatetimeIndex
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns = self.returns.copy()
            self.returns.index = pd.to_datetime(self.returns.index)

        # 未来收益率（用于 IC 计算）
        if future_returns is not None:
            # [修复] 2026-05-20 防御性 copy()，避免外部修改影响内部
            self.future_returns = future_returns.copy()
        else:
            # [新增] 2026-05-20 09-20 自动从 returns 生成 future_returns
            # shift(-1) 将下期收益对齐到当期因子值日期
            self.future_returns = self.returns.shift(-1)

        # 缓存存储
        if config.cache_dir:
            self.cache_store = FactorCacheStore(config.cache_dir)
        else:
            self.cache_store = None

        # 结果存储
        self.results: List[GridSearchResult] = []

    def run(self,
            sort_by: str = 'sharpe_ratio',
            ascending: bool = False,
            verbose: bool = True) -> List[GridSearchResult]:
        """执行网格搜索
        
        Args:
            sort_by: 排序字段，默认 'sharpe_ratio'
            ascending: 是否升序（默认 False，即 Sharpe 高的在前）
            verbose: 是否打印进度
            
        Returns:
            List[GridSearchResult]: 排序后的结果列表
        """
        start_time = time.time()

        # 1) 展开参数网格（笛卡尔积）
        param_combos = self._expand_param_grid()

        # 2) 展开频率→调度器映射
        scheduler_map = self._build_scheduler_map()

        # 总组合数 = param_combos × freqs × topns
        n_param = len(param_combos)
        n_freq = len(self.config.freq_grid)
        n_topn = len(self.config.topn_grid)
        total = n_param * n_freq * n_topn

        # [新增] 2026-05-20 OOS 样本外验证：按时间切分 IS 和 OOS 区间
        oos_ratio = self.config.oos_ratio
        n_total_rows = len(self.returns)
        if oos_ratio > 0 and n_total_rows > 120:
            split_idx = int(n_total_rows * (1.0 - oos_ratio))
            split_date = self.returns.index[split_idx]
            returns_is = self.returns.iloc[:split_idx]
            returns_oos = self.returns.iloc[split_idx:]
        else:
            split_date = None
            returns_is = self.returns
            returns_oos = None

        if verbose:
            print(f"[网格搜索] 因子: {self.config.factor_class.__name__}")
            print(f"  参数组合: {n_param}, 频率: {n_freq}, TopN: {n_topn}")
            print(f"  总计: {total} 次回测")
            print(f"  数据区间: {self.returns.index[0].date()} → {self.returns.index[-1].date()}")
            if split_date:
                print(f"  IS/OOS 切分: {split_date.date()}  (OOS={oos_ratio:.0%})")
            else:
                print(f"  OOS: 关闭 (全量 in-sample)")

        count = 0
        self.results = []

        for combo_idx, params in enumerate(param_combos):
            # 3) 计算因子值（只算一次，所有频率/TopN 共享）
            param_key = self._params_to_key(params)
            factor_values = self._get_factor_values(params, param_key)

            if factor_values is None or factor_values.shape[0] < 60:
                if verbose:
                    print(f"  [跳过] {param_key}: 因子值不足")
                continue

            # 4) 计算 IC 辅助指标（只算一次，只用 IS 区间避免泄漏）
            if self.config.calc_ic:
                fv_is = factor_values.loc[factor_values.index <= split_date] if split_date else factor_values
                ic_mean, ir_value, half_life = self._calc_ic_metrics(fv_is)
            else:
                ic_mean, ir_value, half_life = 0.0, 0.0, None

            # 5) 遍历频率和 TopN
            for freq in self.config.freq_grid:
                scheduler = scheduler_map[freq]

                for top_n in self.config.topn_grid:
                    count += 1
                    if verbose and total <= 50:
                        print(f"  [{count}/{total}] {param_key} freq={freq} topn={top_n}...", end=' ')

                    try:
                        # 构建 Allocator
                        if self.config.allocator_type == 'proportional':
                            from .allocator import ScoreProportional
                            allocator = ScoreProportional()
                        else:
                            allocator = TopNEqualWeight(top_n)

                        # IS 评估
                        pipeline_is = FactorPipeline(
                            returns=returns_is,
                            scheduler=scheduler,
                            allocator=allocator,
                            cost_rate=self.config.cost_rate,
                            rf_annual=self.config.rf_annual,
                        )
                        bt_is = pipeline_is.evaluate(
                            factor_values.loc[:split_date] if split_date else factor_values
                        )

                        # OOS 评估
                        oos_sharpe = 0.0
                        oos_return = 0.0
                        oos_mdd = 0.0
                        if returns_oos is not None:
                            try:
                                pipeline_oos = FactorPipeline(
                                    returns=returns_oos,
                                    scheduler=scheduler,
                                    allocator=allocator,
                                    cost_rate=self.config.cost_rate,
                                    rf_annual=self.config.rf_annual,
                                )
                                bt_oos = pipeline_oos.evaluate(
                                    factor_values.loc[split_date:]
                                )
                                oos_sharpe = bt_oos.sharpe_ratio
                                oos_return = bt_oos.annual_return
                                oos_mdd = bt_oos.max_drawdown
                            except Exception:
                                pass

                        # 汇总结果
                        result = GridSearchResult(
                            factor_name=self._make_factor_name(params),
                            factor_params=params,
                            frequency=freq,
                            top_n=top_n,
                            sharpe_ratio=oos_sharpe if oos_sharpe != 0 else bt_is.sharpe_ratio,
                            annual_return=oos_return if oos_return != 0 else bt_is.annual_return,
                            max_drawdown=oos_mdd if oos_mdd != 0 else bt_is.max_drawdown,
                            calmar_ratio=bt_is.calmar_ratio,
                            total_return=bt_is.total_return,
                            annual_volatility=bt_is.annual_volatility,
                            ic_mean=ic_mean,
                            ir=ir_value,
                            half_life=half_life,
                            n_rebalances=bt_is.n_rebalances,
                            avg_turnover=bt_is.avg_turnover,
                            win_rate=bt_is.win_rate,
                            oos_sharpe_ratio=oos_sharpe,
                            oos_annual_return=oos_return,
                            oos_max_drawdown=oos_mdd,
                            is_sharpe_ratio=bt_is.sharpe_ratio,
                            rank_score=oos_sharpe if oos_sharpe != 0 else bt_is.sharpe_ratio,
                        )
                        self.results.append(result)

                        if verbose and total <= 50:
                            is_str = f"IS={bt_is.sharpe_ratio:.2f}"
                            oos_str = f" OOS={oos_sharpe:.2f}" if oos_sharpe != 0 else ""
                            print(f"{is_str}{oos_str}")

                    except Exception as e:
                        if verbose:
                            print(f"失败: {e}")
                        continue

        # 6) 排序：默认按 OOS Sharpe；无 OOS 时回退 IS Sharpe
        default_sort = 'oos_sharpe_ratio' if split_date else 'sharpe_ratio'
        if sort_by == 'sharpe_ratio' and split_date:
            sort_by = default_sort
        if sort_by in GridSearchResult.__dataclass_fields__:
            self.results.sort(key=lambda r: getattr(r, sort_by), reverse=not ascending)
            for r in self.results:
                r.rank_score = getattr(r, sort_by)
        else:
            self.results.sort(key=lambda r: getattr(r, default_sort), reverse=True)
            for r in self.results:
                r.rank_score = getattr(r, default_sort)

        elapsed = time.time() - start_time
        if verbose:
            sort_label = 'OOS Sharpe' if split_date else 'IS Sharpe'
            print(f"\n[完成] {len(self.results)} 个有效结果，耗时 {elapsed:.1f}s")
            if self.results:
                best = self.results[0]
                oos_info = f" OOS_Sharpe={best.oos_sharpe_ratio:.2f}" if split_date else ""
                print(f"  最优({sort_label}): {best.factor_name} freq={best.frequency} "
                      f"topn={best.top_n} Sharpe={best.sharpe_ratio:.2f}{oos_info}")

        return self.results

    def best(self, sort_by: str = 'oos_sharpe_ratio') -> Optional[GridSearchResult]:
        """获取最优结果
        
        Args:
            sort_by: 排序字段
            
        Returns:
            Optional[GridSearchResult]: 最优结果，没有结果时返回 None
        """
        if not self.results:
            return None

        if sort_by in GridSearchResult.__dataclass_fields__:
            sorted_results = sorted(self.results,
                                    key=lambda r: getattr(r, sort_by),
                                    reverse=True)
            return sorted_results[0]
        return self.results[0]

    def top(self, n: int = 10, sort_by: str = 'sharpe_ratio') -> List[GridSearchResult]:
        """获取前 N 个最优结果
        
        Args:
            n: 返回数量
            sort_by: 排序字段
            
        Returns:
            List[GridSearchResult]: 前 N 个结果
        """
        if sort_by in GridSearchResult.__dataclass_fields__:
            sorted_results = sorted(self.results,
                                    key=lambda r: getattr(r, sort_by),
                                    reverse=True)
        else:
            sorted_results = self.results
        return sorted_results[:n]

    def to_dataframe(self) -> pd.DataFrame:
        """将结果汇总为 DataFrame
        
        Returns:
            pd.DataFrame: 结果表格
        """
        if not self.results:
            return pd.DataFrame()

        rows = []
        for r in self.results:
            rows.append({
                '因子': r.factor_name,
                '频率': r.frequency,
                'TopN': r.top_n,
                'Sharpe': round(r.sharpe_ratio, 3),
                '年化收益': f'{r.annual_return:.1%}',
                '最大回撤': f'{r.max_drawdown:.1%}',
                'Calmar': round(r.calmar_ratio, 2),
                'IC均值': round(r.ic_mean, 4),
                'IR': round(r.ir, 2),
                '半衰期': round(r.half_life, 1) if r.half_life else '-',
                '调仓次数': r.n_rebalances,
                '平均换手': f'{r.avg_turnover:.1%}',
                '胜率': f'{r.win_rate:.1%}',
            })

        return pd.DataFrame(rows)

    # ================= 内部方法 =================

    def _expand_param_grid(self) -> List[Dict[str, Any]]:
        """展开参数网格为笛卡尔积列表
        
        Returns:
            List[Dict]: 每个元素是一个参数组合字典
        """
        if not self.config.param_grid:
            return [{}]  # 无参数，仅一个空组合

        keys = list(self.config.param_grid.keys())
        values = list(self.config.param_grid.values())

        combos = []
        for combo in itertools.product(*values):
            combos.append(dict(zip(keys, combo)))

        return combos

    def _build_scheduler_map(self) -> Dict[str, RebalanceScheduler]:
        """将频率字符串映射为调度器实例
        
        Returns:
            Dict[str, RebalanceScheduler]: {'ME': FixedScheduler('ME'), ...}
        """
        scheduler_map = {}
        for freq in self.config.freq_grid:
            freq_upper = freq.upper()
            if freq_upper in ('ME', 'W', 'M'):
                scheduler_map[freq] = FixedScheduler(freq_upper)
            elif freq.endswith('D') or freq.endswith('d'):
                days = int(freq.replace('D', '').replace('d', ''))
                scheduler_map[freq] = IntervalScheduler(days)
            else:
                try:
                    days = int(freq)
                    scheduler_map[freq] = IntervalScheduler(days)
                except ValueError:
                    raise ValueError(f"无法识别的频率: '{freq}'")
        return scheduler_map

    def _get_factor_values(self,
                           params: Dict[str, Any],
                           param_key: str) -> Optional[pd.DataFrame]:
        """获取因子值（优先从缓存读取）
        
        Args:
            params: 因子参数字典
            param_key: 参数组合的字符串标识
            
        Returns:
            Optional[pd.DataFrame]: 因子值 DataFrame
        """
        start_date = str(self.returns.index[0].date())
        end_date = str(self.returns.index[-1].date())

        # 尝试缓存
        if self.cache_store:
            lookback = params.get('lookback', 0)
            cached = self.cache_store.load(
                self._make_factor_name(params), lookback, start_date, end_date
            )
            if cached is not None:
                return cached

        # 创建因子实例并计算
        try:
            factor = self._create_factor(params)
            symbols = list(self.returns.columns)
            dates = list(self.returns.index)
            
            # [新增] 2026-05-20 09-20 因子值缓存键使用 data 中的日期范围
            # 确保 FactorCalculator 正确处理 data 和 dates 的对齐
            filtered_data = {}
            for field in self.config.factor_data_fields:
                if field in self.data:
                    df = self.data[field]
                    # 对齐到 returns 的日期和标的
                    aligned = df.reindex(
                        index=self.returns.index, columns=self.returns.columns
                    )
                    filtered_data[field] = aligned

            factor_values = factor.calculate(filtered_data, symbols, dates)

            # 写入缓存
            if self.cache_store and factor_values is not None:
                lookback = params.get('lookback', 0)
                self.cache_store.save(
                    self._make_factor_name(params), lookback,
                    start_date, end_date, factor_values
                )

            return factor_values

        except Exception as e:
            warnings.warn(f"计算因子值失败 [{param_key}]: {e}")
            return None

    def _create_factor(self, params: Dict[str, Any]) -> Factor:
        """根据参数创建因子实例
        
        Args:
            params: 因子参数字典
            
        Returns:
            Factor: 因子实例
        """
        # [新增] 2026-05-20 09-20 因子实例化
        # 通过 factor_class 构造函数传入参数
        # 注意：因子类的 __init__ 应接受这些 kwargs
        try:
            return self.config.factor_class(**params)
        except TypeError:
            # 如果因子类不接受 kwargs，尝试无参构造
            warnings.warn(
                f"因子类 {self.config.factor_class.__name__} 不接受参数 {params}，"
                f"使用无参构造"
            )
            return self.config.factor_class()

    def _make_factor_name(self, params: Dict[str, Any]) -> str:
        """生成因子名称（含参数）
        
        Args:
            params: 因子参数字典
            
        Returns:
            str: 因子名称，如 'Momentum_20D' 或 'Reversal_10D'
        """
        prefix = self.config.name_prefix or self.config.factor_class.__name__

        if not params:
            return prefix

        # 构建参数后缀
        param_parts = []
        for k, v in params.items():
            if isinstance(v, float):
                param_parts.append(f"{k}{v:.0f}" if v == int(v) else f"{k}{v}")
            else:
                param_parts.append(f"{k}{v}")

        name = f"{prefix}_" + '_'.join(param_parts)
        return name

    def _params_to_key(self, params: Dict[str, Any]) -> str:
        """参数字典转为简短 key
        
        Args:
            params: 因子参数字典
            
        Returns:
            str: 参数 key
        """
        parts = []
        for k, v in params.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.1f}" if v != int(v) else f"{k}={int(v)}")
            else:
                parts.append(f"{k}={v}")
        return ','.join(parts) if parts else 'default'

    def _calc_ic_metrics(self,
                         factor_values: pd.DataFrame) -> tuple:
        """计算 IC 指标（均值、IR、半衰期）
        
        Args:
            factor_values: 因子值 DataFrame
            
        Returns:
            tuple: (ic_mean, ir, half_life)
        """
        try:
            # 对齐因子值和未来收益率
            common_dates = factor_values.index.intersection(
                self.future_returns.index
            )
            if len(common_dates) < 20:
                return 0.0, 0.0, None

            fv_aligned = factor_values.loc[common_dates]
            fr_aligned = self.future_returns.loc[common_dates]

            # 对齐 columns
            common_cols = fv_aligned.columns.intersection(fr_aligned.columns)
            fv_aligned = fv_aligned[common_cols]
            fr_aligned = fr_aligned[common_cols]

            validator = FactorValidator(
                factor_values=fv_aligned,
                future_returns=fr_aligned,
            )

            # IC
            ic_result = validator.information_coefficient()
            ic_mean = ic_result.get('mean', 0.0) if isinstance(ic_result, dict) else 0.0

            # IR
            try:
                ir_value = validator.information_ratio()
            except Exception:
                ir_value = 0.0

            # 半衰期
            try:
                decay_result = validator.decay_rate(max_lookforward=20)
                half_life = decay_result.get('half_life', None) if isinstance(decay_result, dict) else None
            except Exception:
                half_life = None

            return float(ic_mean), float(ir_value), half_life

        except Exception as e:
            warnings.warn(f"IC 计算失败: {e}")
            return 0.0, 0.0, None

    def __repr__(self) -> str:
        n = len(self.results)
        return (f"FactorGridSearch(factor={self.config.factor_class.__name__}, "
                f"results={n})")
