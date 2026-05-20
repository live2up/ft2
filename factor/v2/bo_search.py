"""
贝叶斯优化参数搜索——Phase A 增强选项

设计思路：
--------
1. 作为网格搜索（FactorGridSearch）的增强替代，通过 --method bo 选择
2. 使用 scikit-optimize 的 Gaussian Process 代理模型
3. 将 lookback 从离散 {20,40,60,80,120} 升级为连续空间 [10, 250]
4. TopN 保持离散（整数），频率保持离散（分类）
5. 典型搜索：15 轮 BO ≈ 15 次回测，覆盖穷举不到的参数区间

使用方式：
--------
>>> from factor.v2.bo_search import FactorBOSearch
>>> bos = FactorBOSearch(
...     factor_class=MomentumFactor,
...     lookback_range=(10, 250),
...     freq_candidates=['ME', 'W', '5D'],
...     topn_candidates=[5, 10, 15, 20],
...     data=data,
...     returns=returns,
... )
>>> result = bos.optimize(n_calls=15)
>>> print(f"最优 lookback={result.lookback}, freq={result.frequency}, Sharpe={result.sharpe:.2f}")

与网格搜索的关系：
--------------
- 网格搜索（grid_search.py）：基线工具，穷举覆盖，结果确定
- BO 搜索（本模块）：进阶工具，连续搜索，同等预算覆盖更大空间
- 两者通过 --method grid / --method bo 选择，不互相替代

依赖说明：
--------
scikit-optimize (skopt) 为可选依赖，未安装时退化为随机搜索并给出提示。
"""

import time
import warnings
from typing import Dict, List, Optional, Any, Type, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from factor.base import Factor

from .scheduler import FixedScheduler, IntervalScheduler, RebalanceScheduler
from .allocator import TopNEqualWeight, WeightAllocator
from .pipeline import FactorPipeline
from .cache_store import FactorCacheStore

warnings.filterwarnings('ignore')


@dataclass
class BOSearchResult:
    """BO 搜索结果
    
    包含最优参数组合和对应的回测绩效，以及 BO 搜索过程的记录。
    """
    # 最优参数
    lookback: float = 0.0
    frequency: str = ''
    top_n: int = 5
    allocator_type: str = 'equal'

    # 回测绩效
    sharpe_ratio: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    n_rebalances: int = 0

    # BO 搜索信息
    n_evaluations: int = 0
    search_time: float = 0.0
    convergence_curve: List[float] = field(default_factory=list)  # 每轮最优 Sharpe

    # 因子信息
    factor_name: str = ''
    factor_class_name: str = ''

    def __repr__(self) -> str:
        return (f"BOSearchResult(lookback={self.lookback:.0f}, "
                f"freq={self.frequency}, topn={self.top_n}, "
                f"Sharpe={self.sharpe_ratio:.2f})")


class FactorBOSearch:
    """贝叶斯优化因子参数搜索
    
    用 Gaussian Process 代理模型替代网格搜索的穷举遍历。
    在同等计算预算（15-25 次回测）下覆盖连续参数空间。
    
    典型流程：
    1. 初始随机采样 5 个点 → 回测
    2. 建 GP 代理模型 → 推荐下一个"最优探索"点
    3. 回测新点 → 更新模型 → 重复 10-15 轮
    4. 收敛后输出最优参数组合
    """

    def __init__(self,
                 factor_class: Type[Factor],
                 lookback_range: Tuple[float, float] = (10, 250),
                 freq_candidates: List[str] = None,
                 topn_candidates: List[int] = None,
                 data: Dict[str, pd.DataFrame] = None,
                 returns: pd.DataFrame = None,
                 cost_rate: float = 0.001,
                 n_initial_points: int = 5,
                 cache_dir: Optional[str] = None,
                 factor_data_fields: List[str] = None,
                 name_prefix: str = ''):
        """初始化 BO 搜索
        
        Args:
            factor_class: 因子类（继承自 Factor）
            lookback_range: lookback 连续搜索范围 (min, max)
            freq_candidates: 频率候选列表，默认 ['ME', 'W', '5D']
            topn_candidates: TopN 候选列表，默认 [5, 10, 15, 20]
            data: OHLCV 数据字典
            returns: 日收益率 DataFrame
            cost_rate: 交易成本率
            n_initial_points: 初始随机采样点数量
            cache_dir: 缓存目录
            factor_data_fields: 因子计算所需数据字段
            name_prefix: 因子名称前缀
        """
        self.factor_class = factor_class
        self.lookback_range = lookback_range
        self.freq_candidates = freq_candidates or ['ME', 'W', '5D']
        self.topn_candidates = topn_candidates or [5, 10, 15, 20]
        self.data = data
        self.returns = returns
        self.cost_rate = cost_rate
        self.n_initial_points = n_initial_points
        self.factor_data_fields = factor_data_fields or ['close']
        self.name_prefix = name_prefix or factor_class.__name__

        # 缓存
        if cache_dir:
            self.cache_store = FactorCacheStore(cache_dir)
        else:
            self.cache_store = None

        # 确保 returns index 为 DatetimeIndex
        if self.returns is not None:
            if not isinstance(self.returns.index, pd.DatetimeIndex):
                self.returns = self.returns.copy()
                self.returns.index = pd.to_datetime(self.returns.index)

        # 收敛曲线记录
        self.convergence_curve: List[float] = []
        self.best_params: Dict[str, Any] = {}
        self.best_sharpe = -np.inf

    def optimize(self, n_calls: int = 15,
                 random_state: int = 42,
                 verbose: bool = True) -> BOSearchResult:
        """执行贝叶斯优化搜索
        
        Args:
            n_calls: 总评估次数（含初始采样），默认 15
            random_state: 随机种子
            verbose: 是否打印进度
            
        Returns:
            BOSearchResult: 最优结果
        """
        start_time = time.time()

        # 构建频率和 TopN 的索引映射
        freq_idx_map = {i: f for i, f in enumerate(self.freq_candidates)}
        topn_idx_map = {i: t for i, t in enumerate(self.topn_candidates)}

        # 搜索空间维度：
        # [0]: lookback (连续)
        # [1]: freq_index (离散整数)
        # [2]: topn_index (离散整数)
        dimensions = [
            (self.lookback_range[0], self.lookback_range[1]),  # lookback
            (0, len(self.freq_candidates) - 1),                 # freq 索引
            (0, len(self.topn_candidates) - 1),                 # topn 索引
        ]

        if verbose:
            total_combos = (
                int(self.lookback_range[1] - self.lookback_range[0]) *
                len(self.freq_candidates) *
                len(self.topn_candidates)
            )
            print(f"[BO搜索] 因子: {self.factor_class.__name__}")
            print(f"  lookback: {self.lookback_range}")
            print(f"  频率候选: {self.freq_candidates}")
            print(f"  TopN候选: {self.topn_candidates}")
            print(f"  总评估: {n_calls} 次 (网格穷举需 ~{total_combos} 次)")

        try:
            from skopt import gp_minimize
            from skopt.space import Integer, Real
            from skopt.utils import use_named_args

            # 定义带名称的空间
            space = [
                Real(self.lookback_range[0], self.lookback_range[1], name='lookback'),
                Integer(0, len(self.freq_candidates) - 1, name='freq_idx'),
                Integer(0, len(self.topn_candidates) - 1, name='topn_idx'),
            ]

            @use_named_args(space)
            def objective(lookback, freq_idx, topn_idx) -> float:
                """BO 目标函数：最大化 Sharpe"""
                lookback = int(round(lookback))
                freq = self.freq_candidates[freq_idx]
                top_n = self.topn_candidates[topn_idx]

                sharpe = self._evaluate_single(lookback, freq, top_n)

                # 更新收敛曲线
                if sharpe > self.best_sharpe:
                    self.best_sharpe = sharpe
                    self.best_params = {
                        'lookback': lookback, 'freq': freq, 'top_n': top_n
                    }
                self.convergence_curve.append(self.best_sharpe)

                if verbose:
                    print(f"  [BO] lookback={lookback} freq={freq} topn={top_n} → Sharpe={sharpe:.3f}")

                # BO 最小化负 Sharpe（即最大化 Sharpe）
                return -sharpe

            # 执行 GP 优化
            result = gp_minimize(
                objective,
                space,
                n_calls=n_calls,
                n_initial_points=self.n_initial_points,
                random_state=random_state,
                n_jobs=1,  # 单线程（因子计算共享数据）
                verbose=False,
            )

            # 解析最优参数
            best_lookback = int(round(result.x[0]))
            best_freq = self.freq_candidates[result.x[1]]
            best_topn = self.topn_candidates[result.x[2]]
            best_sharpe = -result.fun

        except ImportError:
            # [新增] 2026-05-20 scikit-optimize 未安装时退化为随机搜索
            warnings.warn(
                "scikit-optimize 未安装，BO 搜索退化为随机搜索。"
                "安装: pip install scikit-optimize"
            )
            best_lookback, best_freq, best_topn, best_sharpe = self._random_search(
                n_calls, freq_idx_map, topn_idx_map, verbose
            )

        # 最终评估最优参数（获取完整指标）
        final_result = self._evaluate_full(best_lookback, best_freq, best_topn)

        elapsed = time.time() - start_time
        if verbose:
            print(f"\n[BO完成] 最优: lookback={best_lookback}, "
                  f"freq={best_freq}, topn={best_topn}, "
                  f"Sharpe={best_sharpe:.2f}, 耗时 {elapsed:.1f}s")

        return BOSearchResult(
            lookback=best_lookback,
            frequency=best_freq,
            top_n=best_topn,
            sharpe_ratio=best_sharpe,
            annual_return=final_result.get('annual_return', 0.0),
            max_drawdown=final_result.get('max_drawdown', 0.0),
            calmar_ratio=final_result.get('calmar_ratio', 0.0),
            n_rebalances=final_result.get('n_rebalances', 0),
            n_evaluations=n_calls,
            search_time=elapsed,
            convergence_curve=self.convergence_curve,
            factor_name=f"{self.name_prefix}_{best_lookback}D",
            factor_class_name=self.factor_class.__name__,
        )

    # ================= 内部方法 =================

    def _evaluate_single(self, lookback: int, freq: str,
                         top_n: int) -> float:
        """评估单个参数组合，返回 Sharpe
        
        Args:
            lookback: 回看天数
            freq: 频率标识
            top_n: 持仓数量
            
        Returns:
            float: Sharpe 比率（失败时返回 -np.inf）
        """
        try:
            result = self._evaluate_full(lookback, freq, top_n)
            sharpe = result.get('sharpe_ratio', -np.inf)
            return float(sharpe)
        except Exception:
            return -np.inf

    def _evaluate_full(self, lookback: int, freq: str,
                       top_n: int) -> Dict[str, float]:
        """完整评估参数组合
        
        Args:
            lookback: 回看天数
            freq: 频率标识
            top_n: 持仓数量
            
        Returns:
            Dict: 包含 sharpe_ratio, annual_return, max_drawdown 等
        """
        # 1) 获取或计算因子值
        factor_values = self._get_factor_values(lookback)

        if factor_values is None or factor_values.shape[0] < 60:
            return {'sharpe_ratio': -np.inf}

        # 2) 构建调度器
        if freq.upper() in ('ME', 'W', 'M'):
            scheduler = FixedScheduler(freq.upper())
        elif freq.endswith('D') or freq.endswith('d'):
            days = int(freq.replace('D', '').replace('d', ''))
            scheduler = IntervalScheduler(days)
        else:
            scheduler = FixedScheduler('ME')

        # 3) 构建分配器和管道
        allocator = TopNEqualWeight(top_n)
        pipeline = FactorPipeline(
            returns=self.returns,
            scheduler=scheduler,
            allocator=allocator,
            cost_rate=self.cost_rate,
        )
        bt_result = pipeline.evaluate(factor_values)

        return {
            'sharpe_ratio': bt_result.sharpe_ratio,
            'annual_return': bt_result.annual_return,
            'max_drawdown': bt_result.max_drawdown,
            'calmar_ratio': bt_result.calmar_ratio,
            'n_rebalances': bt_result.n_rebalances,
        }

    def _get_factor_values(self, lookback: int) -> Optional[pd.DataFrame]:
        """获取因子值（优先从缓存读取）
        
        Args:
            lookback: 回看天数
            
        Returns:
            Optional[pd.DataFrame]: 因子值
        """
        start_date = str(self.returns.index[0].date())
        end_date = str(self.returns.index[-1].date())
        factor_name = f"{self.name_prefix}_{lookback}D"

        # 尝试缓存
        if self.cache_store:
            cached = self.cache_store.load(factor_name, lookback, start_date, end_date)
            if cached is not None:
                return cached

        # 创建因子并计算
        try:
            factor = self.factor_class(lookback=lookback)
            symbols = list(self.returns.columns)
            dates = list(self.returns.index)

            filtered_data = {}
            for field in self.factor_data_fields:
                if field in self.data:
                    df = self.data[field]
                    aligned = df.reindex(
                        index=self.returns.index, columns=self.returns.columns
                    )
                    filtered_data[field] = aligned

            factor_values = factor.calculate(filtered_data, symbols, dates)

            # 写入缓存
            if self.cache_store and factor_values is not None:
                self.cache_store.save(factor_name, lookback, start_date, end_date, factor_values)

            return factor_values

        except Exception as e:
            warnings.warn(f"BO 因子计算失败 [lookback={lookback}]: {e}")
            return None

    def _random_search(self, n_calls: int,
                       freq_idx_map: Dict[int, str],
                       topn_idx_map: Dict[int, str],
                       verbose: bool) -> Tuple[int, str, int, float]:
        """随机搜索（BO 不可用时的回退方案）
        
        Args:
            n_calls: 随机采样次数
            freq_idx_map: 频率索引映射
            topn_idx_map: TopN 索引映射
            verbose: 是否打印进度
            
        Returns:
            Tuple: (best_lookback, best_freq, best_topn, best_sharpe)
        """
        best_sharpe = -np.inf
        best_lookback = int(self.lookback_range[0])
        best_freq = self.freq_candidates[0]
        best_topn = self.topn_candidates[0]

        for i in range(n_calls):
            lookback = np.random.randint(
                int(self.lookback_range[0]), int(self.lookback_range[1]) + 1
            )
            freq = np.random.choice(self.freq_candidates)
            top_n = np.random.choice(self.topn_candidates)

            sharpe = self._evaluate_single(lookback, freq, top_n)

            if verbose:
                print(f"  [随机 {i+1}/{n_calls}] lookback={lookback} "
                      f"freq={freq} topn={top_n} → Sharpe={sharpe:.3f}")

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_lookback = lookback
                best_freq = freq
                best_topn = top_n

            self.best_sharpe = max(self.best_sharpe, sharpe)
            self.convergence_curve.append(self.best_sharpe)

        return best_lookback, best_freq, best_topn, best_sharpe

    def __repr__(self) -> str:
        return (f"FactorBOSearch(factor={self.factor_class.__name__}, "
                f"lookback={self.lookback_range}, "
                f"freqs={self.freq_candidates}, topns={self.topn_candidates})")
