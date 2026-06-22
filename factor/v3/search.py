"""
factor/v3/search.py — 参数搜索统一入口
=============================================================================

合并 v2 的 grid_search.py 和 bo_search.py。
提供因子参数网格搜索和贝叶斯优化搜索。

[重构] 2026-06-01 从 v2 两个文件合并为 v3/search.py
=============================================================================
"""

import os
import json
import itertools
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .base import FactorCategory, FactorFrequency
from .cache import FactorCacheStore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: 网格搜索
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GridSearchResult:
    """单个网格搜索点的结果

    字段名与 AccountAnalyzer / FactorValidator 方法名一致。
    """
    param_name: str
    lookback: int
    freq: str
    top_n: int = 5
    ic_mean: float = 0.0
    ic_ir: float = 0.0
    sharpe_ratio: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    analyzer: object = None  # AccountAnalyzer 引用


class FactorGridSearch:
    """因子参数网格搜索器

    对 factor_fn 的多个参数组合做批量回测评估，输出按指标排序的结果。
    """

    def __init__(self, returns: pd.DataFrame, cost_rate: float = 0.0,
                 cache_dir: str = '.factor_cache'):
        self.returns = returns
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns = self.returns.copy()
            self.returns.index = pd.to_datetime(self.returns.index)
        self.cost_rate = cost_rate
        self.future_returns = self.returns.shift(-1)
        self.cache = FactorCacheStore(cache_dir)
        self.results: List[GridSearchResult] = []

    def search(self, factor_name: str,
               factor_fn: Callable[[int], pd.DataFrame],
               lookbacks: List[int],
               freqs: List[str] = None,
               top_ns: List[int] = None,
               data: Dict[str, pd.DataFrame] = None) -> List[GridSearchResult]:
        """执行参数网格搜索

        Args:
            factor_name: 因子名称
            factor_fn: 因子计算函数，签名为 (lookback: int) -> pd.DataFrame
            lookbacks: 回看参数列表
            freqs: 频率列表
            top_ns: Top N 列表
            data: 面板数据字典（用于缓存键）

        Returns:
            List[GridSearchResult]: 按 Sharpe 降序排列的结果
        """
        from .validator import FactorValidator
        from .backtest import parse_scheduler
        from .engine_core import FactorEngineCore  # [新增] 2026-06-17 ft2.core 回测引擎

        if freqs is None:
            freqs = ['ME', 'W']
        if top_ns is None:
            top_ns = [5, 10]

        self.results = []
        total = len(lookbacks) * len(freqs) * len(top_ns)

        param_combos = list(itertools.product(lookbacks, freqs, top_ns))
        for i, (lookback, freq, top_n) in enumerate(param_combos):
            # [重构] 2026-06-05 使用 parse_scheduler 统一入口
            scheduler = parse_scheduler(freq)

            # Check cache
            start_date = str(self.returns.index[0].date())
            end_date = str(self.returns.index[-1].date())
            param_key = f"{factor_name}_{lookback}"

            cached = self.cache.load(param_key, lookback, start_date, end_date)
            if cached is not None:
                fv = cached
            else:
                try:
                    fv = factor_fn(lookback)
                    self.cache.save(param_key, lookback, start_date, end_date, fv)
                except Exception as e:
                    logger.warning(f"计算失败 [{factor_name} L={lookback}]: {e}")
                    continue

            # IC validation
            validator = FactorValidator(fv, self.future_returns)
            ic_result = validator.information_coefficient()
            ic_mean = ic_result.get('mean', 0.0)
            ic_ir = ic_result.get('ir', 0.0)
            hr = validator.hit_rate()

            if ic_mean is None or np.isnan(ic_mean):
                continue

            # Pipeline backtest
            # [重构] 2026-06-17 使用 FactorEngineCore (ft2.core) 替代 FactorPipeline
            analyzer = FactorEngineCore.backtest(
                fv, self.returns, top_n=top_n, rebalance=scheduler)
            sr = analyzer.sharpe_ratio()

            result = GridSearchResult(
                param_name=f"{factor_name}_L{lookback}",
                lookback=lookback, freq=freq, top_n=top_n,
                ic_mean=float(ic_mean),
                ic_ir=float(ic_ir) if not np.isnan(ic_ir) else 0.0,
                sharpe_ratio=float(sr) if sr is not None else 0.0,
                annualized_return=analyzer.annualized_return() or 0.0,
                max_drawdown=float(analyzer.max_drawdown()[0]) if analyzer.max_drawdown() else 0.0,
                hit_rate=float(hr) if not np.isnan(hr) else 0.0,
                analyzer=analyzer,
            )
            self.results.append(result)

        self.results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        return self.results

    def best(self, n: int = 10) -> List[GridSearchResult]:
        return self.results[:n]

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        rows = []
        for r in self.results:
            rows.append({
                '参数': r.param_name, 'Lookback': r.lookback,
                '频率': r.freq, 'TopN': r.top_n,
                'IC': round(r.ic_mean, 4), 'IR': round(r.ic_ir, 2),
                'Sharpe': round(r.sharpe_ratio, 2), '年化': f"{r.annualized_return:.1%}",
                '最大回撤': f"{r.max_drawdown:.1%}",
            })
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: 贝叶斯优化
# ═══════════════════════════════════════════════════════════════════════════

class FactorBOSearch:
    """因子参数贝叶斯优化搜索器

    对连续参数空间做贝叶斯优化（GP代理模型 + Expected Improvement）。
    skopt 不可用时退化为随机搜索。
    """

    def __init__(self, returns: pd.DataFrame, cost_rate: float = 0.0,
                 cache_dir: str = '.factor_cache', random_state: int = None):
        self.returns = returns
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns = self.returns.copy()
            self.returns.index = pd.to_datetime(self.returns.index)
        self.cost_rate = cost_rate
        self.future_returns = self.returns.shift(-1)
        self.cache = FactorCacheStore(cache_dir)
        self.rng = np.random.default_rng(random_state)
        self._use_skopt = False
        try:
            import skopt
            self._use_skopt = True
        except ImportError:
            warnings.warn("skopt 不可用，贝叶斯优化退化为随机搜索")

    def search(self, factor_name: str,
               factor_fn: Callable,
               param_space: List[Tuple[float, float, str]],
               n_calls: int = 30,
               freq: str = 'ME',
               top_n: int = 5) -> Dict[str, Any]:
        """贝叶斯优化搜索

        Args:
            factor_name: 因子名称
            factor_fn: 参数 -> 因子值 DataFrame 的函数
            param_space: 参数空间 [(low, high, name), ...]
            n_calls: 优化迭代次数
            freq: 调仓频率
            top_n: 持仓数

        Returns:
            Dict: {'best_params', 'best_score', 'all_results'}
        """
        from .validator import FactorValidator
        from .backtest import parse_scheduler
        from .engine_core import FactorEngineCore  # [新增] 2026-06-17

        # [重构] 2026-06-05 使用 parse_scheduler 统一入口
        scheduler = parse_scheduler(freq)

        # [修复] 2026-06-01 失败值从 1e10 改为 999.0 与 FitnessCalculator 一致
        FAILURE_PENALTY = 999.0

        def objective(params_tuple):
            fv = factor_fn(*params_tuple)
            if fv is None:
                return FAILURE_PENALTY
            analyzer = FactorEngineCore.backtest(
                fv, self.returns, top_n=top_n, rebalance=scheduler)
            sr = analyzer.sharpe_ratio()
            return -(float(sr) if sr is not None else 0.0)

        if self._use_skopt:
            from skopt import gp_minimize
            from skopt.space import Real
            dimensions = [Real(lo, hi, name=name) for lo, hi, name in param_space]
            res = gp_minimize(objective, dimensions, n_calls=n_calls, random_state=int(self.rng.integers(10000)))
            best_params = {d.name: float(v) for d, v in zip(dimensions, res.x)}
            best_score = float(-res.fun)
            return {'best_params': best_params, 'best_score': best_score, 'n_calls': n_calls}
        else:
            # Random search fallback
            best_score = -np.inf
            best_params = {}
            for _ in range(n_calls):
                params = tuple(
                    self.rng.uniform(lo, hi) for lo, hi, _ in param_space
                )
                score = -objective(params)
                if score > best_score or _ == 0:
                    best_score = score
                    best_params = {name: float(v) for v, (_, _, name) in zip(params, param_space)}
            return {'best_params': best_params, 'best_score': best_score, 'n_calls': n_calls}


__all__ = [
    'GridSearchResult', 'FactorGridSearch',
    'FactorBOSearch',
]
