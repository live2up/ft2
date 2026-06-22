"""
factor/v3/backtest.py — 调度器 + 分配器 + 组合器 (不含回测引擎)

合并 v2 的 scheduler.py + allocator.py + combiner.py。
Part 4 回测引擎已迁移至 engine_core.py (ft2.core 驱动)。

模块结构：
  Part 1: 调度器 (RebalanceScheduler / FixedScheduler / IntervalScheduler)
  Part 2: 权重分配器 (WeightAllocator / TopNEqualWeight / ScoreProportional / RiskParity)
  Part 3: 因子组合器 (FactorCombiner / EqualWeightCombiner / FixedWeightCombiner / ExpandingICCombiner)
  Part 4: [已废弃] 回测引擎 → 请使用 factor.v3.engine_core.FactorEngineCore

使用方式：
>>> from factor.v3.engine_core import FactorEngineCore
>>> analyzer = FactorEngineCore.backtest(panel, returns, top_n=3, rebalance='W')
>>> print(f"Sharpe={analyzer.sharpe_ratio():.2f}")

[重构] 2026-06-17 Part 4 迁移到 engine_core.py (ft2.core 统一回测引擎)
=============================================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import warnings


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: 调度器 — 调仓日生成
# ═══════════════════════════════════════════════════════════════════════════

class RebalanceScheduler(ABC):
    """调仓日生成器抽象基类"""

    @abstractmethod
    def generate(self, dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """从交易日序列中生成调仓日列表"""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class FixedScheduler(RebalanceScheduler):
    """固定频率调度器：支持 'ME'(月末) / 'W'(周末) / 'M'(月初)"""

    VALID_FREQS = {'ME', 'W', 'M'}

    def __init__(self, freq: str = 'ME'):
        if freq not in self.VALID_FREQS:
            raise ValueError(f"不支持的模式 '{freq}'，可选: {self.VALID_FREQS}")
        self.freq = freq

    def generate(self, dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        if len(dates) == 0:
            return []
        df = pd.DataFrame({'date': dates})
        df['year'] = dates.year
        df['month'] = dates.month

        if self.freq == 'W':
            iso = dates.isocalendar()
            iso_year = iso['year'].to_numpy(dtype=int)
            iso_week = iso['week'].to_numpy(dtype=int)
            df['year_week'] = [f"{y}-{w:02d}" for y, w in zip(iso_year, iso_week)]
            rebalance = df.groupby('year_week')['date'].last()
        elif self.freq == 'ME':
            df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
            rebalance = df.groupby('year_month')['date'].last()
        elif self.freq == 'M':
            df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
            rebalance = df.groupby('year_month')['date'].first()

        result = sorted(rebalance.tolist())
        if len(result) > 1:
            return result[1:]
        return result

    def __repr__(self) -> str:
        return f"FixedScheduler(freq='{self.freq}')"


class IntervalScheduler(RebalanceScheduler):
    """固定间隔调度器：每 N 个交易日调仓一次"""

    def __init__(self, interval_days: int = 5):
        if interval_days < 1:
            raise ValueError(f"interval_days 必须 >= 1，当前值: {interval_days}")
        self.interval_days = interval_days

    def generate(self, dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        n = len(dates)
        if n <= self.interval_days:
            return []
        result = []
        for i in range(self.interval_days - 1, n, self.interval_days):
            result.append(dates[i])
        return result

    def __repr__(self) -> str:
        return f"IntervalScheduler(interval_days={self.interval_days})"


def recommend_scheduler_from_decay(half_life: float) -> RebalanceScheduler:
    """根据 IC 半衰期自动推荐调度器"""
    if half_life <= 0:
        raise ValueError(f"半衰期必须 > 0，当前值: {half_life}")
    if half_life <= 5:
        return IntervalScheduler(5)
    elif half_life <= 10:
        return FixedScheduler('W')
    elif half_life <= 22:
        return IntervalScheduler(10)
    else:
        return FixedScheduler('ME')


def parse_scheduler(freq: str) -> RebalanceScheduler:
    """字符串频率 → Scheduler 对象

    统一入口，消除各处重复的 freq 字符串解析逻辑。

    Args:
        freq: 频率字符串，支持 'ME'(月末)/'W'(周末)/'M'(月初)/'ND'(每N天)/纯数字N

    Returns:
        RebalanceScheduler 对象

    [新增] 2026-06-05 统一 freq 解析，消除 4 处重复代码
    """
    f = freq.upper()
    if f in ('ME', 'W', 'M'):
        return FixedScheduler(f)
    if f.endswith('D'):
        days = int(f.replace('D', ''))
        if days < 1:
            raise ValueError(f"间隔天数必须 >= 1，当前: {freq}")
        return IntervalScheduler(days)
    try:
        return IntervalScheduler(int(freq))
    except ValueError:
        raise ValueError(f"无法解析频率 '{freq}'，支持: 'ME'/'W'/'M'/'5D'/'10' 等")


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: 权重分配器
# ═══════════════════════════════════════════════════════════════════════════

class WeightAllocator(ABC):
    """权重分配器抽象基类"""

    @abstractmethod
    def allocate(self, scores: np.ndarray) -> np.ndarray:
        """将因子得分转为持仓权重"""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TopNEqualWeight(WeightAllocator):
    """前 N 名等权持有"""

    def __init__(self, top_n: int = 5):
        if top_n < 1:
            raise ValueError(f"top_n 必须 >= 1，当前值: {top_n}")
        self.top_n = top_n

    def allocate(self, scores: np.ndarray) -> np.ndarray:
        n = len(scores)
        if n == 0:
            return np.array([])
        valid_mask = ~np.isnan(scores)
        valid_scores = np.where(valid_mask, scores, -np.inf)
        n_valid = valid_mask.sum()
        actual_n = min(self.top_n, n_valid)
        if actual_n == 0:
            return np.zeros(n)
        top_indices = np.argpartition(-valid_scores, actual_n - 1)[:actual_n]
        top_indices = top_indices[np.argsort(-valid_scores[top_indices])]
        weights = np.zeros(n)
        weights[top_indices] = 1.0 / actual_n
        return weights

    def __repr__(self) -> str:
        return f"TopNEqualWeight(top_n={self.top_n})"


class ScoreProportional(WeightAllocator):
    """正得分按比例分配"""

    def allocate(self, scores: np.ndarray) -> np.ndarray:
        n = len(scores)
        if n == 0:
            return np.array([])
        positive_scores = np.nan_to_num(scores, nan=0.0)
        positive_scores = np.maximum(positive_scores, 0)
        total = positive_scores.sum()
        if total <= 0:
            return np.zeros(n)
        return positive_scores / total

    def __repr__(self) -> str:
        return "ScoreProportional()"


class RiskParity(WeightAllocator):
    """风险平价权重（需要 scipy.optimize）"""

    def __init__(self, window: int = 60):
        if window < 10:
            raise ValueError(f"window 必须 >= 10，当前值: {window}")
        self.window = window

    def allocate(self, scores: np.ndarray,
                 returns_history: Optional[np.ndarray] = None) -> np.ndarray:
        n = len(scores)
        if n == 0:
            return np.array([])
        if returns_history is None or returns_history.shape[1] == 0:
            valid_mask = ~np.isnan(scores)
            n_valid = valid_mask.sum()
            if n_valid == 0:
                return np.zeros(n)
            weights = np.zeros(n)
            weights[valid_mask] = 1.0 / n_valid
            return weights
        try:
            from scipy.optimize import minimize
            # [修复] 2026-06-01 检查协方差矩阵有效性，防止全NaN输入
            if np.all(np.isnan(returns_history)):
                valid_mask = ~np.isnan(scores)
                n_valid = valid_mask.sum()
                if n_valid == 0:
                    return np.zeros(n)
                weights = np.zeros(n)
                weights[valid_mask] = 1.0 / n_valid
                return weights
            cov = np.cov(returns_history, rowvar=False)
            # [修复] 2026-06-01 协方差含 NaN 时退化为等权
            if np.any(np.isnan(cov)):
                valid_mask = ~np.isnan(scores)
                n_valid = valid_mask.sum()
                if n_valid == 0:
                    return np.zeros(n)
                weights = np.zeros(n)
                weights[valid_mask] = 1.0 / n_valid
                return weights
            cov = cov + np.eye(cov.shape[0]) * 1e-6
            m = cov.shape[0]

            def risk_parity_objective(w):
                portfolio_risk = w @ cov @ w
                if portfolio_risk <= 0:
                    return 1e10
                sigma = np.sqrt(portfolio_risk)
                marginal_contrib = cov @ w
                risk_contrib = w * marginal_contrib / sigma
                target = sigma / m
                return np.sum((risk_contrib - target) ** 2)

            w0 = np.ones(m) / m
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1) for _ in range(m)]
            result = minimize(risk_parity_objective, w0, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'maxiter': 1000, 'ftol': 1e-10})
            if result.success:
                opt_weights = np.maximum(result.x, 0)
                total = opt_weights.sum()
                if total > 0:
                    opt_weights = opt_weights / total
            else:
                opt_weights = np.ones(m) / m
            full_weights = np.zeros(n)
            full_weights[:m] = opt_weights
            return full_weights
        except ImportError:
            valid_mask = ~np.isnan(scores)
            n_valid = valid_mask.sum()
            if n_valid == 0:
                return np.zeros(n)
            weights = np.zeros(n)
            weights[valid_mask] = 1.0 / n_valid
            return weights

    def __repr__(self) -> str:
        return f"RiskParity(window={self.window})"


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: 因子组合器
# ═══════════════════════════════════════════════════════════════════════════

def cross_section_zscore(fv: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score 标准化"""
    mean = fv.mean(axis=1)
    std = fv.std(axis=1)
    std_safe = std.replace(0, 1.0)
    return fv.sub(mean, axis=0).div(std_safe, axis=0)


class FactorCombiner(ABC):
    """因子组合器抽象基类"""

    @abstractmethod
    def combine(self, factor_values_list: List[Tuple[str, pd.DataFrame]],
                returns: pd.DataFrame = None) -> pd.DataFrame:
        """组合多个因子"""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class EqualWeightCombiner(FactorCombiner):
    """等权组合"""

    def combine(self, factor_values_list: List[Tuple[str, pd.DataFrame]],
                returns: pd.DataFrame = None) -> pd.DataFrame:
        n = len(factor_values_list)
        if n == 0:
            raise ValueError("factor_values_list 不能为空")
        if n == 1:
            return factor_values_list[0][1].copy()
        return sum(fv for _, fv in factor_values_list) / n


class FixedWeightCombiner(FactorCombiner):
    """固定权重组合"""

    def __init__(self, weights: Dict[str, float]):
        if not weights:
            raise ValueError("weights 不能为空")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("权重之和必须 > 0")
        self.weights = {k: v / total for k, v in weights.items()}

    def combine(self, factor_values_list: List[Tuple[str, pd.DataFrame]],
                returns: pd.DataFrame = None) -> pd.DataFrame:
        if not factor_values_list:
            raise ValueError("factor_values_list 不能为空")
        combined = None
        for name, fv in factor_values_list:
            w = self.weights.get(name, 0.0)
            if w == 0.0:
                continue
            if combined is None:
                combined = fv * w
            else:
                combined = combined + fv * w
        if combined is None:
            raise ValueError("所有因子权重为 0，无法组合")
        return combined

    def __repr__(self) -> str:
        w_str = ', '.join(f'{k}={v:.3f}' for k, v in self.weights.items())
        return f"FixedWeightCombiner({w_str})"


class ExpandingICCombiner(FactorCombiner):
    """Expanding IC 动态加权（无前瞻偏差）"""

    def __init__(self, min_periods: int = 60):
        if min_periods < 1:
            raise ValueError(f"min_periods 必须 >= 1")
        self.min_periods = min_periods

    def combine(self, factor_values_list: List[Tuple[str, pd.DataFrame]],
                returns: pd.DataFrame = None) -> pd.DataFrame:
        if not factor_values_list:
            raise ValueError("factor_values_list 不能为空")
        n_factors = len(factor_values_list)
        if n_factors == 1:
            return factor_values_list[0][1].copy()
        if returns is None:
            raise ValueError("ExpandingICCombiner 需要 returns 参数")
        ic_weight_df = self._compute_time_varying_ic_weights(factor_values_list, returns)
        common_dates = ic_weight_df.index
        common_symbols = factor_values_list[0][1].columns
        combined = pd.DataFrame(0.0, index=common_dates, columns=common_symbols)
        for j, (name, fv) in enumerate(factor_values_list):
            w_col = ic_weight_df.columns[j]
            for t in common_dates:
                if t in fv.index:
                    combined.loc[t] += ic_weight_df.loc[t, w_col] * fv.loc[t]
        return combined

    def _compute_time_varying_ic_weights(self, factor_values_list, returns):
        future_returns = returns.shift(-1)
        n_factors = len(factor_values_list)
        expanding_ic_list = []
        all_dates = None
        for name, fv in factor_values_list:
            common = fv.index.intersection(future_returns.index)
            cols = fv.columns.intersection(future_returns.columns)
            if len(common) < self.min_periods or len(cols) < 3:
                expanding_ic_list.append(None)
                continue
            fv_sub = fv.loc[common, cols]
            fr_sub = future_returns.loc[common, cols]
            daily_ics, dates = [], []
            for date in common:
                row_fv = fv_sub.loc[date].dropna()
                row_fr = fr_sub.loc[date].dropna()
                common_idx = row_fv.index.intersection(row_fr.index)
                if len(common_idx) < 3:
                    daily_ics.append(np.nan)
                    dates.append(date)
                    continue
                ic = row_fv.loc[common_idx].rank().corr(
                    row_fr.loc[common_idx].rank(), method='pearson')
                daily_ics.append(ic)
                dates.append(date)
            ic_s = pd.Series(daily_ics, index=dates)
            expanding_mean_ic = ic_s.expanding(min_periods=self.min_periods).mean()
            expanding_ic_list.append(expanding_mean_ic)
            all_dates = common if all_dates is None else all_dates.intersection(common)
        weight_records = []
        for t in all_dates:
            weights_at_t = []
            for j, ic_s in enumerate(expanding_ic_list):
                if ic_s is not None and t in ic_s.index and not np.isnan(ic_s.loc[t]):
                    weights_at_t.append(max(ic_s.loc[t], 0.0))
                else:
                    weights_at_t.append(0.0)
            total = sum(weights_at_t)
            if total <= 0:
                weights_at_t = [1.0 / n_factors] * n_factors
            else:
                weights_at_t = [w / total for w in weights_at_t]
            weight_records.append(weights_at_t)
        factor_names = [name for name, _ in factor_values_list]
        return pd.DataFrame(weight_records, index=all_dates, columns=factor_names)


# ═══════════════════════════════════════════════════════════════════════════
# Part 4: [已废弃] 回测引擎 — 已迁移至 engine_core.py
# ═══════════════════════════════════════════════════════════════════════════
#
# FactorPipeline / BacktestResult / BacktestSchedule 已废弃。
# 请使用 factor.v3.engine_core.FactorEngineCore 替代：
#
#   >>> from factor.v3.engine_core import FactorEngineCore
#   >>> analyzer = FactorEngineCore.backtest(panel, returns, top_n=3, rebalance='W')
#   >>> analyzer.sharpe_ratio()  # 替代 BacktestResult.sharpe_ratio
#   >>> analyzer.max_drawdown()  # 返回 (value, start_date, end_date) tuple
#
# 保留以下兼容接口供过渡期使用 (内部委托到 FactorEngineCore)：

import warnings as _warnings

@dataclass
class BacktestSchedule:
    """[已废弃] 回测调度计划"""
    returns: pd.DataFrame
    factor_values: pd.DataFrame
    rebalance_dates: List[pd.Timestamp]
    weights_at_rebalance: Dict[pd.Timestamp, np.ndarray]


@dataclass
class BacktestResult:
    """[已废弃] 因子回测结果, 内部委托到 AccountAnalyzer"""
    nav_series: pd.Series = field(default_factory=pd.Series)
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    turnover_cost_ratio: float = 0.0
    avg_turnover: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    yearly_returns: Dict[int, float] = field(default_factory=dict)
    yearly_sharpe: Dict[int, float] = field(default_factory=dict)
    yearly_dd: Dict[int, float] = field(default_factory=dict)
    yearly_vol: Dict[int, float] = field(default_factory=dict)
    n_rebalances: int = 0
    start_date: str = ""
    end_date: str = ""
    scheduler_name: str = ""
    allocator_name: str = ""

    def to_dict(self) -> Dict:
        return {
            '总收益率': f'{self.total_return:.2%}',
            '年化收益率': f'{self.annual_return:.2%}',
            '年化波动率': f'{self.annual_volatility:.2%}',
            '夏普比率': f'{self.sharpe_ratio:.2f}',
            '最大回撤': f'{self.max_drawdown:.2%}',
            '调仓次数': self.n_rebalances,
            '回测区间': f'{self.start_date} → {self.end_date}',
        }

    def yearly_report(self) -> pd.DataFrame:
        years = sorted(set(list(self.yearly_returns.keys())
                          + list(self.yearly_sharpe.keys())
                          + list(self.yearly_dd.keys())))
        if not years:
            return pd.DataFrame()
        records = []
        for year in years:
            records.append({
                '年份': str(year),
                '收益率': f'{self.yearly_returns.get(year, 0):.2%}',
                'Sharpe': f'{self.yearly_sharpe.get(year, 0):.4f}',
                '最大回撤': f'{self.yearly_dd.get(year, 0):.2%}',
                '波动率': f'{self.yearly_vol.get(year, 0):.2%}',
            })
        return pd.DataFrame(records)


class FactorPipeline:
    """[已废弃] 因子→调仓→回测桥接

    内部委托到 FactorEngineCore (ft2.core Engine)。
    新代码请直接使用 factor.v3.engine_core.FactorEngineCore。
    """

    def __init__(self, returns: pd.DataFrame, scheduler, allocator,
                 cost_rate: float = 0.0, rf_annual: float = 0.0):
        _warnings.warn(
            "FactorPipeline 已废弃，请使用 factor.v3.engine_core.FactorEngineCore",
            DeprecationWarning, stacklevel=2)
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)
        self.returns = returns.sort_index()
        self.scheduler = scheduler
        self.allocator = allocator
        self.cost_rate = cost_rate
        self.rf_annual = rf_annual

    def evaluate(self, factor_values: pd.DataFrame):
        """委托到 FactorEngineCore，直接返回 AccountAnalyzer。

        [重构] 2026-06-22 去掉 BacktestResult 翻译层，直接返回 AccountAnalyzer。
        旧代码若依赖 BacktestResult 字段，改为调用 AccountAnalyzer 方法：
          - result.sharpe_ratio → analyzer.sharpe_ratio()
          - result.annual_return → analyzer.annualized_return()
          - result.max_drawdown → analyzer.max_drawdown()[0]
          - result.nav_series → analyzer.daily_returns()
        """
        from .engine_core import FactorEngineCore
        from core.analyzer import AccountAnalyzer

        top_n = getattr(self.allocator, 'top_n', 3)
        return FactorEngineCore.backtest(
            factor_values, self.returns, top_n=top_n,
            rebalance=self.scheduler)

    def __repr__(self) -> str:
        return (f"FactorPipeline[deprecated](scheduler={self.scheduler}, "
                f"allocator={self.allocator})")
