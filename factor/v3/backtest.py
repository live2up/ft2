"""
factor/v3/backtest.py — 回测一体化模块
=============================================================================

合并 v2 的 pipeline.py + scheduler.py + allocator.py + combiner.py。
提供从调度器→权重分配→因子组合→Pipeline回测的完整链路。

模块结构：
  Part 1: 调度器 (RebalanceScheduler / FixedScheduler / IntervalScheduler)
  Part 2: 权重分配器 (WeightAllocator / TopNEqualWeight / ScoreProportional / RiskParity)
  Part 3: 因子组合器 (FactorCombiner / EqualWeightCombiner / FixedWeightCombiner / ExpandingICCombiner)
  Part 4: 回测引擎 (FactorPipeline / BacktestResult / BacktestSchedule)

使用方式：
>>> from factor.v3.backtest import FactorPipeline, FixedScheduler, TopNEqualWeight
>>> pipeline = FactorPipeline(returns, FixedScheduler('ME'), TopNEqualWeight(3))
>>> result = pipeline.evaluate(factor_values)
>>> print(f"Sharpe={result.sharpe_ratio:.2f}")

[重构] 2026-06-01 从 v2 4 个文件合并为 v3/backtest.py
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
# Part 4: 回测引擎 (FactorPipeline)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestSchedule:
    """回测调度计划"""
    returns: pd.DataFrame
    factor_values: pd.DataFrame
    rebalance_dates: List[pd.Timestamp]
    weights_at_rebalance: Dict[pd.Timestamp, np.ndarray]


@dataclass
class BacktestResult:
    """因子回测结果"""
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
            '索提诺比率': f'{self.sortino_ratio:.2f}',
            '卡玛比率': f'{self.calmar_ratio:.2f}',
            '最大回撤': f'{self.max_drawdown:.2%}',
            '平均回撤': f'{self.avg_drawdown:.2%}',
            '日度胜率': f'{self.win_rate:.1%}',
            '盈亏比': f'{self.profit_loss_ratio:.2f}',
            '平均换手率': f'{self.avg_turnover:.2%}',
            '换手成本占比': f'{self.turnover_cost_ratio:.2%}',
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
    """因子→调仓→回测 完整桥接

    职责：
    - 在调仓日取因子值 → 分配权重 → 逐日计算持仓收益 → 累加日频净值
    - 扣除交易成本
    - 计算全套回测指标
    - 支持多频率一键对比
    """

    def __init__(self, returns: pd.DataFrame, scheduler: RebalanceScheduler,
                 allocator: WeightAllocator, cost_rate: float = 0.0,
                 rf_annual: float = 0.0):
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)
        self.returns = returns.sort_index()
        self.scheduler = scheduler
        self.allocator = allocator
        self.cost_rate = cost_rate
        self.rf_annual = rf_annual

    # ── 公开接口 ──

    def evaluate(self, factor_values: pd.DataFrame) -> BacktestResult:
        schedule = self._build_schedule(factor_values)
        if schedule is None:
            return self._empty_result('调度构建失败')
        nav_series, turnover_list = self._accumulate_nav(schedule)
        if len(nav_series) < 2:
            return self._empty_result('日频净值数据点不足')
        return self._calc_metrics(nav_series, turnover_list, schedule.rebalance_dates)

    def compare_frequencies(self, factor_values: pd.DataFrame,
                            freqs: List[str] = None) -> Dict[str, BacktestResult]:
        if freqs is None:
            freqs = ['ME', 'W', '5D']
        results = {}
        for freq in freqs:
            # [重构] 2026-06-05 使用 parse_scheduler 统一入口
            scheduler = parse_scheduler(freq)
            original = self.scheduler
            self.scheduler = scheduler
            try:
                results[freq] = self.evaluate(factor_values)
            finally:
                self.scheduler = original
        return results

    # ── 内部实现 ──

    def _build_schedule(self, factor_values: pd.DataFrame) -> Optional[BacktestSchedule]:
        returns = self.returns
        if not isinstance(factor_values.index, pd.DatetimeIndex):
            factor_values = factor_values.copy()
            factor_values.index = pd.to_datetime(factor_values.index)
        common_dates = returns.index.intersection(factor_values.index)
        if len(common_dates) == 0:
            return None
        common_symbols = returns.columns.intersection(factor_values.columns)
        if len(common_symbols) == 0:
            return None
        returns = returns.loc[common_dates, common_symbols].copy()
        factor_values = factor_values.loc[common_dates, common_symbols].copy()
        rebalance_dates = self.scheduler.generate(returns.index)
        if len(rebalance_dates) < 2:
            return None
        valid_rb_dates = [d for d in rebalance_dates if d in returns.index]
        if len(valid_rb_dates) < 2:
            return None
        weights_at_rebalance = {}
        for rb_date in valid_rb_dates:
            factors_on_date = factor_values.loc[rb_date].values
            weights = self.allocator.allocate(factors_on_date)
            weights_at_rebalance[rb_date] = weights
        return BacktestSchedule(returns=returns, factor_values=factor_values,
                                rebalance_dates=valid_rb_dates,
                                weights_at_rebalance=weights_at_rebalance)

    def _accumulate_nav(self, schedule: BacktestSchedule) -> Tuple[pd.Series, List[float]]:
        returns = schedule.returns
        rb_dates = schedule.rebalance_dates
        nav_dates, nav_values = [], []
        current_nav = 1.0
        prev_weights = None
        turnover_list = []
        nav_dates.append(rb_dates[0])
        nav_values.append(current_nav)
        for i, rb_date in enumerate(rb_dates):
            weights = schedule.weights_at_rebalance[rb_date]
            if prev_weights is not None and self.cost_rate > 0:
                turnover = np.sum(np.abs(weights - prev_weights)) / 2.0
                current_nav *= (1.0 - turnover * self.cost_rate)
                turnover_list.append(turnover)
            elif prev_weights is None and self.cost_rate > 0:
                current_nav *= (1.0 - 0.5 * self.cost_rate)
            prev_weights = weights.copy()
            if i < len(rb_dates) - 1:
                next_date = rb_dates[i + 1]
                mask = (returns.index > rb_date) & (returns.index <= next_date)
            else:
                mask = returns.index > rb_date
            period_returns = returns.loc[mask]
            for dt in period_returns.index:
                daily_ret = _calc_daily_return(weights, period_returns.loc[dt])
                current_nav *= (1.0 + daily_ret)
                nav_dates.append(dt)
                nav_values.append(current_nav)
        if len(nav_dates) < 2:
            return pd.Series(dtype=float), turnover_list
        nav_series = pd.Series(nav_values, index=pd.DatetimeIndex(nav_dates))
        nav_series = nav_series.drop_duplicates().sort_index()
        nav_series = nav_series / nav_series.iloc[0]
        return nav_series, turnover_list

    def _calc_metrics(self, nav_series, turnover_list, rebalance_dates) -> BacktestResult:
        if len(nav_series) < 2:
            return self._empty_result('净值数据点不足')
        result = BacktestResult()
        result.nav_series = nav_series
        result.n_rebalances = len(rebalance_dates) - 1
        result.scheduler_name = repr(self.scheduler)
        result.allocator_name = repr(self.allocator)
        nav_values = nav_series.values
        nav_dates = nav_series.index
        result.start_date = str(nav_dates[0].date())
        result.end_date = str(nav_dates[-1].date())
        result.total_return = float(nav_values[-1] / nav_values[0] - 1)
        daily_ret = nav_series.pct_change().dropna().values
        n_days = len(daily_ret)
        if n_days > 0:
            total_days = (nav_dates[-1] - nav_dates[0]).days
            years = max(total_days / 365.25, 0.01)
            result.annual_return = float((nav_values[-1] / nav_values[0]) ** (1.0 / years) - 1)
            daily_vol = np.std(daily_ret, ddof=1) if n_days > 1 else 0.0
            result.annual_volatility = float(daily_vol * np.sqrt(252))
            rf_daily = self.rf_annual / 252
            excess_daily = daily_ret - rf_daily
            if result.annual_volatility > 0:
                result.sharpe_ratio = float(
                    np.mean(excess_daily) / max(daily_vol, 1e-10) * np.sqrt(252))
            downside = daily_ret[daily_ret < 0]
            if len(downside) > 1:
                downside_vol = np.std(downside, ddof=1)
                if downside_vol > 0:
                    result.sortino_ratio = float(
                        np.mean(excess_daily) / downside_vol * np.sqrt(252))
        peak = np.maximum.accumulate(nav_values)
        drawdowns = (nav_values - peak) / peak
        result.max_drawdown = float(drawdowns.min())
        result.avg_drawdown = float(drawdowns.mean())
        if abs(result.max_drawdown) > 1e-10:
            result.calmar_ratio = float(result.annual_return / abs(result.max_drawdown))
        if turnover_list:
            result.avg_turnover = float(np.mean(turnover_list))
            total_cost = sum(t * self.cost_rate for t in turnover_list)
            total_gross_return = result.total_return + total_cost
            result.turnover_cost_ratio = float(
                total_cost / abs(total_gross_return)) if abs(total_gross_return) > 1e-10 else 0.0
        if n_days > 0:
            wins = daily_ret[daily_ret > 0]
            losses = daily_ret[daily_ret < 0]
            result.win_rate = float(len(wins) / n_days)
            avg_win = np.mean(wins) if len(wins) > 0 else 0.0
            avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0
            result.profit_loss_ratio = float(avg_win / avg_loss) if avg_loss > 0 else 0.0
        result.yearly_returns = _calc_yearly_returns(nav_series)
        result.yearly_sharpe = _calc_yearly_sharpe(nav_series)
        result.yearly_dd = _calc_yearly_dd(nav_series)
        result.yearly_vol = _calc_yearly_vol(nav_series)
        return result

    def _empty_result(self, reason: str = '') -> BacktestResult:
        if reason:
            warnings.warn(f"因子回测返回空结果: {reason}")
        return BacktestResult()

    def __repr__(self) -> str:
        return (f"FactorPipeline(scheduler={self.scheduler}, "
                f"allocator={self.allocator}, cost_rate={self.cost_rate})")


# ── 辅助函数 ──

def _calc_daily_return(weights: np.ndarray, day_returns: pd.Series) -> float:
    nonzero = weights > 0
    if not nonzero.any():
        return 0.0
    sel_ret = day_returns.values[nonzero].copy()
    sel_w = weights[nonzero]
    sel_ret = np.where(np.isnan(sel_ret), 0.0, sel_ret)
    return float(np.dot(sel_w, sel_ret))


def _calc_yearly_returns(nav_series: pd.Series) -> Dict[int, float]:
    yearly = {}
    nav_dates = nav_series.index
    nav_values = nav_series.values
    for year in range(nav_dates[0].year, nav_dates[-1].year + 1):
        year_start = year_end = None
        for i, d in enumerate(nav_dates):
            if d.year == year and year_start is None:
                year_start = i
            if d.year == year:
                year_end = i
        if year_start is not None and year_end is not None and year_end > year_start:
            yearly[year] = float(nav_values[year_end] / nav_values[year_start] - 1)
    return yearly


def _calc_yearly_sharpe(nav_series: pd.Series) -> Dict[int, float]:
    yearly = {}
    daily_ret = nav_series.pct_change().dropna()
    for year in sorted(daily_ret.index.year.unique()):
        year_ret = daily_ret[daily_ret.index.year == year].values
        if len(year_ret) < 10:
            continue
        vol = np.std(year_ret, ddof=1)
        yearly[int(year)] = float(np.mean(year_ret) / vol * np.sqrt(252)) if vol > 1e-10 else 0.0
    return yearly


def _calc_yearly_dd(nav_series: pd.Series) -> Dict[int, float]:
    yearly = {}
    nav_values = nav_series.values
    nav_dates = nav_series.index
    for year in range(nav_dates[0].year, nav_dates[-1].year + 1):
        mask = nav_dates.year == year
        if mask.sum() < 5:
            continue
        year_nav = nav_values[mask]
        peak = np.maximum.accumulate(year_nav)
        yearly[int(year)] = float(((year_nav - peak) / peak).min())
    return yearly


def _calc_yearly_vol(nav_series: pd.Series) -> Dict[int, float]:
    yearly = {}
    daily_ret = nav_series.pct_change().dropna()
    for year in sorted(daily_ret.index.year.unique()):
        year_ret = daily_ret[daily_ret.index.year == year].values
        if len(year_ret) < 5:
            continue
        yearly[int(year)] = float(np.std(year_ret, ddof=1) * np.sqrt(252))
    return yearly
