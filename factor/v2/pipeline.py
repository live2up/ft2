"""
因子→回测 桥接层（FactorPipeline）

设计思路：
--------
1. FactorPipeline：补齐 V1 中"因子 IC 验证"与"实际回测绩效"之间的闭环
2. Input = 因子值 Panel + Scheduler + Allocator
3. Output = 持仓净值 + 回测指标（Sharpe/MaxDD/Calmar/年化/换手成本）
4. compare_frequencies()：同一因子在 ME/W/5D 下的多频率对比

核心逻辑（无前瞻偏差）：
----------------------
  对每个调仓日 d_i:
    1. 取 d_i 日因子值 → Allocator 分配权重 w_i
    2. 逐日计算 d_i→d_{i+1} 持仓收益
    3. 扣除换手成本 = sum(|w_i - w_{i-1}|) / 2 * cost_rate
    4. 更新日频净值
  最终计算 Sharpe/MaxDD/Calmar 等指标

使用方式：
--------
>>> pipeline = FactorPipeline(
...     returns=returns_df,         # 各标的逐日收益率
...     scheduler=FixedScheduler('ME'),
...     allocator=TopNEqualWeight(5)
... )
>>> result = pipeline.evaluate(factor_values)
>>> print(f"Sharpe={result.sharpe_ratio:.2f}, MaxDD={result.max_drawdown:.1%}")

>>> # 多频率对比
>>> comp = pipeline.compare_frequencies(factor_values, ['ME', 'W', '5D'])

依赖说明：
--------
依赖 .scheduler（RebalanceScheduler）和 .allocator（WeightAllocator），
依赖 numpy + pandas，不依赖 ft2 core/signals 模块。

[重构] 2026-05-22 evaluate() 分阶段重构：
  - _build_schedule(): 调度日生成与对齐 + 预计算权重
  - _accumulate_nav(): 逐日净值累积
  - _calc_daily_return(): 单日持仓收益（NaN 填 0，不跳过日期）
  - _calc_metrics(): 指标计算
  修复 3 个 P1 bug：
  - P1-1: 最后一个调仓日后收益丢失 → 追加处理末尾持有期
  - P1-2: 初始建仓成本被归一化消除 → 先记录 NAV 再扣成本
  - P1-3: NaN 跳过整天导致 NAV 缺口 → NaN 填 0 保留日期
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings

from .scheduler import RebalanceScheduler, FixedScheduler, IntervalScheduler
from .allocator import WeightAllocator, TopNEqualWeight


@dataclass
class BacktestSchedule:
    """回测调度计划

    [新增] 2026-05-22 evaluate() 分阶段重构的中间数据结构。
    将调度日生成、因子对齐、权重预计算等步骤的结果封装，
    使 _accumulate_nav() 只需关心"给定权重和收益率，计算净值"。

    Attributes:
        returns: 对齐后的日收益率 DataFrame
        factor_values: 对齐后的因子值 DataFrame
        rebalance_dates: 有效调仓日列表（按时间升序）
        weights_at_rebalance: 每个调仓日的权重字典 {date: np.ndarray}
    """
    returns: pd.DataFrame
    factor_values: pd.DataFrame
    rebalance_dates: List[pd.Timestamp]
    weights_at_rebalance: Dict[pd.Timestamp, np.ndarray]


@dataclass
class BacktestResult:
    """因子回测结果

    包含净值曲线和全套绩效指标，可直接用于 Notebook 报告输出。
    """
    # 净值
    nav_series: pd.Series = field(default_factory=pd.Series)

    # 收益指标
    total_return: float = 0.0          # 总收益率
    annual_return: float = 0.0         # 年化收益率
    annual_volatility: float = 0.0     # 年化波动率

    # 风险调整收益
    sharpe_ratio: float = 0.0          # 夏普比率（年化，无风险利率=0）
    sortino_ratio: float = 0.0         # 索提诺比率
    calmar_ratio: float = 0.0          # 卡玛比率

    # 回撤
    max_drawdown: float = 0.0          # 最大回撤（小数，如 -0.15）
    avg_drawdown: float = 0.0          # 平均回撤

    # 交易
    turnover_cost_ratio: float = 0.0   # 换手成本占总收益比例
    avg_turnover: float = 0.0          # 平均单边换手率

    # 胜率
    win_rate: float = 0.0              # 日度胜率
    profit_loss_ratio: float = 0.0     # 盈亏比

    # 年度收益
    yearly_returns: Dict[int, float] = field(default_factory=dict)

    # 元信息
    n_rebalances: int = 0              # 调仓次数
    start_date: str = ""               # 起始日期
    end_date: str = ""                 # 结束日期
    scheduler_name: str = ""           # 调度器名称
    allocator_name: str = ""           # 分配器名称

    def to_dict(self) -> Dict:
        """转为字典，便于序列化和 Notebook 输出"""
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


class FactorPipeline:
    """因子→调仓→回测 完整桥接

    将因子横截面值转化为实际持仓和回测绩效，是 V2 最核心的新增模块。

    职责：
    - 在调仓日取因子值 → 分配权重 → 逐日计算持仓收益 → 累加日频净值
    - 扣除交易成本（换手成本）
    - 计算全套回测指标
    - 支持多频率一键对比

    [重构] 2026-05-22 evaluate() 分阶段重构，修复 P1-1/2/3：
    - _build_schedule(): 调度日 + 因子对齐 + 预计算权重
    - _accumulate_nav(): 逐日净值累积（含末尾期 + NaN 填 0 + 成本顺序修正）
    - _calc_daily_return(): 单日持仓收益
    - _calc_metrics(): 指标计算
    """

    def __init__(self,
                 returns: pd.DataFrame,
                 scheduler: RebalanceScheduler,
                 allocator: WeightAllocator,
                 cost_rate: float = 0.0,
                 rf_annual: float = 0.0):
        """初始化 Pipeline

        Args:
            returns: 各标的逐日收益率 DataFrame
                     index=日期（pd.DatetimeIndex 或可转的字符串索引），
                     columns=标的代码
                     每个单元格 = 该标的在当日的收益率（如日收益率）
                     注意：returns.loc[date] 表示该日可交易后的收益，
                     即如果调仓日在 T，持仓收益 = returns.loc[T+1:T_next].sum()
            scheduler: 调仓日生成器
            allocator: 权重分配器
            cost_rate: 单边交易成本率（默认 0.0 = 不扣成本）
            rf_annual: 年化无风险利率（默认 0）
        """
        # 确保 index 是 DatetimeIndex
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)

        self.returns = returns.sort_index()
        self.scheduler = scheduler
        self.allocator = allocator
        self.cost_rate = cost_rate
        self.rf_annual = rf_annual

    # ================= 公开接口 =================

    def evaluate(self,
                 factor_values: pd.DataFrame) -> BacktestResult:
        """评估因子回测绩效（日频净值）

        核心流程（无前瞻偏差）：
        1. _build_schedule: 生成调仓日 → 对齐因子与收益 → 预计算权重
        2. _accumulate_nav: 逐日计算净值（含末尾期 + NaN 处理 + 成本扣除）
        3. _calc_metrics: 基于日频净值计算指标

        [重构] 2026-05-22 统一为日频净值计算，替代旧版调仓日频率净值。
        旧版问题：仅记录调仓日净值导致 Sharpe/DD 偏高（期间波动被忽略）。
        新版逐日记录净值，指标更精确、回撤更真实。
        同时修复 P1-1（末尾期丢失）、P1-2（建仓成本消除）、P1-3（NaN 跳过整天）。

        Args:
            factor_values: 因子值 DataFrame
                          index=日期（与 returns 对齐），columns=标的（与 returns 对齐）
                          注意：因子值应已排除未来数据（如 expanding 计算）

        Returns:
            BacktestResult: 完整回测结果（nav_series 为日频序列）
        """
        # 阶段 1: 调度与对齐
        schedule = self._build_schedule(factor_values)
        if schedule is None:
            return self._empty_result('调度构建失败')

        # 阶段 2: 逐日净值累积
        nav_series, turnover_list = self._accumulate_nav(schedule)
        if len(nav_series) < 2:
            return self._empty_result('日频净值数据点不足')

        # 阶段 3: 指标计算
        result = self._calc_metrics(nav_series, turnover_list, schedule.rebalance_dates)
        return result

    def compare_frequencies(self,
                            factor_values: pd.DataFrame,
                            freqs: List[str] = None) -> Dict[str, BacktestResult]:
        """多频率对比：同一因子在不同频率下的回测绩效

        复用同一份因子值，仅切换 Scheduler 重新计算持仓。

        Args:
            factor_values: 因子值 DataFrame
            freqs: 频率列表，支持 'ME', 'W', '5D', '10D', '20D' 等
                   None 时默认 ['ME', 'W', '5D']

        Returns:
            Dict[str, BacktestResult]: {'ME': result_me, 'W': result_w, ...}
        """
        if freqs is None:
            freqs = ['ME', 'W', '5D']

        results = {}
        for freq in freqs:
            # 构建对应频率的调度器
            freq_upper = freq.upper()
            if freq_upper in ('ME', 'W', 'M'):
                scheduler = FixedScheduler(freq_upper)
            elif freq.endswith('D') or freq.endswith('d'):
                try:
                    days = int(freq.replace('D', '').replace('d', ''))
                    scheduler = IntervalScheduler(days)
                except ValueError:
                    raise ValueError(f"无法解析频率 '{freq}'，"
                                     f"支持 'ME'/'W'/'5D'/'10D' 等格式")
            else:
                # 尝试作为数字间隔
                try:
                    days = int(freq)
                    scheduler = IntervalScheduler(days)
                except ValueError:
                    raise ValueError(f"无法解析频率 '{freq}'")

            # 临时替换 scheduler 进行评估
            original_scheduler = self.scheduler
            self.scheduler = scheduler
            try:
                result = self.evaluate(factor_values)
                results[freq] = result
            finally:
                self.scheduler = original_scheduler

        return results

    # ================= 阶段 1: 调度构建 =================

    def _build_schedule(self, factor_values: pd.DataFrame) -> Optional[BacktestSchedule]:
        """构建回测调度计划

        [新增] 2026-05-22 evaluate() 分阶段重构。
        将调度日生成、因子对齐、权重预计算等步骤集中于此，
        使 _accumulate_nav() 只需关心"给定权重和收益率，计算净值"。

        Args:
            factor_values: 因子值 DataFrame

        Returns:
            BacktestSchedule 或 None（失败时）
        """
        returns = self.returns
        if not isinstance(factor_values.index, pd.DatetimeIndex):
            factor_values = factor_values.copy()
            factor_values.index = pd.to_datetime(factor_values.index)

        common_dates = returns.index.intersection(factor_values.index)
        if len(common_dates) == 0:
            warnings.warn("因子回测: 无共同交易日")
            return None

        common_symbols = returns.columns.intersection(factor_values.columns)
        if len(common_symbols) == 0:
            warnings.warn("因子回测: 无共同标的")
            return None

        returns = returns.loc[common_dates, common_symbols].copy()
        factor_values = factor_values.loc[common_dates, common_symbols].copy()

        rebalance_dates = self.scheduler.generate(returns.index)
        if len(rebalance_dates) < 2:
            warnings.warn("因子回测: 调仓日不足（需要 ≥2）")
            return None

        valid_rb_dates = [d for d in rebalance_dates if d in returns.index]
        if len(valid_rb_dates) < 2:
            warnings.warn("因子回测: 有效调仓日不足")
            return None

        # 预计算每个调仓日的权重
        weights_at_rebalance = {}
        for rb_date in valid_rb_dates:
            factors_on_date = factor_values.loc[rb_date].values
            weights = self.allocator.allocate(factors_on_date)
            weights_at_rebalance[rb_date] = weights

        return BacktestSchedule(
            returns=returns,
            factor_values=factor_values,
            rebalance_dates=valid_rb_dates,
            weights_at_rebalance=weights_at_rebalance,
        )

    # ================= 阶段 2: 逐日净值累积 =================

    def _accumulate_nav(self, schedule: BacktestSchedule) -> Tuple[pd.Series, List[float]]:
        """逐日累积净值

        [新增] 2026-05-22 evaluate() 分阶段重构。
        纯计算逻辑：给定调度计划和预计算权重，逐日累积净值。

        修复 3 个 P1 bug：
        - P1-1: 最后一个调仓日后收益丢失 → 追加处理末尾持有期
        - P1-2: 初始建仓成本被归一化消除 → 先记录 NAV=1.0 再扣成本
        - P1-3: NaN 跳过整天导致 NAV 缺口 → NaN 填 0 保留日期

        Args:
            schedule: 回测调度计划

        Returns:
            (nav_series, turnover_list): 日频净值序列和换手率列表
        """
        returns = schedule.returns
        rb_dates = schedule.rebalance_dates

        nav_dates = []
        nav_values = []
        current_nav = 1.0
        prev_weights = None
        turnover_list = []

        # [修复] P1-2: 先记录初始净值 NAV=1.0，再扣建仓成本
        # 旧版：先扣成本再记录 → 归一化后建仓成本消失
        nav_dates.append(rb_dates[0])
        nav_values.append(current_nav)

        for i, rb_date in enumerate(rb_dates):
            weights = schedule.weights_at_rebalance[rb_date]

            # 成本扣除
            if prev_weights is not None and self.cost_rate > 0:
                turnover = np.sum(np.abs(weights - prev_weights)) / 2.0
                current_nav *= (1.0 - turnover * self.cost_rate)
                turnover_list.append(turnover)
            elif prev_weights is None and self.cost_rate > 0:
                build_cost = 0.5 * self.cost_rate
                current_nav *= (1.0 - build_cost)
            prev_weights = weights.copy()

            # 确定持有期: 当前调仓日 → 下一调仓日 或 数据末尾
            if i < len(rb_dates) - 1:
                next_date = rb_dates[i + 1]
                mask = (returns.index > rb_date) & (returns.index <= next_date)
            else:
                # [修复] P1-1: 最后一个调仓日，持有到数据末尾
                # 旧版：valid_rb_dates[:-1] 不包含最后一个调仓日，末尾期收益丢失
                mask = returns.index > rb_date

            period_returns = returns.loc[mask]

            for dt in period_returns.index:
                daily_ret = self._calc_daily_return(weights, period_returns.loc[dt])
                current_nav *= (1.0 + daily_ret)
                nav_dates.append(dt)
                nav_values.append(current_nav)

        if len(nav_dates) < 2:
            return pd.Series(dtype=float), turnover_list

        nav_series = pd.Series(
            nav_values,
            index=pd.DatetimeIndex(nav_dates)
        ).drop_duplicates().sort_index()
        nav_series = nav_series / nav_series.iloc[0]

        return nav_series, turnover_list

    def _calc_daily_return(self, weights: np.ndarray,
                           day_returns: pd.Series) -> float:
        """计算单日持仓收益

        [新增] 2026-05-22 evaluate() 分阶段重构。
        从主循环中提取，消除代码重复，且独立可测。

        [修复] P1-3: NaN 填 0 保留日期，替代旧版 continue 跳过整天。
        旧版问题：持仓标的有 NaN 收益时 continue 跳过整天 → NAV 序列出现缺口
        → pct_change() 跨缺口计算多日收益当作单日 → Sharpe 偏高
        新版：NaN 仓位按 0 收益处理（保守：视为空仓），日期正常记录。

        Args:
            weights: 持仓权重数组，shape (N,)
            day_returns: 当日收益率 Series，index=标的代码

        Returns:
            float: 当日组合收益
        """
        nonzero = weights > 0
        if not nonzero.any():
            return 0.0

        sel_ret = day_returns.values[nonzero].copy()
        sel_w = weights[nonzero]

        # NaN 填 0（保守：NaN 仓位视为空仓，不重归一化权重）
        sel_ret = np.where(np.isnan(sel_ret), 0.0, sel_ret)

        return float(np.dot(sel_w, sel_ret))

    # ================= 阶段 3: 指标计算 =================

    def _calc_metrics(self, nav_series: pd.Series,
                      turnover_list: List[float],
                      rebalance_dates: List[pd.Timestamp]) -> BacktestResult:
        """从日频净值序列计算回测指标

        [重构] 2026-05-22 统一基于日频净值计算指标。
        旧版基于调仓期间收益计算 Sharpe/DD，存在采样偏差（偏高）。
        新版基于日收益率计算，更精确、更接近真实交易体验。

        Args:
            nav_series: 日频净值序列
            turnover_list: 每次调仓的单边换手率列表
            rebalance_dates: 调仓日列表

        Returns:
            BacktestResult: 回测结果
        """
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

            result.annual_return = float(
                (nav_values[-1] / nav_values[0]) ** (1.0 / years) - 1
            )

            daily_vol = np.std(daily_ret, ddof=1) if n_days > 1 else 0.0
            result.annual_volatility = float(daily_vol * np.sqrt(252))

            rf_daily = self.rf_annual / 252
            excess_daily = daily_ret - rf_daily
            if result.annual_volatility > 0:
                result.sharpe_ratio = float(
                    np.mean(excess_daily) / max(daily_vol, 1e-10) * np.sqrt(252)
                )

            downside = daily_ret[daily_ret < 0]
            if len(downside) > 1:
                downside_vol = np.std(downside, ddof=1)
                if downside_vol > 0:
                    result.sortino_ratio = float(
                        np.mean(excess_daily) / downside_vol * np.sqrt(252)
                    )

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
            if abs(total_gross_return) > 1e-10:
                result.turnover_cost_ratio = float(total_cost / abs(total_gross_return))
            else:
                result.turnover_cost_ratio = 0.0

        if n_days > 0:
            wins = daily_ret[daily_ret > 0]
            losses = daily_ret[daily_ret < 0]
            result.win_rate = float(len(wins) / n_days)
            avg_win = np.mean(wins) if len(wins) > 0 else 0.0
            avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0
            result.profit_loss_ratio = float(avg_win / avg_loss) if avg_loss > 0 else 0.0

        result.yearly_returns = self._calc_yearly_returns(nav_series)

        return result

    # ================= 辅助方法 =================

    def _calc_yearly_returns(self, nav_series: pd.Series) -> Dict[int, float]:
        """计算年度收益率

        Args:
            nav_series: 净值序列

        Returns:
            Dict[int, float]: {年份: 收益率}
        """
        yearly = {}
        nav_dates = nav_series.index
        nav_values = nav_series.values

        for year in range(nav_dates[0].year, nav_dates[-1].year + 1):
            year_start = None
            year_end = None
            for i, d in enumerate(nav_dates):
                if d.year == year and year_start is None:
                    year_start = i
                if d.year == year:
                    year_end = i

            if year_start is not None and year_end is not None and year_end > year_start:
                ret = nav_values[year_end] / nav_values[year_start] - 1
                yearly[year] = float(ret)

        return yearly

    def _empty_result(self, reason: str = '') -> BacktestResult:
        """返回空结果

        Args:
            reason: 空结果原因

        Returns:
            BacktestResult: 空结果
        """
        result = BacktestResult()
        if reason:
            warnings.warn(f"因子回测返回空结果: {reason}")
        return result

    def __repr__(self) -> str:
        return (f"FactorPipeline(scheduler={self.scheduler}, "
                f"allocator={self.allocator}, cost_rate={self.cost_rate})")
