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
    2. 计算 d_i → d_{i+1} 持有期收益 = sum(w_i * period_returns)
    3. 扣除换手成本 = sum(|w_i - w_{i-1}|) / 2 * cost_rate
    4. 更新净值
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
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings

from .scheduler import RebalanceScheduler, FixedScheduler, IntervalScheduler
from .allocator import WeightAllocator, TopNEqualWeight


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
    win_rate: float = 0.0              # 月度胜率
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
            '月度胜率': f'{self.win_rate:.1%}',
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
    - 在调仓日取因子值 → 分配权重 → 计算持仓收益 → 累加净值
    - 扣除交易成本（换手成本）
    - 计算全套回测指标
    - 支持多频率一键对比
    """

    def __init__(self,
                 returns: pd.DataFrame,
                 scheduler: RebalanceScheduler,
                 allocator: WeightAllocator,
                 cost_rate: float = 0.001,
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
            cost_rate: 单边交易成本率（默认 0.001 = 10bp）
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

    def evaluate(self,
                 factor_values: pd.DataFrame) -> BacktestResult:
        """评估因子回测绩效
        
        核心流程（无前瞻偏差）：
        1. 用 Scheduler 生成调仓日列表
        2. 每个调仓日 d_i:
           a. 取 d_i 日因子值（仅用 d_i 之前数据）
           b. Allocator 分配权重
           c. 计算 d_i→d_{i+1} 持仓收益
           d. 扣除换手成本
        3. 累加净值并计算指标
        
        Args:
            factor_values: 因子值 DataFrame
                          index=日期（与 returns 对齐），columns=标的（与 returns 对齐）
                          注意：因子值应已排除未来数据（如 expanding 计算）
            
        Returns:
            BacktestResult: 完整回测结果
        """
        # 对齐因子值和收益率数据
        returns = self.returns
        if not isinstance(factor_values.index, pd.DatetimeIndex):
            factor_values = factor_values.copy()
            factor_values.index = pd.to_datetime(factor_values.index)

        # 取交集日期
        common_dates = returns.index.intersection(factor_values.index)
        if len(common_dates) == 0:
            return self._empty_result('无共同交易日')

        # 对齐 columns
        common_symbols = returns.columns.intersection(factor_values.columns)
        if len(common_symbols) == 0:
            return self._empty_result('无共同标的')

        returns = returns.loc[common_dates, common_symbols].copy()
        factor_values = factor_values.loc[common_dates, common_symbols].copy()

        # 生成调仓日列表
        rebalance_dates = self.scheduler.generate(returns.index)
        if len(rebalance_dates) < 2:
            return self._empty_result('调仓日不足（需要 ≥2）')

        # 过滤出存在实际数据的调仓日
        valid_rb_dates = [d for d in rebalance_dates if d in returns.index]
        if len(valid_rb_dates) < 2:
            return self._empty_result('有效调仓日不足')

        # 净值序列（从 1.0 开始）
        nav_dates = []
        nav_values = []
        current_nav = 1.0
        prev_weights = None
        turnover_list = []

        for i, rb_date in enumerate(valid_rb_dates[:-1]):
            next_rb_date = valid_rb_dates[i + 1]

            # 1) 取调仓日因子值
            factors_on_date = factor_values.loc[rb_date].values  # shape (M,)

            # 2) 分配权重
            weights = self.allocator.allocate(factors_on_date)

            # 3) 计算换手成本
            if prev_weights is not None and self.cost_rate > 0:
                turnover = np.sum(np.abs(weights - prev_weights)) / 2.0
                cost = turnover * self.cost_rate
                current_nav *= (1.0 - cost)
                turnover_list.append(turnover)
            elif prev_weights is None:
                # 初始建仓也要算半换手（从空仓到满仓）
                if self.cost_rate > 0:
                    build_cost = 0.5 * self.cost_rate  # 单边建仓
                    current_nav *= (1.0 - build_cost)

            prev_weights = weights.copy()

            # 4) 计算持仓期收益（rb_date → next_rb_date）
            # [新增] 2026-05-20 09-20 使用调仓日次日到下一调仓日收盘的收益
            # 避免在调仓日当天使用当天因子值交易后立即获取当天收益
            period_ret = self._calc_period_return(
                returns, rb_date, next_rb_date, weights
            )

            # 更新净值
            current_nav *= (1.0 + period_ret)

            # 记录净值
            nav_dates.append(next_rb_date)
            nav_values.append(current_nav)

        # 构建净值序列
        nav_series = pd.Series(nav_values, index=pd.DatetimeIndex(nav_dates))

        # 5) 计算回测指标
        result = self._calc_metrics(nav_series, turnover_list, valid_rb_dates)
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

    # ================= 内部方法 =================

    def _calc_period_return(self, returns: pd.DataFrame,
                            start_date: pd.Timestamp,
                            end_date: pd.Timestamp,
                            weights: np.ndarray) -> float:
        """计算调仓期间的持仓收益率
        
        [新增] 2026-05-20 09-20 持仓收益计算
        取 start_date 次日至 end_date（含）的日收益率，按权重加总。
        调仓日在 start_date 执行，收益从下一个交易日开始计算。
        
        Args:
            returns: 收益率 DataFrame
            start_date: 调仓日
            end_date: 下一调仓日
            weights: 权重向量
            
        Returns:
            float: 期间总收益率
        """
        try:
            # 取 start_date 之后到 end_date（含）的收益
            mask = (returns.index > start_date) & (returns.index <= end_date)
            period_returns = returns.loc[mask]

            if len(period_returns) == 0:
                return 0.0

            # [修复] 2026-05-21 持仓收益计算 NaN 处理
            # 旧实现: period_returns.values @ weights 若 period_returns 含 NaN → 全 NaN
            # 新实现: dropna() 过滤掉 NaN 行，再用权重累乘
            period_returns_clean = period_returns.dropna(how='any')
            if len(period_returns_clean) == 0:
                return 0.0

            # 期间每期收益 × 权重 → 组合日收益 → 累乘
            daily_portfolio_ret = period_returns_clean.values @ weights
            # 使用对数累加避免浮点精度问题，再转回简单收益率
            log_ret = np.log1p(daily_portfolio_ret).sum()
            period_ret = np.expm1(log_ret)

            return float(period_ret)

        except Exception as e:
            warnings.warn(f"计算期间收益失败 [{start_date}→{end_date}]: {e}")
            return 0.0

    def _calc_metrics(self, nav_series: pd.Series,
                      turnover_list: List[float],
                      rebalance_dates: List[pd.Timestamp]) -> BacktestResult:
        """从净值序列计算全套回测指标
        
        Args:
            nav_series: 净值序列（仅含调仓日时点）
            turnover_list: 每次调仓的单边换手率列表
            rebalance_dates: 调仓日列表
            
        Returns:
            BacktestResult: 回测结果
        """
        if len(nav_series) < 2:
            return self._empty_result('净值数据点不足')

        result = BacktestResult()
        result.nav_series = nav_series
        result.n_rebalances = len(rebalance_dates) - 1  # 实际调仓次数
        result.scheduler_name = repr(self.scheduler)
        result.allocator_name = repr(self.allocator)

        nav_values = nav_series.values
        nav_dates = nav_series.index

        # 起止日期
        result.start_date = str(nav_dates[0].date())
        result.end_date = str(nav_dates[-1].date())

        # 总收益率
        result.total_return = float(nav_values[-1] / nav_values[0] - 1)

        # 调仓日之间的收益率（用于计算 Sharpe 等）
        period_returns = np.diff(nav_values) / nav_values[:-1]

        # 年化收益率
        n_periods = len(period_returns)
        if n_periods > 0:
            # 估算年化：按调仓频率折算
            n_days = (nav_dates[-1] - nav_dates[0]).days
            if n_days > 0:
                years = n_days / 365.25
                result.annual_return = float(
                    (nav_values[-1] / nav_values[0]) ** (1.0 / max(years, 0.01)) - 1
                )
            else:
                result.annual_return = 0.0

            # 年化波动率
            period_vol = np.std(period_returns, ddof=1) if n_periods > 1 else 0.0
            # [修复] 2026-05-20 年化因子说明：非等间隔调仓（如月末）各期间隔不同，
            # 用 period_vol * sqrt(N) 年化是近似值。等间隔场景（IntervalScheduler）精确。
            annual_factor = max(n_periods / max(years, 0.01), 1)
            result.annual_volatility = float(period_vol * np.sqrt(annual_factor))

            # 夏普比率（年化）
            rf_period = self.rf_annual / annual_factor
            excess = period_returns - rf_period
            if result.annual_volatility > 0:
                result.sharpe_ratio = float(
                    (np.mean(excess) / max(period_vol, 1e-10)) * np.sqrt(annual_factor)
                )

            # 索提诺比率：只考虑下行波动
            downside = period_returns[period_returns < 0]
            if len(downside) > 1:
                downside_vol = np.std(downside, ddof=1)
                if downside_vol > 0:
                    result.sortino_ratio = float(
                        (np.mean(period_returns) / downside_vol) * np.sqrt(annual_factor)
                    )

        # 最大回撤
        peak = np.maximum.accumulate(nav_values)
        drawdowns = (nav_values - peak) / peak
        result.max_drawdown = float(drawdowns.min())  # 负值
        result.avg_drawdown = float(drawdowns.mean())

        # 卡玛比率
        if abs(result.max_drawdown) > 1e-10:
            result.calmar_ratio = float(result.annual_return / abs(result.max_drawdown))

        # 换手统计
        if turnover_list:
            result.avg_turnover = float(np.mean(turnover_list))
            # 换手成本占总收益比例
            total_cost = sum(t * self.cost_rate for t in turnover_list)
            total_gross_return = result.total_return + total_cost
            if abs(total_gross_return) > 1e-10:
                result.turnover_cost_ratio = float(total_cost / abs(total_gross_return))
            else:
                result.turnover_cost_ratio = 0.0

        # 月度胜率
        result.win_rate, result.profit_loss_ratio = self._calc_win_rate(period_returns)

        # 年度收益
        result.yearly_returns = self._calc_yearly_returns(nav_series)

        return result

    def _calc_win_rate(self, period_returns: np.ndarray) -> Tuple[float, float]:
        """计算月度胜率和盈亏比
        
        将调仓期收益聚合到月度，计算月度正收益比例。
        
        Args:
            period_returns: 调仓期间收益率数组
            
        Returns:
            Tuple[float, float]: (胜率, 盈亏比)
        """
        if len(period_returns) == 0:
            return 0.0, 0.0

        wins = period_returns[period_returns > 0]
        losses = period_returns[period_returns < 0]
        n_total = len(period_returns)

        win_rate = len(wins) / n_total if n_total > 0 else 0.0

        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        return float(win_rate), float(profit_loss_ratio)

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
