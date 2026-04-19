# signals/backtest.py - 轻量回测模块
"""
轻量回测器 - 快速测试信号效果

功能：
1. 输入：信号生成器 + K线数据
2. 计算：仓位、收益、回撤、夏普等
3. 输出：BacktestResult 对象 + Notebook 报告

架构：
  SignalGenerator → 信号 → 仓位转换 → 收益计算 → 指标分析 → 报告

使用示例：
    from signals import MASignal, run_backtest
    
    result = run_backtest(
        generator=MASignal(5, 20),
        data=df,
        initial_capital=1e6
    )
    
    print(f"收益率: {result.total_return:.2f}%")
    print(f"夏普: {result.sharpe:.2f}")
    result.export_report("MA5_20报告")
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import pandas as pd
import numpy as np

from .generator import SignalGenerator
from .threshold import ThresholdPolicy


@dataclass
class BacktestResult:
    """回测结果（纯数据对象）"""
    # 基础数据
    nav: pd.Series          # 策略净值
    benchmark_nav: pd.Series  # 基准净值
    positions: pd.Series    # 仓位序列
    signals: pd.Series      # 信号序列
    data: pd.DataFrame      # 原始K线数据
    
    # 收益指标
    initial_capital: float
    final_nav: float
    total_return: float
    annual_return: float
    benchmark_total_return: float
    benchmark_annual_return: float
    excess_return: float
    excess_annual: float
    
    # 风险指标
    max_drawdown: float
    annual_vol: float
    downside_vol: float
    max_drawdown_duration: float
    
    # 风险调整后收益
    sharpe: float
    sortino: float
    calmar: float
    information_ratio: float
    
    # 交易质量
    trade_count: int
    win_rate: float
    profit_loss_ratio: float
    avg_hold_days: float
    max_consecutive_losses: int
    
    def __repr__(self):
        return (
            f"BacktestResult("
            f"总收益={self.total_return:.2f}%, "
            f"年化={self.annual_return:.2f}%, "
            f"夏普={self.sharpe:.2f}, "
            f"最大回撤={self.max_drawdown:.2f}%, "
            f"胜率={self.win_rate:.1f}%, "
            f"交易次数={self.trade_count})"
        )


def run_backtest(
    generator: SignalGenerator,
    data: pd.DataFrame,
    initial_capital: float = 1_000_000,
    threshold: ThresholdPolicy = None,
    signal_to_position: Callable = None,
    long_only: bool = True,
) -> BacktestResult:
    """
    运行轻量回测
    
    Args:
        generator: 信号生成器
        data: K线数据（需包含 close 列）
        initial_capital: 初始资金
        threshold: 阈值策略（可选）
        signal_to_position: 自定义信号转仓位函数
        long_only: 只做多（默认True）
        
    Returns:
        BacktestResult: 回测结果对象
    """
    # 1. 生成信号
    signal_values = generator.generate(data)
    
    # 2. 转换为仓位
    if signal_to_position:
        positions = signal_values.apply(signal_to_position)
    elif threshold:
        positions = signal_values.apply(lambda x: threshold.apply(x).value)
    elif long_only:
        positions = (signal_values > 0).astype(int)
    else:
        positions = np.sign(signal_values)
    
    # 仓位延迟1天（避免未来函数）
    positions = positions.shift(1).fillna(0)
    
    # 3. 计算收益（仓位再延迟1天，信号发生后才交易）
    df = data.copy()
    df['return'] = df['close'].pct_change()
    
    strategy_returns = positions.shift(1) * df['return']
    benchmark_returns = df['return']
    
    nav = initial_capital * (1 + strategy_returns).cumprod()
    benchmark_nav = initial_capital * (1 + benchmark_returns).cumprod()
    
    # 4. 计算指标
    final_nav = nav.iloc[-1]
    total_return = (final_nav / initial_capital - 1) * 100
    benchmark_total_return = (benchmark_nav.iloc[-1] / initial_capital - 1) * 100
    
    days = (nav.index[-1] - nav.index[0]).days
    years = days / 365.25
    annual_return = ((final_nav / initial_capital) ** (1 / years) - 1) * 100
    benchmark_annual_return = ((benchmark_nav.iloc[-1] / initial_capital) ** (1 / years) - 1) * 100
    
    excess_return = total_return - benchmark_total_return
    excess_annual = annual_return - benchmark_annual_return
    
    # 风险指标
    cumulative_max = nav.expanding().max()
    drawdown = (nav - cumulative_max) / cumulative_max * 100
    max_drawdown = drawdown.min()
    
    benchmark_cumulative_max = benchmark_nav.expanding().max()
    benchmark_drawdown = (benchmark_nav - benchmark_cumulative_max) / benchmark_cumulative_max * 100
    
    daily_vol = strategy_returns.std()
    annual_vol = daily_vol * np.sqrt(252) * 100
    
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
    
    in_drawdown = drawdown < 0
    drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()
    max_drawdown_duration = drawdown_periods[in_drawdown].value_counts().max() if in_drawdown.any() else 0
    
    # 风险调整后收益
    risk_free_rate = 3.0
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    tracking_error = (strategy_returns - benchmark_returns).std() * np.sqrt(252)
    information_ratio = (annual_return - benchmark_annual_return) / tracking_error if tracking_error > 0 else 0
    
    # 交易质量
    position_changes = positions.diff()
    trade_count = (position_changes != 0).sum()
    
    buy_dates = position_changes[position_changes == 1].index
    sell_dates = position_changes[position_changes == -1].index
    
    trades = []
    for buy_date, sell_date in zip(buy_dates[:-1], sell_dates):
        buy_price = data.loc[buy_date, 'close']
        sell_price = data.loc[sell_date, 'close']
        trade_return = (sell_price / buy_price - 1) * 100
        trades.append(trade_return)
    
    winning_trades = [t for t in trades if t > 0]
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
    
    avg_win = np.mean(winning_trades) if winning_trades else 0
    losing_trades = [t for t in trades if t <= 0]
    avg_loss = abs(np.mean(losing_trades)) if losing_trades else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    
    hold_days = []
    for buy_date, sell_date in zip(buy_dates[:-1], sell_dates):
        hold_days.append((sell_date - buy_date).days)
    avg_hold_days = np.mean(hold_days) if hold_days else 0
    
    consecutive_losses = 0
    max_consecutive_losses = 0
    for t in trades:
        if t <= 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0
    
    return BacktestResult(
        nav=nav,
        benchmark_nav=benchmark_nav,
        positions=positions,
        signals=signal_values,
        data=data,
        initial_capital=initial_capital,
        final_nav=final_nav,
        total_return=total_return,
        annual_return=annual_return,
        benchmark_total_return=benchmark_total_return,
        benchmark_annual_return=benchmark_annual_return,
        excess_return=excess_return,
        excess_annual=excess_annual,
        max_drawdown=max_drawdown,
        annual_vol=annual_vol,
        downside_vol=downside_vol,
        max_drawdown_duration=max_drawdown_duration,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        information_ratio=information_ratio,
        trade_count=trade_count,
        win_rate=win_rate,
        profit_loss_ratio=profit_loss_ratio,
        avg_hold_days=avg_hold_days,
        max_consecutive_losses=max_consecutive_losses,
    )
