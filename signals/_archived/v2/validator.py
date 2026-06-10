"""
signals/v2/validator.py — 内置验证框架
=============================================================================

提供完整的回测和验证能力，不再需要外部脚本手动实现。


============================================================================
                         架构层级（竖式）
============================================================================

 ┌─────────────────────────────────────────────────────────────────────┐
 │  输入：Expression + OHLCV DataFrame + 初始资金                        │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 1  —  信号生成                                                  │
 │                                                                      │
 │  signal_values = expr.generate(data)  →  pd.Series (1.0 or 0.0)     │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 2  —  信号 → 持仓 (T+1 防未来函数)                             │
 │                                                                      │
 │  long_only=True:                                                     │
 │    positions = (signal > 0).astype(int)   # 满仓 1 / 空仓 0          │
 │    positions = positions.shift(1).fillna(0) # 今日信号→明日执行       │
 │                                                                      │
 │  long_only=False:                                                    │
 │    positions = np.sign(signal_values)      # 多空双向                 │
 │    positions = positions.shift(1).fillna(0)                          │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 3  —  净值计算 (_compute_backtest)                              │
 │                                                                      │
 │  strategy_returns  = positions × close.pct_change()                  │
 │  benchmark_returns = close.pct_change()        # 买入持有             │
 │                                                                      │
 │  nav           = capital × (1 + strategy_returns).cumprod()          │
 │  benchmark_nav = capital × (1 + benchmark_returns).cumprod()         │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 4  —  绩效指标计算 (BacktestResult)                             │
 │                                                                      │
 │  ┌─────────────────────┬──────────────────────────────────────┐     │
 │  │   收益类             │  总收益 / 年化收益 / 超额收益         │     │
 │  ├─────────────────────┼──────────────────────────────────────┤     │
 │  │   风险类             │  最大回撤 / 年化波动 / 下行波动       │     │
 │  │                     │  最大回撤持续期                       │     │
 │  ├─────────────────────┼──────────────────────────────────────┤     │
 │  │   风险调整收益       │  Sharpe / Sortino / Calmar / IR      │     │
 │  │                     │  (无风险利率 = 3%)                   │     │
 │  ├─────────────────────┼──────────────────────────────────────┤     │
 │  │   交易质量           │  交易次数 / 胜率 / 盈亏比             │     │
 │  │                     │  平均持仓天数 / 最大连续亏损           │     │
 │  └─────────────────────┴──────────────────────────────────────┘     │
 └─────────────────────────────────────────────────────────────────────┘


============================================================================
                         Walk-Forward 验证 (walk_forward)
============================================================================

 ┌──────────────────────────────────────────────────────────────────────┐
 │  滚动窗口验证：训练集→测试集→滑动→重复                                │
 │                                                                       │
 │  train_size='2Y', test_size='1Y', step='6M'                          │
 │                                                                       │
 │  ┌──────────┬──────────┐                                             │
 │  │  train   │  test    │  → 记录 test_sharpe                         │
 │  └──────────┴──────────┘                                             │
 │            ┌──────────┬──────────┐                                    │
 │            │  train   │  test    │  → 记录 (滑动 6M)                  │
 │            └──────────┴──────────┘                                    │
 │                       ┌──────────┬──────────┐                         │
 │                       │  train   │  test    │  → ...                  │
 │                       └──────────┴──────────┘                         │
 │                                                                       │
 │  输出 WalkForwardResult:                                              │
 │    · mean_sharpe / negative_count / stability_score                  │
 │    · decay_curve (过拟合衰减曲线)                                     │
 │    · windows (每个窗口详细指标)                                        │
 └──────────────────────────────────────────────────────────────────────┘


============================================================================
                         参数规范
============================================================================

 ┌─────────────────────┬─────────────────────────────────────────────┐
 │       参数           │       说明 / 默认值                          │
 ├─────────────────────┼─────────────────────────────────────────────┤
 │ initial_capital     │ 初始资金 (默认 1,000,000)                     │
 │ long_only           │ 仅做多? True=满仓/空仓, False=多空双向       │
 │ risk_free_rate      │ 无风险利率 (固定 3%)                         │
 │ train_size / test   │ Walk-Forward 窗口参数                        │
 │ anchor_start        │ True=锚定起点(扩展窗口), False=滚动窗口       │
 └─────────────────────┴─────────────────────────────────────────────┘

============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field

# [修复] 2026-05-20 消除硬编码绝对路径，改为 os.path 相对路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

from .expression import Expression, TreeNode


@dataclass
class BacktestResult:
    """回测结果（纯数据对象）"""
    nav: pd.Series
    benchmark_nav: pd.Series
    positions: pd.Series
    signals: pd.Series
    data: pd.DataFrame

    initial_capital: float
    final_nav: float
    total_return: float
    annual_return: float
    benchmark_total_return: float
    benchmark_annual_return: float
    excess_return: float
    excess_annual: float

    max_drawdown: float
    annual_vol: float
    downside_vol: float
    max_drawdown_duration: float

    sharpe: float
    sortino: float
    calmar: float
    information_ratio: float

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
            f"交易={self.trade_count})"
        )

    @property
    def annual_return_pct(self) -> float:
        return self.annual_return / 100.0


@dataclass
class WalkForwardResult:
    """Walk-Forward验证结果"""
    windows: List[Dict]
    summary: Dict
    decay_curve: Dict

    @property
    def mean_sharpe(self) -> float:
        return self.summary.get('mean_sharpe', 0.0)

    @property
    def negative_sharpe_count(self) -> int:
        return self.summary.get('negative_sharpe_count', 0)

    @property
    def total_windows(self) -> int:
        return len(self.windows)

    @property
    def stability_score(self) -> float:
        """参数稳定性得分 = 均值/标准差"""
        sharpes = [w.get('sharpe', 0) for w in self.windows]
        if len(sharpes) < 2:
            return 1.0
        mean_s = np.mean(sharpes)
        std_s = np.std(sharpes, ddof=1)
        if std_s == 0:
            return 0.0
        return mean_s / std_s

    def __repr__(self):
        return (
            f"WalkForwardResult("
            f"窗口={self.total_windows}, "
            f"均值夏普={self.mean_sharpe:.2f}, "
            f"稳定性={self.stability_score:.2f})"
        )


# [弃用] 2026-06-10 建议用 core_backtest() 替代
#   简化引擎的 Sharpe 比 core 引擎虚高 15~20%，且无费率/交易记录
#   仅 gp_optimizer/grid_search 因性能原因保留使用
def _compute_backtest(positions: pd.Series, data: pd.DataFrame,
                       signals: pd.Series = None,
                       initial_capital: float = 1_000_000,
                       commission: float = 0.0,
                       slippage: float = 0.0) -> BacktestResult:
    """核心回测计算逻辑
    [修复] 预留 commission/slippage 交易成本接口，默认0(先筛选信号再考虑成本)
    修复交易次数统计为完整回合数"""
    df = data.copy()
    df['return'] = df['close'].pct_change()

    # [新增] 交易成本: 每次换手扣减 commission(双边佣金) + slippage(滑点)
    # position_change != 0 表示发生交易，成本 = |仓位变化| * (commission + slippage)
    position_changes = positions.diff().fillna(0)
    cost_rate = commission * 2 + slippage  # 买入佣金 + 卖出佣金 + 单边滑点
    transaction_costs = position_changes.abs() * cost_rate

    strategy_returns = (positions * df['return'] - transaction_costs).fillna(0)
    benchmark_returns = df['return'].fillna(0)

    nav = initial_capital * (1 + strategy_returns).cumprod()
    benchmark_nav = initial_capital * (1 + benchmark_returns).cumprod()

    final_nav = float(nav.iloc[-1])
    total_return = (final_nav / initial_capital - 1) * 100
    benchmark_total_return = (float(benchmark_nav.iloc[-1]) / initial_capital - 1) * 100

    try:
        days = (nav.index[-1] - nav.index[0]).days
    except TypeError:
        days = len(nav) - 1
    years = max(days / 365.25, 0.1)
    annual_return = ((final_nav / initial_capital) ** (1 / years) - 1) * 100 if years > 0 and final_nav > 0 else 0
    benchmark_annual_return = ((float(benchmark_nav.iloc[-1]) / initial_capital) ** (1 / years) - 1) * 100

    excess_return = total_return - benchmark_total_return
    excess_annual = annual_return - benchmark_annual_return

    cumulative_max = nav.expanding().max()
    drawdown = (nav - cumulative_max) / cumulative_max * 100
    max_drawdown = float(drawdown.min())

    daily_vol = strategy_returns.std()
    annual_vol = daily_vol * np.sqrt(252) * 100

    downside_returns = strategy_returns[strategy_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0

    in_drawdown = drawdown < 0
    if in_drawdown.any():
        drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()
        max_drawdown_duration = float(drawdown_periods[in_drawdown].value_counts().max())
    else:
        max_drawdown_duration = 0

    risk_free_rate = 3.0
    sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
    sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    tracking_error = (strategy_returns - benchmark_returns).std() * np.sqrt(252)
    information_ratio = (annual_return - benchmark_annual_return) / tracking_error if tracking_error > 0 else 0

    # 交易质量
    # [修复] 交易次数改为完整回合数（一买一卖=1个回合），而非持仓变化次数
    # 旧实现: trade_count = position_changes != 0 的次数，0→1→0→1→0 = 4次
    # 新实现: trade_count = 完整买入→卖出回合数，0→1→0→1→0 = 2次
    position_changes = positions.diff()
    buy_dates = position_changes[position_changes == 1].index
    sell_dates = position_changes[position_changes == -1].index
    trade_count = min(len(buy_dates), len(sell_dates))
    # 如果最后还有未平仓的买入，算半个回合（不计入完整交易，但在hold_days里体现）
    open_trades = len(buy_dates) - len(sell_dates)

    trades = []
    for bd, sd in zip(buy_dates[:-1], sell_dates):
        if bd in data.index and sd in data.index:
            buy_price = float(data.loc[bd, 'close'])
            sell_price = float(data.loc[sd, 'close'])
            if buy_price > 0:
                trades.append((sell_price / buy_price - 1) * 100)

    winning_trades = [t for t in trades if t > 0]
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

    avg_win = np.mean(winning_trades) if winning_trades else 0
    losing_trades = [t for t in trades if t <= 0]
    avg_loss = abs(np.mean(losing_trades)) if losing_trades else 1
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    hold_days_vals = []
    for bd, sd in zip(buy_dates[:-1], sell_dates):
        hold_days_vals.append((sd - bd).days)
    avg_hold_days = float(np.mean(hold_days_vals)) if hold_days_vals else 0

    consecutive_losses = 0
    max_consecutive_losses = 0
    for t in trades:
        if t <= 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0

    if signals is None:
        signals = positions.copy()

    return BacktestResult(
        nav=nav, benchmark_nav=benchmark_nav, positions=positions,
        signals=signals, data=data,
        initial_capital=initial_capital, final_nav=final_nav,
        total_return=total_return, annual_return=annual_return,
        benchmark_total_return=benchmark_total_return,
        benchmark_annual_return=benchmark_annual_return,
        excess_return=excess_return, excess_annual=excess_annual,
        max_drawdown=max_drawdown, annual_vol=annual_vol,
        downside_vol=downside_vol, max_drawdown_duration=max_drawdown_duration,
        sharpe=sharpe, sortino=sortino, calmar=calmar,
        information_ratio=information_ratio,
        trade_count=trade_count, win_rate=win_rate,
        profit_loss_ratio=profit_loss_ratio,
        avg_hold_days=avg_hold_days,
        max_consecutive_losses=int(max_consecutive_losses),
    )


# [弃用] 2026-06-10 建议用 run_backtest_with_core() / core_backtest() 替代
def run_backtest(expr_or_gen: Any, data: pd.DataFrame,
                 initial_capital: float = 1_000_000,
                 long_only: bool = True,
                 commission: float = 0.0,
                 slippage: float = 0.0) -> BacktestResult:
    """
    运行单次回测，支持 Expression 和 旧 SignalGenerator

    Args:
        expr_or_gen: Expression实例 或 旧SignalGenerator
        data: K线数据
        initial_capital: 初始资金
        long_only: 仅做多
        commission: 单边佣金费率 (默认0，筛选阶段不扣成本)
        slippage: 滑点费率 (默认0，筛选阶段不扣成本)
    """
    # 判断输入类型
    if hasattr(expr_or_gen, 'generate'):
        # 旧 SignalGenerator
        signal_values = expr_or_gen.generate(data)
    elif hasattr(expr_or_gen, 'generate_from_features'):
        # Expression with feature_df
        signal_values = expr_or_gen.generate(data)
    elif isinstance(expr_or_gen, Expression):
        signal_values = expr_or_gen.generate(data)
    else:
        raise ValueError(f"不支持的类型: {type(expr_or_gen)}")

    if long_only:
        positions = (signal_values > 0).astype(int)
    else:
        positions = np.sign(signal_values)

    positions = positions.shift(1).fillna(0)
    return _compute_backtest(positions, data, signal_values, initial_capital,
                              commission=commission, slippage=slippage)


def run_backtest_from_signal(signals: Union[pd.Series, np.ndarray],
                              prices: Union[pd.Series, np.ndarray],
                              data: Optional[pd.DataFrame] = None,
                              initial_capital: float = 1_000_000,
                              long_only: bool = True) -> BacktestResult:
    """
    直接从信号序列回测 — 不再需要构造假的generator！

    Args:
        signals: 信号序列
        prices: 价格序列（收盘价）
        data: 完整的OHLCV DataFrame（可选，不提供则会自动构造）
        initial_capital: 初始资金
        long_only: 仅做多
    """
    if isinstance(signals, np.ndarray):
        signals = pd.Series(signals)

    if isinstance(prices, pd.Series):
        prices_arr = prices.values
        idx = prices.index
    else:
        prices_arr = prices
        idx = pd.RangeIndex(len(prices))

    if data is not None:
        idx = data.index
        if isinstance(signals, pd.Series) and signals.index.equals(idx):
            pass
        else:
            arr = np.array(signals).ravel()
            if len(arr) == len(idx):
                signals = pd.Series(arr, index=idx)
            else:
                s = pd.Series(np.nan, index=idx)
                start = len(idx) - len(arr)
                s.iloc[start:] = arr
                signals = s.fillna(0)

    if long_only:
        positions = (signals > 0).astype(int)
    else:
        positions = signals.apply(np.sign) if hasattr(signals, 'apply') else np.sign(signals)

    positions = positions.shift(1).fillna(0)

    if data is None:
        data = pd.DataFrame({
            'open': prices_arr,
            'high': prices_arr,
            'low': prices_arr,
            'close': prices_arr,
            'volume': np.ones(len(prices_arr)),
        }, index=idx)
    else:
        data = data.copy()
        if not positions.index.equals(data.index):
            positions.index = data.index

    return _compute_backtest(positions, data, signals, initial_capital)


def walk_forward(expr_or_gen: Any, data: pd.DataFrame,
                 train_size: str = '2Y',
                 test_size: str = '1Y',
                 step: str = '6M',
                 initial_capital: float = 1_000_000,
                 anchor_start: bool = False) -> WalkForwardResult:
    """
    Walk-Forward验证

    Args:
        expr_or_gen: Expression 或 旧 SignalGenerator
        data: K线数据
        train_size: 训练窗口 ('2Y' = 2年)
        test_size: 测试窗口
        step: 步进长度
        initial_capital: 初始资金
        anchor_start: True=锚定起始点(扩展窗口), False=滚动窗口

    Returns:
        WalkForwardResult
    """
    # 解析时间窗口
    def _parse_timedelta(s: str) -> pd.Timedelta:
        s = s.strip()
        unit_map = {'Y': 'days', 'M': 'days', 'D': 'days', 'y': 'days'}
        unit = 'days'
        for u in unit_map:
            if s.endswith(u):
                unit = 'years' if u.lower() == 'y' else 'months'
                s = s[:-1]
                break
        val = int(s)
        if unit == 'years':
            return pd.Timedelta(days=int(val * 365.25))
        elif unit == 'months':
            return pd.Timedelta(days=int(val * 30.5))
        return pd.Timedelta(days=val)

    train_td = _parse_timedelta(train_size)
    test_td = _parse_timedelta(test_size)
    step_td = _parse_timedelta(step)

    total_span = data.index[-1] - data.index[0]
    windows = []

    train_start = data.index[0]
    while train_start + train_td + test_td < data.index[-1]:
        train_end = train_start + train_td
        test_end = train_end + test_td

        train_data = data.loc[(data.index >= train_start) & (data.index < train_end)]
        test_data = data.loc[(data.index >= train_end) & (data.index < test_end)]

        if len(train_data) < 30 or len(test_data) < 10:
            break

        try:
            # [修复] Walk-Forward 必须重置 _feature_df 和 _feature_space，消除前瞻偏差
            # 旧实现: 只重置 _feature_space=None，但 _feature_df 优先于 _feature_space
            #         generate() 走 _feature_df 分支时，用的是全量数据计算的特征
            #         导致 train/test 子集实际上用了包含未来数据的特征矩阵
            # 新实现: 同时重置 _feature_df=None 和 _feature_space=None，
            #         强制每个窗口用各自的子数据重新计算特征，确保无前瞻偏差
            if isinstance(expr_or_gen, Expression):
                expr_or_gen._feature_df = None
                expr_or_gen._feature_space = None
            train_result = run_backtest(expr_or_gen, train_data, initial_capital)

            # 测试集回测 — 同样需要重置，防止用训练集的特征
            if isinstance(expr_or_gen, Expression):
                expr_or_gen._feature_df = None
                expr_or_gen._feature_space = None
            test_result = run_backtest(expr_or_gen, test_data, initial_capital)

            windows.append({
                'train_start': str(train_start.date()),
                'train_end': str(train_end.date()),
                'test_start': str(train_end.date()),
                'test_end': str(test_end.date()),
                'train_sharpe': train_result.sharpe,
                'test_sharpe': test_result.sharpe,
                'train_annual': train_result.annual_return,
                'test_annual': test_result.annual_return,
                'train_drawdown': train_result.max_drawdown,
                'test_drawdown': test_result.max_drawdown,
                'train_trades': train_result.trade_count,
                'test_trades': test_result.trade_count,
            })
        except Exception as e:
            windows.append({
                'train_start': str(train_start.date()),
                'train_end': str(train_end.date()),
                'test_start': str(train_end.date()),
                'test_end': str(test_end.date()),
                'error': str(e),
                'train_sharpe': 0,
                'test_sharpe': 0,
            })

        if anchor_start:
            train_start = data.index[0]
            train_td += step_td
            if train_td > test_td * 6:
                break
        else:
            train_start += step_td

    # 汇总统计
    test_sharpes = [w.get('test_sharpe', 0) for w in windows if 'error' not in w]
    train_sharpes = [w.get('train_sharpe', 0) for w in windows if 'error' not in w]
    test_annuals = [w.get('test_annual', 0) for w in windows if 'error' not in w]

    summary = {
        'total_windows': len(windows),
        'mean_test_sharpe': float(np.mean(test_sharpes)) if test_sharpes else 0,
        'min_test_sharpe': float(np.min(test_sharpes)) if test_sharpes else 0,
        'max_test_sharpe': float(np.max(test_sharpes)) if test_sharpes else 0,
        'negative_count': sum(1 for s in test_sharpes if s < 0),
        'positive_count': sum(1 for s in test_sharpes if s >= 0),
        'mean_sharpe': float(np.mean(test_sharpes)) if test_sharpes else 0,
        'negative_sharpe_count': sum(1 for s in test_sharpes if s < 0),
        'std_test_sharpe': float(np.std(test_sharpes, ddof=1)) if len(test_sharpes) > 1 else 0,
        'stability_score': float(np.mean(test_sharpes) / np.std(test_sharpes, ddof=1)) if len(test_sharpes) > 1 and np.std(test_sharpes) > 0 else 0,
        'mean_test_annual': float(np.mean(test_annuals)) if test_annuals else 0,
        'mean_test_sharpe_2': float(np.mean(test_sharpes)) if test_sharpes else 0,
    }

    # 衰减曲线
    decay_curve = {
        'train_sharpes': train_sharpes,
        'test_sharpes': test_sharpes,
        'overfit_ratios': [t / s if s > 0 else 0 for t, s in zip(train_sharpes, test_sharpes)],
    }

    return WalkForwardResult(windows=windows, summary=summary, decay_curve=decay_curve)


def compare(*expressions: Any, data: pd.DataFrame,
            initial_capital: float = 1_000_000) -> pd.DataFrame:
    """
    多表达式对比回测

    Args:
        expressions: Expression列表
        data: K线数据
        initial_capital: 初始资金

    Returns:
        DataFrame: 对比表
    """
    results = []
    for expr in expressions:
        try:
            result = run_backtest(expr, data, initial_capital)
            results.append({
                '表达式': expr.source[:50] if hasattr(expr, 'source') else str(expr)[:50],
                '夏普': round(result.sharpe, 3),
                '年化(%)': round(result.annual_return, 2),
                '回撤(%)': round(result.max_drawdown, 2),
                '胜率(%)': round(result.win_rate, 1),
                '交易': result.trade_count,
                '复杂度': getattr(expr, 'complexity', 0),
            })
        except Exception as e:
            results.append({
                '表达式': str(expr)[:50],
                '夏普': 'ERROR',
                '年化(%)': str(e)[:30],
                '回撤(%)': 0,
                '胜率(%)': 0,
                '交易': 0,
                '复杂度': 0,
            })
    return pd.DataFrame(results)


# ============================================================
# [新增] 2026-06-10 core_backtest — 纯信号→core引擎 一键回测
# ============================================================
# 设计原则:
#   1. 只接受 pd.Series 信号（预计算），不接受 Expression/多态
#   2. 信号时序: signal[t] → bar[t] 触发建仓/清仓（引擎逐bar递进天然T+1）
#   3. 费率由 core.AccountManager 自动处理（ETF万3佣金+千1印花税）
#   4. 返回 AccountAnalyzer → 链式调用 to_notebook/set_benchmark/to_excel
#
# 调用链: signal + ohlcv → embed _signal → Engine.run() → AccountAnalyzer
# ============================================================

def core_backtest(
    signal: 'pd.Series',
    data: 'pd.DataFrame',
    symbol: str = '399317.SZ',
    freq: str = '1d',
    initial_capital: float = 1_000_000,
    start_date: str = None,
    note_fields: List[str] = None,
) -> 'AccountAnalyzer':
    """
    用 ft2.core 引擎回测择时信号（纯信号入口，不依赖 Expression）。

    Args:
        signal: pd.Series，index=日期，value>0 做多、value<=0 空仓
        data: OHLCV DataFrame，必须含 close/open 列，index=DatetimeIndex
        symbol: 交易标的代码（'399317.SZ' 等）
        freq: 数据频率（'1d' 等，作为引擎缓存 key）
        initial_capital: 初始资金
        start_date: 回测起始日期（'2020-01-01'），None=从数据首位开始。
                    设置后策略和基准统一从该日期起步，净值起点一致。
        note_fields: 附加写入 TradeRecord.note 的 bar 字段名（如 ['close','volume']）

    Returns:
        AccountAnalyzer: 含完整指标/交易记录，可链式调用:
            analyzer.set_benchmark(...).to_notebook("标题")

    Example:
        >>> from signals.v2 import core_backtest
        >>> # signal 已预先计算好
        >>> analyzer = core_backtest(my_signal, ohlcv_df, symbol='399317.SZ',
        ...                          start_date='2020-01-01')
        >>> analyzer.set_benchmark(bench_nav, '买入持有').to_notebook("ATR择时")
    """
    from core.engine import Engine
    from core.account import OrderSide
    from core.storage import context
    from core.analyzer import AccountAnalyzer

    # ── 1. 信号对齐（全长保留，缺失填充 0=空仓）──
    if not isinstance(signal, pd.Series):
        raise TypeError(f"signal 必须是 pd.Series，收到 {type(signal)}")
    if not isinstance(signal.values[0], (int, float, np.floating, np.integer)):
        raise TypeError(f"signal 值必须是数值类型，收到 {type(signal.values[0])}")

    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if 'eob' not in df.columns:
        df['eob'] = df.index

    # [修复] 2026-06-10 去掉 common_idx 截断 → 用全量数据 + fillna(0)
    #   旧逻辑: df = df.loc[common_idx] — 信号没有的日期被丢弃
    #   结果: 空仓期的净值记录丢失，引擎从第一个信号日开始 → daily_assets 缺头
    #   新逻辑: 信号对齐到数据全长，缺失处填 0 → 空仓期保留 INITIAL_CAPITAL 净值
    df['_signal'] = signal.reindex(df.index).fillna(0).values

    # [新增] 2026-06-10 start_date 截断 → 策略和基准统一起点
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        df = df.loc[df.index >= start_ts].copy()
        if len(df) < 2:
            import warnings
            warnings.warn(f"core_backtest: start_date={start_date} 后数据不足2条")
            return AccountAnalyzer(account=None)

    if len(df) < 2:
        import warnings
        warnings.warn("core_backtest: 有效数据不足2条")
        return AccountAnalyzer(account=None)

    # ── 2. 引擎初始化 ──
    engine = Engine(init_cash=initial_capital)
    context.mode = 'backtest'
    context.unsubscribe(symbol, freq)
    context.subscribe(symbol, freq, count=300)
    engine.add_data(symbol, freq, df)

    note_fields = note_fields or []

    class _SignalStrategy:
        def on_bar(self, ctx, bars):
            bar = bars[0]
            sig = bar.get('_signal', 0)
            has_pos = bool(ctx.account.get_position())

            target_long = sig > 0

            note_parts = [f"signal={sig:.4f}"]
            for f in note_fields:
                if f in bar:
                    note_parts.append(f"{f}={bar[f]}")
            note = ', '.join(note_parts)

            if target_long and not has_pos:
                try:
                    ctx.account.order_percent(symbol, 1.0, OrderSide.Buy, note=note)
                except ValueError:
                    pass
            elif not target_long and has_pos:
                try:
                    ctx.account.order_percent(symbol, 1.0, OrderSide.Sell, note=note)
                except ValueError:
                    pass

    start_time = df['eob'].iloc[0]
    end_time = df['eob'].iloc[-1]

    try:
        engine.run(_SignalStrategy, start_time, end_time)
    except Exception as e:
        import warnings
        warnings.warn(f"core_backtest: Engine 回测异常: {e}")

    return AccountAnalyzer(engine.account)


def core_backtest_with_benchmark(
    signal: 'pd.Series',
    data: 'pd.DataFrame',
    symbol: str = '399317.SZ',
    freq: str = '1d',
    initial_capital: float = 1_000_000,
    start_date: str = None,
    note_fields: List[str] = None,
    bench_label: str = '基准',
) -> 'AccountAnalyzer':
    """
    core_backtest + 基准对比（一次调用完成信号回测 + 基准注入）。

    基准通过 BenchHolder 策略运行（与信号共用同一份数据），
    daily_assets 自动注入到 signal analyzer 中。
    当设置 start_date 时，策略和基准同时从该日期截断，确保起点一致。

    Returns:
        AccountAnalyzer: 已注入基准的 analyzer，可直接 to_notebook() 输出对比报告

    Example:
        >>> analyzer = core_backtest_with_benchmark(sig, df, start_date='2020-01-01')
        >>> analyzer.to_notebook("策略 vs 基准")
    """
    from core.engine import Engine
    from core.account import BenchHolder
    from core.storage import context
    from core.analyzer import AccountAnalyzer

    # [修复] 2026-06-10 start_date 统一截断 → 策略和基准同一天起步
    bench_data = data.copy()
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        bench_data = bench_data.loc[bench_data.index >= start_ts].copy()

    # ── 1. 基准回测 ──
    bench_engine = Engine(init_cash=initial_capital)
    context.mode = 'backtest'
    context.unsubscribe(symbol, freq)
    context.subscribe(symbol, freq, count=3000)
    bench_engine.add_data(symbol, freq, bench_data)

    bench_dt_start = bench_data.index[0].to_pydatetime() if hasattr(bench_data.index[0], 'to_pydatetime') else bench_data.index[0]
    bench_dt_end = bench_data.index[-1].to_pydatetime() if hasattr(bench_data.index[-1], 'to_pydatetime') else bench_data.index[-1]
    bench_engine.run(BenchHolder, bench_dt_start, bench_dt_end)
    bench_analyzer = AccountAnalyzer(bench_engine.account)

    # ── 2. 信号回测（传入 start_date，内部截断到同一天）──
    analyzer = core_backtest(
        signal, data, symbol=symbol, freq=freq,
        initial_capital=initial_capital, start_date=start_date,
        note_fields=note_fields)

    # ── 3. 注入基准 ──
    analyzer.set_benchmark(bench_analyzer.daily_assets, bench_label)
    return analyzer


# [复用] 2026-06-10 重构: run_backtest_with_core 现在对 Series 输入委托给 core_backtest
#   Expression/callable/ndarray 的兼容性保留，但信号解析后统一走 core_backtest
def run_backtest_with_core(
    expr_or_signals,
    data: 'pd.DataFrame',
    symbol: str = '399317.SZ',
    freq: str = '1d',
    initial_capital: float = 1_000_000,
    long_only: bool = True,
    start_date: str = None,
    note_fields: List[str] = None,
) -> 'AccountAnalyzer':
    """
    用 core 引擎回测择时信号，替代简化版 run_backtest()。

    信号时序：signal[t] → bar[t] 调仓（引擎逐 bar 递进天然等价 T+1）
    首日收益由 bar[t+1] 的快照捕获。

    Args:
        expr_or_signals: Expression / callable / pd.Series / np.ndarray / list
        data: OHLCV DataFrame（index=DatetimeIndex）
        symbol: 交易标的代码
        freq: 数据频率（'1d' 等）
        initial_capital: 初始资金
        long_only: True=仅做多（信号>0买入，信号<=0卖出）
        note_fields: 附加 bar 字段写入 TradeRecord.note（如 ['rsi']）

    Returns:
        AccountAnalyzer: 含完整指标/交易记录的链式分析器
            analyzer.set_benchmark(...).to_notebook(...)

    Example:
        >>> from signals.v2 import Expression, FeatureSpace, run_backtest_with_core
        >>> fs = FeatureSpace()
        >>> expr = Expression("thr_mean(ATR{7})", fs)
        >>> analyzer = run_backtest_with_core(expr, df, symbol='399317.SZ')
        >>> analyzer.set_benchmark(bench_nav, '买入持有').to_notebook("ATR突破")
    """
    from core.analyzer import AccountAnalyzer

    # ── 1. 信号标准化（多态分发 → 统一为 pd.Series）──
    if isinstance(expr_or_signals, pd.Series):
        signal_values = expr_or_signals
    elif isinstance(expr_or_signals, np.ndarray):
        signal_values = pd.Series(
            expr_or_signals.ravel(), index=data.index[:len(expr_or_signals)])
    elif hasattr(expr_or_signals, 'generate'):
        signal_values = expr_or_signals.generate(data)
    elif callable(expr_or_signals):
        signal_values = expr_or_signals(data)
    else:
        signal_values = pd.Series(
            list(expr_or_signals), index=data.index[:len(expr_or_signals)])

    if len(signal_values.dropna()) < 2:
        import warnings
        warnings.warn("run_backtest_with_core: 有效信号值不足")
        return AccountAnalyzer(account=None)

    # ── 2. 委托给 core_backtest（纯信号 → core 引擎）──
    return core_backtest(
        signal_values, data, symbol=symbol, freq=freq,
        initial_capital=initial_capital, start_date=start_date,
        note_fields=note_fields)


class Validator:
    """验证器 — 统一入口"""

    @staticmethod
    def run_backtest(expr_or_gen, data, initial_capital=1_000_000, long_only=True):
        return run_backtest(expr_or_gen, data, initial_capital, long_only)

    @staticmethod
    def run_backtest_from_signal(signals, prices, data=None, initial_capital=1_000_000, long_only=True):
        return run_backtest_from_signal(signals, prices, data, initial_capital, long_only)

    @staticmethod
    def run_backtest_with_core(expr_or_signals, data, symbol='399317.SZ',
                                freq='1d', initial_capital=1_000_000,
                                long_only=True, note_fields=None):
        return run_backtest_with_core(expr_or_signals, data, symbol, freq,
                                       initial_capital, long_only, note_fields)

    @staticmethod
    def walk_forward(expr_or_gen, data, train_size='2Y', test_size='1Y', step='6M',
                     initial_capital=1_000_000, anchor_start=False):
        return walk_forward(expr_or_gen, data, train_size, test_size, step,
                            initial_capital, anchor_start)

    @staticmethod
    def compare(*expressions, data=None, initial_capital=1_000_000):
        return compare(*expressions, data=data, initial_capital=initial_capital)
