# signals/examples.py - 使用示例
"""
signals 模块完整使用示例

运行方式：
  python ft2/signals/examples.py
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, r'd:\01-Doc\程序化\ft2')

from signals import (
    Signal, SignalType, SignalDirection,
    SignalGenerator, MASignal, MACDSignal, RSISignal, KDJSignal, 
    BOLLSignal, VOLSignal, RSRSMSignal, CompositeSignal,
    VotingCombiner, ScoringCombiner, EqualWeightCombiner,
    SimpleThreshold, PercentileThreshold, DualThreshold,
    run_backtest, BacktestResult,
    ICAnalyzer,
)


def generate_sample_data(days: int = 300) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(42)
    returns = np.random.randn(days) * 0.02
    close = 10 * (1 + returns).cumprod()
    
    high = close * (1 + np.random.rand(days) * 0.01)
    low = close * (1 - np.random.rand(days) * 0.01)
    open_price = low + (high - low) * np.random.rand(days)
    volume = np.random.randint(1e6, 1e8, days)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }, index=dates)
    
    df.index.name = 'date'
    return df


def example_basic_signals():
    print("\n" + "=" * 60)
    print("示例 1: 基础信号生成")
    print("=" * 60)
    
    data = generate_sample_data(300)
    
    ma = MASignal(short_period=5, long_period=20)
    macd = MACDSignal(fast=12, slow=26, signal=9)
    rsi = RSISignal(period=14)
    
    ma_series = ma.generate(data)
    macd_series = macd.generate(data)
    rsi_series = rsi.generate(data)
    
    print(f"MA 信号: {ma.name}")
    print(f"  最新值: {ma_series.iloc[-1]:.4f}")
    
    print(f"\nMACD 信号: {macd.name}")
    print(f"  最新值: {macd_series.iloc[-1]:.4f}")
    
    print(f"\nRSI 信号: {rsi.name}")
    print(f"  最新值: {rsi_series.iloc[-1]:.2f}")
    
    ma_latest = ma.generate_latest(data)
    print(f"\nMA 最新: value={ma_latest.value:.4f}, direction={ma_latest.direction.name}")


def example_signal_combination():
    print("\n" + "=" * 60)
    print("示例 2: 信号融合")
    print("=" * 60)
    
    data = generate_sample_data(300)
    
    generators = [
        MASignal(short_period=5, long_period=20),
        MACDSignal(fast=12, slow=26, signal=9),
        RSISignal(period=14),
    ]
    
    signal_dict = {gen.name: gen.generate(data) for gen in generators}
    signals_df = pd.DataFrame(signal_dict)
    
    print(f"\n信号数据形状: {signals_df.shape}")
    print(f"\n最新信号值:")
    print(signals_df.tail())
    
    voting = VotingCombiner()
    voting_result = voting.combine_series(
        type('SignalSeries', (), {'signals': signals_df})()
    )
    print(f"\n投票融合（最新）: {voting_result.iloc[-1]}")
    
    weights = {'MA5_20': 0.4, 'Macd12_26_9': 0.35, 'RSI14': 0.25}
    scoring = ScoringCombiner(weights=weights)
    scoring_result = scoring.combine_series(
        type('SignalSeries', (), {'signals': signals_df})()
    )
    print(f"打分融合（最新）: {scoring_result.iloc[-1]:.4f}")
    
    equal = EqualWeightCombiner()
    equal_result = equal.combine_series(
        type('SignalSeries', (), {'signals': signals_df})()
    )
    print(f"等权融合（最新）: {equal_result.iloc[-1]:.4f}")


def example_threshold():
    print("\n" + "=" * 60)
    print("示例 3: 阈值策略")
    print("=" * 60)
    
    data = generate_sample_data(300)
    
    ma = MASignal(short_period=5, long_period=20)
    signal = ma.generate(data)
    
    print(f"\n信号统计:")
    print(f"  均值: {signal.mean():.4f}, 标准差: {signal.std():.4f}")
    
    simple = SimpleThreshold(upper_threshold=0, lower_threshold=0)
    directions = simple.apply_series(signal)
    
    print(f"\n简单阈值 (0, 0):")
    print(f"  做多: {(directions == 1).sum()} 天")
    print(f"  空仓: {(directions == 0).sum()} 天")
    
    percentile = PercentileThreshold(upper_percentile=70, lower_percentile=30)
    percentile.fit(signal)
    directions_pct = percentile.apply_series(signal)
    
    print(f"\n分位数阈值 (70%, 30%):")
    print(f"  做多: {(directions_pct == 1).sum()} 天")
    print(f"  空仓: {(directions_pct == 0).sum()} 天")


def example_backtest():
    print("\n" + "=" * 60)
    print("示例 4: 轻量回测")
    print("=" * 60)
    
    data = generate_sample_data(300)
    
    result = run_backtest(
        generator=MASignal(short_period=5, long_period=20),
        data=data,
        initial_capital=1_000_000
    )
    
    print(f"\n--- 回测结果 ---")
    print(f"总收益率: {result.total_return:.2f}%")
    print(f"年化收益: {result.annual_return:.2f}%")
    print(f"最大回撤: {result.max_drawdown:.2f}%")
    print(f"夏普比率: {result.sharpe:.2f}")
    print(f"胜率: {result.win_rate:.1f}%")
    print(f"盈亏比: {result.profit_loss_ratio:.2f}")
    print(f"交易次数: {result.trade_count}")


def example_ic_analysis():
    print("\n" + "=" * 60)
    print("示例 5: IC 分析")
    print("=" * 60)
    
    data = generate_sample_data(300)
    
    ma = MASignal(short_period=5, long_period=20)
    signals = ma.generate(data)
    
    ic_result = ICAnalyzer.analyze(
        signals=signals,
        prices=data['close'],
        rolling_windows=[30, 60, 120],
        holding_periods=[1, 3, 5, 10, 20],
        annualize=False
    )
    
    basic = ic_result['basic']
    summary = basic['summary']
    print(f"\n--- IC 分析结果 ---")
    print(f"IC 均值: {summary['ic_mean']:.4f}")
    print(f"IC 标准差: {summary['ic_std']:.4f}")
    print(f"IC_IR: {summary['ic_ir']:.2f}")
    print(f"IC>0 占比: {summary['ic_positive_ratio']:.1f}%")
    
    print(f"\n--- 持有期 IC ---")
    for key, val in basic.items():
        if key != 'summary':
            period = key.replace('IC_', '').replace('d', '')
            print(f"  {period}日: Pearson={val['pearson_ic']:.4f}, Rank={val['rank_ic']:.4f}")
    
    sig = ic_result['significance']
    print(f"\n--- 统计显著性 ---")
    print(f"Pearson IC: {sig['pearson_ic']:.4f}")
    print(f"t 统计量: {sig['t_statistic']:.4f}")
    print(f"p 值: {sig['p_value']:.6f}")
    print(f"显著性: {'✅ 显著' if sig['significant'] else '❌ 不显著'}")


def example_multi_signal_backtest():
    print("\n" + "=" * 60)
    print("示例 6: 多信号融合回测")
    print("=" * 60)
    
    data = generate_sample_data(300)
    
    composite = CompositeSignal([
        MASignal(short_period=5, long_period=20),
        MACDSignal(fast=12, slow=26, signal=9),
        RSISignal(period=14),
    ], name='CompositeTrend')
    
    result = run_backtest(
        generator=composite,
        data=data,
        initial_capital=1_000_000
    )
    
    print(f"\n--- 组合信号回测结果 ---")
    print(f"信号名称: {composite.name}")
    print(f"总收益率: {result.total_return:.2f}%")
    print(f"夏普比率: {result.sharpe:.2f}")
    print(f"最大回撤: {result.max_drawdown:.2f}%")


def main():
    print("=" * 60)
    print("signals 模块使用示例")
    print("=" * 60)
    
    example_basic_signals()
    example_signal_combination()
    example_threshold()
    example_backtest()
    example_ic_analysis()
    example_multi_signal_backtest()
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
