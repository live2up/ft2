"""
案例 02: 多品种回测 — 双均线轮动策略

演示:
- 同时订阅多个品种
- 在 on_bar 中按品种分发
- 动态资金分配（平均分配）
- account.order_percent 按净值比例下单
- account.get_position 查询持仓
"""

import sys
sys.path.insert(0, r'd:\01-Doc\Quant\ft2')

import datetime
import numpy as np
import pandas as pd
from core import Engine, context, account, AccountAnalyzer, OrderSide


# ═══════════════════════════════════════════════════════════════════
# 1. 构造多品种数据
# ═══════════════════════════════════════════════════════════════════

def make_multi_data(symbols, n_days=300, seed_base=42):
    """构造多个品种的模拟日线数据"""
    all_data = {}
    for i, sym in enumerate(symbols):
        np.random.seed(seed_base + i * 137)
        dates = pd.bdate_range(start='2018-01-02', periods=n_days)
        returns = np.random.randn(n_days) * 0.014 + 0.0002 * (i + 1)  # 不同漂移
        close = (8 + i * 2) * np.exp(np.cumsum(returns))
        open_  = close * (1 + np.random.randn(n_days) * 0.002)
        high   = np.maximum(open_, close) * (1 + np.abs(np.random.randn(n_days) * 0.004))
        low    = np.minimum(open_, close) * (1 - np.abs(np.random.randn(n_days) * 0.004))
        volume = np.random.randint(5e4, 5e5, n_days)
        all_data[sym] = pd.DataFrame({
            'eob':    [d.replace(hour=15, minute=15, second=1) for d in dates],
            'open':   np.round(open_, 2),
            'high':   np.round(high, 2),
            'low':    np.round(low, 2),
            'close':  np.round(close, 2),
            'volume': volume,
        })
    return all_data


# ═══════════════════════════════════════════════════════════════════
# 2. 多品种轮动策略
# ═══════════════════════════════════════════════════════════════════

class MultiSymbolRotation:
    """双均线轮动：买入 MA5 > MA20 的品种，平均分配仓位"""

    def __init__(self, symbols):
        self.symbols = symbols
        self.positions = {s: False for s in symbols}  # 当前是否持仓

    def on_bar(self, context, bars):
        # 当前 bar 涉及的品种
        bar_symbols = {b['symbol'] for b in bars}

        # 按品种计算信号
        signals = {}  # symbol → True/False（是否持有）
        for symbol in self.symbols:
            if symbol not in bar_symbols:
                continue
            df = context.data(symbol, '1d', count=30, fields='close')
            if df is None or len(df) < 21:
                continue
            close = df['close'].values
            ma5  = close[-5:].mean()
            ma20 = close[-20:].mean()
            signals[symbol] = ma5 > ma20

        # 计算应持有的品种数
        target_symbols = [s for s, sig in signals.items() if sig]
        n_target = len(target_symbols)

        if n_target == 0:
            # 全部卖出
            for sym in self.symbols:
                if self.positions[sym]:
                    account.order_percent(sym, 1.0, OrderSide.Sell,
                                          note="信号消失→清仓")
                    self.positions[sym] = False
            return

        # 每品种分配仓位比例
        alloc = 1.0 / n_target

        # 调仓
        for symbol in self.symbols:
            should_hold = symbol in target_symbols
            if should_hold and not self.positions[symbol]:
                account.order_percent(symbol, alloc, OrderSide.Buy,
                                      note=f"MA5>MA20 平均分配 {alloc:.0%}")
                self.positions[symbol] = True
            elif not should_hold and self.positions[symbol]:
                account.order_percent(symbol, 1.0, OrderSide.Sell,
                                      note="信号消失→卖出")
                self.positions[symbol] = False


# ═══════════════════════════════════════════════════════════════════
# 3. 运行回测
# ═══════════════════════════════════════════════════════════════════

def main():
    symbols = ['399317.SZ', '000001.SZ', '600519.SH']

    all_data = make_multi_data(symbols, n_days=300)

    engine = Engine()
    context.mode = 'backtest'

    # 订阅所有品种
    context.subscribe(symbols, '1d', count=100)

    # 加载数据
    for sym in symbols:
        engine.add_data(sym, '1d', all_data[sym])

    # 确定统一回测区间
    dates = sorted(engine.timeline.keys())
    start_time = dates[30]
    end_time = dates[-1]

    # 运行
    engine.run(lambda: MultiSymbolRotation(symbols), start_time, end_time)

    # 分析
    analyzer = AccountAnalyzer(account)
    print("=" * 60)
    print(f"  品种: {symbols}")
    print(f"  最终净值: {account.get_account()['nav']:,.2f}")
    print(f"  累计收益率: {analyzer.return_rate() * 100:.2f}%")
    print(f"  夏普比率:   {analyzer.sharpe_ratio():.2f}")
    print(f"  最大回撤:   {analyzer.max_drawdown()[0] * 100:.2f}%")
    print(f"  交易次数:   {len(account.trade_records)}")

    # 按品种统计
    from collections import Counter
    sym_counts = Counter(t.symbol for t in account.trade_records)
    for sym in symbols:
        print(f"    {sym}: {sym_counts.get(sym, 0)} 笔")
    print("=" * 60)

    # 生成报告
    analyzer.to_notebook("多品种均线轮动策略")
    print("\n报告已生成")


if __name__ == '__main__':
    main()
