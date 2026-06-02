"""
案例 01: 基础回测 — MA20 金叉死叉策略

演示:
- 构造模拟日线数据（OHLCV + eob）
- 订阅数据、加载到 Engine
- 编写策略类（on_bar 模式）
- 运行回测并生成报告

策略: 收盘价上穿 MA20 → 满仓买入；下穿 MA20 → 清仓卖出
"""

import sys
sys.path.insert(0, r'd:\01-Doc\Quant\ft2')

import datetime
import numpy as np
import pandas as pd
from core import Engine, context, account, AccountAnalyzer, OrderSide


# ═══════════════════════════════════════════════════════════════════
# 1. 构造模拟数据
# ═══════════════════════════════════════════════════════════════════

def make_dummy_data(symbol: str, n_days: int = 500, seed: int = 42):
    """构造模拟日线 OHLCV 数据（含 eob 列）"""
    np.random.seed(seed)
    dates = pd.bdate_range(start='2020-01-02', periods=n_days)

    returns = np.random.randn(n_days) * 0.015
    close = 10.0 * np.exp(np.cumsum(returns))

    open_  = close * (1 + np.random.randn(n_days) * 0.002)
    high   = np.maximum(open_, close) * (1 + np.abs(np.random.randn(n_days) * 0.005))
    low    = np.minimum(open_, close) * (1 - np.abs(np.random.randn(n_days) * 0.005))
    volume = np.random.randint(1e4, 1e6, n_days)

    df = pd.DataFrame({
        'eob':    [d.replace(hour=15, minute=15, second=1) for d in dates],
        'open':   np.round(open_, 2),
        'high':   np.round(high, 2),
        'low':    np.round(low, 2),
        'close':  np.round(close, 2),
        'volume': volume,
    })
    return df


# ═══════════════════════════════════════════════════════════════════
# 2. 编写策略
# ═══════════════════════════════════════════════════════════════════

class MACrossStrategy:
    """MA20 金叉买入 / 死叉卖出"""

    def __init__(self):
        self.in_position = False
        self.last_signal = None  # 记录上一次信号

    def on_bar(self, context, bars):
        for bar in bars:
            symbol = bar['symbol']

            # 取最近 30 根 bar 的 close（context.data 天然防未来数据）
            df = context.data(symbol, '1d', count=30, fields='close')
            if df is None or len(df) < 21:
                continue

            close_arr = df['close'].values
            ma20 = close_arr[-20:].mean()
            prev_ma20 = close_arr[-21:-1].mean()   # 上一根 bar 的 MA20
            current_price = bar['close']

            prev_signal = 'above' if close_arr[-2] > prev_ma20 else 'below'
            curr_signal = 'above' if current_price > ma20 else 'below'

            # 金叉（下→上穿）
            if prev_signal == 'below' and curr_signal == 'above' and not self.in_position:
                account.order_percent(
                    symbol, 1.0, OrderSide.Buy,
                    note=f"金叉买入 close={current_price:.2f} ma20={ma20:.2f}"
                )
                self.in_position = True

            # 死叉（上→下穿）
            elif prev_signal == 'above' and curr_signal == 'below' and self.in_position:
                account.order_percent(
                    symbol, 1.0, OrderSide.Sell,
                    note=f"死叉卖出 close={current_price:.2f} ma20={ma20:.2f}"
                )
                self.in_position = False


# ═══════════════════════════════════════════════════════════════════
# 3. 运行回测
# ═══════════════════════════════════════════════════════════════════

def main():
    symbol = '399317.SZ'

    # 准备数据
    df = make_dummy_data(symbol, n_days=500)

    # 初始化引擎
    engine = Engine()
    context.mode = 'backtest'
    context.subscribe(symbol, '1d', count=100)

    # 加载数据
    engine.add_data(symbol, '1d', df)

    # 确定回测区间（跳过前30天预热）
    dates = sorted(d['eob'] for d in df.to_dict('records'))
    start_time = dates[30]
    end_time = dates[-1]

    # 运行
    engine.run(MACrossStrategy, start_time, end_time)

    # 分析
    analyzer = AccountAnalyzer(account)
    print("=" * 50)
    print(f"  最终净值: {account.get_account()['nav']:,.2f}")
    print(f"  累计收益率: {analyzer.return_rate() * 100:.2f}%")
    print(f"  夏普比率:   {analyzer.sharpe_ratio():.2f}")
    print(f"  最大回撤:   {analyzer.max_drawdown()[0] * 100:.2f}%")
    print(f"  交易次数:   {len(account.trade_records)}")
    print("=" * 50)

    # 生成报告
    analyzer.to_notebook("MA20金叉策略回测")
    print("\n报告已生成，请在浏览器中打开查看")


if __name__ == '__main__':
    main()
