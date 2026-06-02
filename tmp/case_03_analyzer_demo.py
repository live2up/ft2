"""
案例 03: 分析器用法 — AccountAnalyzer 全功能演示

演示:
- 多时间区间指标切片（getTimeRange）
- 所有 @metric 指标查询
- returns() 多区间批量计算
- 最佳/最差交易查询
- 导出自定义 daily_assets 数据
- to_notebook 报告生成
- to_excel 报告生成
"""

import sys
sys.path.insert(0, r'd:\01-Doc\Quant\ft2')

import datetime
import numpy as np
import pandas as pd
from core import Engine, context, account, AccountAnalyzer, OrderSide


# ═══════════════════════════════════════════════════════════════════
# 1. 先跑一个回测
# ═══════════════════════════════════════════════════════════════════

def make_data(n_days=500):
    np.random.seed(42)
    dates = pd.bdate_range(start='2020-01-02', periods=n_days)
    returns = np.random.randn(n_days) * 0.015
    close = 10.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        'eob':    [d.replace(hour=15, minute=15, second=1) for d in dates],
        'open':   np.round(close * (1 + np.random.randn(n_days) * 0.002), 2),
        'high':   np.round(close * (1 + np.abs(np.random.randn(n_days) * 0.005)), 2),
        'low':    np.round(close * (1 - np.abs(np.random.randn(n_days) * 0.005)), 2),
        'close':  np.round(close, 2),
        'volume': np.random.randint(1e4, 1e6, n_days),
    })


class SimpleStrategy:
    """简单突破策略：收盘价创20日新高买入，创20日新低卖出"""

    def __init__(self):
        self.in_position = False

    def on_bar(self, context, bars):
        for bar in bars:
            sym = bar['symbol']
            df = context.data(sym, '1d', count=25, fields='close')
            if df is None or len(df) < 22:
                continue
            close_arr = df['close'].values
            high20 = close_arr[-21:-1].max()
            low20  = close_arr[-21:-1].min()
            price  = bar['close']

            if price > high20 and not self.in_position:
                account.order_percent(sym, 1.0, OrderSide.Buy,
                                      note=f"突破20日高点 {price:.2f}>{high20:.2f}")
                self.in_position = True
            elif price < low20 and self.in_position:
                account.order_percent(sym, 1.0, OrderSide.Sell,
                                      note=f"跌破20日低点 {price:.2f}<{low20:.2f}")
                self.in_position = False


def run_backtest():
    symbol = '399317.SZ'
    df = make_data(500)
    engine = Engine()
    context.mode = 'backtest'
    context.subscribe(symbol, '1d', count=100)
    engine.add_data(symbol, '1d', df)
    dates = sorted(d['eob'] for d in df.to_dict('records'))
    engine.run(SimpleStrategy, dates[30], dates[-1])
    return AccountAnalyzer(account)


# ═══════════════════════════════════════════════════════════════════
# 2. 演示分析器
# ═══════════════════════════════════════════════════════════════════

def main():
    analyzer = run_backtest()

    # ── 2.1 直接调用指标方法（返回纯数字） ──
    print("=" * 60)
    print("【直接调用指标方法】")
    print(f"  累计收益率:   {analyzer.return_rate() * 100:.2f}%")
    print(f"  年化收益率:   {analyzer.annualized_return() * 100:.2f}%")
    print(f"  年化波动率:   {analyzer.volatility() * 100:.2f}%")
    print(f"  夏普比率:     {analyzer.sharpe_ratio():.2f}")
    print(f"  索提诺比率:   {analyzer.sortino_ratio():.2f}")
    print(f"  Ulcer Index:  {analyzer.ulcer_index():.2f}")
    print(f"  UPI:          {analyzer.upi():.2f}")
    print(f"  VaR(95%):     {analyzer.var(0.95) * 100:.2f}%")
    print(f"  CVaR(95%):    {analyzer.cvar(0.95) * 100:.2f}%")

    # ── 2.2 最大回撤（返回 (值, 起始日, 结束日)） ──
    dd = analyzer.max_drawdown()
    if dd:
        print(f"  最大回撤:     {dd[0] * 100:.2f}% ({dd[1]} → {dd[2]})")

    # ── 2.3 交易指标 ──
    print(f"\n  交易次数:     {len(analyzer.trade_profits)}")
    print(f"  胜率:         {analyzer.win_rate() * 100:.2f}%")
    print(f"  平均盈亏比:   {analyzer.avg_profit_loss_ratio():.2f}")
    print(f"  平均持仓天数: {analyzer.avg_holding_period():.1f}")
    print(f"  凯利仓位:     {analyzer.kelly_criterion() * 100:.1f}%")
    print(f"  半凯利仓位:   {analyzer.kelly_fraction(0.5) * 100:.1f}%")
    print("=" * 60)

    # ── 2.4 时间区间切片 ──
    print("\n【时间区间切片】")
    for period in ['1m', '3m', '1y']:
        analyzer.getTimeRange(period)
        ret = analyzer.return_rate()
        vol = analyzer.volatility()
        if ret is not None:
            print(f"  {period:>4s}: 收益率={ret*100:6.2f}%  波动率={vol*100:5.2f}%")

    # 恢复全数据
    analyzer.getTimeRange('all')

    # ── 2.5 批量多区间收益率 ──
    print("\n【批量多区间收益率】")
    rets = analyzer.returns('1m,3m,6m,1y,all')
    for p, r in rets.items():
        if r is not None:
            print(f"  {p:>4s}: {r*100:6.2f}%")

    # ── 2.6 最佳/最差交易 ──
    print("\n【最佳5笔交易】")
    for t in analyzer.get_largest_profit_trades(5):
        print(f"  {t['symbol']} {t['open_time'].strftime('%Y-%m-%d')}→"
              f"{t['close_time'].strftime('%Y-%m-%d')}: "
              f"盈利 {t['profit']:,.2f}")

    print("\n【最差5笔交易】")
    for t in analyzer.get_largest_loss_trades(5):
        print(f"  {t['symbol']} {t['open_time'].strftime('%Y-%m-%d')}→"
              f"{t['close_time'].strftime('%Y-%m-%d')}: "
              f"亏损 {t['profit']:,.2f}")

    # ── 2.7 统一收集所有 @metric ──
    print("\n【所有 @metric 指标摘要】")
    all_m = analyzer.metrics()
    for name, m in sorted(all_m.items(), key=lambda x: (x[1]['group'], x[1]['order'])):
        val = m['value']
        if isinstance(val, tuple):
            val = val[0]
        print(f"  [{m['group']:<4s}] {name:<12s} = {val}")

    # ── 2.8 导出 daily_assets ──
    print(f"\n【每日资产数据】共 {len(analyzer.daily_assets)} 天")
    assets = analyzer.get_daily_total_assets()
    dates_sorted = sorted(assets.keys())
    print(f"  首日({dates_sorted[0]}): {assets[dates_sorted[0]]:,.2f}")
    print(f"  末日({dates_sorted[-1]}): {assets[dates_sorted[-1]]:,.2f}")

    # ── 2.9 方式 2：直接传入 daily_assets（不依赖 account） ──
    print("\n【方式2：直接传入 daily_assets】")
    analyzer2 = AccountAnalyzer(daily_assets=assets)
    print(f"  累计收益率: {analyzer2.return_rate() * 100:.2f}%")
    print(f"  夏普比率:   {analyzer2.sharpe_ratio():.2f}")
    # （注意：这种方式没有 trade_profits，交易指标返回 None）

    # ── 2.10 费用配置 ──
    print("\n【当前费用配置】")
    print(f"  {account.fee_config}")

    # ── 2.11 生成报告 ──
    analyzer.to_notebook("突破策略回测分析")
    analyzer.to_excel("突破策略回测")

    print("\n✅ 报告已生成")


if __name__ == '__main__':
    main()
