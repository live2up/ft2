"""
案例 05: 高级用法 — 信号备注追溯、快照恢复、阈值扫描

演示:
- TradeRecord.note 信号备注，回溯每笔交易触发原因
- 复盘指定日期的持仓状态
- 多参数阈值扫描（参数优化）
- 快照状态查询
"""

import sys
sys.path.insert(0, r'd:\01-Doc\Quant\ft2')

import numpy as np
import pandas as pd
from core import Engine, context, account, AccountAnalyzer, OrderSide


# ═══════════════════════════════════════════════════════════════════
# 1. 数据
# ═══════════════════════════════════════════════════════════════════

def make_data(n_days=600):
    np.random.seed(123)
    dates = pd.bdate_range(start='2019-01-02', periods=n_days)
    returns = np.random.randn(n_days) * 0.013
    close = 10.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        'eob':    [d.replace(hour=15, minute=15, second=1) for d in dates],
        'open':   np.round(close * (1 + np.random.randn(n_days) * 0.002), 2),
        'high':   np.round(close * (1 + np.abs(np.random.randn(n_days) * 0.004)), 2),
        'low':    np.round(close * (1 - np.abs(np.random.randn(n_days) * 0.004)), 2),
        'close':  np.round(close, 2),
        'volume': np.random.randint(3e4, 3e5, n_days),
    })


# ═══════════════════════════════════════════════════════════════════
# 2. 带参数 + 信号备注的策略
# ═══════════════════════════════════════════════════════════════════

class MovingAverageStrategy:
    """通用 MA 策略：支持自定义快慢均线周期，每笔交易附带信号备注"""

    def __init__(self, fast_period=5, slow_period=20, name="MA"):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = name
        self.in_position = False

    def on_bar(self, context, bars):
        for bar in bars:
            sym = bar['symbol']
            needed = max(self.fast_period, self.slow_period) + 5
            df = context.data(sym, '1d', count=needed + 1, fields='close')
            if df is None or len(df) < needed:
                continue
            close = df['close'].values
            fast_ma = close[-self.fast_period:].mean()
            slow_ma = close[-self.slow_period:].mean()
            prev_fast = close[-self.fast_period-1:-1].mean()
            prev_slow = close[-self.slow_period-1:-1].mean()
            price = bar['close']

            cross_up   = prev_fast <= prev_slow and fast_ma > slow_ma
            cross_down = prev_fast >= prev_slow and fast_ma < slow_ma

            if cross_up and not self.in_position:
                # [关键] note 字段记录详细信号信息
                account.order_percent(
                    sym, 1.0, OrderSide.Buy,
                    note=f"{self.name}({self.fast_period},{self.slow_period})" +
                         f" 金叉 | fast={fast_ma:.2f} slow={slow_ma:.2f} " +
                         f"price={price:.2f}"
                )
                self.in_position = True
            elif cross_down and self.in_position:
                account.order_percent(
                    sym, 1.0, OrderSide.Sell,
                    note=f"{self.name}({self.fast_period},{self.slow_period})" +
                         f" 死叉 | fast={fast_ma:.2f} slow={slow_ma:.2f} " +
                         f"price={price:.2f}"
                )
                self.in_position = False


# ═══════════════════════════════════════════════════════════════════
# 3. 阈值扫描器
# ═══════════════════════════════════════════════════════════════════

def scan_parameters(symbol, df, fast_list, slow_list):
    """扫描 MA(fast, slow) 参数组合，返回排序结果"""
    from core.storage import context as ctx
    from core.engine import Engine as Eng
    from core.account import AccountManager

    dates = sorted(d['eob'] for d in df.to_dict('records'))
    start_time = dates[60]
    end_time = dates[-1]
    results = []

    for fast in fast_list:
        for slow in slow_list:
            if fast >= slow:
                continue

            # 每个参数组合独立初始化（重置全局状态）
            new_account = AccountManager(init_cash=1e6)
            new_ctx = type(ctx)()
            new_ctx.mode = 'backtest'
            new_ctx.subscribe(symbol, '1d', count=max(fast, slow) + 50)

            engine = Eng()
            engine.add_data(symbol, '1d', df)

            # 注入到 engine.run 使用的全局
            import core.account as acc_mod
            import core.storage as stg_mod
            orig_account = acc_mod.account
            orig_context = stg_mod.context
            acc_mod.account = new_account
            stg_mod.context = new_ctx

            try:
                engine.run(MovingAverageStrategy(fast, slow, "MA"), start_time, end_time)
                analyzer = AccountAnalyzer(new_account)
                sharpe = analyzer.sharpe_ratio()
                ret = analyzer.return_rate()
                dd = analyzer.max_drawdown()
                max_dd = dd[0] if dd else None
                trades = len(new_account.trade_records)
                results.append({
                    'fast': fast, 'slow': slow,
                    'sharpe': sharpe or 0,
                    'return': ret or 0,
                    'max_dd': max_dd or 0,
                    'trades': trades,
                })
            finally:
                acc_mod.account = orig_account
                stg_mod.context = orig_context

    return sorted(results, key=lambda x: x['sharpe'], reverse=True)


def main():
    symbol = '399317.SZ'
    df = make_data(600)

    # ═══════════════════════════════════════════════════════════════
    # 3A. 单次回测 + 信号备注追溯
    # ═══════════════════════════════════════════════════════════════
    engine = Engine()
    context.mode = 'backtest'
    context.subscribe(symbol, '1d', count=100)
    engine.add_data(symbol, '1d', df)

    dates = sorted(d['eob'] for d in df.to_dict('records'))
    engine.run(MovingAverageStrategy(10, 30, "MA"), dates[60], dates[-1])

    analyzer = AccountAnalyzer(account)

    print("=" * 70)
    print("【信号备注追溯 — 每笔交易都记录了触发原因】")
    print("-" * 70)
    for i, t in enumerate(account.trade_records[:10]):
        side = 'BUY ' if t.side == OrderSide.Buy else 'SELL'
        print(f"  #{i+1:2d} {t.created_at.strftime('%Y-%m-%d')} {side} "
              f"{t.symbol} x{t.volume} @{t.price:.2f} "
              f"金额={t.amount:,.0f} 费={t.fee:.2f}")
        if t.note:
            print(f"      备注: {t.note}")
    print(f"  ... 共 {len(account.trade_records)} 笔交易")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════
    # 3B. 复盘指定日期 — 查询快照
    # ═══════════════════════════════════════════════════════════════
    print("\n【复盘指定日期】")
    for s in account.snapshots[::50][:5]:  # 每50个快照取一个
        print(f"  {s.created_at.strftime('%Y-%m-%d')}: "
              f"现金={s.cash:,.0f} 净值={s.nav:,.0f} "
              f"持仓数={len(s.positions)}")
        for sym, pos in s.positions.items():
            print(f"    {sym}: 数量={pos.volume} 成本={pos.cost_price:.2f} 现价={pos.price:.2f}")
    print(f"  共 {len(account.snapshots)} 个快照")

    # ═══════════════════════════════════════════════════════════════
    # 3C. 交易盈利分析
    # ═══════════════════════════════════════════════════════════════
    print(f"\n【交易盈亏】共 {len(analyzer.trade_profits)} 笔完整交易")
    profits = [t['profit'] for t in analyzer.trade_profits]
    print(f"  总盈亏:     {sum(profits):,.2f}")
    print(f"  盈利笔数:   {sum(1 for p in profits if p > 0)}")
    print(f"  亏损笔数:   {sum(1 for p in profits if p < 0)}")
    print(f"  平均盈利:   {analyzer.avg_profit('amount'):,.2f}")
    print(f"  平均亏损:   {analyzer.avg_loss('amount'):,.2f}")

    # ═══════════════════════════════════════════════════════════════
    # 3D. 参数扫描
    # ═══════════════════════════════════════════════════════════════
    print("\n【参数扫描 — MA(fast, slow) 夏普排序 Top5】")
    top = scan_parameters(symbol, df,
                          fast_list=[3, 5, 10],
                          slow_list=[15, 20, 30])
    for r in top[:5]:
        print(f"  MA({r['fast']:2d},{r['slow']:2d})  "
              f"夏普={r['sharpe']:.3f}  收益={r['return']*100:.1f}%  "
              f"回撤={r['max_dd']*100:.1f}%  笔数={r['trades']}")

    # ═══════════════════════════════════════════════════════════════
    # 3E. 生成报告
    # ═══════════════════════════════════════════════════════════════
    analyzer.to_notebook("MA策略（含信号备注）")
    print("\n✅ 报告已生成")


if __name__ == '__main__':
    main()
