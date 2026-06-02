"""
案例 04: 自定义指标 — @metric 装饰器扩展

演示:
- 为 AccountAnalyzer 添加自定义指标
- 继承 AccountAnalyzer 并注册新指标
- 新指标自动出现在 to_notebook / to_excel 中
- 费用配置与最低佣金
"""

import sys
sys.path.insert(0, r'd:\01-Doc\Quant\ft2')

import numpy as np
import pandas as pd
from core import Engine, context, account, AccountAnalyzer, OrderSide
from core.analyzer import metric  # 关键：导入装饰器


# ═══════════════════════════════════════════════════════════════════
# 1. 自定义分析器：继承 + 新增指标
# ═══════════════════════════════════════════════════════════════════

class MyAnalyzer(AccountAnalyzer):
    """扩展分析器 — 只需添加 @metric 方法即可"""

    # ── 收益组 ──
    @metric(name='月均收益率', group='收益', fmt='.2%', desc='平均每月收益率', order=12)
    def monthly_avg_return(self):
        """按自然月分组的平均收益率"""
        data = self._ensure_sliced_data()
        if data is None:
            return None
        assets = data['daily_assets']
        if len(assets) < 2:
            return None
        dates = sorted(assets.keys())
        # 按月分组
        monthly_returns = []
        for mon in pd.date_range(dates[0], dates[-1], freq='MS'):
            mon_dates = [d for d in dates if d.year == mon.year and d.month == mon.month]
            if len(mon_dates) >= 2:
                r = assets[mon_dates[-1]] / assets[mon_dates[0]] - 1
                monthly_returns.append(r)
        if not monthly_returns:
            return None
        return np.mean(monthly_returns)

    # ── 风险组 ──
    @metric(name='Calmar比率', group='风险', fmt='.2f', desc='年化收益/最大回撤的比值', order=33)
    def calmar_ratio(self):
        ann_ret = self.annualized_return()
        dd = self.max_drawdown()
        if ann_ret is None or dd is None or dd[0] == 0:
            return None
        return ann_ret / dd[0]

    @metric(name='最大连胜次数', group='风险', fmt='.0f', desc='连续盈利的最大交易数', order=25)
    def max_consecutive_wins(self):
        """计算连续盈利的最大次数"""
        if not self._trade_profits:
            return None
        max_streak = cur = 0
        for t in self._trade_profits:
            if t['profit'] > 0:
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 0
        return max_streak if max_streak > 0 else None

    @metric(name='最大连亏次数', group='风险', fmt='.0f', desc='连续亏损的最大交易数', order=26)
    def max_consecutive_losses(self):
        if not self._trade_profits:
            return None
        max_streak = cur = 0
        for t in self._trade_profits:
            if t['profit'] < 0:
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 0
        return max_streak if max_streak > 0 else None

    # ── 交易组 ──
    @metric(name='总盈利金额', group='交易', fmt='.0f', desc='所有盈利交易的总金额', order=43)
    def total_profit_amount(self):
        if not self._trade_profits:
            return None
        return sum(t['profit'] for t in self._trade_profits if t['profit'] > 0)

    @metric(name='总亏损金额', group='交易', fmt='.0f', desc='所有亏损交易的总金额（取绝对值）', order=44)
    def total_loss_amount(self):
        if not self._trade_profits:
            return None
        total = sum(t['profit'] for t in self._trade_profits if t['profit'] < 0)
        return abs(total) if total != 0 else None


# ═══════════════════════════════════════════════════════════════════
# 2. 构造数据和回测
# ═══════════════════════════════════════════════════════════════════

def make_data(n_days=500):
    np.random.seed(777)
    dates = pd.bdate_range(start='2020-01-02', periods=n_days)
    returns = np.random.randn(n_days) * 0.015 + 0.0003
    close = 10.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        'eob':    [d.replace(hour=15, minute=15, second=1) for d in dates],
        'open':   np.round(close * (1 + np.random.randn(n_days) * 0.002), 2),
        'high':   np.round(close * (1 + np.abs(np.random.randn(n_days) * 0.004)), 2),
        'low':    np.round(close * (1 - np.abs(np.random.randn(n_days) * 0.004)), 2),
        'close':  np.round(close, 2),
        'volume': np.random.randint(5e4, 5e5, n_days),
    })


class BollStrategy:
    """布林带策略：下轨买入，上轨卖出"""

    def __init__(self):
        self.in_position = False

    def on_bar(self, context, bars):
        for bar in bars:
            sym = bar['symbol']
            df = context.data(sym, '1d', count=30, fields='close')
            if df is None or len(df) < 25:
                continue
            c = df['close'].values
            ma = c[-20:].mean()
            std = c[-20:].std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            price = bar['close']

            if price < lower and not self.in_position:
                account.order_percent(sym, 1.0, OrderSide.Buy,
                                      note=f"布林下轨买入 {price:.2f}<{lower:.2f}")
                self.in_position = True
            elif price > upper and self.in_position:
                account.order_percent(sym, 1.0, OrderSide.Sell,
                                      note=f"布林上轨卖出 {price:.2f}>{upper:.2f}")
                self.in_position = False


def main():
    symbol = '399317.SZ'

    # 准备
    df = make_data(500)
    engine = Engine()
    context.mode = 'backtest'
    context.subscribe(symbol, '1d', count=100)
    engine.add_data(symbol, '1d', df)

    # 修改费用配置（类ETF：佣金万1，最低1元，印花税0）
    account.fee_config = {
        'commission_rate': 0.0001,
        'stamp_tax_rate': 0.0,
        'min_commission': 1.0
    }

    dates = [d['eob'] for d in df.to_dict('records')]
    engine.run(BollStrategy, dates[30], dates[-1])

    # ── 使用自定义分析器 ──
    analyzer = MyAnalyzer(account)
    print("=" * 60)
    print("【自定义分析器 — 扩展指标】")
    print(f"  费用配置(ETF模式): {account.fee_config}")

    # 新指标自动出现在 metrics() 中
    all_m = analyzer.metrics()
    for name in sorted(all_m.keys()):
        m = all_m[name]
        val = m['value']
        if isinstance(val, tuple):
            val = val[0]
        if val is not None:
            if m['fmt'].endswith('%') and isinstance(val, (int, float)):
                d = int(m['fmt'][1:-1])
                print(f"  [{m['group']:<4s}] {name:<14s} = {val*100:.{d}f}%")
            else:
                print(f"  [{m['group']:<4s}] {name:<14s} = {val}")

    print(f"\n  交易次数: {len(analyzer.trade_profits)}")
    print(f"  胜率:     {analyzer.win_rate() * 100:.1f}%")
    print("=" * 60)

    # to_notebook 自动拾取新指标
    analyzer.to_notebook("布林带策略（含自定义指标）")

    print("\n✅ 自定义指标报告已生成")


if __name__ == '__main__':
    main()
