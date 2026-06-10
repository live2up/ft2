"""
factor/v4/backtest.py — 因子轮动回测 (ft2.core Engine 驱动)

替换 v3 的自研 Pipeline，复用 signals.v4 的 EngineCore 统一架构。
"""
import numpy as np, pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

from core.engine import Engine
from core.storage import context
from core.account import OrderSide


@dataclass
class FactorBacktestResult:
    """因子轮动回测结果"""
    sharpe: float
    cagr: float
    max_drawdown: float
    nav: pd.Series
    trades: int
    position: pd.DataFrame

    def __repr__(self):
        return (f"FactorResult(sharpe={self.sharpe:.3f}, CAGR={self.cagr:.1%}, "
                f"MDD={self.max_drawdown:.1%}, trades={self.trades})")


class FactorBacktest:
    """
    因子轮动回测（ft2.core Engine 驱动）

    用法:
        panel = expr.rank_panel(assets)  # DataFrame(日期×品种), 值=截面排名
        result = FactorBacktest.run(panel, top_n=3, rebalance='W')

    Args:
        panel: DataFrame(index=DatetimeIndex, columns=品种代码), 值越大越好
        assets: {品种代码: OHLCV DataFrame}
        top_n: 持仓品种数
        rebalance: 调仓频率 'W'(周) 'M'(月) 'Q'(季)
        initial_capital: 初始资金
    """

    @staticmethod
    def run(panel: pd.DataFrame,
            assets: Dict[str, pd.DataFrame],
            top_n: int = 3,
            rebalance: str = 'W',
            initial_capital: float = 1_000_000) -> FactorBacktestResult:
        """
        Args:
            panel: 因子排名面板, index=日期, columns=品种
            assets: 各品种 OHLCV 数据
            top_n: 持仓 Top N
            rebalance: 调仓频率 'D'/'W'/'M'
        """
        # 对齐日期
        dates = panel.index.sort_values()
        symbols = panel.columns.tolist()

        # 调仓日期
        rebalance_dates = pd.Series(dates).resample(rebalance).last().dropna()
        rebalance_set = set(rebalance_dates)

        engine = Engine(init_cash=initial_capital)
        context.mode = 'backtest'

        # 注册数据
        for code in symbols:
            if code in assets:
                df = assets[code].copy()
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if 'eob' not in df.columns:
                    df['eob'] = df.index
                context.subscribe(code, '1d')
                engine.add_data(code, '1d', df)

        # 策略状态
        cash = float(initial_capital)
        positions: Dict[str, float] = {}  # code → shares
        nav_history = []
        trade_count = 0

        class _RotationStrategy:
            def on_bar(self, ctx, bars):
                nonlocal cash, positions, trade_count

                if len(bars) == 0 or not bars[0].get('eob'):
                    return

                current_date = bars[0]['eob']
                # 总净值
                total_nav = cash
                for code, shares in positions.items():
                    for b in bars:
                        if b.get('symbol') == code:
                            total_nav += shares * b.get('close', 0)
                nav_history.append(total_nav)

                # 只在调仓日交易
                if current_date not in rebalance_set:
                    return
                if current_date not in panel.index:
                    return

                # 获取当天排名
                row = panel.loc[current_date]
                top_codes = row.nlargest(top_n).index.tolist()

                # 平仓：清掉不在 Top N 的
                for code in list(positions.keys()):
                    if code not in top_codes:
                        for b in bars:
                            if b.get('symbol') == code:
                                price = b.get('close', 0)
                                if price > 0:
                                    cash += positions[code] * price
                                    trade_count += 1
                                del positions[code]
                                break

                # 开仓：买入新 Top N
                weight = 1.0 / len(top_codes)
                for code in top_codes:
                    if code in positions:
                        continue  # 已持有
                    for b in bars:
                        if b.get('symbol') == code:
                            price = b.get('close', 0)
                            if price > 0:
                                target_value = total_nav * weight
                                shares = target_value / price
                                cost = shares * price
                                if cost <= cash:
                                    cash -= cost
                                    positions[code] = shares
                                    trade_count += 1
                            break

        engine.add_strategy(_RotationStrategy())
        engine.run()

        # 计算结果
        nav = pd.Series(nav_history, index=dates[:len(nav_history)])
        ret = nav.pct_change().dropna()
        sharpe = float(ret.mean() / (ret.std() + 1e-10) * np.sqrt(252))
        cagr = float(nav.iloc[-1] ** (252 / len(nav)) - 1)
        dd = nav / nav.cummax() - 1
        max_dd = float(dd.min())

        pos_df = pd.DataFrame(index=panel.index, columns=symbols, data=0.0)

        return FactorBacktestResult(
            sharpe=sharpe, cagr=cagr, max_drawdown=max_dd,
            nav=nav, trades=trade_count, position=pos_df,
        )
