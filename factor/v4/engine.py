# [重构] 2026-06-12 从 backtest.py 重命名，对齐 signals/v4/engine.py
"""
factor/v4/engine.py — 因子轮动回测引擎 (ft2.core 驱动)

对齐 signals/v4 EngineCore 架构:
  full  — Engine.run() + AccountManager.order_percent() → AccountAnalyzer (完整指标/交易记录/notebook)
  fast  — Engine.run() + 自管净值 → FastResult (~0.5s/次)

用法:
  from factor.v4 import EngineCore

  # full 模式 — 验证 + 报告
  analyzer = EngineCore.backtest(panel, assets, mode='full', top_n=3, rebalance='W')
  analyzer.to_notebook("因子轮动回测")

  # fast 模式 — 搜索
  result = EngineCore.backtest(panel, assets, mode='fast', top_n=3, rebalance='W')
  # result.sharpe, result.cagr, result.max_drawdown
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from core.engine import Engine
from core.account import OrderSide, BenchHolder
from core.storage import context
from core.analyzer import AccountAnalyzer


@dataclass
class FastResult:
    """fast 模式回测结果 (轻量, 对齐 signals FastResult)"""
    sharpe: float
    cagr: float
    total_return: float
    max_drawdown: float
    annual_vol: float
    trades: int
    win_rate: float
    calmar: float
    nav: np.ndarray

    def __repr__(self):
        return (f"FastResult(SR={self.sharpe:.3f}, CAGR={self.cagr:.1%}, "
                f"MDD={self.max_drawdown:.1%}, trades={self.trades})")


class EngineCore:
    """因子轮动回测引擎 — ft2.core 驱动, full/fast 双模式

    对齐 signals/v4 EngineCore:
      - full: AccountManager.order_percent() → AccountAnalyzer
      - fast: 自管净值, 不生成 TradeRecord/snapshots
    """

    @staticmethod
    def backtest(panel: pd.DataFrame,
                 assets: Dict[str, pd.DataFrame],
                 top_n: int = 3,
                 rebalance: str = 'W',
                 mode: str = 'full',
                 initial_capital: float = 1_000_000,
                 start_date: str = None,
                 bench_label: str = None) -> Union[AccountAnalyzer, FastResult]:
        """因子轮动回测统一入口

        Args:
            panel: 因子排名面板, index=日期, columns=品种, 值越大越好
            assets: {品种代码: OHLCV DataFrame} (需含 close/high/low/open/volume)
            top_n: 持仓品种数
            rebalance: 调仓频率 'D'/'W'/'M'
            mode: 'full' → AccountAnalyzer, 'fast' → FastResult
            initial_capital: 初始资金
            start_date: 回测起始日, None=数据首位
            bench_label: full 模式下基准标签 (自动跑 BenchHolder)

        Returns:
            mode='full': AccountAnalyzer (可 to_notebook/set_benchmark)
            mode='fast': FastResult (轻量指标)
        """
        if mode == 'fast':
            return EngineCore._run_fast(panel, assets, top_n, rebalance,
                                       initial_capital, start_date)
        else:
            return EngineCore._run_full(panel, assets, top_n, rebalance,
                                       initial_capital, start_date, bench_label)

    # ============================================================
    # full 模式 — Engine + AccountManager → AccountAnalyzer
    # ============================================================

    @staticmethod
    def _run_full(panel, assets, top_n, rebalance, initial_capital, start_date,
                  bench_label):
        """full 模式: Engine.run() + AccountManager.order_percent()"""
        panel = EngineCore._prepare_panel(panel, start_date)
        dates = panel.index.sort_values()
        symbols = panel.columns.tolist()

        rebalance_dates = dates.to_series().resample(rebalance).last().dropna()
        rebalance_set = set(rebalance_dates)

        engine = Engine(init_cash=initial_capital)
        context.mode = 'backtest'
        engine.account.fee_config['commission_rate'] = 0.0
        engine.account.fee_config['stamp_tax_rate'] = 0.0
        engine.account.fee_config['min_commission'] = 0.0

        for code in symbols:
            if code in assets:
                df = assets[code].copy()
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if 'eob' not in df.columns:
                    df['eob'] = df.index
                # 嵌入因子排名
                if code in panel.columns:
                    df['_rank'] = panel[code].reindex(df.index).fillna(0).values
                else:
                    df['_rank'] = 0
                context.unsubscribe(code, '1d')
                context.subscribe(code, '1d', count=300)
                engine.add_data(code, '1d', df)

        # 策略状态
        symbols_set = set(symbols)

        class _FullRotationStrategy:
            def on_bar(self, ctx, bars):
                if len(bars) == 0:
                    return

                # 判断是否调仓日 (取首个 bar 的 eob)
                current_date = None
                for b in bars:
                    dt = b.get('eob')
                    if dt is not None:
                        current_date = dt
                        break
                if current_date is None or current_date not in rebalance_set:
                    return
                if current_date not in panel.index:
                    return

                # 获取当天排名
                row = panel.loc[current_date]
                top_codes = set(row.nlargest(top_n).index.tolist())

                # 平仓: 不在 Top N 的品种全部卖出
                positions = ctx.account.get_positions()
                for code, pos in list(positions.items()):
                    if pos.shares > 0 and code not in top_codes:
                        try:
                            ctx.account.order_percent(
                                code, 1.0, OrderSide.Sell,
                                note=f"rebalance: out of Top{top_n}",
                            )
                        except (ValueError, RuntimeError):
                            pass

                # 开仓: 买入 Top N 中未持有的品种 (等权分配)
                if top_codes:
                    weight = 1.0 / len(top_codes)
                    for code in top_codes & symbols_set:
                        pos = positions.get(code)
                        if pos and pos.shares > 0:
                            continue  # 已持有
                        try:
                            ctx.account.order_percent(
                                code, weight, OrderSide.Buy,
                                note=f"rebalance: Top{top_n}",
                            )
                        except (ValueError, RuntimeError):
                            pass

        start_time = panel.index[0].to_pydatetime()
        end_time = panel.index[-1].to_pydatetime()
        engine.run(_FullRotationStrategy, start_time, end_time)
        analyzer = AccountAnalyzer(engine.account)

        # 基准
        if bench_label is not None:
            bench_eng = Engine(init_cash=initial_capital)
            bench_eng.account.fee_config['commission_rate'] = 0.0
            bench_eng.account.fee_config['stamp_tax_rate'] = 0.0
            bench_eng.account.fee_config['min_commission'] = 0.0
            bench_symbol = bench_label
            if bench_symbol in assets:
                bench_df = assets[bench_symbol].copy()
                if not isinstance(bench_df.index, pd.DatetimeIndex):
                    bench_df.index = pd.to_datetime(bench_df.index)
                if 'eob' not in bench_df.columns:
                    bench_df['eob'] = bench_df.index
                context.unsubscribe(bench_symbol, '1d')
                context.subscribe(bench_symbol, '1d', count=3000)
                bench_eng.add_data(bench_symbol, '1d', bench_df)
                bench_eng.run(BenchHolder, start_time, end_time)
                bench_an = AccountAnalyzer(bench_eng.account)
                analyzer.set_benchmark(bench_an.daily_assets, bench_label)

        return analyzer

    # ============================================================
    # fast 模式 — Engine.run() 驱动 + 自管净值
    # ============================================================

    @staticmethod
    def _run_fast(panel, assets, top_n, rebalance, initial_capital, start_date):
        """fast 模式: 自管净值, 不调 order_percent (~0.5s/次)"""
        panel = EngineCore._prepare_panel(panel, start_date)
        dates = panel.index.sort_values()
        symbols = panel.columns.tolist()

        rebalance_dates = dates.to_series().resample(rebalance).last().dropna()
        rebalance_set = {pd.Timestamp(d.date()) for d in rebalance_dates}

        engine = Engine(init_cash=initial_capital)
        context.mode = 'backtest'
        engine.account.fee_config['commission_rate'] = 0.0
        engine.account.fee_config['stamp_tax_rate'] = 0.0
        engine.account.fee_config['min_commission'] = 0.0

        for code in symbols:
            if code in assets:
                df = assets[code].copy()
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if 'eob' not in df.columns:
                    df['eob'] = df.index
                if code in panel.columns:
                    df['_rank'] = panel[code].reindex(df.index).fillna(0).values
                else:
                    df['_rank'] = 0
                context.unsubscribe(code, '1d')
                context.subscribe(code, '1d', count=300)
                engine.add_data(code, '1d', df)

        cash = float(initial_capital)
        positions: Dict[str, float] = {}  # code → shares
        nav_history = []
        trade_count = 0
        wins = 0
        last_total_cost = 0.0

        class _FastRotationStrategy:
            def on_bar(self, ctx, bars):
                nonlocal cash, positions, trade_count, wins, last_total_cost

                if len(bars) == 0:
                    return

                # 计算净值
                total_nav = cash
                for code, shares in list(positions.items()):
                    for b in bars:
                        if b.get('symbol') == code:
                            total_nav += shares * b.get('close', b.get('open', 0))
                nav_history.append(total_nav)

                # 判断调仓日 (strip time from eob for comparison)
                current_date = None
                for b in bars:
                    dt = b.get('eob')
                    if dt is not None:
                        current_date = pd.Timestamp(dt).normalize()
                        break
                if current_date is None:
                    return
                if current_date not in rebalance_set:
                    return
                if current_date not in panel.index:
                    return

                row = panel.loc[current_date]
                top_codes = set(row.nlargest(top_n).index.tolist())

                # 平仓：清掉不在 Top N 的
                buy_cost = 0.0
                for code in list(positions.keys()):
                    if code not in top_codes:
                        for b in bars:
                            if b.get('symbol') == code:
                                price = b.get('close', b.get('open', 0))
                                if price > 0:
                                    cash += positions[code] * price
                                    trade_count += 1
                                del positions[code]
                                break

                # 开仓：买入新 Top N (等权)
                if top_codes:
                    weight = 1.0 / len(top_codes)
                    for code in top_codes & set(symbols):
                        if code in positions:
                            continue
                        for b in bars:
                            if b.get('symbol') == code:
                                price = b.get('close', b.get('open', 0))
                                if price > 0:
                                    target_value = total_nav * weight
                                    shares = target_value / price
                                    cost = shares * price
                                    if cost <= cash:
                                        cash -= cost
                                        positions[code] = shares
                                        trade_count += 1
                                        buy_cost += cost
                                break

                if buy_cost > 0:
                    last_total_cost = buy_cost

        start_time = panel.index[0].to_pydatetime()
        end_time = panel.index[-1].to_pydatetime()
        engine.run(_FastRotationStrategy, start_time, end_time)

        # 最终平仓
        if positions:
            final_nav = nav_history[-1] if nav_history else initial_capital
        else:
            final_nav = nav_history[-1] if nav_history else initial_capital

        nav_arr = np.array(nav_history, dtype=float)
        # 确保最后一点是最终净值
        nav_arr[-1] = final_nav

        total_return = final_nav / initial_capital - 1
        n_days = len(nav_arr)
        years = max((n_days - 1) / 252, 0.1)
        cagr = (final_nav / initial_capital) ** (1 / years) - 1

        cummax = np.maximum.accumulate(nav_arr)
        drawdown = nav_arr / cummax - 1
        max_dd = float(np.min(drawdown))

        daily_ret = np.diff(nav_arr) / nav_arr[:-1]
        annual_vol = float(np.std(daily_ret, ddof=1) * np.sqrt(252))
        sharpe = (cagr - 0.02) / annual_vol if annual_vol > 0 else 0
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        win_rate = wins / trade_count if trade_count > 0 else 0

        return FastResult(
            sharpe=float(sharpe), cagr=float(cagr),
            total_return=float(total_return), max_drawdown=float(max_dd),
            annual_vol=float(annual_vol), trades=trade_count,
            win_rate=float(win_rate), calmar=float(calmar), nav=nav_arr,
        )

    # ============================================================
    # 工具方法
    # ============================================================

    @staticmethod
    def _prepare_panel(panel, start_date):
        """标准化面板数据"""
        panel = panel.copy()
        if not isinstance(panel.index, pd.DatetimeIndex):
            panel.index = pd.to_datetime(panel.index)
        if start_date is not None:
            panel = panel.loc[panel.index >= pd.Timestamp(start_date)]
        return panel
