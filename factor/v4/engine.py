# [重构] 2026-06-12 从 backtest.py 重命名，对齐 signals/v4/engine.py
"""
factor/v4/engine.py — 因子轮动回测引擎 (ft2.core 驱动)

对齐 signals/v4 EngineCore 架构:
  full  — Engine.run() + AccountManager.order_percent() → AccountAnalyzer (完整指标/交易记录/notebook)
  fast  — Engine.run() + 自管净值 → AccountAnalyzer (~0.5s/次)

用法:
  from factor.v4 import EngineCore

  # 两种模式统一返回 AccountAnalyzer，调用接口一致
  analyzer = EngineCore.backtest(panel, assets, mode='fast', top_n=3, rebalance='W')
  print(analyzer.sharpe_ratio(), analyzer.max_drawdown(), analyzer.metrics())

  # full 模式 — 验证 + 报告
  analyzer = EngineCore.backtest(panel, assets, mode='full', top_n=3, rebalance='W')
  analyzer.to_notebook("因子轮动回测")
"""
import pandas as pd
from typing import Dict

from core.engine import Engine
from core.account import OrderSide, BenchHolder
from core.storage import context
from core.analyzer import AccountAnalyzer






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
                 bench_label: str = None,
                 buffer: int = 0) -> AccountAnalyzer:
        """因子轮动回测统一入口

        Args:
            panel: 因子排名面板, index=日期, columns=品种, 值越大越好
            assets: {品种代码: OHLCV DataFrame} (需含 close/high/low/open/volume)
            top_n: 持仓品种数
            rebalance: 调仓频率 'D'/'W'/'M'
            mode: 'full'/'fast', 统一返回 AccountAnalyzer (接口一致)
            initial_capital: 初始资金
            start_date: 回测起始日, None=数据首位
            bench_label: full 模式下基准标签 (自动跑 BenchHolder)
            buffer: 排名缓冲数, 持仓品种排名滑出 top_n+buffer 名才剔除。
                    0=无缓冲(严格Top-N), >0=容忍滑出buffer个名次。
                    例: top_n=3, buffer=2 → 持仓排名在6名及以下才卖出

        Returns:
            AccountAnalyzer: 统一分析器 (可 sharpe_ratio()/metrics()/to_notebook())
        """
        if mode == 'fast':
            return EngineCore._run_fast(panel, assets, top_n, rebalance,
                                       initial_capital, start_date, buffer)
        else:
            return EngineCore._run_full(panel, assets, top_n, rebalance,
                                       initial_capital, start_date, bench_label, buffer)

    # ============================================================
    # full 模式 — Engine + AccountManager → AccountAnalyzer
    # ============================================================

    @staticmethod
    def _run_full(panel, assets, top_n, rebalance, initial_capital, start_date,
                  bench_label, buffer=0):
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
        # [重构] 2026-06-18 buffer: 相对偏移量, >0启用缓冲, 0=严格Top-N
        _use_buffer = buffer > 0
        _buffer_rank = top_n + buffer  # 缓冲截断排名

        class _FullRotationStrategy:
            def on_bar(self, ctx, bars):
                if len(bars) == 0:
                    return

                # 判断是否调仓日 (strip time from eob for comparison)
                current_date = None
                for b in bars:
                    dt = b.get('eob')
                    if dt is not None:
                        current_date = pd.Timestamp(dt).normalize()
                        break
                if current_date is None or current_date not in rebalance_set:
                    return
                if current_date not in panel.index:
                    return

                # 获取当天排名
                row = panel.loc[current_date]
                top_codes = set(row.nlargest(top_n).index.tolist())

                # [重构] 2026-06-18 缓冲区逻辑：持仓排名滑出 top_n+buffer 名才剔除
                # 无缓冲时退化为原始逻辑：不在top_codes即卖出
                if _use_buffer:
                    buffer_codes = set(row.nlargest(_buffer_rank).index.tolist())
                else:
                    buffer_codes = top_codes

                # 平仓: 不在缓冲区内的品种全部卖出
                # 排名跌出前 buffer_rank 名才卖出，减少无谓换手
                # 无缓冲时 buffer_codes=top_codes，退化为原始逻辑
                positions = ctx.account.get_position()  # None → 所有持仓 dict
                # [修复] 2026-06-18 卖出前预计算保留品种数，避免卖出后positions未及时更新
                n_keep = sum(1 for code, pos in positions.items()
                            if pos.get('volume', 0) > 0 and code in buffer_codes)
                for code, pos in list(positions.items()):
                    if pos.get('volume', 0) > 0 and code not in buffer_codes:
                        try:
                            ctx.account.order_percent(
                                code, 1.0, OrderSide.Sell,
                                note=f"rebalance: rank > {_buffer_rank}",
                            )
                        except (ValueError, RuntimeError):
                            pass

                # 开仓: 买入 Top N 中未持有的品种 (等权分配)
                # 持仓总数不超过top_n：缓冲保留的品种占位，只补买空位
                n_slots = top_n - n_keep  # 可补买的空位数
                if top_codes and n_slots > 0:
                    weight = 1.0 / len(top_codes)
                    n_bought = 0
                    for code in top_codes & symbols_set:
                        if n_bought >= n_slots:
                            break  # 空位已用完
                        pos = positions.get(code)
                        if pos and pos.get('volume', 0) > 0:
                            continue  # 已持有（包括缓冲保留的）
                        try:
                            ctx.account.order_percent(
                                code, weight, OrderSide.Buy,
                                note=f"rebalance: Top{top_n}",
                            )
                            n_bought += 1
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
    def _run_fast(panel, assets, top_n, rebalance, initial_capital, start_date,
                  buffer=0):
        """fast 模式: 自管净值 → AccountAnalyzer, 与 full 接口统一"""
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
        # [重构] 2026-06-16 直接构建 daily_assets dict → AccountAnalyzer
        daily_assets = {}
        # [重构] 2026-06-18 buffer: 相对偏移量, >0启用缓冲
        _use_buffer = buffer > 0
        _buffer_rank = top_n + buffer

        class _FastRotationStrategy:
            def on_bar(self, ctx, bars):
                nonlocal cash, positions

                if len(bars) == 0:
                    return

                # [重构] 2026-06-16 日期提取前置，确保 date/nav 一一对应
                current_date = None
                for b in bars:
                    dt = b.get('eob')
                    if dt is not None:
                        current_date = pd.Timestamp(dt).normalize()
                        break
                if current_date is None:
                    return

                # 计算净值 (调仓日先记旧持仓净值为准，因交易不改变NAV恒等式)
                total_nav = cash
                for code, shares in list(positions.items()):
                    for b in bars:
                        if b.get('symbol') == code:
                            total_nav += shares * b.get('close', b.get('open', 0))
                daily_assets[current_date] = float(total_nav)

                if current_date not in rebalance_set:
                    return
                if current_date not in panel.index:
                    return

                row = panel.loc[current_date]
                top_codes = set(row.nlargest(top_n).index.tolist())

                # [重构] 2026-06-18 缓冲区逻辑：持仓排名滑出 top_n+buffer 名才剔除
                if _use_buffer:
                    buffer_codes = set(row.nlargest(_buffer_rank).index.tolist())
                else:
                    buffer_codes = top_codes

                # 平仓：清掉不在缓冲区内的
                for code in list(positions.keys()):
                    if code not in buffer_codes:
                        for b in bars:
                            if b.get('symbol') == code:
                                price = b.get('close', b.get('open', 0))
                                if price > 0:
                                    cash += positions[code] * price
                                del positions[code]
                                break

                # 开仓：买入新 Top N (等权)
                # [修复] 2026-06-18 卖出后持仓数已更新，限制总持仓不超过top_n
                n_slots = top_n - len(positions)  # 可补买的空位数
                if top_codes and n_slots > 0:
                    weight = 1.0 / len(top_codes)
                    n_bought = 0
                    for code in top_codes & set(symbols):
                        if n_bought >= n_slots:
                            break  # 空位已用完
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
                                        n_bought += 1
                                break

        start_time = panel.index[0].to_pydatetime()
        end_time = panel.index[-1].to_pydatetime()
        engine.run(_FastRotationStrategy, start_time, end_time)

        # [重构] 2026-06-16 返回 AccountAnalyzer，与 full 模式接口统一
        return AccountAnalyzer(daily_assets=daily_assets)

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
