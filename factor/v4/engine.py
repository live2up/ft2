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
import numpy as np
import pandas as pd
from typing import Dict, Union

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
                 assets: Dict[str, pd.DataFrame] = None,
                 returns: pd.DataFrame = None,
                 top_n: int = 3,
                 rebalance: Union[str, object] = 'W',
                 mode: str = 'fast',
                 initial_capital: float = 1_000_000,
                 start_date: str = None,
                 bench_label: str = None,
                 buffer: int = 0) -> AccountAnalyzer:
        """因子轮动回测统一入口

        Args:
            panel: 因子排名面板, index=日期, columns=品种, 值越大越好
            assets: {品种代码: OHLCV DataFrame} (优先于 returns)
            returns: 日收益率 DataFrame (自动合成 OHLCV assets)
            top_n: 持仓品种数
            rebalance: 调仓频率 'D'/'W'/'M'/'ME'/'5D' 或 RebalanceScheduler 对象
            mode: 'fast'/'fast', 统一返回 AccountAnalyzer (接口一致)
            initial_capital: 初始资金
            start_date: 回测起始日, None=数据首位
            bench_label: full 模式下基准标签 (自动跑 BenchHolder)
            buffer: 排名缓冲数, 持仓品种排名滑出 top_n+buffer 名才剔除。
                    0=无缓冲(严格Top-N), >0=容忍滑出buffer个名次。
                    例: top_n=3, buffer=2 → 持仓排名在6名及以下才卖出

        Returns:
            AccountAnalyzer: 统一分析器 (可 sharpe_ratio()/metrics()/to_notebook())

        [重构] 2026-06-18 新增 returns 参数 (自动合成 OHLCV), rebalance 支持 Scheduler 对象
        """
        # 数据源: assets 优先, 否则从 returns 合成
        if assets is None and returns is not None:
            assets = EngineCore._synthetic_assets(returns)
        elif assets is None:
            raise ValueError("需要 assets 或 returns 参数")

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

        rebalance_set = EngineCore._make_rebalance_set(dates, rebalance)

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

        rebalance_set = EngineCore._make_rebalance_set(dates, rebalance)

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

        # [重构] 2026-06-20 策略状态改为实例属性, Engine.run_fast() 读取 self.daily_assets
        class _FastRotationStrategy:
            def __init__(self):
                self.cash = float(initial_capital)
                self.positions: Dict[str, float] = {}
                self.daily_assets = {}
                self.use_buffer = buffer > 0
                self.buffer_rank = top_n + buffer

            def on_bar(self, ctx, bars):
                if len(bars) == 0:
                    return

                # 日期提取前置，确保 date/nav 一一对应
                current_date = None
                for b in bars:
                    dt = b.get('eob')
                    if dt is not None:
                        current_date = pd.Timestamp(dt).normalize()
                        break
                if current_date is None:
                    return

                # 计算净值 (调仓日先记旧持仓净值为准，因交易不改变NAV恒等式)
                total_nav = self.cash
                for code, shares in list(self.positions.items()):
                    for b in bars:
                        if b.get('symbol') == code:
                            total_nav += shares * b.get('close', b.get('open', 0))
                # key 用 datetime.date, 与 full 模式 _aggregate_daily_assets 一致
                self.daily_assets[current_date.date()] = float(total_nav)

                if current_date not in rebalance_set:
                    return
                if current_date not in panel.index:
                    return

                row = panel.loc[current_date]
                top_codes = set(row.nlargest(top_n).index.tolist())

                # 缓冲区逻辑：持仓排名滑出 top_n+buffer 名才剔除
                if self.use_buffer:
                    buffer_codes = set(row.nlargest(self.buffer_rank).index.tolist())
                else:
                    buffer_codes = top_codes

                # 平仓：清掉不在缓冲区内的
                for code in list(self.positions.keys()):
                    if code not in buffer_codes:
                        for b in bars:
                            if b.get('symbol') == code:
                                price = b.get('close', b.get('open', 0))
                                if price > 0:
                                    self.cash += self.positions[code] * price
                                del self.positions[code]
                                break

                # 开仓：买入新 Top N (等权)
                n_slots = top_n - len(self.positions)
                if top_codes and n_slots > 0:
                    weight = 1.0 / len(top_codes)
                    n_bought = 0
                    for code in top_codes & set(symbols):
                        if n_bought >= n_slots:
                            break
                        if code in self.positions:
                            continue
                        for b in bars:
                            if b.get('symbol') == code:
                                price = b.get('close', b.get('open', 0))
                                if price > 0:
                                    target_value = total_nav * weight
                                    shares = target_value / price
                                    cost = shares * price
                                    if cost <= self.cash:
                                        self.cash -= cost
                                        self.positions[code] = shares
                                        n_bought += 1
                                break

        start_time = panel.index[0].to_pydatetime()
        end_time = panel.index[-1].to_pydatetime()
        # [重构] 2026-06-20 使用 Engine.run_fast()
        return engine.run_fast(_FastRotationStrategy(), start_time, end_time)

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

    @staticmethod
    def _synthetic_assets(returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """从日收益率 DataFrame 合成 OHLCV assets dict

        close = cumprod(1+r), open/high/low = close, volume = 1。
        Engine 的 order_percent/自管持仓只看 close 价格，不影响回测结果。

        [新增] 2026-06-18 从 v3 engine_core 上移，v4 自包含
        """
        assets = {}
        dates = returns.index
        for code in returns.columns:
            ret_col = returns[code].values
            close_prices = np.cumprod(1.0 + np.nan_to_num(ret_col, nan=0.0))
            df = pd.DataFrame({
                'open': close_prices,
                'high': close_prices,
                'low': close_prices,
                'close': close_prices,
                'volume': 1.0,
            }, index=dates)
            df['eob'] = dates
            assets[code] = df
        return assets

    @staticmethod
    def _make_rebalance_set(dates: pd.DatetimeIndex,
                            rebalance: Union[str, object]) -> set:
        """生成调仓日集合，兼容字符串和 v3 RebalanceScheduler 对象

        字符串: 使用 pd.resample (与 v4 原逻辑一致)
        Scheduler: 调用 generate() (保持 v3 兼容)

        [新增] 2026-06-18 统一调仓日生成，支持 Scheduler 对象透传
        """
        # Scheduler 对象 → 调用 generate
        if hasattr(rebalance, 'generate'):
            rb_dates = rebalance.generate(dates)
            return {pd.Timestamp(d.date()) for d in rb_dates}

        # 字符串
        rebalance = str(rebalance)
        if rebalance.endswith('D') and rebalance[:-1].isdigit():
            # 固定间隔: '5D' → 每5个交易日
            interval = int(rebalance[:-1])
            rb_dates = [dates[i] for i in range(interval - 1, len(dates), interval)]
            return {pd.Timestamp(d.date()) for d in rb_dates}

        # 标准频率: 'D' / 'W' / 'M' / 'ME'
        freq_map = {'M': 'MS', 'ME': 'ME'}
        freq = freq_map.get(rebalance, rebalance)
        rebalance_series = dates.to_series().resample(freq).last().dropna()
        return {pd.Timestamp(d.date()) for d in rebalance_series}
