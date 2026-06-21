# [重构] 2026-06-12 从 backtest.py 重命名，对齐 signals/v4/engine.py
"""
factor/v4/engine.py — 因子轮动回测引擎 (ft2.core 驱动)

对齐 signals/v4 EngineCore 架构:
  full   — Engine.run() + AccountManager.order_percent() → AccountAnalyzer (完整指标/交易记录/notebook)
  fast   — Engine.run_fast() + FastAccount.order_percent() → AccountAnalyzer (~400ms/次, 1566天5品种)
  vector — 纯矩阵向量化, 跳过事件驱动循环 → AccountAnalyzer (~50~100x 快于 full)

用法:
  from factor.v4 import EngineCore

  # 三种模式统一返回 AccountAnalyzer，调用接口一致
  analyzer = EngineCore.backtest(panel, assets, mode='vector', top_n=3, rebalance='W')
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
    """因子轮动回测引擎 — ft2.core 驱动, full/fast/vector 三模式

    对齐 signals/v4 EngineCore:
      - full:   AccountManager.order_percent() → AccountAnalyzer
      - fast:   FastAccount, 无 TradeRecord/snapshots
      - vector: 纯矩阵向量化, 跳过事件驱动循环
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
                 buffer: int = 0,
                 fee_config: dict = None) -> AccountAnalyzer:
        """因子轮动回测统一入口

        Args:
            panel: 因子排名面板, index=日期, columns=品种, 值越大越好
            assets: {品种代码: OHLCV DataFrame} (优先于 returns)
            returns: 日收益率 DataFrame (自动合成 OHLCV assets)
            top_n: 持仓品种数
            rebalance: 调仓频率 'D'/'W'/'M'/'ME'/'5D' 或 RebalanceScheduler 对象
            mode: 'fast'/'full', 统一返回 AccountAnalyzer (接口一致)
            initial_capital: 初始资金
            start_date: 回测起始日, None=数据首位
            bench_label: full 模式下基准标签 (自动跑 BenchHolder)
            buffer: 排名缓冲数, 持仓品种排名滑出 top_n+buffer 名才剔除。
                    0=无缓冲(严格Top-N), >0=容忍滑出buffer个名次。
                    例: top_n=3, buffer=2 → 持仓排名在6名及以下才卖出
            fee_config: 费率配置 dict, None=零费率。
                        例: {'commission_rate': 0.0003, 'stamp_tax_rate': 0.001, 'min_commission': 5.0}
                        向量化模式不模拟 lot_size，用连续权重近似（偏差 <1%）

        Returns:
            AccountAnalyzer: 统一分析器 (可 sharpe_ratio()/metrics()/to_notebook())

        [重构] 2026-06-18 新增 returns 参数 (自动合成 OHLCV), rebalance 支持 Scheduler 对象
        [新增] 2026-06-21 mode='vector' 纯矩阵向量化路径
        """
        # 数据源: assets 优先, 否则从 returns 合成
        if assets is None and returns is not None:
            assets = EngineCore._synthetic_assets(returns)
        elif assets is None:
            raise ValueError("需要 assets 或 returns 参数")

        if mode == 'vector':
            return EngineCore._run_vectorized(panel, assets, top_n, rebalance,
                                             initial_capital, start_date, buffer, fee_config)
        elif mode == 'fast':
            return EngineCore._run_fast(panel, assets, top_n, rebalance,
                                       initial_capital, start_date, buffer, fee_config)
        else:
            return EngineCore._run_full(panel, assets, top_n, rebalance,
                                       initial_capital, start_date, bench_label, buffer, fee_config)

    # ============================================================
    # full 模式 — Engine + AccountManager → AccountAnalyzer
    # ============================================================

    @staticmethod
    def _run_full(panel, assets, top_n, rebalance, initial_capital, start_date,
                  bench_label, buffer=0, fee_config=None):
        """full 模式: Engine.run() + AccountManager.order_percent()"""
        panel = EngineCore._prepare_panel(panel, start_date)
        dates = panel.index.sort_values()
        symbols = panel.columns.tolist()

        rebalance_set = EngineCore._make_rebalance_set(dates, rebalance)

        engine = Engine(init_cash=initial_capital, fee_config=fee_config)
        context.mode = 'backtest'

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
            bench_eng = Engine(init_cash=initial_capital, fee_config=fee_config)
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
                  buffer=0, fee_config=None):
        """fast 模式: ctx.account 下单 (指向 FastAccount) → AccountAnalyzer"""
        panel = EngineCore._prepare_panel(panel, start_date)
        dates = panel.index.sort_values()
        symbols = panel.columns.tolist()

        rebalance_set = EngineCore._make_rebalance_set(dates, rebalance)

        engine = Engine(init_cash=initial_capital, fee_config=fee_config)
        context.mode = 'backtest'

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

        class _FastRotationStrategy:
            def __init__(self):
                self.use_buffer = buffer > 0
                self.buffer_rank = top_n + buffer

            def on_bar(self, ctx, bars):
                if len(bars) == 0:
                    return

                current_date = None
                for b in bars:
                    dt = b.get('eob')
                    if dt is not None:
                        current_date = pd.Timestamp(dt).normalize()
                        break
                if current_date is None:
                    return

                # 引擎自动记录日末净值 (drive_timeline 在 on_bar 后调用 mark())
                if current_date not in rebalance_set:
                    return
                if current_date not in panel.index:
                    return

                row = panel.loc[current_date]
                top_codes = set(row.nlargest(top_n).index.tolist())

                if self.use_buffer:
                    buffer_codes = set(row.nlargest(self.buffer_rank).index.tolist())
                else:
                    buffer_codes = top_codes

                # [修复] 2026-06-21 对齐 _run_full：平仓用 get_position() 获取持仓信息
                positions = ctx.account.get_position()
                n_keep = sum(1 for code, pos in positions.items()
                            if pos.get('volume', 0) > 0 and code in buffer_codes)
                for code, pos in list(positions.items()):
                    if pos.get('volume', 0) > 0 and code not in buffer_codes:
                        try:
                            ctx.account.order_percent(code, 1.0, OrderSide.Sell)
                        except (ValueError, RuntimeError):
                            pass

                # [修复] 2026-06-21 对齐 _run_full：开仓用 order_percent (而非 order_volume)
                n_slots = top_n - n_keep
                if top_codes and n_slots > 0:
                    weight = 1.0 / len(top_codes)
                    n_bought = 0
                    for code in top_codes & set(symbols):
                        if n_bought >= n_slots:
                            break
                        pos = positions.get(code)
                        if pos and pos.get('volume', 0) > 0:
                            continue
                        try:
                            ctx.account.order_percent(code, weight, OrderSide.Buy)
                            n_bought += 1
                        except (ValueError, RuntimeError):
                            pass

        start_time = panel.index[0].to_pydatetime()
        end_time = panel.index[-1].to_pydatetime()
        return engine.run_fast(_FastRotationStrategy(), start_time, end_time)

    # ============================================================
    # vector 模式 — 纯矩阵向量化, 跳过事件驱动循环
    # ============================================================

    @staticmethod
    def _run_vectorized(panel, assets, top_n, rebalance, initial_capital, start_date,
                        buffer=0, fee_config=None):
        """向量化因子轮动：纯矩阵运算，不经过 Engine._drive_timeline。

        连续权重近似 (无 lot_size)，费率模拟与 fast/full 对齐。
        调仓时序: 当日收盘后用旧权重计收益 → 调仓 → 记录净值 (与 _drive_timeline 一致)
        """
        panel = EngineCore._prepare_panel(panel, start_date)
        dates = panel.index.sort_values()
        symbols = panel.columns.tolist()
        symbols_set = set(symbols)
        n = len(dates)
        sym_to_idx = {c: i for i, c in enumerate(symbols)}

        # 构建价格 + 收益率矩阵 (T × N)
        price_arr = np.full((n, len(symbols)), np.nan)
        for code in symbols:
            if code in assets:
                price_arr[:, sym_to_idx[code]] = assets[code]['close'].reindex(dates).values
        returns_arr = np.diff(price_arr, axis=0) / price_arr[:-1]
        returns_arr = np.nan_to_num(returns_arr, nan=0.0)

        # 调仓日
        rebalance_set = EngineCore._make_rebalance_set(dates, rebalance)
        use_buffer = buffer > 0
        buffer_rank = top_n + buffer

        fee = fee_config or {'commission_rate': 0.0, 'stamp_tax_rate': 0.0, 'min_commission': 0.0}

        # 预计算每日 top_codes
        panel_arr = panel.values  # T × N
        top_indices = np.argsort(-panel_arr, axis=1)[:, :max(top_n, buffer_rank if use_buffer else top_n)]

        # 权重矩阵: weight_matrix[i] = 第i天开始时的持仓权重 (即前一天收盘调仓后的权重)
        weight_matrix = np.zeros((n, len(symbols)))
        held = set()
        current_weights = {}

        nav = float(initial_capital)
        daily_nav = {dates[0].date(): round(nav, 2)}

        for i in range(n):
            date = dates[i]
            is_rebalance = date in rebalance_set and date in panel.index

            # 1. 当日收益: 用旧权重 × 当日收益率 (i-1 → i)
            if i > 0 and current_weights:
                ret = 0.0
                for c, w in current_weights.items():
                    j = sym_to_idx.get(c)
                    if j is not None:
                        ret += w * returns_arr[i - 1, j]
                nav *= (1.0 + ret)

            # 2. 调仓 (收盘后)
            if is_rebalance:
                top_codes = set()
                for j in range(top_n):
                    code = symbols[top_indices[i, j]]
                    if code in symbols_set:
                        top_codes.add(code)

                if use_buffer:
                    buffer_codes = set()
                    for j in range(buffer_rank):
                        code = symbols[top_indices[i, j]]
                        if code in symbols_set:
                            buffer_codes.add(code)
                    keep = held & buffer_codes
                else:
                    keep = held & top_codes

                n_keep = len(keep)
                n_slots = top_n - n_keep
                new_codes = [c for c in top_codes if c not in held][:n_slots]
                target_codes = keep | set(new_codes)
                n_target = len(target_codes)

                if n_target > 0:
                    wt = 1.0 / n_target
                    new_weights = {c: wt for c in target_codes}
                else:
                    new_weights = {}

                # 手续费
                if current_weights:
                    sell_val = buy_val = 0.0
                    for c, w in current_weights.items():
                        nw = new_weights.get(c, 0.0)
                        if w > nw:
                            sell_val += (w - nw) * nav
                    for c, w in new_weights.items():
                        ow = current_weights.get(c, 0.0)
                        if w > ow:
                            buy_val += (w - ow) * nav
                    if sell_val > 0:
                        nav -= max(sell_val * fee['commission_rate'], fee['min_commission'])
                        nav -= sell_val * fee['stamp_tax_rate']
                    if buy_val > 0:
                        nav -= max(buy_val * fee['commission_rate'], fee['min_commission'])

                current_weights = new_weights
                held = target_codes

            weight_matrix[i] = [current_weights.get(c, 0.0) for c in symbols]
            daily_nav[date.date()] = round(nav, 2)

        return AccountAnalyzer(daily_assets=daily_nav)

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
