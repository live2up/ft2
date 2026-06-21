"""
factor/v3/engine_core.py — 因子轮动回测引擎 (ft2.core 驱动)

参考 v4/engine.py EngineCore 架构，适配 v3 数据格式（returns DataFrame 合成 OHLCV）。

与 v4 结构完全对齐：
  - 同架构: fast/full 双模式，均走 ft2.core Engine
  - 同输出: 统一返回 AccountAnalyzer
  - 同接口: backtest(panel, assets/returns, top_n, rebalance, mode)

v3 特有：输入 returns DataFrame 时自动合成 OHLCV assets dict，无缝桥接 ft2.core。

用法:
  >>> from factor.v3.engine_core import FactorEngineCore
  >>> analyzer = FactorEngineCore.backtest(panel, returns, top_n=3, rebalance='W')
  >>> print(analyzer.sharpe_ratio(), analyzer.metrics())

[新增] 2026-06-17 统一 v3 回测引擎到 ft2.core
[对齐] 2026-06-21 fast 模式对齐 v4: Engine.run_fast() + FastAccount
"""
import pandas as pd
import numpy as np
from typing import Dict, Union

from core.engine import Engine
from core.account import OrderSide, BenchHolder
from core.storage import context
from core.analyzer import AccountAnalyzer


# ── v3 调度器兼容 ──

def scheduler_to_rebalance(scheduler) -> str:
    """将 v3 RebalanceScheduler 对象转为 rebalance 字符串"""
    cls_name = scheduler.__class__.__name__
    if cls_name == 'FixedScheduler':
        return scheduler.freq  # 'ME' / 'W' / 'M'
    elif cls_name == 'IntervalScheduler':
        return f'{scheduler.interval_days}D'
    return 'W'


def allocator_to_top_n(allocator) -> int:
    """从 v3 WeightAllocator 对象提取 top_n"""
    cls_name = allocator.__class__.__name__
    if cls_name == 'TopNEqualWeight':
        return allocator.top_n
    return 3


class FactorEngineCore:
    """因子轮动回测引擎 — ft2.core 驱动, fast/full 双模式

    完全对齐 v4 EngineCore 架构:
      - fast: Engine.run_fast() + FastAccount.order_percent() → AccountAnalyzer
      - full: Engine.run() + AccountManager.order_percent() → AccountAnalyzer
      统一返回 AccountAnalyzer，调用 sharpe_ratio()/metrics()/to_notebook()
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
                 fee_config: dict = None) -> AccountAnalyzer:           # [对齐] 2026-06-20 v4 对齐
        """因子轮动回测统一入口

        Args:
            panel: 因子排名面板, index=日期, columns=品种, 值越大越好
            assets: {品种代码: OHLCV DataFrame} (优先于 returns)
            returns: 日收益率 DataFrame (自动合成 OHLCV assets)
            top_n: 持仓品种数
            rebalance: 调仓频率 'D'/'W'/'M'/'ME'/'5D' 或 RebalanceScheduler 对象
            mode: 'full'/'fast', 统一返回 AccountAnalyzer
            initial_capital: 初始资金
            start_date: 回测起始日, None=数据首位
            bench_label: full 模式下基准标签 (自动跑 BenchHolder)
            buffer: 排名缓冲数, 持仓品种排名滑出 top_n+buffer 名才剔除。
                    0=无缓冲(严格Top-N), >0=容忍滑出buffer个名次。
                    例: top_n=3, buffer=2 → 持仓排名在6名及以下才卖出
            fee_config: 费率配置 dict, None=零费率。
                        例: {'commission_rate': 0.0003, 'stamp_tax_rate': 0.001, 'min_commission': 5.0}

        Returns:
            AccountAnalyzer: 统一分析器 (可 sharpe_ratio()/metrics()/to_notebook())
        """
        # 数据源: assets 优先, 否则从 returns 合成
        if assets is not None:
            asset_dict = assets
        elif returns is not None:
            asset_dict = FactorEngineCore._synthetic_assets(returns)
        else:
            raise ValueError("需要 assets 或 returns 参数")

        if mode == 'fast':
            return FactorEngineCore._run_fast(panel, asset_dict, top_n, rebalance,
                                             initial_capital, start_date, buffer, fee_config)
        else:
            return FactorEngineCore._run_full(panel, asset_dict, top_n, rebalance,
                                             initial_capital, start_date, bench_label,
                                             buffer, fee_config)

    # ============================================================
    # fast 模式 — Engine.run_fast() + FastAccount (对齐 v4 _run_fast)
    # ============================================================

    @staticmethod
    def _run_fast(panel, assets, top_n, rebalance, initial_capital,
                  start_date=None, buffer=0, fee_config=None):
        """fast 模式: ctx.account 下单 (指向 FastAccount), 对齐 v4"""
        panel = FactorEngineCore._prepare_panel(panel, start_date)
        dates = panel.index.sort_values()
        symbols = panel.columns.tolist()

        rebalance_set = FactorEngineCore._make_rebalance_set(dates, rebalance)

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

                # 引擎自动记录日末净值
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

                # 平仓
                for code in list(ctx.account.positions.keys()):
                    if code not in buffer_codes:
                        ctx.account.order_percent(code, 1.0, OrderSide.Sell)

                # 开仓：等权买入
                n_slots = top_n - len(ctx.account.positions)
                if top_codes and n_slots > 0:
                    weight = 1.0 / len(top_codes)
                    total_nav = ctx.account.get_account()['nav']
                    n_bought = 0
                    for code in top_codes & set(symbols):
                        if n_bought >= n_slots:
                            break
                        if code in ctx.account.positions:
                            continue
                        price = ctx.account._get_price(code)
                        if price <= 0:
                            continue
                        shares = (total_nav * weight) / price
                        ctx.account.order_volume(code, shares, OrderSide.Buy)
                        n_bought += 1

        start_time = panel.index[0].to_pydatetime()
        end_time = panel.index[-1].to_pydatetime()
        return engine.run_fast(_FastRotationStrategy(), start_time, end_time)

    # ============================================================
    # full 模式 — Engine.run() + AccountManager.order_percent() → AccountAnalyzer
    # ============================================================

    @staticmethod
    def _run_full(panel, assets, top_n, rebalance,
                  initial_capital, start_date, bench_label, buffer=0,
                  fee_config=None):  # [对齐] 2026-06-20 v4 对齐
        """full 模式: Engine.run() + AccountManager.order_percent(), 对齐 v4"""
        panel = FactorEngineCore._prepare_panel(panel, start_date)

        dates = panel.index.sort_values()
        symbols = [s for s in panel.columns.tolist() if s in assets]

        # 调仓日: 兼容字符串和 Scheduler 对象
        rebalance_set = FactorEngineCore._make_rebalance_set(dates, rebalance)

        engine = Engine(init_cash=initial_capital, fee_config=fee_config)
        context.mode = 'backtest'

        for code in symbols:
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

        symbols_set = set(symbols)
        # [对齐] 2026-06-20 buffer: 相对偏移量, >0启用缓冲, 0=严格Top-N
        _use_buffer = buffer > 0
        _buffer_rank = top_n + buffer  # 缓冲截断排名

        class _FullRotationStrategy:
            def on_bar(self, ctx, bars):
                if len(bars) == 0:
                    return

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

                row = panel.loc[current_date]
                top_codes = set(row.nlargest(top_n).index.tolist())

                # [对齐] 2026-06-20 缓冲区逻辑：持仓排名滑出 top_n+buffer 名才剔除
                if _use_buffer:
                    buffer_codes = set(row.nlargest(_buffer_rank).index.tolist())
                else:
                    buffer_codes = top_codes

                # 平仓: 不在缓冲区内的品种全部卖出
                positions = ctx.account.get_position()
                # [对齐] 2026-06-20 卖出前预计算保留品种数，避免卖出后positions未及时更新
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
        if bench_label is not None and bench_label in assets:
            bench_eng = Engine(init_cash=initial_capital, fee_config=fee_config)
            bench_df = assets[bench_label].copy()
            if not isinstance(bench_df.index, pd.DatetimeIndex):
                bench_df.index = pd.to_datetime(bench_df.index)
            if 'eob' not in bench_df.columns:
                bench_df['eob'] = bench_df.index
            context.unsubscribe(bench_label, '1d')
            context.subscribe(bench_label, '1d', count=3000)
            bench_eng.add_data(bench_label, '1d', bench_df)
            bench_eng.run(BenchHolder, start_time, end_time)
            bench_an = AccountAnalyzer(bench_eng.account)
            analyzer.set_benchmark(bench_an.daily_assets, bench_label)

        return analyzer

    # ============================================================
    # 工具方法
    # ============================================================

    @staticmethod
    def _make_rebalance_set(dates: pd.DatetimeIndex,
                            rebalance: Union[str, object]) -> set:
        """生成调仓日集合，兼容字符串和 v3 Scheduler 对象

        字符串: 使用 pd.resample (与 v4 一致)
        Scheduler: 调用 generate() (保持 v3 兼容)
        """
        # v3 Scheduler 对象 → 调用 generate
        if hasattr(rebalance, 'generate'):
            rb_dates = rebalance.generate(dates)
            return {pd.Timestamp(d.date()) for d in rb_dates}

        # 字符串: 与 v4 完全一致
        rebalance = str(rebalance)
        if rebalance.endswith('D') and rebalance[:-1].isdigit():
            # 固定间隔: '5D' → 每5个交易日 (对齐 v3 IntervalScheduler)
            interval = int(rebalance[:-1])
            rb_dates = [dates[i] for i in range(interval - 1, len(dates), interval)]
            return {pd.Timestamp(d.date()) for d in rb_dates}

        # 标准频率: 'D' / 'W' / 'M' / 'ME' (与 v4 完全一致)
        freq_map = {'M': 'MS', 'ME': 'ME'}
        freq = freq_map.get(rebalance, rebalance)
        rebalance_series = dates.to_series().resample(freq).last().dropna()
        return {pd.Timestamp(d.date()) for d in rebalance_series}

    @staticmethod
    def _prepare_panel(panel, start_date):
        """标准化因子面板"""
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

        Returns:
            {品种代码: OHLCV DataFrame}
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
