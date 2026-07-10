"""
signals/v5/engine.py — 统一回测引擎 (ft2.core 驱动)

[v5] 2026-07-10 从 v4 原样迁移，与 AST 版本无关。
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Union

from core.engine import Engine
from core.account import OrderSide, BenchHolder
from core.storage import context
from core.analyzer import AccountAnalyzer


class SigEngine:
    """v5 统一回测引擎 — ft2.core 驱动, full/fast 双模式

    [v5] 2026-07-10 从 v4 原样迁移，接口保持完全一致。
    """

    @staticmethod
    def backtest(
        signal: Union[pd.Series, np.ndarray, list],
        data: pd.DataFrame,
        symbol: str = '399317.SZ',
        freq: str = '1d',
        initial_capital: float = 1_000_000,
        start_date: str = None,
        mode: str = 'full',
        fee_config: dict = None,
        note_fields: List[str] = None,
        bench_label: str = None,
    ):
        """统一回测入口。两种模式均返回 AccountAnalyzer。"""
        if isinstance(signal, np.ndarray):
            signal = pd.Series(signal.ravel(), index=data.index[:len(signal)])
        elif isinstance(signal, list):
            signal = pd.Series(signal, index=data.index[:len(signal)])
        elif not isinstance(signal, pd.Series):
            raise TypeError(f"signal 类型不支持: {type(signal)}")

        if mode == 'fast':
            return SigEngine._run_fast(signal, data, symbol, freq,
                                       initial_capital, start_date, fee_config)
        else:
            return SigEngine._run_full(signal, data, symbol, freq,
                                       initial_capital, start_date,
                                       note_fields, bench_label, fee_config)

    @staticmethod
    def backtest_stateful(
        data: pd.DataFrame,
        buy_expr: str,
        sell_expr: str = None,
        max_hold: int = None,
        symbol: str = '399317.SZ',
        freq: str = '1d',
        initial_capital: float = 1_000_000,
        start_date: str = None,
        mode: str = 'fast',
        fee_config: dict = None,
        note_fields: list = None,
        bench_label: str = None,
        extra_features: dict = None,
    ):
        """非对称 BUY/SELL 状态机回测（一站式）"""
        from .expression import stateful_signal
        signal = stateful_signal(data, buy_expr, sell_expr, max_hold, extra_features)
        return SigEngine.backtest(signal, data, symbol, freq, initial_capital,
                                   start_date, mode, fee_config, note_fields,
                                   bench_label)

    @staticmethod
    def _run_full(signal, data, symbol, freq, initial_capital, start_date,
                  note_fields, bench_label, fee_config):
        """full 模式: 完整 Engine.run() → AccountAnalyzer"""
        df = SigEngine._prepare_data(data, signal, start_date)
        engine = Engine(init_cash=initial_capital, fee_config=fee_config)
        context.mode = 'backtest'

        context.unsubscribe(symbol, freq)
        context.subscribe(symbol, freq, count=300)
        engine.add_data(symbol, freq, df)

        note_fields = note_fields or []

        class _Strategy:
            def on_bar(self, ctx, bars):
                bar = bars[0]
                sig = bar.get('_signal', 0)
                has_pos = bool(ctx.account.get_position())

                note_parts = [f"signal={sig:.4f}"]
                for f in note_fields:
                    if f in bar:
                        note_parts.append(f"{f}={bar[f]}")
                note = ', '.join(note_parts)

                if sig > 0 and not has_pos:
                    try:
                        ctx.account.order_percent(symbol, 1.0, OrderSide.Buy, note=note)
                    except (ValueError, RuntimeError):
                        pass
                elif sig <= 0 and has_pos:
                    try:
                        ctx.account.order_percent(symbol, 1.0, OrderSide.Sell, note=note)
                    except (ValueError, RuntimeError):
                        pass

        engine.run(_Strategy, df['eob'].iloc[0], df['eob'].iloc[-1])
        analyzer = AccountAnalyzer(engine.account)

        if bench_label is not None:
            bench_df = data.copy()
            if start_date:
                bench_df = bench_df.loc[bench_df.index >= pd.Timestamp(start_date)]
            if 'eob' not in bench_df.columns:
                bench_df['eob'] = bench_df.index
            bench_eng = Engine(init_cash=initial_capital, fee_config=fee_config)
            context.unsubscribe(symbol, freq)
            context.subscribe(symbol, freq, count=3000)
            bench_eng.add_data(symbol, freq, bench_df)
            bench_eng.run(BenchHolder,
                          bench_df.index[0].to_pydatetime(),
                          bench_df.index[-1].to_pydatetime())
            bench_an = AccountAnalyzer(bench_eng.account)
            analyzer.set_benchmark(bench_an.daily_assets, bench_label)

        return analyzer

    @staticmethod
    def _run_fast(signal, data, symbol, freq, initial_capital, start_date, fee_config):
        """fast 模式: Engine.run_fast() 时间线驱动"""
        df = SigEngine._prepare_data(data, signal, start_date)
        engine = Engine(init_cash=initial_capital, fee_config=fee_config)
        context.mode = 'backtest'

        context.unsubscribe(symbol, freq)
        context.subscribe(symbol, freq, count=300)
        engine.add_data(symbol, freq, df)

        class _FastStrategy:
            def on_bar(self, ctx, bars):
                bar = bars[0]
                sig = bar.get('_signal', 0)
                has_pos = bool(ctx.account.get_position())

                if sig > 0 and not has_pos:
                    try:
                        ctx.account.order_percent(symbol, 1.0, OrderSide.Buy)
                    except (ValueError, RuntimeError):
                        pass
                elif sig <= 0 and has_pos:
                    try:
                        ctx.account.order_percent(symbol, 1.0, OrderSide.Sell)
                    except (ValueError, RuntimeError):
                        pass

        return engine.run_fast(_FastStrategy(), df['eob'].iloc[0], df['eob'].iloc[-1])

    @staticmethod
    def _prepare_data(data, signal, start_date):
        """信号嵌入 + start_date 截断"""
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if 'eob' not in df.columns:
            df['eob'] = df.index

        df['_signal'] = signal.reindex(df.index).fillna(0).values

        if start_date is not None:
            df = df.loc[df.index >= pd.Timestamp(start_date)].copy()

        return df
