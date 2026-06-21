"""
signals/v3/engine.py — 统一回测引擎 (ft2.core 驱动)
=============================================================================
v3 核心设计: 探索、测试、回测三个阶段全部走 ft2.core Engine。

两种模式:
  full  — Engine.run() + AccountManager → AccountAnalyzer (完整指标/交易记录/notebook)
  fast  — Engine.run_fast() + FastAccount → AccountAnalyzer (~0.5s/次)

两种模式统一返回 AccountAnalyzer，指标计算走同一套 @metric 方法，保证一致性。
差异仅在:
  full: 通过 AccountManager.order_percent() 下单 → TradeRecord/snapshots
  fast: 通过 FastAccount.order_percent() 下单，不生成 TradeRecord/snapshots
快约 6 倍，适合 GP/网格搜索。无费率模式下 fast/full 差异 < 0.001。

用法:
  from signals.v3 import EngineV3

  # full 模式 — 验证
  analyzer = EngineV3.backtest(signal, data, mode='full', start_date='2020-01-01')
  analyzer.set_benchmark(...).to_notebook("策略")

  # fast 模式 — 搜索 (同样返回 AccountAnalyzer)
  analyzer = EngineV3.backtest(signal, data, mode='fast', start_date='2020-01-01')
  print(analyzer.sharpe_ratio(), analyzer.annualized_return(), analyzer.max_drawdown())
=============================================================================
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Union

from core.engine import Engine
from core.account import OrderSide, BenchHolder
from core.storage import context
from core.analyzer import AccountAnalyzer


class EngineV3:
    """v3 统一回测引擎 — ft2.core 驱动, full/fast 双模式"""

    # ============================================================
    # 公共入口
    # ============================================================

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
        """
        统一回测入口。两种模式均返回 AccountAnalyzer，指标计算走同一套 @metric 方法。

        Args:
            signal: 信号序列 (pd.Series / np.ndarray / list), index=日期
                    value>0 做多, <=0 空仓
            data: OHLCV DataFrame, index=DatetimeIndex
            symbol: 标的代码
            freq: 频率
            initial_capital: 初始资金
            start_date: 回测起始日 ('2020-01-01'), None=从数据首位
            mode: 'full' → AccountAnalyzer (含交易记录), 'fast' → AccountAnalyzer (无交易记录, ~6x快)
            fee_config: 费率配置 dict, None=零费率。
                        例: {'commission_rate': 0.0003, 'stamp_tax_rate': 0.001, 'min_commission': 5.0}
            note_fields: full 模式下写入 TradeRecord.note 的字段
            bench_label: full 模式下基准标签 (自动跑 BenchHolder)

        Returns:
            AccountAnalyzer: 统一分析器 (可 sharpe_ratio()/annualized_return()/metrics()/to_notebook())
        """
        # 信号标准化
        if isinstance(signal, np.ndarray):
            signal = pd.Series(signal.ravel(), index=data.index[:len(signal)])
        elif isinstance(signal, list):
            signal = pd.Series(signal, index=data.index[:len(signal)])
        elif not isinstance(signal, pd.Series):
            raise TypeError(f"signal 类型不支持: {type(signal)}")

        if mode == 'fast':
            return EngineV3._run_fast(signal, data, symbol, freq,
                                       initial_capital, start_date, fee_config)
        else:
            return EngineV3._run_full(signal, data, symbol, freq,
                                       initial_capital, start_date,
                                       note_fields, bench_label, fee_config)

    # ============================================================
    # full 模式 — Engine + AccountManager → AccountAnalyzer
    # ============================================================

    @staticmethod
    def _run_full(signal, data, symbol, freq, initial_capital, start_date,
                  note_fields, bench_label, fee_config):
        """full 模式: 完整 Engine.run() → AccountAnalyzer"""
        df = EngineV3._prepare_data(data, signal, start_date)
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
                    except ValueError:
                        pass
                elif sig <= 0 and has_pos:
                    try:
                        ctx.account.order_percent(symbol, 1.0, OrderSide.Sell, note=note)
                    except ValueError:
                        pass

        engine.run(_Strategy, df['eob'].iloc[0], df['eob'].iloc[-1])
        analyzer = AccountAnalyzer(engine.account)

        # 基准 (同一费率)
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

    # ============================================================
    # fast 模式 — Engine.run_fast() 驱动 + FastAccount → AccountAnalyzer
    # ============================================================

    @staticmethod
    def _run_fast(signal, data, symbol, freq, initial_capital, start_date, fee_config):
        """
        fast 模式: Engine.run_fast() 时间线驱动, 策略通过 ctx.account 下单 → AccountAnalyzer。

        与 full 模式统一返回 AccountAnalyzer，指标计算走同一套 @metric 方法，
        保证 fast/full 指标一致。差异: 不生成 TradeRecord/snapshots, ~6x快。
        fast 模式下 ctx.account 指向 FastAccount，接口兼容。
        """
        df = EngineV3._prepare_data(data, signal, start_date)
        engine = Engine(init_cash=initial_capital, fee_config=fee_config)
        context.mode = 'backtest'

        context.unsubscribe(symbol, freq)
        context.subscribe(symbol, freq, count=300)
        engine.add_data(symbol, freq, df)

        class _FastStrategy:
            def on_bar(self, ctx, bars):
                bar = bars[0]
                sig = bar.get('_signal', 0)
                price = bar.get('close', bar.get('open', 0))
                if price <= 0:
                    return
                dt = bar.get('eob')
                if dt is None:
                    return

                ctx.account.mark()

                if sig > 0 and not ctx.account.positions:
                    ctx.account.order_percent(symbol, 1.0, OrderSide.Buy)
                elif sig <= 0 and ctx.account.positions:
                    ctx.account.order_percent(symbol, 1.0, OrderSide.Sell)

        return engine.run_fast(_FastStrategy(), df['eob'].iloc[0], df['eob'].iloc[-1])

    # ============================================================
    # 数据准备
    # ============================================================

    @staticmethod
    def _prepare_data(data, signal, start_date):
        """信号嵌入 + start_date 截断"""
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if 'eob' not in df.columns:
            df['eob'] = df.index

        # 信号: 全长填充, 缺失=0 (空仓)
        df['_signal'] = signal.reindex(df.index).fillna(0).values

        # start_date 截断
        if start_date is not None:
            df = df.loc[df.index >= pd.Timestamp(start_date)].copy()

        return df
