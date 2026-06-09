"""
signals/v3/engine.py — 统一回测引擎 (ft2.core 驱动)
=============================================================================
v3 核心设计: 探索、测试、回测三个阶段全部走 ft2.core Engine。

两种模式:
  full  — Engine.run() + AccountManager → AccountAnalyzer (完整指标/交易记录/notebook)
  fast  — Engine.run() + 轻量策略          → BacktestResult   (仅 Sharpe/CAGR/MDD, 无记录)

fast 模式与 full 模式的费率、时间线、净值计算完全一致，差异仅在是否生成
TradeRecord/snapshots。快约 6 倍，适合 GP/网格搜索。

用法:
  from signals.v3 import EngineV3

  # full 模式 — 验证
  analyzer = EngineV3.backtest(signal, data, mode='full', start_date='2020-01-01')
  analyzer.set_benchmark(...).to_notebook("策略")

  # fast 模式 — 搜索
  result = EngineV3.backtest(signal, data, mode='fast', start_date='2020-01-01')
  # result.sharpe, result.cagr, result.mdd, result.trades
=============================================================================
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from dataclasses import dataclass

from core.engine import Engine
from core.account import OrderSide, BenchHolder
from core.storage import context
from core.analyzer import AccountAnalyzer

# 费率配置 (对齐 ETF 万3 佣金 + 千1 印花税)
COMMISSION_RATE = 0.0003
STAMP_DUTY = 0.001         # 卖出时单边征收
MIN_COMMISSION = 5.0       # 最低佣金


@dataclass
class FastResult:
    """fast 模式回测结果 (轻量, 无交易记录)"""
    sharpe: float
    cagr: float          # 年化收益率 (小数, 如 0.148)
    total_return: float   # 累计收益率
    max_drawdown: float   # 最大回撤 (负数, 如 -0.105)
    annual_vol: float     # 年化波动率
    trades: int           # 交易回合数
    win_rate: float       # 胜率
    calmar: float         # Calmar 比率
    nav: np.ndarray       # 每日净值序列

    def __repr__(self):
        return (f"FastResult(SR={self.sharpe:.3f}, CAGR={self.cagr:.1%}, "
                f"MDD={self.max_drawdown:.1%}, trades={self.trades})")


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
        with_fees: bool = False,
        note_fields: List[str] = None,
        bench_label: str = None,
    ):
        """
        统一回测入口。

        Args:
            signal: 信号序列 (pd.Series / np.ndarray / list), index=日期
                    value>0 做多, <=0 空仓
            data: OHLCV DataFrame, index=DatetimeIndex
            symbol: 标的代码
            freq: 频率
            initial_capital: 初始资金
            start_date: 回测起始日 ('2020-01-01'), None=从数据首位
            mode: 'full' → AccountAnalyzer, 'fast' → FastResult
            with_fees: 是否扣除费率 (指数择时默认 False)
            note_fields: full 模式下写入 TradeRecord.note 的字段
            bench_label: full 模式下基准标签 (自动跑 BenchHolder)

        Returns:
            mode='full': AccountAnalyzer (可 to_notebook/set_benchmark)
            mode='fast': FastResult (轻量指标)
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
                                       initial_capital, start_date, with_fees)
        else:
            return EngineV3._run_full(signal, data, symbol, freq,
                                       initial_capital, start_date,
                                       note_fields, bench_label, with_fees)

    # ============================================================
    # full 模式 — Engine + AccountManager → AccountAnalyzer
    # ============================================================

    @staticmethod
    def _run_full(signal, data, symbol, freq, initial_capital, start_date,
                  note_fields, bench_label, with_fees):
        """full 模式: 完整 Engine.run() → AccountAnalyzer"""
        df = EngineV3._prepare_data(data, signal, start_date)
        engine = Engine(init_cash=initial_capital)
        context.mode = 'backtest'

        # [新增] 2026-06-10 指数择时默认不扣费率
        if not with_fees:
            engine.account.fee_config['commission_rate'] = 0.0
            engine.account.fee_config['stamp_tax_rate'] = 0.0
            engine.account.fee_config['min_commission'] = 0.0

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

        # 基准
        if bench_label is not None:
            bench_df = data.copy()
            if start_date:
                bench_df = bench_df.loc[bench_df.index >= pd.Timestamp(start_date)]
            bench_eng = Engine(init_cash=initial_capital)
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
    # fast 模式 — Engine 时间线驱动 + 轻量净值累积
    # ============================================================

    @staticmethod
    def _run_fast(signal, data, symbol, freq, initial_capital, start_date, with_fees):
        """
        fast 模式: numpy 直算净值, 不生成 TradeRecord/snapshots。

        费率公式与 core.AccountManager 完全一致:
          买入成本 = price * shares + max(price*shares * COMMISSION_RATE, MIN_COMMISSION)
          卖出收入 = price * shares - max(price*shares * COMMISSION_RATE, MIN_COMMISSION)
                   - price*shares * STAMP_DUTY

        速度: ~0.3s/次 (vs full 模式 ~3s/次)
        """
        df = EngineV3._prepare_data(data, signal, start_date)
        closes = df['close'].values
        signals = df['_signal'].values
        n = len(df)

        nav = np.full(n, initial_capital, dtype=float)
        position = 0          # 0=空仓, 1=满仓
        buy_nav = 0.0         # 买入时的净值 (用于计算持仓股数)
        trade_rounds = 0
        wins = 0

        for i in range(1, n):
            # 继承前一日净值
            if position == 0:
                nav[i] = nav[i-1]
            else:
                # 持仓: 按价格变动调整净值
                nav[i] = nav[i-1] * (closes[i] / closes[i-1])

            sig = signals[i]  # signal[i] 决定 i 时刻的操作

            if sig > 0 and position == 0:
                # 买入
                cash = nav[i]
                if with_fees:
                    commission = max(cash * COMMISSION_RATE, MIN_COMMISSION)
                    investable = cash - commission
                else:
                    investable = cash
                if investable > 0:
                    position = 1
                    buy_nav = investable
                    trade_rounds += 1

            elif sig <= 0 and position == 1:
                # 卖出
                sell_value = nav[i]
                if with_fees:
                    commission = max(sell_value * COMMISSION_RATE, MIN_COMMISSION)
                    stamp = sell_value * STAMP_DUTY
                    nav[i] = sell_value - commission - stamp
                else:
                    nav[i] = sell_value
                if nav[i] > buy_nav:
                    wins += 1
                position = 0
                buy_nav = 0.0

        # 最终仍持仓 => 按最后价平仓
        if position == 1:
            sell_value = nav[-1]
            if with_fees:
                commission = max(sell_value * COMMISSION_RATE, MIN_COMMISSION)
                stamp = sell_value * STAMP_DUTY
                nav[-1] = sell_value - commission - stamp
            else:
                nav[-1] = sell_value

        # 指标计算
        final_nav = nav[-1]
        total_return = final_nav / initial_capital - 1
        years = max((n - 1) / 252, 0.1)
        cagr = (final_nav / initial_capital) ** (1 / years) - 1

        cummax = np.maximum.accumulate(nav)
        drawdown = nav / cummax - 1
        max_dd = float(np.min(drawdown))

        daily_ret = np.diff(nav) / nav[:-1]
        # 去掉换手日的异常收益 (买入/卖出当天 nav 包含了费用跳变)
        daily_ret_clean = daily_ret[np.abs(daily_ret) < 0.5]
        if len(daily_ret_clean) < 10:
            daily_ret_clean = daily_ret

        annual_vol = float(np.std(daily_ret_clean, ddof=1) * np.sqrt(252))
        sharpe = (cagr - 0.02) / annual_vol if annual_vol > 0 else 0
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        win_rate = wins / trade_rounds if trade_rounds > 0 else 0

        return FastResult(
            sharpe=float(sharpe), cagr=float(cagr), total_return=float(total_return),
            max_drawdown=float(max_dd), annual_vol=float(annual_vol),
            trades=trade_rounds, win_rate=float(win_rate), calmar=float(calmar),
            nav=nav,
        )

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
