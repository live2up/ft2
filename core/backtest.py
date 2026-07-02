"""
core/backtest.py — 简化回测入口 (Engine 二次封装)
============================================================

定位：core.Engine 之上的薄封装，消除重复的使用模板代码。
      不替代 Engine，内部委托 Engine.run / Engine.run_fast。

封装的模板:
  1. 创建 Engine + context.mode 配置
  2. 数据注入 (add_data 逐个传入 + eob 列 + DatetimeIndex + subscribe)
  3. 基准注入 (BenchHolder + set_benchmark, full 模式自动)

不封装的:
  - Strategy 的 on_bar 决策逻辑 (用户自己写, 保留完整事件驱动能力)
  - start_time/end_time (用户传入, 因场景而异)

用法 — 自定义 Strategy (完整事件驱动能力):
  >>> from core import Backtester
  >>> bt = Backtester()
  >>> bt.set_init_cash(1_000_000).set_benchmark('399317.SZ', df_399317, '国证A指')
  >>> bt.add_data('600000.SH', df_600000, symbol_name='浦发银行')
  >>> analyzer = bt.run(MyStrategy(), start_time, end_time, mode='full')

逃生舱: bt.engine 可直接拿原始 Engine 做高级操作。

[新增] 2026-06-30 Engine 二次封装, 简化调用不丢灵活性
"""

import pandas as pd
from typing import Dict, Optional, Any

from .engine import Engine
from .account import BenchHolder
from .storage import context
from .analyzer import AccountAnalyzer




# ============================================================
# Backtester — Engine 二次封装
# ============================================================

class Backtester:
    """简化回测入口 — 封装 Engine 使用模板, 不封装 Strategy 决策

    封装:
      - Engine 创建 (init_cash / fee_config)
      - context.mode = 'backtest'
      - 数据注入 (add_data 逐个传入 + eob 列 + DatetimeIndex + subscribe)
      - 基准注入 (BenchHolder + set_benchmark, full 模式自动)

    不封装:
      - Strategy 的 on_bar (用户自定义, 保留完整事件驱动能力)
      - start_time/end_time (用户传入, 因场景而异)

    Attributes:
        engine: 原始 Engine 实例 (add_data 后可用, run 前可配置, run 后可查询)
    """

    def __init__(self):
        """构造空 Backtester, 数据在 add_data() 时传入"""
        self._symbol_data: Dict[str, pd.DataFrame] = {}
        self._symbol_names: Dict[str, str] = {}  # 品种名称（可选，仅用于显示）
        self._init_cash: float = 1e6
        self._fee_config: Optional[dict] = None
        self._bench_label: Optional[str] = None
        self._freq: str = '1d'

        # engine 在 add_data 时创建
        self.engine: Optional[Engine] = None

    # ── 配置 (链式) ──

    def set_init_cash(self, init_cash: float) -> 'Backtester':
        """设置初始资金 (对齐 Engine.__init__ 的 init_cash 参数)"""
        self._init_cash = init_cash
        return self

    def set_fee_config(self, fee_config: dict) -> 'Backtester':
        """设置费率配置 (对齐 Engine.__init__ 的 fee_config 参数)"""
        self._fee_config = fee_config
        return self

    def set_benchmark(self, bench_label: str, bench_data: pd.DataFrame,
                      symbol_name: str = None) -> 'Backtester':
        """设置基准品种 (标签+数据一起传, full 模式自动跑 BenchHolder)

        对齐 FacEngine 的 bench_label 参数，基准数据只存不加入主引擎。

        Args:
            bench_label: 基准品种代码 (如 '399317.SZ')
            bench_data: OHLCV DataFrame
            symbol_name: 品种名称 (可选, 仅用于显示)
        """
        self._bench_label = bench_label

        # 标准化数据
        df = bench_data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if 'eob' not in df.columns:
            df['eob'] = df.index

        # 基准数据只存 _symbol_data, 不加入主引擎
        self._symbol_data[bench_label] = df
        if symbol_name:
            self._symbol_names[bench_label] = symbol_name
        elif 'name' in df.columns:
            self._symbol_names[bench_label] = df['name'].iloc[-1]

        return self

    def set_freq(self, freq: str) -> 'Backtester':
        """设置数据频率 (默认 '1d', 支持 '1w' 等多频率)"""
        self._freq = freq
        return self

    # ── 数据注入 ──

    def add_data(self, symbol: str, data: pd.DataFrame, symbol_name: str = None) -> 'Backtester':
        """添加策略品种数据，对齐 Engine.add_data 风格

        注意: 基准数据请通过 set_benchmark(label, data) 传入，不走 add_data。

        封装:
          - Engine 创建 (首次 add_data 时)
          - context.mode = 'backtest'
          - DatetimeIndex + eob 列 + subscribe + add_data
          - symbol_name 透传给 Engine (Engine 自动提取 df 列作为后备)

        Args:
            symbol: 品种代码（如 '600000.SH', '399317.SZ'）
            data: OHLCV DataFrame，需满足:
                * index: 日期（DatetimeIndex / 可转换的日期字符串）
                * columns: 应含 OHLCV 字段（open/high/low/close/amount/volume 等）
                  可额外带 symbol_name/name 列（Engine 自动识别为品种名称）
                * eob 列（可选）：若不传，Backtester 自动设 eob = index
            symbol_name: 品种名称（可选，显式传入时优先于 df 列）

        Returns:
            Backtester (链式)

        Example:
            >>> from core import Backtester
            >>> bt = Backtester().set_init_cash(1e6).set_benchmark('399317.SZ', df_399317, '国证A指')
            >>> bt.add_data('600000.SH', df_600000, symbol_name='浦发银行')
            >>> analyzer = bt.run(MyStrategy(), start, end)
        """
        # 首次 add_data 时创建 Engine
        if self.engine is None:
            self.engine = Engine(init_cash=self._init_cash, fee_config=self._fee_config)
            context.mode = 'backtest'

        # 标准化数据
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if 'eob' not in df.columns:
            df['eob'] = df.index

        # subscribe + add_data
        context.unsubscribe(symbol, self._freq)
        context.subscribe(symbol, self._freq, count=300)
        self.engine.add_data(symbol, self._freq, df, symbol_name=symbol_name)

        return self

    # ── 执行 (接受任意 Strategy) ──

    def run(self, strategy: Any, start_time, end_time,
            mode: str = 'full') -> AccountAnalyzer:
        """执行回测 — 用户传 Strategy, 保留完整事件驱动能力

        Args:
            strategy: 带 on_bar 的对象或类 (同 Engine.run)
            start_time: 回测起始时间
            end_time: 回测结束时间
            mode: 'full' (快照+交易记录+基准) / 'fast' (仅净值, 无交易记录)

        Returns:
            AccountAnalyzer

        Example:
            >>> bt = Backtester().set_init_cash(1e6).set_benchmark('399317.SZ', df_399317)
            >>> bt.add_data('600000.SH', df_600000)
            >>> analyzer = bt.run(MyStrategy(), start, end, mode='full')
        """
        if self.engine is None:
            raise RuntimeError("未注入数据, 请先调 add_data()")

        if mode == 'fast':
            return self.engine.run_fast(strategy, start_time, end_time)

        # full
        self.engine.run(strategy, start_time, end_time)
        analyzer = AccountAnalyzer(self.engine.account)

        # 基准注入
        if self._bench_label and self._bench_label in self._symbol_data:
            analyzer = self._inject_benchmark(analyzer, start_time, end_time)

        return analyzer

    # ── 内部: 基准注入 (封装 BenchHolder 模板) ──

    def _inject_benchmark(self, analyzer: AccountAnalyzer,
                          start_time, end_time) -> AccountAnalyzer:
        """跑基准 Engine + BenchHolder, 注入对比"""
        bench_label = self._bench_label

        # 直接使用已标准化的数据
        if bench_label not in self._symbol_data:
            raise ValueError(f"基准品种 {bench_label} 未通过 set_benchmark() 注入")

        bench_df = self._symbol_data[bench_label].copy()

        # 创建基准 Engine
        bench_eng = Engine(init_cash=self._init_cash, fee_config=self._fee_config)

        # 基准数据需要单独 subscribe + add_data（因为没加入主引擎）
        context.unsubscribe(bench_label, self._freq)
        context.subscribe(bench_label, self._freq, count=3000)

        # 获取品种名称（用于显示，没有则回退到代码）
        bench_name = self._symbol_names.get(bench_label)

        # 传入 symbol_name（如果有的话）
        bench_eng.add_data(bench_label, self._freq, bench_df, symbol_name=bench_name)

        bench_eng.run(BenchHolder, start_time, end_time)
        bench_an = AccountAnalyzer(bench_eng.account)

        analyzer.set_benchmark(bench_an.daily_assets, bench_name or bench_label)
        return analyzer
