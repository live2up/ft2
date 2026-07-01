"""
core/backtest.py — 简化回测入口 (Engine 二次封装)
============================================================

定位：core.Engine 之上的薄封装，消除重复的使用模板代码。
      不替代 Engine，内部委托 Engine.run / Engine.run_fast。

封装的模板:
  1. 创建 Engine + context.mode 配置
  2. 数据注入 (add_data 循环 + eob 列 + DatetimeIndex + subscribe)
  3. 基准注入 (BenchHolder + set_benchmark)
  4. start_time/end_time 计算

不封装的:
  - Strategy 的 on_bar 决策逻辑 (用户自己写, 保留完整事件驱动能力)
  - vector 矩阵旁路 (FacEngine 自有, 不重复)

用法 1 — 自定义 Strategy (完整事件驱动能力):
  >>> from core import Backtester
  >>> bt = Backtester()
  >>> bt.set_init_cash(1_000_000).set_benchmark('399317.SZ').add_datas(assets)
  >>> analyzer = bt.run(MyStrategy(), start_time, end_time, mode='full')

用法 2 — positions 快捷入口 (离线选品场景):
  >>> positions = {date: {code: weight}, ...}
  >>> analyzer = (Backtester()
  ...             .set_init_cash(1_000_000)
  ...             .set_benchmark('399317.SZ')
  ...             .add_datas(assets)
  ...             .run_positions(positions, start_time, end_time, mode='full'))

逃生舱: bt.engine 可直接拿原始 Engine 做高级操作。

[新增] 2026-06-30 Engine 二次封装, 简化调用不丢灵活性
"""

import pandas as pd
from typing import Dict, Optional, Any

from .engine import Engine
from .account import OrderSide, BenchHolder
from .storage import context
from .analyzer import AccountAnalyzer


# ============================================================
# PositionsStrategy — 目标持仓策略 (positions 快捷入口内部用)
# ============================================================

class PositionsStrategy:
    """目标持仓策略 — 读 positions dict, 调仓日 order_percent

    适合离线选品场景 (因子/组合), 不适合运行时决策 (止损/加仓)。
    运行时决策请写自定义 Strategy 传给 Backtester.run()。

    Args:
        positions: {date: {code: weight}}
                  date: pd.Timestamp / date / str (内部归一化为 .normalize())
                  有 key 的日期才调仓, 无 key 不操作
    """

    def __init__(self, positions: Dict):
        self._positions = self._normalize_positions(positions)

    @staticmethod
    def _normalize_positions(positions) -> Dict:
        """归一化 positions 的 key 为 Timestamp.normalize()"""
        normalized = {}
        for d, pos in positions.items():
            ts = pd.Timestamp(d).normalize()
            normalized[ts] = pos
        return normalized

    def on_bar(self, ctx, bars):
        if not bars:
            return
        current_date = pd.Timestamp(bars[0].get('eob')).normalize()
        target = self._positions.get(current_date)
        if target is None:
            return

        # 平仓: 不在目标内的品种全部卖出
        for code, hold in list(ctx.account.get_position().items()):
            if hold.get('volume', 0) <= 0:
                continue
            if code not in target:
                try:
                    ctx.account.order_percent(code, 1.0, OrderSide.Sell)
                except (ValueError, RuntimeError):
                    pass

        # 开仓/调整: 目标品种调到指定权重
        for code, weight in target.items():
            try:
                ctx.account.order_percent(code, weight, OrderSide.Buy)
            except (ValueError, RuntimeError):
                pass


# ============================================================
# Backtester — Engine 二次封装
# ============================================================

class Backtester:
    """简化回测入口 — 封装 Engine 使用模板, 不封装 Strategy 决策

    封装:
      - Engine 创建 (init_cash / fee_config)
      - context.mode = 'backtest'
      - 数据注入循环 (add_datas + eob 列 + DatetimeIndex + subscribe)
      - 基准注入 (BenchHolder + set_benchmark, full 模式自动)
      - 品种名称表注入

    不封装:
      - Strategy 的 on_bar (用户自定义, 保留完整事件驱动能力)
      - start_time/end_time (用户传入, 因场景而异)

    Attributes:
        engine: 原始 Engine 实例 (add_datas 后可用, run 前可配置, run 后可查询)
    """

    def __init__(self):
        """构造空 Backtester, 数据在 add_datas() 时传入"""
        self._assets: Dict[str, pd.DataFrame] = {}
        self._init_cash: float = 1e6
        self._fee_config: Optional[dict] = None
        self._bench_label: Optional[str] = None
        self._symbol_names: Dict[str, str] = {}
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

    def set_benchmark(self, bench_label: str) -> 'Backtester':
        """设置基准品种代码 (full 模式自动跑 BenchHolder 注入对比)"""
        self._bench_label = bench_label
        return self

    def set_freq(self, freq: str) -> 'Backtester':
        """设置数据频率 (默认 '1d', 支持 '1w' 等多频率)"""
        self._freq = freq
        return self

    def set_symbol_names(self, names: Dict[str, str]) -> 'Backtester':
        """设置品种名称表 (analyzer 报告查表显示品种名称)"""
        self._symbol_names = names
        return self

    # ── 数据注入 ──

    def add_datas(self, assets: Dict[str, pd.DataFrame]) -> 'Backtester':
        """创建 Engine 并批量注入 OHLCV 数据

        封装:
          - Engine(init_cash, fee_config) 创建
          - context.mode = 'backtest'
          - for code: DatetimeIndex + eob 列 + subscribe + add_data
          - symbol_name 透传 (显式 set_symbol_names > df['symbol_name'] > df['name'] > 空)

        assets 规范:
            {品种代码: OHLCV DataFrame}
            - key: 品种代码（如 '600000.SH', '399317.SZ'），与 Engine 和策略中的 code 一致
            - value: pd.DataFrame，需满足:
                * index: 日期（DatetimeIndex / 可转换的日期字符串），
                  内部自动转 pd.Timestamp
                * columns: 应含 OHLCV 字段（open/high/low/close/amount/volume 等）
                * eob 列（可选）：若不传，Backtester 自动设 eob = index
                  （这是便利封装，省掉手动加 eob 列的模板步骤）
            - 品种名称：通过 set_symbol_names() 显式指定（L156），
              或 DataFrame 含 'symbol_name'/'name' 列自动提取
              优先级：set_symbol_names > df['symbol_name'] > df['name'] > 空

        Args:
            assets: {品种代码: OHLCV DataFrame}

        Returns:
            Backtester (链式)

        Example:
            >>> df_600000 = pd.DataFrame({'open': [...], 'close': [...], ...}, index=dates)
            >>> df_399317 = pd.DataFrame({'open': [...], 'close': [...], ...}, index=dates)
            >>> assets = {'600000.SH': df_600000, '399317.SZ': df_399317}
            >>> from core import Backtester
            >>> bt = (Backtester()
            ...     .set_init_cash(1e6)
            ...     .set_benchmark('399317.SZ')
            ...     .set_symbol_names({'600000.SH': '浦发银行', '399317.SZ': '国证A指'})
            ...     .add_datas(assets)
            ...     .run(MyStrategy(), start, end))
        """
        self._assets = assets
        self.engine = Engine(init_cash=self._init_cash, fee_config=self._fee_config)
        context.mode = 'backtest'

        for code, df in assets.items():
            df = df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            if 'eob' not in df.columns:
                df['eob'] = df.index
            context.unsubscribe(code, self._freq)
            context.subscribe(code, self._freq, count=300)
            symbol_name = self._symbol_names.get(code)
            self.engine.add_data(code, self._freq, df, symbol_name=symbol_name)

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
            >>> bt = (Backtester()
            ...     .set_init_cash(1e6)
            ...     .set_benchmark('399317.SZ')
            ...     .add_datas(assets)
            ...     .run(MyStrategy(), start, end, mode='full'))
        """
        if self.engine is None:
            raise RuntimeError("未注入数据, 请先调 add_datas()")

        if mode == 'fast':
            return self.engine.run_fast(strategy, start_time, end_time)

        # full
        self.engine.run(strategy, start_time, end_time)
        analyzer = AccountAnalyzer(self.engine.account)

        # 基准注入
        if self._bench_label and self._bench_label in self._assets:
            analyzer = self._inject_benchmark(analyzer, start_time, end_time)

        return analyzer

    def run_positions(self, positions: Dict, start_time, end_time,
                      mode: str = 'full') -> AccountAnalyzer:
        """positions 快捷入口 — 内部构建 PositionsStrategy

        适合离线选品场景 (因子轮动/多风格组合)。
        运行时决策场景 (止损/加仓/条件单) 请写自定义 Strategy 传给 run()。

        Args:
            positions: {date: {code: weight}}, 有 key 的日期才调仓
            start_time: 回测起始时间
            end_time: 回测结束时间
            mode: 'full' / 'fast'

        Returns:
            AccountAnalyzer

        Example:
            >>> positions = {date: {'600000': 0.5, '600001': 0.5}, ...}
            >>> analyzer = (Backtester()
            ...             .set_init_cash(1e6)
            ...             .set_benchmark('399317.SZ')
            ...             .add_datas(assets)
            ...             .run_positions(positions, start, end, mode='full'))
        """
        strategy = PositionsStrategy(positions)
        return self.run(strategy, start_time, end_time, mode=mode)

    # ── 内部: 基准注入 (封装 BenchHolder 模板) ──

    def _inject_benchmark(self, analyzer: AccountAnalyzer,
                          start_time, end_time) -> AccountAnalyzer:
        """跑基准 Engine + BenchHolder, 注入对比"""
        bench_label = self._bench_label
        bench_eng = Engine(init_cash=self._init_cash, fee_config=self._fee_config)
        bench_df = self._assets[bench_label].copy()
        if not isinstance(bench_df.index, pd.DatetimeIndex):
            bench_df.index = pd.to_datetime(bench_df.index)
        if 'eob' not in bench_df.columns:
            bench_df['eob'] = bench_df.index
        context.unsubscribe(bench_label, self._freq)
        context.subscribe(bench_label, self._freq, count=3000)
        bench_symbol_name = self._symbol_names.get(bench_label)
        bench_eng.add_data(bench_label, self._freq, bench_df, symbol_name=bench_symbol_name)
        bench_eng.run(BenchHolder, start_time, end_time)
        bench_an = AccountAnalyzer(bench_eng.account)
        analyzer.set_benchmark(bench_an.daily_assets, bench_label)
        return analyzer
