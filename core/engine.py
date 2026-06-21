#这个是回测引擎，尽量精简导入的模块
from collections import OrderedDict
from .storage import context, _Cache
from .account import AccountManager, FastAccount
import pandas as pd

class Engine:
    """回测引擎 — 时间线驱动

    与 GM 架构的核心差异：
        GM 的引擎是 C SDK（.pyd/.dll），Python 层只做事件分发。GM 的 cache 和 account
        放全局 context，因为每进程只有一个数据源（C SDK 推送），不存在多引擎场景。
        
        本框架是纯 Python 引擎，同一进程可运行多个 Engine 实例（基准+策略）。
        因此 _cache、bar_data_set、account 归 Engine 实例所有，天然隔离。

    [重构] 2026-06-09 _cache 和 bar_data_set 移入 Engine 实例
    [重构] 2026-06-09 account 移入 Engine 实例：
        - 每实例独立账户，无需 account.reset()
        - 策略通过 context.account 访问（委托到活跃 Engine）
        - 实盘时替换为 RealBroker，接口不变
    """
    def __init__(self, init_cash=1e6, fee_config=None):
        self.init_cash = init_cash
        self.timeline = OrderedDict()
        self.cache_count = 100
        self._cache = _Cache()
        self.bar_data_set = set()
        self.account = AccountManager(init_cash=init_cash, fee_config=fee_config)
        self.fast_account = None  # 每次 run_fast() 时重建

    def set_cache_count(self, cache_count):
        self.cache_count = cache_count

    # [重构] 2026-06-09 缓存管理方法移入 Engine（原 context 委托方法）
    def _init_cache(self, symbol, freq, format, fields, count):
        self._cache.init_cache(symbol, freq, format, fields, count)

    def _has_cache(self, symbol, freq):
        return self._cache.has_cache(symbol, freq)

    def _rm_cache(self, symbol, freq):
        self._cache.rm_cache(symbol, freq)

    def _add_data_to_cache(self, symbol, freq, data):
        if data is None:
            return
        self._cache.add_data(symbol, freq, data)

    def _add_bar2bar_data_cache(self, bar):
        """逐 bar 加入缓存（去重）"""
        kk = (bar["symbol"], bar["frequency"], bar["eob"])
        if kk in self.bar_data_set:
            from .storage import logger
            logger.debug("bar data %s 已存在, 跳过不加入", kk)
        else:
            self._add_data_to_cache(bar["symbol"], bar["frequency"], bar)
            self.bar_data_set.add(kk)

    def add_data(self, symbol, freq, data):
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')

        params = context.get_subscribe_params(symbol, freq)
        if params is None:
            sample_bar = data[0]
            fields = list(sample_bar.keys())
            count = self.cache_count
            format_ = None
        else:
            sample_bar = data[0]
            available_fields = set(sample_bar.keys())
            requested_fields = params['fields'] or list(available_fields)
            fields = [f for f in requested_fields if f in available_fields]
            count = params.get('count', self.cache_count)
            format_ = params.get('format')

        if not self._has_cache(symbol, freq):
            self._init_cache(symbol, freq, format=format_, fields=fields, count=count)

        for bar in data:
            bar['symbol'] = symbol
            bar['frequency'] = freq
            eob = bar.get('eob')
            if eob is None:
                continue

            eob = self._normalize_eob(eob)
            bar['eob'] = eob

            if eob in self.timeline:
                for b in self.timeline[eob]:
                    if b['symbol'] == symbol and b['frequency'] == freq:
                        b.update(bar)
                        break
                else:
                    self.timeline[eob].append(bar)
            else:
                self.timeline[eob] = [bar]

    @staticmethod
    def _normalize_eob(eob):
        """规范化 bar 结束时间：纯日期(00:00:00) → 15:00，已有时间的不动"""
        ts = pd.Timestamp(eob)
        if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
            return ts.replace(hour=15, minute=0, second=0)
        return ts

    def run(self, strategy, start_time, end_time):
        if isinstance(strategy, type):
            strategy = strategy()
        self._drive_timeline(strategy, start_time, end_time, snapshot=True)

    def run_fast(self, strategy, start_time, end_time):
        """快速回测：替换 self.account 为 FastAccount，策略统一用 ctx.account 下单。

        跑完时间线后从 FastAccount.daily_assets 构造 AccountAnalyzer，
        跳过快照聚合和 FIFO 交易匹配，约 6x 快于 full 模式。

        策略契约：
        - ctx.account 在 fast 模式下指向 FastAccount (接口兼容 AccountManager)
        - 策略调用 ctx.account.order_percent/order_volume 下单 (自动扣费+记净值)
        - 非交易日: ctx.account.mark() 记录净值

        Returns:
            AccountAnalyzer: 统一分析器 (仅资产指标可用，交易指标返回 None)
        """
        from .analyzer import AccountAnalyzer
        if isinstance(strategy, type):
            strategy = strategy()

        # 替换 account 为 FastAccount，策略 ctx.account 透明切换
        original_account = self.account
        self.fast_account = FastAccount(self.init_cash, original_account.fee_config)
        self.account = self.fast_account

        try:
            self._drive_timeline(strategy, start_time, end_time, snapshot=False)
            daily = self.fast_account.daily_assets
            if not daily:
                raise RuntimeError(
                    "fast 模式要求策略在 on_bar 中调用 ctx.account.mark() 记录净值，"
                    "或通过 ctx.account.order_percent/order_volume 自动记录。"
                )
            return AccountAnalyzer(daily_assets=daily)
        finally:
            self.account = original_account

    # ── 内部时间线驱动 (run/run_fast 共用) ──
    def _drive_timeline(self, strategy, start_time, end_time, snapshot=True):
        """时间线驱动循环：逐 bar → on_bar → (可选)take_snapshot

        start/end 取传入值与时间线边界的交集：传入超出范围自动 clamp。
        init_snapshot 锚定时间线上 start_time 之前的最后一根 bar（而非日历-1天）。
        """
        sorted_times = sorted(self.timeline.keys())
        if not sorted_times:
            return

        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        if end_time.hour == 0 and end_time.minute == 0 and end_time.second == 0:
            end_time = end_time.replace(hour=23, minute=59, second=59)

        # clamp 到时间线范围
        t_min, t_max = sorted_times[0], sorted_times[-1]
        if start_time < t_min:
            start_time = t_min
        if end_time > t_max:
            end_time = t_max
        if start_time > end_time:
            return

        prev_engine = context._active_engine
        context._active_engine = self

        try:
            _add_bar = self._add_bar2bar_data_cache
            _snapshot = self.account.take_snapshot if snapshot else None

            if snapshot:
                # init_snapshot = 时间线上 start 之前的最后一根 bar
                # 若无更早 bar（start_time 即首根），fallback 到日历前一日
                prev = [t for t in sorted_times if t < start_time]
                if prev:
                    init_date = prev[-1]
                else:
                    init_date = start_time - pd.Timedelta(days=1)
                self.account.init_snapshot(init_date)

            for current_time, bars in sorted(self.timeline.items()):
                context._current_time = current_time

                for bar in bars:
                    _add_bar(bar)

                if start_time <= current_time <= end_time:
                    strategy.on_bar(context, bars)
                    if _snapshot:
                        _snapshot()
        finally:
            context._active_engine = prev_engine


# [重构] 2026-06-09 移除全局 engine 实例
#   旧架构：全局 engine + 全局 account，需手动 reset()
#   新架构：每次回测创建 Engine(init_cash=...) 新实例，内置独立账户
