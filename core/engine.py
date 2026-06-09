#这个是回测引擎，尽量精简导入的模块
from collections import OrderedDict
from .storage import context, _Cache
from .account import account
import pandas as pd

class Engine:
    """回测引擎 — 时间线驱动

    [重构] 2026-06-09 _cache 和 bar_data_set 移入 Engine 实例：
        - 每个 Engine 实例持有独立的缓存，天然隔离多轮/多引擎回测
        - context.data() 通过 context._active_engine 委托到当前活跃引擎的缓存
        - run() 入口注册活跃引擎，出口还原
    """
    def __init__(self):
        self.timeline = OrderedDict()
        self.cache_count=100
        self._cache = _Cache()
        self.bar_data_set = set()

    def set_cache_count(self,cache_count):
        self.cache_count=cache_count

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
            format_=None
        else:
            sample_bar = data[0]
            available_fields = set(sample_bar.keys())
            requested_fields = params['fields'] or list(available_fields)
            fields = [f for f in requested_fields if f in available_fields]
            count = params.get('count', self.cache_count)
            format_=params.get('format')

        if not self._has_cache(symbol, freq):
            self._init_cache(symbol, freq, format=format_, fields=fields, count=count)

        for bar in data:
            bar['symbol'] = symbol
            bar['frequency'] = freq
            eob = bar.get('eob')
            if eob is None:
                continue

            # [新增] 2026-06-03 规范化 eob 时间
            #   纯日期(00:00:00) → 15:00，日/周/月线通用
            #   已有具体时间的（如盘中多周期）保持不动
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
        # [修复] 2026-05-30 hasattr(类, 'on_bar') 对类也返回 True，
        #   导致类不会被实例化。改用 isinstance(strategy, type) 判断。
        # 支持策略实例或策略类
        if isinstance(strategy, type):
            strategy = strategy()
        
        # [新增] 2026-06-09 自动适配时间类型：str/date/datetime/Timestamp → pd.Timestamp
        #   统一内部时间类型，用户可直接传 '2023-01-01' 字符串
        #   纯日期结尾(00:00:00) → 推到 23:59:59，确保最后一天 bar 不被排除
        start_time = pd.Timestamp(start_time)
        end_time = pd.Timestamp(end_time)
        if end_time.hour == 0 and end_time.minute == 0 and end_time.second == 0:
            end_time = end_time.replace(hour=23, minute=59, second=59)
        
        # [重构] 2026-06-09 注册为活跃引擎，context.data() 委托到本实例的缓存
        prev_engine = context._active_engine
        context._active_engine = self

        try:
            _add_bar = self._add_bar2bar_data_cache
            _snapshot = account.take_snapshot

            # [重构] 2026-06-02 初始盘前快照：独立于循环，语义固定为 snapshots[0]
            #   时间锚定在回测区间开始前一天，避免大量预热数据导致基准日过远、年化被压低
            from datetime import timedelta
            account.init_snapshot(start_time - timedelta(days=1))
            
            for current_time, bars in sorted(self.timeline.items()):
                context._current_time = current_time
                
                for bar in bars:
                    _add_bar(bar)
                    
                if start_time <= current_time <= end_time:
                    strategy.on_bar(context, bars)
                    _snapshot()
        finally:
            # 还原上一个活跃引擎（支持嵌套，如 walk-forward 场景）
            context._active_engine = prev_engine

engine=Engine()
