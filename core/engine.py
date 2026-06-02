#这个是回测引擎，尽量精简导入的模块
from collections import OrderedDict
from .storage import context
from .account import account
import pandas as pd

class Engine:
    def __init__(self):
        self.timeline = OrderedDict()
        self.cache_count=100

    def set_cache_count(self,cache_count):
        self.cache_count=cache_count

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

        if not context._has_cache(symbol, freq):
            context._init_cache(symbol, freq, format=format_, fields=fields, count=count)

        for bar in data:
            bar['symbol'] = symbol
            bar['frequency'] = freq
            eob = bar.get('eob')
            if eob is None:
                continue
                
            if eob in self.timeline:
                for b in self.timeline[eob]:
                    if b['symbol'] == symbol and b['frequency'] == freq:
                        b.update(bar)
                        break
                else:
                    self.timeline[eob].append(bar)
            else:
                self.timeline[eob] = [bar]

    def run(self, strategy, start_time, end_time):
        # [修复] 2026-05-30 hasattr(类, 'on_bar') 对类也返回 True，
        #   导致类不会被实例化。改用 isinstance(strategy, type) 判断。
        # 支持策略实例或策略类
        if isinstance(strategy, type):
            strategy = strategy()
        
        _add_bar = context._add_bar2bar_data_cache
        _snapshot = account.take_snapshot

        # [重构] 2026-06-02 初始盘前快照：独立于循环，语义固定为 snapshots[0]
        #   时间锚定在首根 bar 前一天，确保基准日期始终在交易日之前
        sorted_times = sorted(self.timeline.keys())
        if sorted_times:
            from datetime import timedelta
            account.init_snapshot(sorted_times[0] - timedelta(days=1))
        
        for current_time, bars in sorted(self.timeline.items()):
            context._current_time = current_time
            
            for bar in bars:
                _add_bar(bar)
                
            if start_time <= current_time <= end_time:
                strategy.on_bar(context, bars)
                _snapshot()

engine=Engine()
