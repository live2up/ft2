import datetime
from collections import defaultdict, deque
from typing import Dict, List,Union
import pandas as pd
import logging
logger = logging.getLogger(__name__)

class Context:
    """回测上下文 — 全局配置 + 当前时钟

    与 GM 架构的核心差异：
        GM 的 context 是重对象，持有 _cache、bar_data_set、accounts、callbacks 等全部状态。
        因为 GM 的引擎在 C SDK，Python 层需要 context 集中管理一切。
        
        本框架是纯 Python，Engine 实例管理运行时数据，context 只保留跨模块共享的：
        - _subscribed（订阅配置）: Engine.add_data() 和 account._get_price() 共用
        - mode / _current_time（模式+时钟）: context.now 供策略和 account 读取
        - _active_engine（活跃引擎引用）: context.data() 委托目标

    [重构] 2026-06-09 _cache 和 bar_data_set 移入 Engine 实例：
        - 缓存是每次回测独有的运行时状态，生命周期应与 Engine 一致
        - context.data() 委托给 context._active_engine 的缓存
        - 删除了 reset()：Engine 实例天然隔离，无需手动清理
    """
    def __init__(self):
        self._subscribed = {}
        self.mode = None
        self._current_time = None
        self._active_engine = None   # 当前活跃的 Engine 实例

    def is_backtest_model(self):
        return self.mode == 'backtest'

    @property 
    def now(self):
        return self._current_time if self.mode == 'backtest' else datetime.datetime.now()
    
    def subscribe(self, symbols: Union[str, List[str]], freq='1d', count=100, fields=None,format='df'):
        if isinstance(symbols, str):
            symbols = [symbols]
        # [修复] 2026-06-09 fields 字符串转列表，避免 engine.add_data() 中逐字符遍历
        if isinstance(fields, str):
            fields = [f.strip() for f in fields.split(',')]
        # [新增] 2026-06-09 自动补 eob：account._get_price() 内部依赖 eob 做时间比对
        #   用户只需指定业务字段（如 'close'），框架自动确保 eob 被缓存
        if fields is not None and 'eob' not in fields:
            fields.append('eob')
            
        for symbol in symbols:
            self._subscribed[(symbol, freq)] = {
                'fields': fields,
                'count': count,
                'format': format
            }

    def get_subscribe_params(self, symbol: str, freq: str) -> dict:
        return self._subscribed.get((symbol, freq))

    def unsubscribe(self, symbols: Union[str, List[str]], freq='1d'):
        if isinstance(symbols, str):
            symbols = [symbols]
            
        for symbol in symbols:
            if (symbol, freq) in self._subscribed:
                del self._subscribed[(symbol, freq)]
                # [重构] 2026-06-09 委托给活跃 Engine 清除缓存
                if self._active_engine:
                    self._active_engine._rm_cache(symbol, freq)

    @property  
    def symbols(self, freq=None):
        if freq:
            return {s[0] for s in self._subscribed if s[1] == freq}
        return {s[0] for s in self._subscribed}
    
    # [重构] 2026-06-09 data() 委托给活跃 Engine 的缓存
    #   context 不再持有 _cache，由 context._active_engine 提供
    def data(self, symbol: str, frequency: str, count: int = 1, fields: Union[str, List[str]] = None):
        if not frequency:
            frequency = "1d"
        if count < 1:
            count = 1
        if self._active_engine is None:
            raise RuntimeError("context.data() 调用时没有活跃的 Engine，请先调用 engine.run()")
        return self._active_engine._cache.get_data(symbol, frequency, count, fields)

    # [新增] 2026-06-09 account 委托给活跃 Engine
    #   策略统一通过 context.account 访问，回测走 SimAccount，实盘可替换为 RealBroker
    @property
    def account(self):
        if self._active_engine is None:
            raise RuntimeError("context.account 调用时没有活跃的 Engine")
        return self._active_engine.account


# ============================================================================
# 缓存数据结构（Engine 实例持有）
# ============================================================================

class _Cache:
    def __init__(self):
        self._col_cache = {}
        self._row_cache = {}
        self._initialized = set()
        self._data_loader = None

    def set_data_loader(self, loader):
        self._data_loader = loader
    def init_cache(self, symbol, freq, format, fields, count):
        key = (symbol, freq)
        if format == "col":
            self._col_cache[key] = _ColQuote(symbol, freq, format, fields, count)
        else:
            self._row_cache[key] = _RowQuote(symbol, freq, format, fields, count)

        if key in self._initialized:
            self._initialized.remove(key)

    def rm_cache(self, symbol, freq):
        key = (symbol, freq)
        if key in self._col_cache:
            del self._col_cache[key]
        if key in self._row_cache:
            del self._row_cache[key]
        if key in self._initialized:
            self._initialized.remove(key)

    def has_cache(self, symbol, freq):
        key = (symbol, freq)
        if key in self._col_cache:
            return True
        if key in self._row_cache:
            return True
        return False

    def add_data(self, symbol, freq, data: Dict):
        key = (symbol, freq)
        if key in self._col_cache:
            self._col_cache[key].add_data(data)
        if key in self._row_cache:
            self._row_cache[key].add_data(data)

    def get_data(self, symbol, freq, count, fields):
        key = (symbol, freq)
        if key in self._col_cache:
            q = self._col_cache[key]
        elif key in self._row_cache:
            q = self._row_cache[key]
        else:
            raise ValueError(f"请先订阅{symbol}的{freq}周期数据")

        if key not in self._initialized:
            miss_count = q.miss_count(count)
            if miss_count != 0 and self._data_loader:
                data = self._data_loader.load_history(
                    symbol=symbol,
                    frequency=freq,
                    count=miss_count+1,
                    end_time=q.earliest_time() or datetime.datetime.now()
                )
                if freq == "1d":
                    for item in data[::-1]:
                        if 'eob' in item:
                            item["eob"] = item["eob"].replace(hour=15, minute=15, second=1)
                            if context.now < item["eob"]:
                                continue
                        q.add_data(item, left=True)
                else:
                    for item in data[::-1]:
                        q.add_data(item, left=True)
            self._initialized.add(key)

        return q.get_data(fields, count)


class _RowQuote:
    def __init__(self, symbol, freq, format, fields, count):
        self._symbol = symbol
        self._freq = freq
        self._format = format
        self._fields = fields
        self._earliest_time = None
        self._data = deque(maxlen=count)

    def add_data(self, data: Dict, left=False):
        if left and self.full():
            return
        if left and self._earliest_time is not None:
            if (self._freq == "tick") and (data["created_at"] >= self._earliest_time):
                return
            if (self._freq != "tick") and (data["eob"] >= self._earliest_time):
                return
        if self._earliest_time is None:
            if self._freq == "tick":
                self._earliest_time = data["created_at"]
            else:
                self._earliest_time = data["eob"]
        newdata = {}
        for field in self._fields:
            newdata[field] = data.get(field)
        if left:
            self._data.appendleft(newdata)
            return
        self._data.append(newdata)

    def get_data(self, fields, count):
        data_len = len(self._data)
        start = data_len - count
        if start < 0:
            start = 0
        result = []
        for i in range(start, data_len):
            if not fields:
                result.append(self._data[i])
            else:
                result.append({k: v for k, v in self._data[i].items() if k in fields})
        if self._format == "df":
            return pd.DataFrame(result)
        return result

    def miss_count(self, count):
        if count <= len(self._data):
            return 0
        return self._data.maxlen - len(self._data)

    def earliest_time(self):
        return self._earliest_time

    def full(self):
        return len(self._data) == self._data.maxlen


class _ColQuote:
    def __init__(self, symbol, freq, format, fields, count):
        self._symbol = symbol
        self._freq = freq
        self._format = format
        self._fields = fields
        self._earliest_time = None
        self._data = {}
        for field in fields:
            if field == "symbol":
                continue
            self._data[field] = deque(maxlen=count)

    def add_data(self, data: Dict, left=False):
        if left and self.full():
            return
        if left and self._earliest_time is not None:
            if (self._freq == "tick") and (data["created_at"] >= self._earliest_time):
                return
            if (self._freq != "tick") and (data["eob"] >= self._earliest_time):
                return
        if self._earliest_time is None:
            if self._freq == "tick":
                self._earliest_time = data["created_at"]
            else:
                self._earliest_time = data["eob"]
        for field in self._fields:
            if field == "symbol":
                continue
            if field in ["bid_p", "bid_v", "ask_p", "ask_v"]:
                quotes = data.get("quotes")
                if quotes and len(quotes) != 0:
                    item = quotes[0].get(field)
                else:
                    item = None
            else:
                item = data.get(field)
            if left:
                self._data[field].appendleft(item)
                continue
            self._data[field].append(item)

    def get_data(self, fields, count):
        if not fields:
            fields = self._fields
        result = {}
        for field in self._fields:
            if field not in fields:
                continue
            if field == "symbol" and field in fields:
                result["symbol"] = self._symbol
                continue
            q = self._data[field]
            q_len = len(q)
            start = q_len - count
            if start < 0:
                start = 0
            l = []
            for i in range(start, q_len):
                l.append(q[i])
            result[field] = l
        return result

    def miss_count(self, count):
        for q in self._data.values():
            if count <= len(q):
                return 0
            return q.maxlen - len(q)

    def earliest_time(self):
        return self._earliest_time

    def full(self):
        for q in self._data.values():
            return len(q) == q.maxlen


context=Context()
