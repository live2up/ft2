"""
factor/v4/base.py — 因子基类与因子库 (v4 独立版)

[重构] 2026-06-22 从 v3/base.py 复制为 v4 独立版，与 v3 解耦。

内容:
  FactorCategory      — 因子分类枚举 (11种)
  FactorFrequency     — 因子频率枚举 (8种)
  FactorMetadata      — 因子元数据 dataclass
  Factor              — 因子抽象基类 (装饰器可注册)
  LibraryEntry        — 因子库条目 dataclass
  FactorLibrary       — 自增长因子库 (JSON 持久化)
"""
import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional


class FactorCategory(Enum):
    PRICE = auto()
    VOLUME = auto()
    VALUE = auto()
    GROWTH = auto()
    PROFIT = auto()
    MOMENTUM = auto()
    REVERSAL = auto()
    LIQUIDITY = auto()
    VOLATILITY = auto()
    TECHNICAL = auto()
    CUSTOM = auto()


class FactorFrequency(Enum):
    DAILY = 'D'
    WEEKLY = 'W'
    MONTHLY = 'M'
    QUARTERLY = 'Q'
    YEARLY = 'Y'
    MINUTE_1 = '1min'
    MINUTE_5 = '5min'
    MINUTE_60 = '60min'


@dataclass
class FactorMetadata:
    """因子元数据"""
    category: FactorCategory = FactorCategory.CUSTOM
    frequency: FactorFrequency = FactorFrequency.DAILY
    description: str = ''
    author: str = ''
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    source: str = ''


_FACTOR_REGISTRY: Dict[str, type] = {}


class Factor(ABC):
    """因子抽象基类"""

    @abstractmethod
    def calculate(self, data: Dict) -> 'pd.Series':
        """计算因子值"""
        pass


class FactorRegistry:
    """因子注册器"""

    @staticmethod
    def register(cls):
        name = cls.__name__
        _FACTOR_REGISTRY[name] = cls
        return cls

    @staticmethod
    def get(name: str) -> Optional[type]:
        return _FACTOR_REGISTRY.get(name)

    @staticmethod
    def list() -> List[str]:
        return list(_FACTOR_REGISTRY.keys())

    @staticmethod
    def clear():
        _FACTOR_REGISTRY.clear()


def factor(cls):
    """因子注册装饰器"""
    return FactorRegistry.register(cls)


@dataclass
class LibraryEntry:
    """因子库条目"""
    alpha_id: str
    expression: str
    fitness: float = 0.0
    ic_mean: float = 0.0
    icir: float = 0.0
    sharpe: float = 0.0
    turnover: float = 0.0
    category: str = 'CUSTOM'
    source: str = 'manual'
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict] = None


class FactorLibrary:
    """自增长因子库 — 支持注册、查询、TopN、JSON 持久化"""

    def __init__(self):
        self._entries: Dict[str, LibraryEntry] = {}

    def register(self, entry: LibraryEntry) -> bool:
        if entry.alpha_id in self._entries:
            warnings.warn(f"因子 {entry.alpha_id} 已存在，跳过")
            return False
        self._entries[entry.alpha_id] = entry
        return True

    def register_batch(self, entries: List[LibraryEntry]) -> int:
        count = 0
        for e in entries:
            if self.register(e):
                count += 1
        return count

    def get(self, alpha_id: str) -> Optional[LibraryEntry]:
        return self._entries.get(alpha_id)

    def remove(self, alpha_id: str) -> bool:
        if alpha_id in self._entries:
            del self._entries[alpha_id]
            return True
        return False

    def size(self) -> int:
        return len(self._entries)

    def top(self, n: int = 10, sort_by: str = 'sharpe') -> List[LibraryEntry]:
        return sorted(self._entries.values(),
                      key=lambda e: getattr(e, sort_by, 0) or 0,
                      reverse=True)[:n]

    def by_time(self, start: str, end: str) -> List[LibraryEntry]:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        return [e for e in self._entries.values()
                if start_dt <= datetime.fromisoformat(e.created_at) <= end_dt]

    def by_source(self, source: str) -> List[LibraryEntry]:
        return [e for e in self._entries.values() if e.source == source]

    def by_category(self, category: str) -> List[LibraryEntry]:
        return [e for e in self._entries.values() if e.category == category]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            [asdict(e) for e in self._entries.values()],
            indent=indent,
            ensure_ascii=False,
        )

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> 'FactorLibrary':
        lib = cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            lib.register(LibraryEntry(**item))
        return lib
