"""
factor/v3/base.py — 因子基类 + 因子库
=============================================================================

从 v2/base.py 移植基础组件，新增 FactorLibrary 自增长因子库。

模块结构：
  Part 1: 基础枚举与元数据 (FactorCategory / FactorFrequency / FactorMetadata)
  Part 2: 因子基类 (Factor / FactorMeta / FactorRegistry / @factor)
  Part 3: 因子库 (FactorLibrary / LibraryEntry)  # [新增] v3

[重构] 2026-06-01 从 v2/base.py 移植，新增 FactorLibrary
=============================================================================
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Callable, Type
from enum import Enum
from functools import wraps
import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: 枚举与元数据
# ═══════════════════════════════════════════════════════════════════════════

class FactorCategory(Enum):
    """因子分类枚举"""
    PRICE = "price"           # 价格类
    VOLUME = "volume"         # 成交量类
    VALUE = "value"           # 估值类
    GROWTH = "growth"         # 成长类
    PROFIT = "profit"         # 盈利类
    MOMENTUM = "momentum"     # 动量类
    REVERSAL = "reversal"     # 反转类
    LIQUIDITY = "liquidity"   # 流动性
    VOLATILITY = "volatility" # 波动率
    TECHNICAL = "technical"   # 技术指标
    CUSTOM = "custom"         # 自定义


class FactorFrequency(Enum):
    """因子计算频率"""
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1m"
    MINUTE_1 = "1mT"
    MINUTE_5 = "5mT"
    MINUTE_15 = "15mT"
    MINUTE_30 = "30mT"
    MINUTE_60 = "60mT"


@dataclass
class FactorMetadata:
    """因子元数据"""
    name: str
    description: str
    category: FactorCategory
    frequency: FactorFrequency
    author: str = ""
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: 因子基类与注册器
# ═══════════════════════════════════════════════════════════════════════════

class Factor(ABC):
    """因子基类"""

    def __init__(self, metadata: FactorMetadata):
        self.metadata = metadata
        self._cache = {}

    @abstractmethod
    def calculate(self, data: Dict[str, pd.DataFrame],
                  symbols: List[str], dates: List[date]) -> pd.DataFrame:
        pass

    def validate_input(self, data, symbols, dates) -> bool:
        if not data or not symbols or dates is None or len(dates) == 0:
            return False
        for field, df in data.items():
            if df.shape[0] != len(dates) or df.shape[1] != len(symbols):
                return False
        return True

    def clear_cache(self):
        self._cache.clear()

    def __str__(self) -> str:
        return f"Factor(name={self.metadata.name}, category={self.metadata.category.value})"

    def __repr__(self) -> str:
        return (f"Factor(name={self.metadata.name}, "
                f"description={self.metadata.description}, "
                f"category={self.metadata.category.value})")


class FactorRegistry:
    """因子注册器"""
    _factors: Dict[str, Type[Factor]] = {}

    @classmethod
    def register(cls, factor_class: Type[Factor]) -> Type[Factor]:
        # [修复] 2026-06-01 用 except Exception 替代裸 except
        # 裸 except 会捕获 KeyboardInterrupt/SystemExit
        try:
            temp_metadata = FactorMetadata(
                name=factor_class.__name__, description="",
                category=FactorCategory.CUSTOM, frequency=FactorFrequency.DAILY)
            factor_instance = factor_class(temp_metadata)
            factor_name = factor_instance.metadata.name
        except Exception:
            factor_name = factor_class.__name__
        cls._factors[factor_name] = factor_class
        return factor_class

    @classmethod
    def get_factor(cls, name: str) -> Optional[Type[Factor]]:
        return cls._factors.get(name)

    @classmethod
    def list_factors(cls) -> List[str]:
        return list(cls._factors.keys())

    @classmethod
    def clear(cls):
        cls._factors.clear()


def factor(name: str = None, description: str = "",
           category: FactorCategory = FactorCategory.CUSTOM,
           frequency: FactorFrequency = FactorFrequency.DAILY):
    """因子装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper._is_factor = True
        wrapper._factor_name = name or func.__name__
        wrapper._factor_description = description
        wrapper._factor_category = category
        wrapper._factor_frequency = frequency
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: 因子库 FactorLibrary — [新增] v3
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LibraryEntry:
    """因子库条目

    记录一个已验证因子的完整信息，支持按多种维度查询。

    Attributes:
        alpha_id: 因子唯一标识 (如 'gtja_001', 'gp_round1_003')
        expression: 表达式字符串
        category: 因子分类
        fitness: 适应度值 (发现时的最优值)
        ic_mean: IC 均值 (选填)
        sharpe: Pipeline 回测 Sharpe (选填)
        discovered_at: 发现轮次 (0=种子公式, 1+=GP 轮次)
        source: 来源 ('formula' | 'gp' | 'manual')
    """
    alpha_id: str
    expression: str
    category: FactorCategory = FactorCategory.CUSTOM
    fitness: float = 0.0
    ic_mean: float = 0.0
    sharpe: float = 0.0
    discovered_at: int = 0
    source: str = 'manual'

    def __repr__(self) -> str:
        return (f"LibraryEntry({self.alpha_id}, fitness={self.fitness:.3f}, "
                f"source={self.source})")


class FactorLibrary:
    """自增长因子库

    记录所有已验证的因子 (种子公式 + GP 发现)，
    支持入库、查询、导出种子表达式供下一轮 GP 使用。

    典型用法：
    >>> lib = FactorLibrary()
    >>> lib.register(LibraryEntry('alpha001', formula, category=FactorCategory.MOMENTUM))
    >>> seeds = lib.seed_expressions(50)  # 取 Top 50 作为下一轮种子

    [新增] 2026-06-01 v3 核心组件
    """

    def __init__(self):
        self._entries: Dict[str, LibraryEntry] = {}

    def register(self, entry: LibraryEntry) -> bool:
        """入库一个因子 (已存在的跳过)

        Returns:
            bool: True=新增, False=已存在 (未覆盖)
        """
        if entry.alpha_id in self._entries:
            return False
        self._entries[entry.alpha_id] = entry
        return True

    def register_batch(self, entries: List[LibraryEntry]) -> int:
        """批量入库，返回新增数量"""
        count = 0
        for e in entries:
            if self.register(e):
                count += 1
        return count

    def seed_expressions(self, n: int = 100,
                         sort_by: str = 'fitness') -> List[str]:
        """获取 Top N 因子表达式作为 GP 种子

        Args:
            n: 返回的表达式数量
            sort_by: 排序字段 ('fitness' / 'ic_mean' / 'sharpe')

        Returns:
            List[str]: 表达式字符串列表
        """
        entries = sorted(
            self._entries.values(),
            key=lambda e: getattr(e, sort_by, 0),
            reverse=True,
        )
        return [e.expression for e in entries[:n]]

    def top(self, n: int = 20, sort_by: str = 'fitness') -> List[LibraryEntry]:
        """查询 Top N 因子"""
        return sorted(
            self._entries.values(),
            key=lambda e: getattr(e, sort_by, 0),
            reverse=True,
        )[:n]

    def by_category(self, category: FactorCategory) -> List[LibraryEntry]:
        """按分类查询"""
        return [e for e in self._entries.values() if e.category == category]

    def by_source(self, source: str) -> List[LibraryEntry]:
        """按来源查询 ('formula' | 'gp' | 'manual')"""
        return [e for e in self._entries.values() if e.source == source]

    def by_round(self, round_num: int) -> List[LibraryEntry]:
        """按发现轮次查询"""
        return [e for e in self._entries.values() if e.discovered_at == round_num]

    def size(self) -> int:
        """当前库存因子数"""
        return len(self._entries)

    def all_expressions(self) -> List[str]:
        """获取所有表达式"""
        return [e.expression for e in self._entries.values()]

    def to_dataframe(self) -> pd.DataFrame:
        """导出为 DataFrame"""
        rows = []
        for e in self._entries.values():
            rows.append({
                'alpha_id': e.alpha_id,
                'expression': e.expression[:80],
                'category': e.category.value,
                'fitness': round(e.fitness, 3),
                'ic_mean': round(e.ic_mean, 4),
                'sharpe': round(e.sharpe, 2),
                'round': e.discovered_at,
                'source': e.source,
            })
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        n_formula = len(self.by_source('formula'))
        n_gp = len(self.by_source('gp'))
        n_manual = len(self.by_source('manual'))
        return (f"FactorLibrary(total={self.size()}, "
                f"formula={n_formula}, gp={n_gp}, manual={n_manual})")
