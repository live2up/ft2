"""
因子基类模块（V2 自包含版本）

[重构] 2026-05-20 从 factor/base.py 移植，使 V2 模块完全独立不依赖 V1。
保留核心抽象：Factor/FactorRegistry/FactorFrequency/FactorMeta/@factor 装饰器。

设计思路：
---------
1. Factor基类：定义因子计算接口和基本属性
2. FactorRegistry：因子注册器，支持因子发现和实例化
3. 装饰器：@factor 用于标记因子计算函数
4. 元类：自动注册因子类

使用方式：
---------
>>> from factor.v2.base import Factor, FactorRegistry, FactorMetadata, FactorFrequency
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Callable, Type
from enum import Enum
import pandas as pd
import numpy as np
from functools import wraps


class FactorCategory(Enum):
    """因子分类枚举"""
    PRICE = "price"           # 价格类因子
    VOLUME = "volume"         # 成交量类因子
    VALUE = "value"           # 估值类因子
    GROWTH = "growth"         # 成长类因子
    PROFIT = "profit"         # 盈利类因子
    MOMENTUM = "momentum"     # 动量类因子
    REVERSAL = "reversal"     # 反转类因子
    LIQUIDITY = "liquidity"   # 流动性因子
    VOLATILITY = "volatility" # 波动率因子
    TECHNICAL = "technical"   # 技术指标因子
    CUSTOM = "custom"         # 自定义因子


class FactorFrequency(Enum):
    """因子计算频率枚举"""
    DAILY = "1d"              # 日频
    WEEKLY = "1w"             # 周频
    MONTHLY = "1m"            # 月频
    MINUTE_1 = "1m"           # 1分钟
    MINUTE_5 = "5m"           # 5分钟
    MINUTE_15 = "15m"         # 15分钟
    MINUTE_30 = "30m"         # 30分钟
    MINUTE_60 = "60m"         # 60分钟


@dataclass
class FactorMetadata:
    """因子元数据"""
    name: str                    # 因子名称
    description: str             # 因子描述
    category: FactorCategory     # 因子分类
    frequency: FactorFrequency   # 计算频率
    author: str = ""             # 作者
    version: str = "1.0.0"       # 版本号
    created_at: datetime = field(default_factory=datetime.now)  # 创建时间
    updated_at: datetime = field(default_factory=datetime.now)  # 更新时间
    parameters: Dict[str, Any] = field(default_factory=dict)    # 参数配置
    dependencies: List[str] = field(default_factory=list)       # 依赖因子


class Factor(ABC):
    """因子基类

    所有因子必须继承此类并实现calculate方法
    """

    def __init__(self, metadata: FactorMetadata):
        self.metadata = metadata
        self._cache = {}  # 缓存计算结果

    @abstractmethod
    def calculate(self, data: Dict[str, pd.DataFrame],
                  symbols: List[str],
                  dates: List[date]) -> pd.DataFrame:
        """
        计算因子值

        Args:
            data: 数据字典，key为数据字段名，value为DataFrame（index为日期，columns为标的）
            symbols: 标的列表
            dates: 日期列表

        Returns:
            pd.DataFrame: 因子值DataFrame（index为日期，columns为标的）
        """
        pass

    def validate_input(self, data: Dict[str, pd.DataFrame],
                      symbols: List[str],
                      dates: List[date]) -> bool:
        """
        验证输入数据

        Args:
            data: 输入数据
            symbols: 标的列表
            dates: 日期列表

        Returns:
            bool: 输入是否有效
        """
        if not data:
            return False
        if not symbols:
            return False
        if dates is None or len(dates) == 0:
            return False

        # 检查数据完整性
        for field, df in data.items():
            if df.shape[0] != len(dates) or df.shape[1] != len(symbols):
                return False

        return True

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()

    def get_cache_key(self, symbols: List[str], dates: List[date]) -> str:
        """
        生成缓存键

        Args:
            symbols: 标的列表
            dates: 日期列表

        Returns:
            str: 缓存键
        """
        if hasattr(dates[0], 'strftime'):
            start_date = dates[0].strftime('%Y%m%d')
            end_date = dates[-1].strftime('%Y%m%d')
        else:
            start_date = str(dates[0])
            end_date = str(dates[-1])

        return f"{self.metadata.name}_{'_'.join(symbols[:3])}_{start_date}_{end_date}"

    def calculate_with_cache(self, data: Dict[str, pd.DataFrame],
                           symbols: List[str],
                           dates: List[date]) -> pd.DataFrame:
        """
        带缓存的因子计算

        Args:
            data: 输入数据
            symbols: 标的列表
            dates: 日期列表

        Returns:
            pd.DataFrame: 因子值
        """
        cache_key = self.get_cache_key(symbols, dates)

        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.validate_input(data, symbols, dates):
            raise ValueError("输入数据验证失败")

        result = self.calculate(data, symbols, dates)
        self._cache[cache_key] = result

        return result

    def __str__(self) -> str:
        """字符串表示"""
        return f"Factor(name={self.metadata.name}, category={self.metadata.category.value})"

    def __repr__(self) -> str:
        """详细表示"""
        return (f"Factor(name={self.metadata.name}, "
                f"description={self.metadata.description}, "
                f"category={self.metadata.category.value}, "
                f"frequency={self.metadata.frequency.value})")


class FactorRegistry:
    """因子注册器

    管理所有注册的因子类，支持因子发现和实例化
    """

    _factors: Dict[str, Type[Factor]] = {}

    @classmethod
    def register(cls, factor_class: Type[Factor]) -> Type[Factor]:
        """
        注册因子类

        Args:
            factor_class: 因子类

        Returns:
            Type[Factor]: 注册的因子类
        """
        try:
            temp_metadata = FactorMetadata(
                name=factor_class.__name__,
                description="",
                category=FactorCategory.CUSTOM,
                frequency=FactorFrequency.DAILY
            )
            factor_instance = factor_class(temp_metadata)
            factor_name = factor_instance.metadata.name
        except:
            factor_name = factor_class.__name__

        cls._factors[factor_name] = factor_class
        return factor_class

    @classmethod
    def get_factor(cls, name: str) -> Optional[Type[Factor]]:
        """
        获取因子类

        Args:
            name: 因子名称

        Returns:
            Optional[Type[Factor]]: 因子类，如果不存在返回None
        """
        return cls._factors.get(name)

    @classmethod
    def list_factors(cls) -> List[str]:
        """
        列出所有注册的因子

        Returns:
            List[str]: 因子名称列表
        """
        return list(cls._factors.keys())

    @classmethod
    def create_factor(cls, name: str, **kwargs) -> Optional[Factor]:
        """
        创建因子实例

        Args:
            name: 因子名称
            **kwargs: 传递给因子构造函数的参数

        Returns:
            Optional[Factor]: 因子实例，如果不存在返回None
        """
        factor_class = cls.get_factor(name)
        if factor_class is None:
            return None

        return factor_class(**kwargs)

    @classmethod
    def discover(cls) -> int:
        """自动发现并注册所有 Factor 子类

        使用 __subclasses__() 递归遍历所有已导入的 Factor 子类，
        避免因遗漏手工注册而导致因子不可用。

        注意：需要先 import 因子定义所在的模块，discover() 才能发现。

        Returns:
            int: 新发现的因子类数量
        """
        def _get_all_subclasses(base_cls):
            """递归获取所有子类"""
            subclasses = set()
            for sub in base_cls.__subclasses__():
                subclasses.add(sub)
                subclasses.update(_get_all_subclasses(sub))
            return subclasses

        all_subclasses = _get_all_subclasses(Factor)
        count = 0
        for sub_cls in all_subclasses:
            if sub_cls.__name__ == 'Factor' or sub_cls.__name__ in cls._factors:
                continue
            try:
                cls.register(sub_cls)
                count += 1
            except Exception:
                pass
        return count

    @classmethod
    def clear(cls):
        """清空注册表"""
        cls._factors.clear()


def factor(name: str = None,
          description: str = "",
          category: FactorCategory = FactorCategory.CUSTOM,
          frequency: FactorFrequency = FactorFrequency.DAILY,
          author: str = "",
          version: str = "1.0.0"):
    """
    因子装饰器，用于标记因子计算函数

    Args:
        name: 因子名称，默认使用函数名
        description: 因子描述
        category: 因子分类
        frequency: 计算频率
        author: 作者
        version: 版本号

    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._is_factor = True
        wrapper._factor_name = name or func.__name__
        wrapper._factor_description = description
        wrapper._factor_category = category
        wrapper._factor_frequency = frequency
        wrapper._factor_author = author
        wrapper._factor_version = version

        return wrapper
    return decorator


class FactorMeta(type(ABC)):
    """因子元类，自动注册因子类"""

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)

        if name != 'Factor' and 'Factor' in [base.__name__ for base in bases]:
            FactorRegistry.register(cls)

        return cls
