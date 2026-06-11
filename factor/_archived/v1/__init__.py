"""
ft2.factor.v1 - 因子挖掘模块 v1（已存档）

自 2026-05-25 起，v1 模块归档至本子目录。
新开发请使用 factor.v2，v1 仅保留用于向后兼容。

包含：
- base: 因子基类（ABC + 注册器 + 装饰器 + 元类）
- calculator: 因子计算引擎（批量计算 + 依赖图）
- validator: 因子检验器（IC/IR/衰减等）
- combiner: 因子组合器（多因子合成 + 正交化）
- manager: 因子管理器（版本控制 + 元数据 + 持久化）
"""

from .base import (
    Factor, FactorRegistry, FactorMeta,
    FactorMetadata, FactorCategory, FactorFrequency,
    factor as factor_decorator
)
from .calculator import FactorCalculator, DataSource, FactorDependencyGraph, create_sample_data
from .validator import FactorValidator, ValidationMetric, ValidationResult, validation_metric
from .combiner import (
    FactorCombiner, CombinationMethod, OrthogonalizationMethod,
    StandardizationMethod, CombinationResult
)
from .manager import (
    FactorManager, StorageFormat, FactorStatus,
    FactorVersion, FactorLibraryEntry
)

__all__ = [
    # 基础类
    'Factor',
    'FactorRegistry',
    'FactorMeta',
    'FactorMetadata',
    'FactorCategory',
    'FactorFrequency',
    'factor_decorator',

    # 计算引擎
    'FactorCalculator',
    'DataSource',
    'FactorDependencyGraph',
    'create_sample_data',

    # 检验器
    'FactorValidator',
    'ValidationMetric',
    'ValidationResult',
    'validation_metric',

    # 组合器
    'FactorCombiner',
    'CombinationMethod',
    'OrthogonalizationMethod',
    'StandardizationMethod',
    'CombinationResult',

    # 管理器
    'FactorManager',
    'StorageFormat',
    'FactorStatus',
    'FactorVersion',
    'FactorLibraryEntry',
]