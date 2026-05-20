"""
ft2.factor - 因子挖掘模块

包含：
- base: 因子基类
- calculator: 因子计算引擎
- validator: 因子检验器
- combiner: 因子组合器
- manager: 因子管理器

设计原则：
1. 松耦合设计：因子模块独立于回测框架
2. 统一接口：所有因子遵循相同的计算接口
3. 完整检验：提供全面的因子检验指标
4. 灵活组合：支持多种因子组合方法
5. 版本管理：完整的因子版本控制
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