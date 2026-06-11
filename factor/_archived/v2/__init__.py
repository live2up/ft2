"""
factor.v2 - 因子挖掘 V2 模块（完全自包含，零 V1 依赖）

V2 增量重构，地基先行、网格优先：
- base: Factor/FactorRegistry/FactorFrequency 等基类（V2 自有）
- scheduler: 调仓日生成抽象层
- allocator: 权重分配器抽象层
- pipeline: 因子→回测桥接层
- validator: 因子检验器（IC/IR/衰减/命中率）
- cache_store: 因子值持久化缓存
- grid_search: 因子参数网格搜索（Phase A 主力工具）
- bo_search: 贝叶斯优化（网格之后的可选增强）
- gp_primitives: GP 符号回归原语集（Phase C）
- gp_evaluator: GP 多频率适应度评估器
- expression: 因子表达式引擎（字符串→AST→面板因子值）
- gp_miner: 因子GP矿机（符号回归自动发现因子）

设计原则：
1. factor/v2/ 为完全自包含的独立包，无任何对 factor/（V1）的 import 依赖
2. 网格搜索保留为基线工具，新方法可插拔
3. 遵循 AGENTS.md 注释规范
"""

from .base import (
    Factor,
    FactorRegistry,
    FactorMeta,
    FactorMetadata,
    FactorCategory,
    FactorFrequency,
    factor as factor_decorator,
)
from .scheduler import (
    RebalanceScheduler,
    FixedScheduler,
    IntervalScheduler,
    recommend_scheduler_from_decay,
)
from .allocator import (
    WeightAllocator,
    TopNEqualWeight,
    ScoreProportional,
    RiskParity,
)
from .pipeline import (
    FactorPipeline,
    BacktestResult,
    BacktestSchedule,
)
from .combiner import (
    FactorCombiner,
    EqualWeightCombiner,
    FixedWeightCombiner,
    ExpandingICCombiner,
    cross_section_zscore,
)
from .cache_store import (
    FactorCacheStore,
)
from .grid_search import (
    FactorGridSearch,
    FactorGridConfig,
    GridSearchResult,
)
from .bo_search import (
    FactorBOSearch,
    BOSearchResult,
)
from .gp_primitives import (
    ts_rank,
    ts_zscore,
    delay,
    correlation,
    decay_linear,
    cs_rank,
    signed_power,
    EXTENDED_PRIMITIVES,
    get_primitive,
    list_primitives,
)
from .gp_evaluator import (
    GPMultiFreqEvaluator,
    GPEvaluationResult,
)
from .validator import (
    FactorValidator,
    ValidationMetric,
    ValidationResult,
    validation_metric,
)
from .expression import (
    FactorExpression,
    ASTNode,
    NodeType,
    parse_expression,
)
from .gp_miner import (
    FactorGPMiner,
    Individual,
    DEFAULT_GP_CONFIG,
)
from .alpha101 import (
    ALPHA101,
    ALPHA101_CATEGORIES,
)
from .alpha191 import (
    ALPHA191,
    ALPHA191_CATEGORIES,
)
from .expression_factor import (
    ExpressionFactor,
    expression_factor,
    AlphaExplorer,
    AlphaResult,
)

__all__ = [
    # 基类（V2 自包含）
    'Factor',
    'FactorRegistry',
    'FactorMeta',
    'FactorMetadata',
    'FactorCategory',
    'FactorFrequency',
    'factor_decorator',
    # 调度器
    'RebalanceScheduler',
    'FixedScheduler',
    'IntervalScheduler',
    'recommend_scheduler_from_decay',
    # 分配器
    'WeightAllocator',
    'TopNEqualWeight',
    'ScoreProportional',
    'RiskParity',
    # 管道
    'FactorPipeline',
    'BacktestResult',
    'BacktestSchedule',
    # 组合器
    'FactorCombiner',
    'EqualWeightCombiner',
    'FixedWeightCombiner',
    'ExpandingICCombiner',
    'cross_section_zscore',
    # 缓存
    'FactorCacheStore',
    # 网格搜索
    'FactorGridSearch',
    'FactorGridConfig',
    'GridSearchResult',
    # BO 搜索
    'FactorBOSearch',
    'BOSearchResult',
    # GP 原语
    'ts_rank',
    'ts_zscore',
    'delay',
    'correlation',
    'decay_linear',
    'cs_rank',
    'signed_power',
    'EXTENDED_PRIMITIVES',
    'get_primitive',
    'list_primitives',
    # GP 评估器
    'GPMultiFreqEvaluator',
    'GPEvaluationResult',
    # 检验器（V2 自包含）
    'FactorValidator',
    'ValidationMetric',
    'ValidationResult',
    'validation_metric',
    # 因子表达式引擎
    'FactorExpression',
    'ASTNode',
    'NodeType',
    'parse_expression',
    # 因子 GP 矿机
    'FactorGPMiner',
    'Individual',
    'DEFAULT_GP_CONFIG',
    # WQ 101 因子公式
    'ALPHA101',
    'ALPHA101_CATEGORIES',
    # 国泰 191 因子公式
    'ALPHA191',
    'ALPHA191_CATEGORIES',
    # 表达式因子适配器 + 批量探索器
    'ExpressionFactor',
    'expression_factor',
    'AlphaExplorer',
    'AlphaResult',
]
