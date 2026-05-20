"""
factor.v2 - 因子挖掘 V2 模块

V2 增量重构，地基先行、网格优先：
- scheduler: 调仓日生成抽象层
- allocator: 权重分配器抽象层
- pipeline: 因子→回测桥接层
- cache_store: 因子值持久化缓存
- grid_search: 因子参数网格搜索（Phase A 主力工具）
- bo_search: 贝叶斯优化（网格之后的可选增强）
- gp_primitives: GP 符号回归原语集（Phase C）
- gp_evaluator: GP 多频率适应度评估器

设计原则：
1. 所有新增模块在 factor/v2/ 下，不影响 V1 factor/ 模块
2. 网格搜索保留为基线工具，新方法可插拔
3. 遵循 AGENTS.md 注释规范
"""

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

__all__ = [
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
]
