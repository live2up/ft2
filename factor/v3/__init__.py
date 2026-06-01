"""
factor/v3 — 因子发现引擎

以机器因子探索为核心，构建可插拔 GP 适应度、迭代发现循环、
统一公式库的因子引擎。

模块结构 (9 文件):
  base.py         — FactorCategory / FactorMetadata / FactorLibrary
  primitives.py   — 19 个时序/截面原语
  engine.py       — 表达式引擎 (Tokenizer + Parser + AST + ExpressionFactor + AlphaExplorer)
  backtest.py     — 回测一体化 (Scheduler + Allocator + Combiner + Pipeline)
  formulas.py     — WQ101 + GT191 公式字典
  validator.py    — IC/IR/Bootstrap/换手率 检验
  search.py       — 网格搜索 + BO 搜索
  discover.py     — GP引擎 + 可插拔适应度 + 迭代发现流水线
  cache.py        — 因子值缓存

Quick Start:
  >>> from factor.v3 import (
  ...     FactorPipeline, FixedScheduler, TopNEqualWeight,  # backtest
  ...     FactorExpression, ExpressionFactor, AlphaExplorer,  # engine
  ...     ALPHA101, ALPHA191,                                 # formulas
  ...     FactorLibrary, FactorDiscoveryEngine,               # discover
  ... )
[重构] 2026-06-01 v3 创建
"""

# ── base ──
from .base import (
    FactorCategory, FactorFrequency, FactorMetadata,
    Factor, FactorRegistry, factor,
    LibraryEntry, FactorLibrary,
)

# ── primitives ──
from .primitives import (
    ts_rank, ts_zscore, delay, correlation, decay_linear,
    cs_rank, cs_zscore, signed_power,
    ts_sum, ts_mean, ts_std, ts_max, ts_min,
    sma, covariance, regbeta,
    ts_argmin, ts_argmax, ifelse,
    EXTENDED_PRIMITIVES, get_primitive, list_primitives,
)

# ── engine ──
from .engine import (
    NodeType, ASTNode, FactorExpression, parse_expression,
    ExpressionFactor, expression_factor,
    AlphaExplorer, AlphaResult,
    VARIABLE_MAP, UNARY_FUNCTIONS, BINARY_FUNCTIONS, PRIMITIVE_FUNCTIONS,
    evaluate_node, Parser, tokenize,
)

# ── backtest ──
from .backtest import (
    RebalanceScheduler, FixedScheduler, IntervalScheduler,
    recommend_scheduler_from_decay,
    WeightAllocator, TopNEqualWeight, ScoreProportional, RiskParity,
    FactorCombiner, EqualWeightCombiner, FixedWeightCombiner, ExpandingICCombiner,
    cross_section_zscore,
    BacktestResult, BacktestSchedule, FactorPipeline,
)

# ── formulas ──
from .formulas import (
    ALPHA101, ALPHA101_CATEGORIES,
    ALPHA191, ALPHA191_CATEGORIES,
    BASIC_FACTORS, BASIC_FACTORS_CATEGORIES,
)

# ── validator ──
from .validator import (
    FactorValidator, ValidationResult, ValidationMetric, validation_metric,
)

# ── search ──
from .search import (
    GridSearchResult, FactorGridSearch, FactorBOSearch,
)

# ── discover ──
from .discover import (
    FitnessMode, FitnessCalculator,
    ICIRFitness, SharpeFitness, MultiFreqFitness,
    make_fitness_calculator,
    Individual, GPEngine,
    DiscoveryReport, FactorDiscoveryEngine,
    DEFAULT_GP_CONFIG, TERMINALS, PRIMITIVE_WITH_PARAMS,
)

# ── cache ──
from .cache import FactorCacheStore


__all__ = [
    # base
    'FactorCategory', 'FactorFrequency', 'FactorMetadata',
    'Factor', 'FactorRegistry', 'factor',
    'LibraryEntry', 'FactorLibrary',
    # primitives
    'ts_rank', 'ts_zscore', 'delay', 'correlation', 'decay_linear',
    'cs_rank', 'cs_zscore', 'signed_power',
    'ts_sum', 'ts_mean', 'ts_std', 'ts_max', 'ts_min',
    'sma', 'covariance', 'regbeta',
    'ts_argmin', 'ts_argmax', 'ifelse',
    'EXTENDED_PRIMITIVES', 'get_primitive', 'list_primitives',
    # engine
    'NodeType', 'ASTNode', 'FactorExpression', 'parse_expression',
    'ExpressionFactor', 'expression_factor',
    'AlphaExplorer', 'AlphaResult',
    'VARIABLE_MAP', 'UNARY_FUNCTIONS', 'BINARY_FUNCTIONS', 'PRIMITIVE_FUNCTIONS',
    'evaluate_node', 'Parser', 'tokenize',
    # backtest
    'RebalanceScheduler', 'FixedScheduler', 'IntervalScheduler',
    'recommend_scheduler_from_decay',
    'WeightAllocator', 'TopNEqualWeight', 'ScoreProportional', 'RiskParity',
    'FactorCombiner', 'EqualWeightCombiner', 'FixedWeightCombiner', 'ExpandingICCombiner',
    'cross_section_zscore',
    'BacktestResult', 'BacktestSchedule', 'FactorPipeline',
    # formulas
    'ALPHA101', 'ALPHA101_CATEGORIES',
    'ALPHA191', 'ALPHA191_CATEGORIES',
    'BASIC_FACTORS', 'BASIC_FACTORS_CATEGORIES',
    # validator
    'FactorValidator', 'ValidationResult', 'ValidationMetric', 'validation_metric',
    # search
    'GridSearchResult', 'FactorGridSearch', 'FactorBOSearch',
    # discover
    'FitnessMode', 'FitnessCalculator',
    'ICIRFitness', 'SharpeFitness', 'MultiFreqFitness',
    'make_fitness_calculator',
    'Individual', 'GPEngine',
    'DiscoveryReport', 'FactorDiscoveryEngine',
    'DEFAULT_GP_CONFIG', 'TERMINALS', 'PRIMITIVE_WITH_PARAMS',
    # cache
    'FactorCacheStore',
]
