"""
factor/v3 — 因子发现引擎

以机器因子探索为核心，构建可插拔 GP 适应度、迭代发现循环、
统一公式库的因子引擎。

模块结构:
  base.py         — FactorCategory / FactorMetadata / FactorLibrary (含 save/load)
  primitives.py   — 23 个时序/截面原语
  engine.py       — 表达式引擎 (Tokenizer + Parser + AST + ExpressionFactor + AlphaExplorer)
  backtest.py     — 回测一体化 (Scheduler + Allocator + Combiner + Pipeline)
  formulas/       — 公式数据库 (wq101.py / gt191.py / basic.py)
  discovered/     — GP 发现因子存档 (json 持久化 + 时间戳查询)
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
[重构] 2026-06-04 v3 重组 formulas/ + discovered/ 子目录
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
    cs_rank, cs_zscore, cs_mean, signed_power,
    ts_sum, ts_mean, ts_std, ts_max, ts_min,
    sma, covariance, regbeta,
    ts_argmin, ts_argmax, ifelse, ts_skew, delta, winsorize,
    ts_regression_residual,
    EXTENDED_PRIMITIVES, get_primitive, list_primitives,
)

# ── engine ──
from .engine import (
    NodeType, ASTNode, FactorExpression, parse_expression,
    ExpressionFactor, expression_factor,
    AlphaExplorer, AlphaResult,
    VARIABLE_MAP, UNARY_FUNCTIONS, BINARY_FUNCTIONS, PRIMITIVE_FUNCTIONS,
    evaluate_node, Parser, tokenize,
    register_terminal, registered_terminals,  # [新增] 2026-06-05 自定义终端
)

# ── backtest ──
from .backtest import (
    RebalanceScheduler, FixedScheduler, IntervalScheduler,
    recommend_scheduler_from_decay, parse_scheduler,
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
    INDUSTRY_ALPHA, INDUSTRY_ALPHA_CATEGORIES,
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
    ICIRFitness, SharpeFitness, MultiFreqFitness, WQBFitness,
    make_fitness_calculator,
    Individual, GPEngine,
    DiscoveryReport, FactorDiscoveryEngine,
    DEFAULT_GP_CONFIG, TERMINALS, SAFE_GP_TERMINALS, PRIMITIVE_WITH_PARAMS,
)

# ── industry_fitness ──
from .industry_fitness import IndustryFitness

# ── cache ──
from .cache import FactorCacheStore

# ── discovered ──
from .discovered import load_discovered, merge_discovered


__all__ = [
    # base
    'FactorCategory', 'FactorFrequency', 'FactorMetadata',
    'Factor', 'FactorRegistry', 'factor',
    'LibraryEntry', 'FactorLibrary',
    # primitives
    'ts_rank', 'ts_zscore', 'delay', 'correlation', 'decay_linear',
    'cs_rank', 'cs_zscore', 'cs_mean', 'signed_power',
    'ts_sum', 'ts_mean', 'ts_std', 'ts_max', 'ts_min',
    'sma', 'covariance', 'regbeta',
    'ts_argmin', 'ts_argmax', 'ifelse', 'ts_skew', 'delta', 'winsorize',
    'EXTENDED_PRIMITIVES', 'get_primitive', 'list_primitives',
    # engine
    'NodeType', 'ASTNode', 'FactorExpression', 'parse_expression',
    'ExpressionFactor', 'expression_factor',
    'AlphaExplorer', 'AlphaResult',
    'VARIABLE_MAP', 'UNARY_FUNCTIONS', 'BINARY_FUNCTIONS', 'PRIMITIVE_FUNCTIONS',
    'evaluate_node', 'Parser', 'tokenize',
    'register_terminal', 'registered_terminals',  # [新增] 2026-06-05
    # backtest
    'RebalanceScheduler', 'FixedScheduler', 'IntervalScheduler',
    'recommend_scheduler_from_decay', 'parse_scheduler',
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
    'ICIRFitness', 'SharpeFitness', 'MultiFreqFitness', 'WQBFitness',
    'make_fitness_calculator',
    'Individual', 'GPEngine',
    'DiscoveryReport', 'FactorDiscoveryEngine',
    'DEFAULT_GP_CONFIG', 'TERMINALS', 'SAFE_GP_TERMINALS', 'PRIMITIVE_WITH_PARAMS',
    # cache
    'FactorCacheStore',
]
