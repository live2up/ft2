"""
signals/v2/__init__.py - V2核心模块导出

signals v2 是基于AIdev探索结论的底层重构模块。
核心变革：从"一个指标一个类"到"特征空间+表达式引擎"。

V2扩展模块：
- grid_search   : 参数网格搜索编排器
- gp_optimizer  : 遗传算法择时优化器（基于 AIdev GP V5）
- pysr_adapter  : PySR 符号回归适配器（Julia后端）
"""

from .features import (
    FeatureSpace,
    register_feature,
    DEFAULT_CONFIG,
    calc_atr,
    calc_stddev,
    calc_bbwidth,
    calc_hv,
    calc_natr,
    calc_trima,
    calc_sma_val,
    calc_ema,
    calc_tsf,
    calc_adx,
    calc_rsi,
    calc_cci,
    calc_macd,
    calc_var,
    calc_linearreg,
    calc_vol_ratio,
    calc_vol_chg,
    calc_trend_strength,
    calc_vol_regime,
    calc_mom_chg,
    calc_up_ratio,
)

from .expression import (
    NodeType,
    TreeNode,
    Expression,
    parse_expression,
    parse_and_build,
    persist,
    regime_switch,
    np_persist,
)

from .validator import (
    Validator,
    run_backtest,
    run_backtest_from_signal,
    walk_forward,
    WalkForwardResult,
)

from .pipeline import (
    SignalPipeline,
    pipe_and,
    pipe_or,
    pipe_vote,
    pipe_weighted,
)

from .registry import (
    ExpressionRegistry,
    DEFAULT_REGISTRY,
)

from .presets import (
    PRESETS,
    ExpressionPreset,
)

from .explainer import (
    Explainer,
    ExplanationReport,
    RegimePerformance,
    explain_signal,
)

from .decay_monitor import (
    DecayMonitor,
    DecayResult,
    AlertLevel,
    check_decay,
)

from .grid_search import (
    GridSearch,
    GridResult,
)

from .gp_optimizer import (
    GPOptimizer,
    Individual,
    GenerationHistory,
    NODE_CONFIG,
    SEED_EXPRESSIONS,
)

from .pysr_adapter import (
    PySRAdapter,
    SRFormulaResult,
    DEFAULT_PYSR_CONFIG,
)

__all__ = [
    # Features
    'FeatureSpace', 'register_feature', 'DEFAULT_CONFIG',
    'calc_atr', 'calc_stddev', 'calc_bbwidth', 'calc_hv', 'calc_natr',
    'calc_trima', 'calc_sma_val', 'calc_ema', 'calc_tsf', 'calc_adx',
    'calc_rsi', 'calc_cci', 'calc_macd', 'calc_var', 'calc_linearreg',
    'calc_vol_ratio', 'calc_vol_chg', 'calc_trend_strength',
    'calc_vol_regime', 'calc_mom_chg', 'calc_up_ratio',
    # Expression
    'NodeType', 'TreeNode', 'Expression', 'parse_expression',
    'parse_and_build', 'persist', 'regime_switch', 'np_persist',
    # Validator
    'Validator', 'run_backtest', 'run_backtest_from_signal', 'walk_forward', 'WalkForwardResult',
    # Pipeline
    'SignalPipeline', 'pipe_and', 'pipe_or', 'pipe_vote', 'pipe_weighted',
    # Registry
    'ExpressionRegistry', 'DEFAULT_REGISTRY',
    # Presets
    'PRESETS', 'ExpressionPreset',
    # Explainer
    'Explainer', 'ExplanationReport', 'RegimePerformance', 'explain_signal',
    # Decay Monitor
    'DecayMonitor', 'DecayResult', 'AlertLevel', 'check_decay',
    # Grid Search
    'GridSearch', 'GridResult',
    # GP Optimizer
    'GPOptimizer', 'Individual', 'GenerationHistory',
    'NODE_CONFIG', 'SEED_EXPRESSIONS',
    # PySR Adapter
    'PySRAdapter', 'SRFormulaResult', 'DEFAULT_PYSR_CONFIG',
]
