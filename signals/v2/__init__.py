"""
signals/v2/__init__.py — V2 核心模块导出 & 架构总览
=============================================================================

signals v2 是基于 AIdev 探索结论的底层重构模块。
核心变革：从"一个指标一个类"到"特征空间 + 表达式引擎"。


============================================================================
                         模块架构总览（竖式）
============================================================================

 ┌─────────────────────────────────────────────────────────────────────┐
 │  数据层                                                               │
 │                                                                      │
 │  OHLCV DataFrame (d2_api / 通达信 / 任何数据源)                       │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  特征工厂层  ───  features.py                                        │
 │                                                                      │
 │  FeatureSpace  —  声明式配置 → 纯函数计算 → 55+ 个特征列              │
 │  ├── DEFAULT_CONFIG  : 函数调用式 + 列名引用式 + 布尔开关             │
 │  ├── _FEATURE_CALC_REGISTRY : 55+ calc_xxx() 纯函数                  │
 │  ├── normalize : close / pct / raw 三种策略                          │
 │  ├── differences : [A, B, op] 衍生特征计算                           │
 │  └── register_feature() : 自定义特征注册入口                          │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  信号表达层  ───  expression.py                                      │
 │                                                                      │
 │  Expression  —  字符串 → Tokenizer → Parser → AST → 信号序列         │
 │  ├── TreeNode (7种节点) : FEATURE / CONSTANT / OPERATOR / FUNCTION   │
 │  │                       THRESHOLD / PERSIST / SWITCH                │
 │  ├── 阈值函数 : thr_0 / thr_mean / thr_med                           │
 │  ├── 运算符重载 : & / | / - / +                                     │
 │  └── 自省 + 序列化 : save/load JSON                                  │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  管线编排层  ───  pipeline.py                                        │
 │                                                                      │
 │  SignalPipeline  —  声明式多阶段管线                                 │
 │  ├── signal → persist → threshold → combine                          │
 │  └── pipe_and / pipe_or / pipe_vote / pipe_weighted                  │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  回测验证层  ───  validator.py                                       │
 │                                                                      │
 │  run_backtest()  —  信号 → 持仓(T+1) → 净值 → 绩效指标               │
 │  walk_forward()  —  滚动窗口验证，过拟合检测                         │
 │  compare()       —  多表达式横向对比                                  │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
 ┌──────────┐ ┌──────────┐ ┌──────────────┐
 │ presets  │ │ registry │ │  explainer    │
 │ 模板库    │ │ 注册发现  │ │  白盒子解释   │
 └──────────┘ └──────────┘ └──────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  高级优化层                                                           │
 │                                                                      │
 │  grid_search.py   — 参数网格搜索 (所有 ? 展开遍历)                    │
 │  gp_optimizer.py  — 遗传算法优化 (交叉/变异/精英保留)                 │
 │  pysr_adapter.py  — PySR 符号回归 (Julia 后端)                       │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
│  监控 & 扩展层                                                        │
│                                                                      │
│  decay_monitor.py   — 因子衰减监控 (IC 趋势 → 预警)                  │
│  ic_analyzer.py     — IC 分析器 (Pearson/Rank IC / 显著性 / 衰减)     │
│  market_breadth.py  — 市场广度特征 (涨跌比/新高新低/麦克莱伦)         │
│  timeframe.py       — 多周期特征 (周线/月线 FeatureSpace 扩展)        │
└─────────────────────────────────────────────────────────────────────┘


============================================================================
                         参数规范对照（模块归属）
============================================================================

┌─────────────────────┬──────────────────┬─────────────────────────┐
│       配置项          │      语法类型    │      所在模块            │
├─────────────────────┼──────────────────┼─────────────────────────┤
│ features            │ 函数调用式        │ features.py (Config)    │
│ differences         │ 列名引用式 [A,B,op]│ features.py (Config)   │
│ regime              │ 布尔开关          │ features.py (Config)    │
│ market_breadth      │ 布尔开关          │ features.py (Config)    │
│ Expression          │ 数学公式 AST      │ expression.py           │
│ INDICATORS          │ 混用(业务映射)    │ run_baseline.py (应用层)│
└─────────────────────┴──────────────────┴─────────────────────────┘

============================================================================
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

from .scoring import (
    ScoredSignal,
    CompositeScorer,
    three_zone_backtest,
    BacktestResult,
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

from .ic_analyzer import (
    ICAnalyzer,
)

from .timeframe import (
    resample_ohlcv,
    align_to_daily,
    compute_multitimeframe_features,
    resample_and_compute,
)

from .market_breadth import (
    calc_advance_decline_ratio,
    calc_sector_diffusion,
    calc_new_high_low,
    calc_mcclellan_oscillator,
    calc_arms_index,
    register_breadth_features,
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
    # Scoring
    'ScoredSignal', 'CompositeScorer', 'three_zone_backtest', 'BacktestResult',
    # IC Analyzer
    'ICAnalyzer',
    # Timeframe
    'resample_ohlcv', 'align_to_daily',
    'compute_multitimeframe_features', 'resample_and_compute',
    # Market Breadth
    'calc_advance_decline_ratio', 'calc_sector_diffusion',
    'calc_new_high_low', 'calc_mcclellan_oscillator', 'calc_arms_index',
    'register_breadth_features',
]
