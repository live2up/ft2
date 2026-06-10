"""
signals/v3/__init__.py — V3 独立包 (零外部依赖 v2)
=============================================================================
v3 核心变革: 探索、测试、回测三个阶段全部走 ft2.core Engine。

v3 现在是完全独立的包——复制了 v2 中所有被依赖的模块，
不再 import signals.v2。所有导入为包内相对导入。

============================================================================
                         V3 架构总览
============================================================================

 ┌───────────────────────────────────────────────────────────────────────┐
 │  数据层 (不变)                                                          │
 │  d2_api / tdx / 通达信 → OHLCV DataFrame                               │
 └─────────────────────────┬─────────────────────────────────────────────┘
                           │
                           ▼
 ┌───────────────────────────────────────────────────────────────────────┐
 │  特征层 (v3 独立)                                                       │
 │  signals.v3.FeatureSpace / Expression / parse_expression               │
 │  signals.v3.pipeline / presets / registry                              │
 └─────────────────────────┬─────────────────────────────────────────────┘
                           │ signal + feature_df
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
 ┌──────────────────┐ ┌──────────┐ ┌──────────────┐
 │ search/           │ │ validate/│ │ monitor/     │
 │                  │ │          │ │              │
 │ GPSearch (fast)  │ │ single   │ │ explainer    │
 │ GridSearch (fast)│ │ compare  │ │ decay        │
 │                  │ │ wf       │ │ ic           │
 └────────┬─────────┘ └────┬─────┘ └──────────────┘
          │                │
          │     ┌──────────┘
          ▼     ▼
 ┌────────────────────────────────────────────┐
 │            engine.py — EngineV3             │
 │                                            │
 │  backtest(signal, data, mode='fast|full')   │
 │                                            │
 │  fast → Engine.run() + 自管净值              │
 │  full → Engine.run() + AccountManager      │
 │         → AccountAnalyzer → to_notebook    │
 └────────────────────────────────────────────┘

============================================================================
                          快速上手
============================================================================

from signals.v3 import EngineV3

# 探索 (fast)
result = EngineV3.backtest(signal, data, mode='fast')
# → FastResult(sharpe=1.13, cagr=0.148, ...)

# 验证 (full)
analyzer = EngineV3.backtest(signal, data, mode='full',
                              start_date='2020-01-01', bench_label='399317.SZ')
analyzer.to_notebook("策略回测")

# 搜索
from signals.v3.search import GPSearch
gs = GPSearch(fs, train, test, start_date='2020-01-01')
gs.run()
for ind in gs.elite_set(5):
    analyzer = EngineV3.backtest(ind.evaluate(features), data, mode='full')
    analyzer.to_notebook(f"GP发现: {ind.expression_str}")

============================================================================
"""
# ── v3 本地模块 (全部自包含) ──
from .features import (
    FeatureSpace, register_feature, DEFAULT_CONFIG,
)
from .expression import (
    Expression, parse_expression, parse_and_build,
    persist, regime_switch, np_persist,
    NodeType, TreeNode,
)
from .pipeline import (
    SignalPipeline, pipe_and, pipe_or, pipe_vote, pipe_weighted,
)
from .registry import (
    ExpressionRegistry, DEFAULT_REGISTRY,
)
from .presets import (
    PRESETS, ExpressionPreset,
)

# ── v3 核心引擎 ──
from .engine import EngineV3, FastResult

# ── v3 搜索 ──
from .search import GPSearch, GridSearch

# ── v3 验证 ──
from .validate import validate_single, compare_signals, walkforward_validate, walkforward_validate_expr

# ── v3 打分 ──
from .scoring import ScoredSignal, CompositeScorer, three_zone_backtest, BacktestResult

# ── v3 分析 ──
from .monitor import ICAnalyzer, DecayMonitor, DecayResult, AlertLevel, check_decay
from .monitor import Explainer, ExplanationReport, RegimePerformance, explain_signal

__all__ = [
    # v3 引擎
    'EngineV3', 'FastResult',
    # 搜索
    'GPSearch', 'GridSearch',
    # 验证
    'validate_single', 'compare_signals', 'walkforward_validate', 'walkforward_validate_expr',
    # 打分
    'ScoredSignal', 'CompositeScorer', 'three_zone_backtest', 'BacktestResult',
    # 特征/表达式
    'FeatureSpace', 'register_feature', 'DEFAULT_CONFIG',
    'Expression', 'parse_expression', 'parse_and_build',
    'persist', 'regime_switch', 'np_persist',
    'NodeType', 'TreeNode',
    # 管线
    'SignalPipeline', 'pipe_and', 'pipe_or', 'pipe_vote', 'pipe_weighted',
    # 注册表/预设
    'ExpressionRegistry', 'DEFAULT_REGISTRY',
    'PRESETS', 'ExpressionPreset',
    # 分析
    'ICAnalyzer', 'DecayMonitor', 'DecayResult', 'AlertLevel', 'check_decay',
    'Explainer', 'ExplanationReport', 'RegimePerformance', 'explain_signal',
]
