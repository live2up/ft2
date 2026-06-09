"""
signals/v3/__init__.py — V3 统一架构
=============================================================================
v3 核心变革: 探索、测试、回测三个阶段全部走 ft2.core Engine。

与 v2 的关系:
  - 继承 v2 的 FeatureSpace, Expression, 算法层 (GP/网格)
  - 替换 v2 的简化引擎 (run_backtest) 为 v3.EngineV3 (full/fast)
  - fast 模式 ≈ 简化引擎速度, 但费率和时间线与 full 完全一致
  - 搜索出来的信号直接就是真实 Sharpe, 不需要两阶段验证

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
 │  特征层 (继承 v2)                                                       │
 │  signals.v2.FeatureSpace / Expression / parse_expression               │
 │  signals.v2.pipeline / presets / registry                              │
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
│  fast → Engine.run() + 自管净值(无TradeRecord) │
│  full → Engine.run() + AccountManager      │
│         → AccountAnalyzer → to_notebook    │
│                                            │
│  同一 Engine.run() 时间线, fast/full 一致    │
 └────────────────────────────────────────────┘
          │
          ▼
 ┌────────────────────────────────────────────┐
 │              ft2.core                       │
 │  Engine / AccountManager / AccountAnalyzer  │
 │  Notebook / BenchHolder                     │
 └────────────────────────────────────────────┘


============================================================================
                          模式对比
============================================================================

┌──────────┬──────────────┬──────────────┬────────────────────┐
│          │ full (验证)   │ fast (搜索)   │ v2 简化引擎 (已弃) │
├──────────┼──────────────┼──────────────┼────────────────────┤
│ 引擎     │ core.Engine   │ core.Engine   │ numpy/pandas       │
│ 费率     │ ETF万3+千1    │ ETF万3+千1    │ 0                  │
│ 时间线   │ eob 递进      │ eob 递进      │ shift(1)           │
│ 账户     │ 完整          │ 无            │ 无                 │
│ 输出     │ AccountAnlzr  │ FastResult    │ BacktestResult     │
│ 速度/次  │ ~3s           │ ~0.5s         │ ~0.1s              │
│ GP 4000  │ 3.3h          │ 33min         │ 8min               │
│ Sharpe   │ 真实          │ =full(一致)   │ 虚高20%            │
└──────────┴──────────────┴──────────────┴────────────────────┘


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
import sys, os

# 从 v2 继承 (不重复实现)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from signals.v2.features import (
    FeatureSpace, register_feature, DEFAULT_CONFIG,
)
from signals.v2.expression import (
    Expression, parse_expression, parse_and_build,
    persist, regime_switch, np_persist,
    NodeType, TreeNode,
)
from signals.v2.pipeline import (
    SignalPipeline, pipe_and, pipe_or, pipe_vote, pipe_weighted,
)
from signals.v2.registry import (
    ExpressionRegistry, DEFAULT_REGISTRY,
)
from signals.v2.presets import (
    PRESETS, ExpressionPreset,
)

# v3 核心
from .engine import EngineV3, FastResult
from .search import GPSearch, GridSearch
from .validate import validate_single, compare_signals, walkforward_validate
from .scoring import ScoredSignal, CompositeScorer, three_zone_backtest, BacktestResult

__all__ = [
    # v3 引擎
    'EngineV3', 'FastResult',
    # 搜索
    'GPSearch', 'GridSearch',
    # 验证
    'validate_single', 'compare_signals', 'walkforward_validate',
    # 打分
    'ScoredSignal', 'CompositeScorer', 'three_zone_backtest', 'BacktestResult',
    # v2 继承
    'FeatureSpace', 'register_feature', 'DEFAULT_CONFIG',
    'Expression', 'parse_expression', 'parse_and_build',
    'persist', 'regime_switch', 'np_persist',
    'NodeType', 'TreeNode',
    'SignalPipeline', 'pipe_and', 'pipe_or', 'pipe_vote', 'pipe_weighted',
    'ExpressionRegistry', 'DEFAULT_REGISTRY',
    'PRESETS', 'ExpressionPreset',
]
