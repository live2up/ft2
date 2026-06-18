"""
factor/v4 — 因子发现引擎 (基于 signals.v4 AST DSL + ft2.core Engine)

与 v3 的区别：
  - 自研 Parser + 原语 → signals.v4 Expression + registry (共用)
  - 自研回测 Pipeline → ft2.core Engine (统一时间线+费率)
  - 23 原语 → 67 原语 (与 signals 共享)
  - 自研 ASTNode GP → Python ast GP (原生 infix/if-else/and/or)

模块结构:
  base.py          — FactorLibrary + FactorMetadata (复用 v3)
  engine.py        — EngineCore (ft2.core Engine 驱动, fast/full 双模式, 统一返回 AccountAnalyzer)
  expression.py    — FactorExpression (基于 signals.v4 AST DSL)
  gp_engine.py     — GP 因子组合优化引擎 (Python AST 原生, 种子驱动)
  discover.py      — 发现流水线 (适配 signals.v4)
  validator.py     — IC/IR/Bootstrap 检验 (复用 v3)
  search.py        — 网格搜索 + BO (适配 signals.v4)
  cache.py         — 因子值缓存 (复用 v3)
  industry_fitness.py — 行业适应度 (复用 v3)
  llm/             — LLM 因子生成器
  formulas/        — 公式库 (WQ101/GT191，语法适配 V4)

Quick Start:
  >>> from factor.v4 import FactorExpression, EngineCore, FactorLibrary, GPEngine
  >>> from signals.v4 import Expression
  >>> expr = Expression("cs_rank(ts_roc(CLOSE, 20))")
  >>> panel = expr.rank_panel(assets)       # 因子值面板
  >>> result = EngineCore.backtest(panel, assets, mode='fast', top_n=3, rebalance='W')
"""

# ── base (复用 v3) ──
from .base import (
    FactorCategory, FactorFrequency, FactorMetadata,
    Factor, FactorRegistry, factor,
    LibraryEntry, FactorLibrary,
)

# ── engine ──
from .engine import EngineCore

# ── expression ──
from .expression import FactorExpression

# ── validator ──
from .validator import FactorValidator, ValidationResult

# ── cache ──
from .cache import FactorCacheStore

# ── gp_engine ──
from .gp_engine import GPEngine, Individual

__all__ = [
    'EngineCore',
    'FactorExpression',
    'GPEngine', 'Individual',
    'FactorCategory', 'FactorMetadata', 'FactorLibrary',
    'FactorValidator', 'ValidationResult', 'FactorCacheStore',
]
