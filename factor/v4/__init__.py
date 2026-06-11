"""
factor/v4 — 因子发现引擎 (基于 signals.v4 AST DSL + ft2.core Engine)

与 v3 的区别：
  - 自研 Parser + 原语 → signals.v4 Expression + registry (共用)
  - 自研回测 Pipeline → ft2.core Engine (统一时间线+费率)
  - 23 原语 → 63 原语 (与 signals 共享)

模块结构:
  base.py          — FactorLibrary + FactorMetadata (复用 v3)
  backtest.py      — 因子轮动回测 (ft2.core Engine 驱动)
  discover.py      — GP 发现引擎 (适配 signals.v4)
  validator.py     — IC/IR/Bootstrap 检验 (复用 v3)
  search.py        — 网格搜索 + BO (适配 signals.v4)
  cache.py         — 因子值缓存 (复用 v3)
  industry_fitness.py — 行业适应度 (复用 v3)
  formulas/        — 公式库 (WQ101/GT191，语法适配 V4)

Quick Start:
  >>> from factor.v4 import FactorBacktest, FactorExpression, FactorLibrary
  >>> from signals.v4 import Expression
  >>> expr = Expression("cs_rank(ts_roc(CLOSE, 20))")
  >>> panel = expr.rank_panel(assets)       # 因子值面板
  >>> result = FactorBacktest.run(panel)    # ft2.core Engine 回测
"""

# ── base (复用 v3) ──
from .base import (
    FactorCategory, FactorFrequency, FactorMetadata,
    Factor, FactorRegistry, factor,
    LibraryEntry, FactorLibrary,
)

# ── engine ──
from .engine import V4FactorExpression

# ── backtest ──
from .backtest import FactorBacktest

# ── validator ──
from .validator import FactorValidator, ValidationResult

# ── cache ──
from .cache import FactorCacheStore

__all__ = [
    'V4FactorExpression',
    'FactorBacktest',
    'FactorCategory', 'FactorMetadata', 'FactorLibrary',
    'FactorValidator', 'ValidationResult', 'FactorCacheStore',
]
