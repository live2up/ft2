"""
factor/v4 — 因子发现引擎 (基于 signals.v4 AST DSL + ft2.core Engine)

与 v3 的区别：
  - 自研 Parser + 原语 → signals.v4 Expression + registry (共用)
  - 自研回测 Pipeline → ft2.core Engine (统一时间线+费率)
  - 23 原语 → 67 原语 (与 signals 共享)
  - 自研 ASTNode GP → Python ast GP (原生 infix/if-else/and/or)

模块结构:
  base.py          — FactorLibrary + FactorMetadata
  engine.py        — FacEngine (ft2.core Engine 驱动, fast/full 双模式, 统一返回 AccountAnalyzer)
  expression.py    — FactorExpression (基于 signals.v4 AST DSL)
  gp_engine.py     — GP 因子组合优化引擎 (Python AST 原生, 种子驱动)
  validator.py     — IC/IR/Bootstrap 检验
  search.py        — 网格搜索 + BO (适配 signals.v4)
  cache.py         — 因子值缓存
  industry_fitness.py — 行业适应度 + FitnessCalculator 基类
  llm/             — LLM 因子生成器
  formulas/        — 公式库 (WQ101/GT191，语法适配 V4)

Quick Start:
  >>> from factor.v4 import FactorExpression, FacEngine, FactorLibrary, GPEngine
  >>> expr = FactorExpression("cs_rank(ts_roc(CLOSE, 20))")
  >>> panel = expr.evaluate_ranked(data_dict)   # ndarray(T,N) 截面排名
  >>> result = FacEngine.backtest(panel, assets, mode='fast', top_n=3, rebalance='W')
"""

# ── base (v4 独立版) ──
from .base import (
    FactorCategory, FactorFrequency, FactorMetadata,
    Factor, FactorRegistry, factor,
    LibraryEntry, FactorLibrary,
)

# ── engine ──
from .engine import FacEngine

# ── expression ──
from .expression import FactorExpression

# ── validator ──
from .validator import FactorValidator, ValidationResult

# ── cache ──
from .cache import FactorCacheStore

# ── gp_engine ──
from .gp_engine import GPEngine, Individual

# ── knowledge 因子知识树 ──
from .knowledge import FactorKnowledgeBase, ExplorationRecord, ExplorationStatus, ALPHA_SOURCES

__all__ = [
    'FacEngine',
    'FactorExpression',
    'GPEngine', 'Individual',
    'FactorCategory', 'FactorMetadata', 'FactorLibrary',
    'FactorValidator', 'ValidationResult', 'FactorCacheStore',
    'FactorKnowledgeBase', 'ExplorationRecord', 'ExplorationStatus', 'ALPHA_SOURCES',
]
