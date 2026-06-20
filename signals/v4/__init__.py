"""
signals/v4/__init__.py — V4 信号模块（Python AST DSL + ft2.core Engine）
=============================================================================
V4 核心：Python ast 解析表达式 → 真实回测 → 交互式报告。

用法：
  from signals.v4 import Expression, EngineCore

  expr = Expression("(CLOSE / ts_mean(CLOSE, 50) - 1) * 100")
  signal = expr.generate(ohlcv_df)
  analyzer = EngineCore.backtest(signal, ohlcv_df, mode='fast')
  print(f"Sharpe={analyzer.sharpe_ratio():.3f}")
=============================================================================
"""
# V4 DSL
from .ast_dsl import (
    parse_expression, evaluate, get_variables, get_functions,
    DSLSecurityError, DSLSyntaxError,
)
from .expression import Expression
from .registry import FUNC_REGISTRY, is_valid_variable, register_function, register_variable, unregister_function, unregister_variable

# V4 引擎（full/fast 统一返回 AccountAnalyzer）
from .engine import EngineCore

__all__ = [
    'Expression',
    'EngineCore',
    'parse_expression', 'evaluate',
    'get_variables', 'get_functions',
    'FUNC_REGISTRY', 'is_valid_variable',
    'register_function', 'register_variable',
    'DSLSecurityError', 'DSLSyntaxError',
]
