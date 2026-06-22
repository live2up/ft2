"""
signals/v4/__init__.py — V4 信号模块（Python AST DSL + ft2.core Engine）
=============================================================================
V4 核心：Python ast 解析表达式 → 真实回测 → 交互式报告。

用法：
  from signals.v4 import Expression, SigEngine
  from signals.v4.search import GPSearch, GridSearch
  from signals.v4.validate import validate_single, compare_signals, validate_walkforward
  from signals.v4.market_breadth import calc_advance_decline_ratio, ...

  expr = Expression("(CLOSE / ts_mean(CLOSE, 50) - 1) * 100")
  signal = expr.generate(ohlcv_df)
  analyzer = SigEngine.backtest(signal, ohlcv_df, mode='fast')
  print(f"Sharpe={analyzer.sharpe_ratio():.3f}")
=============================================================================
"""
# V4 DSL（兼容重导出 → 权威源在 utils/ast）
from .ast_dsl import (
    parse_expression, evaluate, get_variables, get_functions,
    DSLSecurityError, DSLSyntaxError,
)
from .expression import Expression, stateful_signal
from .registry import FUNC_REGISTRY, is_valid_variable, register_function, register_variable, unregister_function, unregister_variable

# V4 引擎（full/fast 统一返回 AccountAnalyzer）
from .engine import SigEngine

# V4 子模块
from . import market_breadth
from . import wf_result
from . import search
from . import validate

__all__ = [
    'Expression',
    'stateful_signal',
    'SigEngine',
    'parse_expression', 'evaluate',
    'get_variables', 'get_functions',
    'FUNC_REGISTRY', 'is_valid_variable',
    'register_function', 'register_variable',
    'DSLSecurityError', 'DSLSyntaxError',
    'market_breadth',
    'wf_result',
    'search',
    'validate',
]
