"""
signals/v5/__init__.py — V5 信号模块（Python AST DSL + ft2.core Engine + v5 GP）
=============================================================================
V5 核心：Python ast 解析表达式 → 真实回测 → 交互式报告。
V5 升级：GP 引擎基于 utils/gp/v5 核心，支持权重聚焦/方向追踪/缓存等 v5 特性。

用法：
  from signals.v5 import Expression, SigEngine
  from signals.v5.search import SigGPEngine, GridSearch
  from signals.v5.validate import validate_single, compare_signals, validate_walkforward
  from signals.v5.market_breadth import calc_advance_decline_ratio, ...

  expr = Expression("(CLOSE / ts_mean(CLOSE, 50) - 1) * 100")
  signal = expr.generate(ohlcv_df)
  analyzer = SigEngine.backtest(signal, ohlcv_df, mode='fast')
  print(f"Sharpe={analyzer.sharpe_ratio():.3f}")

  # v5 GP 符号回归
  from signals.v5.expression import _build_gp_data_dict
  data_dict = _build_gp_data_dict(ohlcv_df)
  engine = SigGPEngine(data=data_dict, fitness_calculator=..., ...)
  engine.run()
=============================================================================
"""
# V5 DSL（兼容重导出 → 权威源在 utils/ast）
from .ast_dsl import (
    parse_expression, evaluate, get_variables, get_functions,
    DSLSecurityError, DSLSyntaxError,
)
from .expression import Expression, stateful_signal, _SignalFromAST, _build_gp_data_dict
from .registry import FUNC_REGISTRY, is_valid_variable, register_function, register_variable, unregister_function, unregister_variable

# V5 引擎（full/fast 统一返回 AccountAnalyzer）
from .engine import SigEngine

# V5 子模块
from . import market_breadth
from . import wf_result
from . import search
from . import validate

__all__ = [
    'Expression',
    'stateful_signal',
    '_SignalFromAST',
    '_build_gp_data_dict',
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
