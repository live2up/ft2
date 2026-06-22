"""
utils/ast — 公共 AST 基础设施 (signals 和 factor 模块共享)

提供:
  - AST DSL: parse, validate, evaluate (Python AST 表达式引擎)
  - 函数注册表: 67+ 时序/截面/数学原语
  - 截面解析器: 单遍 bottom-up 处理嵌套/组合截面函数

[重构] 2026-06-22 从 signals/v4 和 factor/v4 提取到 utils 公共层
"""

from .dsl import (
    parse_expression, evaluate, get_variables, get_functions,
    DSLSecurityError, DSLSyntaxError,
)
from .registry import (
    FUNC_REGISTRY, is_valid_variable,
    register_function, register_variable,
    unregister_function, unregister_variable,
)
from .resolver import (
    CsResolver,
    CROSS_SECTIONAL_FUNCTIONS,
    _has_any_cs,
    _is_outer_cs_rank_call,
    _eval_colwise,
)

__all__ = [
    # dsl
    'parse_expression', 'evaluate', 'get_variables', 'get_functions',
    'DSLSecurityError', 'DSLSyntaxError',
    # registry
    'FUNC_REGISTRY', 'is_valid_variable',
    'register_function', 'register_variable',
    'unregister_function', 'unregister_variable',
    # resolver
    'CsResolver', 'CROSS_SECTIONAL_FUNCTIONS',
    '_has_any_cs', '_is_outer_cs_rank_call', '_eval_colwise',
]
