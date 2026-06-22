"""
utils/ast — 公共 AST 基础设施 (signals 和 factor 模块共享)

提供:
  - AST DSL: parse, validate, evaluate (Python AST 表达式引擎)
  - 函数注册表: 67+ 时序/截面/数学原语
  - 截面解析器: 单遍 bottom-up 处理嵌套/组合截面函数

使用:
    from utils.ast import (
        parse_expression,           # 解析表达式字符串 → AST
        CsResolver,                 # 截面函数嵌套解析
        register_function,          # 注册自定义函数
        register_variable,          # 注册自定义变量
    )

注册自定义函数:
    from utils.ast.registry import register_function, register_variable
    register_function('my_func', lambda x, w: ...)
    register_variable('MY_VAR')

    # 兼容旧路径 (通过重导出链仍可用):
    #   from signals.v4.registry import register_function, register_variable

[重构] 2026-06-22 从 signals/v4 和 factor/v4 提取到 utils 公共层
"""

from .dsl import (
    parse_expression, evaluate, get_variables, get_functions,
    normalize_data_keys,
    DSLSecurityError, DSLSyntaxError,
)
from .registry import (
    FUNC_REGISTRY, is_valid_variable,
    register_function, register_variable,
    unregister_function, unregister_variable,
)
from .resolver import (
    CsResolver,
    _get_cs_functions,
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
    'CsResolver', '_get_cs_functions',
    '_has_any_cs', '_is_outer_cs_rank_call', '_eval_colwise',
]
