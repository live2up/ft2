"""
signals/v4/ast_dsl.py — 兼容重导出

[重构] 2026-06-22 逻辑已迁移至 utils/ast/dsl.py,
此文件保留向后兼容。
"""
from utils.ast.dsl import *  # noqa: F401,F403
from utils.ast.dsl import (  # noqa: F401
    parse_expression, evaluate, get_variables, get_functions,
    DSLSecurityError, DSLSyntaxError,
)
