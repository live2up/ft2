"""
factor/v4/cs_resolver.py — 兼容重导出

[重构] 2026-06-22 逻辑已迁移至 utils/ast/resolver.py,
此文件保留向后兼容。
"""
from utils.ast.resolver import (
    CsResolver,
    _get_cs_functions,
    _has_any_cs,
    _is_outer_cs_rank_call,
    _eval_colwise,
    _cross_sectional_rank,
)

__all__ = [
    'CsResolver',
    '_get_cs_functions',
    '_has_any_cs',
    '_is_outer_cs_rank_call',
    '_eval_colwise',
    '_cross_sectional_rank',
]
