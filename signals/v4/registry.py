"""
signals/v4/registry.py — 兼容重导出

[重构] 2026-06-22 逻辑已迁移至 utils/ast/registry.py,
此文件保留向后兼容。
"""
from utils.ast.registry import *  # noqa: F401,F403
from utils.ast.registry import FUNC_REGISTRY, SAFE_CONSTANTS, is_valid_variable, VALID_VAR_PREFIXES  # noqa: F401
from utils.ast.registry import register_function, register_variable, unregister_function, unregister_variable  # noqa: F401
