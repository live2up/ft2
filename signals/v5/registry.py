"""
signals/v5/registry.py — 兼容重导出

[重构] 2026-07-07 从 v4 升级到 v5。
权威源已迁移至 utils/ast, 此文件保留向后兼容。新代码请使用:

    from utils.ast import register_function, register_variable

旧路径仍可用但建议迁移:

    from signals.v5.registry import ...   # 仍生效，重导出链
"""
from utils.ast import *  # noqa: F401,F403
from utils.ast import FUNC_REGISTRY, SAFE_CONSTANTS, is_valid_variable, VALID_VAR_PREFIXES  # noqa: F401
from utils.ast import register_function, register_variable, unregister_function, unregister_variable  # noqa: F401
