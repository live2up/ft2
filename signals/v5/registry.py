"""
signals/v5/registry.py — 便捷重导出

直接从 utils.ast.v2 重导出，方便 signals/v5 用户一站式导入。
新代码推荐直接使用:

    from utils.ast.v2 import register_function, register_variable
"""
from utils.ast.v2 import *  # noqa: F401,F403
from utils.ast.v2 import FUNC_REGISTRY, SAFE_CONSTANTS, is_valid_variable, VALID_VAR_PREFIXES  # noqa: F401
from utils.ast.v2 import register_function, register_variable, unregister_function, unregister_variable  # noqa: F401
