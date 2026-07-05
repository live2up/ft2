"""
utils/gp/v5/ — 遗传编程引擎 v5
=============================================================================

提供:
  from utils.gp.v5 import GPEngine, Individual, TreeGenConfig
  from utils.gp.v5.config import DEFAULT_GP_CONFIG, DEFAULT_TREE_GEN_CONFIG
"""

from .config import (
    TreeGenConfig, Individual, DEFAULT_GP_CONFIG, DEFAULT_TREE_GEN_CONFIG,
    GP_VARIABLES, GP_CONSTANTS, TS_FUNCTIONS, TS_FUNCTIONS_2ARG,
    MATH_FUNCTIONS, _fill_weights,
)
from .engine import GPEngine

# 保留 _ExpressionFromAST 的兼容别名 — 由因子/信号模块各自注入
# 用法: from factor.v5.expression import _ExpressionFromAST
#       gp = GPEngine(data=data, evaluator=lambda d,t: _ExpressionFromAST(t).evaluate(d))

__all__ = [
    'GPEngine', 'Individual', 'TreeGenConfig',
    'DEFAULT_GP_CONFIG', 'DEFAULT_TREE_GEN_CONFIG',
    'GP_VARIABLES', 'GP_CONSTANTS', 'TS_FUNCTIONS', 'TS_FUNCTIONS_2ARG',
    'MATH_FUNCTIONS', '_fill_weights',
]
