"""
utils/gp/v5/ — 遗传编程引擎 v5

模块结构:
  config.py     — TreeGenConfig + Individual + 常量 (不变)
  ast_utils.py  — AST 纯函数工具 (抽取自 engine.py)
  tree_gen.py   — 随机树生成 + 变异算子 (抽取自 engine.py)
  cache.py      — FitnessCache (抽取自 engine.py)
  engine.py     — GPEngine 主类 (精简后)

提供:
  from utils.gp.v5 import GPEngine, Individual, TreeGenConfig
  from utils.gp.v5.config import DEFAULT_GP_CONFIG, DEFAULT_TREE_GEN_CONFIG
"""

from .config import (
    TreeGenConfig, Individual, DEFAULT_GP_CONFIG, DEFAULT_TREE_GEN_CONFIG,
    GP_VARIABLES, GP_CONSTANTS, _fill_weights,
)
from .ast_utils import (
    _expr_str, _collect_replaceable, _replace_subtree, _simplify_ast, _canonicalize_key,
)
from .tree_gen import (
    _grow_tree, _random_tree, _random_tree_explore,
    _mutate_subtree, _MUTATE_OPS,
)
from .cache import FitnessCache
from .engine import GPEngine

__all__ = [
    'GPEngine', 'Individual', 'TreeGenConfig',
    'DEFAULT_GP_CONFIG', 'DEFAULT_TREE_GEN_CONFIG',
    'GP_VARIABLES', 'GP_CONSTANTS', '_fill_weights',
    '_expr_str', '_collect_replaceable', '_replace_subtree',
    '_simplify_ast', '_canonicalize_key',
    '_grow_tree', '_random_tree', '_random_tree_explore',
    '_mutate_subtree', '_MUTATE_OPS',
    'FitnessCache',
]
