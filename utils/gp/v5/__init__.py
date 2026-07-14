"""
utils/gp/v5 — 遗传编程引擎核心 (AST 原生)

═══════════════════════════════════════════════════════════════
架构 (5 个实质文件):

  1. 引擎层  (engine.py)     → 遗传编程主循环 + 岛屿模型
     GPEngine                  — 种群管理 / 方向演化 / Motif 种子 / ε-greedy

  2. 配置层  (config.py)     → 树生成概率 + 个体定义
     TreeGenConfig             — 变量/函数/结构权重偏置
     Individual                — 个体封装 (tree / fitness / depth)
     DEFAULT_GP_CONFIG         — 引擎默认参数
     DEFAULT_TREE_GEN_CONFIG   — 树生成默认权重

  3. 缓存层  (cache.py)       → Fitness 双级缓存
     FitnessCache              — 内存 + SQLite 持久化 / 懒加载 / 溯源

  4. 生成层  (tree_gen.py)    → 随机树生成 + 4 种变异算子
     _grow_tree / _random_tree — 全/受限树生成
     _mutate_subtree           — 子树替换变异
     _mutate_param             — 参数变异
     _mutate_logic             — 逻辑/条件变异

  5. 工具层  (ast_utils.py)    → AST 纯函数
     _expr_str / _simplify_ast / _canonicalize_key  — 表达式规范化
     _collect_replaceable / _replace_subtree         — 子树替换
     _extract_subtrees                             — 子结构提取

  依赖方向: 引擎 ← 配置 + 缓存 + 生成 + 工具
═══════════════════════════════════════════════════════════════

[重构] 2026-07-06 从 factor/v5/gp_engine.py 抽取，供 factor 和 signals 共享
[重构] 2026-07-14 补充包级 re-export，对齐 utils.ast.v2 风格
"""

# ── 引擎层 (engine.py) ──
from .engine import GPEngine

# ── 配置层 (config.py) ──
from .config import (
    TreeGenConfig,
    Individual,
    DEFAULT_GP_CONFIG,
    DEFAULT_TREE_GEN_CONFIG,
)

# ── 缓存层 (cache.py) ──
from .cache import FitnessCache

# tree_gen / ast_utils 为内部工具，不通过包级导出

__all__ = [
    # engine
    'GPEngine',
    # config
    'TreeGenConfig',
    'Individual',
    'DEFAULT_GP_CONFIG',
    'DEFAULT_TREE_GEN_CONFIG',
    # cache
    'FitnessCache',
]
