"""
utils/gp/v6 — 遗传编程引擎核心 (AST 原生) + AURORA 增强

═══════════════════════════════════════════════════════════════
架构 (6 个实质文件):

  1. 引擎层  (engine.py)       → 遗传编程主循环 + 岛屿模型
     GPEngine                    — 种群管理 / 方向演化 / Motif 种子 / ε-greedy

  2. AURORA层 (aurora_engine.py) → AURORA 启发式 GP (v6 新增)
     AURORAEngine                — Archive + DNS 局部竞争 + 编码器 + 灭绝
     FactorEncoder               — 轻量 MLP: 因子统计 → 潜在向量 φ
     AuroraArchive               — 非结构化 archive, DNS 支配关系替换
     extract_factor_stats        — 因子输出矩阵 → 8 维统计摘要

  3. 配置层  (config.py)       → 树生成概率 + 个体定义
     TreeGenConfig               — 变量/函数/结构权重偏置
     Individual                  — 个体封装 (tree / fitness / depth)
     EncodedIndividual           — AURORA 个体 (Individual + phi + stats)
     DEFAULT_GP_CONFIG           — 引擎默认参数
     AURA_DEFAULT_CONFIG         — AURORA 默认参数
     DEFAULT_TREE_GEN_CONFIG     — 树生成默认权重

  4. 缓存层  (cache.py)         → Fitness 双级缓存
     FitnessCache                — 内存 + SQLite 持久化 / 懒加载 / 溯源

  5. 生成层  (tree_gen.py)      → 随机树生成 + 4 种变异算子
     _grow_tree / _random_tree   — 全/受限树生成
     _mutate_subtree / _mutate_param / _mutate_logic — 变异算子

  6. 工具层  (ast_utils.py)      → AST 纯函数
     _expr_str / _simplify_ast / _canonicalize_key  — 表达式规范化

  依赖方向: AURORA ← 引擎 ← 配置 + 缓存 + 生成 + 工具
═══════════════════════════════════════════════════════════════

[重构] 2026-07-20 从 v5 复制为 v6，新增 AURORA 增强层
"""

# ── 引擎层 (engine.py) ──
from .engine import GPEngine

# ── AURORA 层 (aurora_engine.py) [v6 新增] ──
from .aurora_engine import (
    AURORAEngine,
    FactorEncoder,
    AuroraArchive,
    extract_factor_stats,
)

# ── 配置层 (config.py) ──
from .config import (
    TreeGenConfig,
    Individual,
    EncodedIndividual,
    DEFAULT_GP_CONFIG,
    AURA_DEFAULT_CONFIG,
    DEFAULT_TREE_GEN_CONFIG,
)

# ── 缓存层 (cache.py) ──
from .cache import FitnessCache

# tree_gen / ast_utils 为内部工具，不通过包级导出

__all__ = [
    # engine
    'GPEngine',
    # AURORA engine
    'AURORAEngine',
    'FactorEncoder',
    'AuroraArchive',
    'extract_factor_stats',
    # config
    'TreeGenConfig',
    'Individual',
    'EncodedIndividual',
    'DEFAULT_GP_CONFIG',
    'AURA_DEFAULT_CONFIG',
    'DEFAULT_TREE_GEN_CONFIG',
    # cache
    'FitnessCache',
]
