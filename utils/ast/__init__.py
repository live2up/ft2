"""
utils/ast — 公共 AST 基础设施 (signals 和 factor 模块共享)

═══════════════════════════════════════════════════════════════
五层架构:

  1. 语法层  (dsl.py)        → 定义"能写什么"
     parse_expression()    — Python AST 解析 (白名单/黑名单安全校验)
     evaluate()            — 递归求值 AST 节点
     eval_colwise()        — 面板逐列求值 (2D 安全)
     cross_sectional_rank()— 截面排名 0~1
     normalize_data_keys() — 数据键 ALL_CAPS 规范化

  2. 原语层  (functions.py)  → 定义"能算什么"
     FUNC_REGISTRY          — 72 时序/截面/数学/特征函数
     FUNC_CATEGORIES        — 按类别索引

  3. 变量层  (variables.py)  → 定义"能引用什么"
     VALID_VAR_PREFIXES     — 70+ 合法变量前缀
     VAR_CATEGORIES         — 按类别索引

  4. 编排层  (resolver.py)   → 截面函数嵌套解算
     CsResolver.resolve()   — 单遍 bottom-up AST 变换
     自动发现 cs_* 前缀函数, 处理任意深度嵌套/组合

  5. 规格层  (spec.py)       → AST 构建与约束
     make_var/make_call/... — 类型安全 AST 节点构建器 (供 GP 引擎)
     normalize_expression() — 表达式规范化
     describe_expression()  — 表达式结构化描述 (供 LLM)
     grammar_spec_for_llm() — 语法规格 (供 LLM prompt)

  向后兼容: registry.py 重导出 functions.py + variables.py

  依赖方向: 语法 ← 原语 + 变量 ← 编排 ← 规格
═══════════════════════════════════════════════════════════════

命名约定 (对齐 WQ101 行业标准):
  变量:   ALL_CAPS          CLOSE, REL_CLOSE, SECTOR_UP
  函数:   prefix_snake      ts_roc, cs_rank, expanding_std
  窗口:   参数名 d (day)      ts_mean(x, d)  ← 不对齐 WQ101 的 w
  统计:   样本 ddof=1         ts_std, ts_skew, cs_zscore

[重构] 2026-06-22 从 signals/v4 和 factor/v4 提取到 utils 公共层
[规范化] 2026-06-22 拆分为 5 层清晰架构
"""

# ── 语法层 (dsl.py) ──
from .dsl import (
    parse_expression, evaluate, get_variables, get_functions,
    normalize_data_keys,
    eval_colwise, cross_sectional_rank,
    ast_depth, ast_node_count,
    DSLSecurityError, DSLSyntaxError,
)

# ── 原语层 (functions.py) ──
from .functions import (
    FUNC_REGISTRY, SAFE_CONSTANTS,
    register_function, unregister_function,
    FUNC_CATEGORIES, get_func_category,
)

# ── 变量层 (variables.py) ──
from .variables import (
    VALID_VAR_PREFIXES, is_valid_variable,
    register_variable, unregister_variable,
    VAR_CATEGORIES, get_var_category,
)

# ── 编排层 (resolver.py) ──
from .resolver import (
    CsResolver,
    _get_cs_functions, _has_any_cs, _is_outer_cs_rank_call,
)

# ── 基类层 (expr_base.py) ──
from .expr_base import AstExpression

# ── 规格层 (spec.py) ──
from .spec import (
    # 构建器
    make_var, make_const, make_call,
    make_binop, make_unaryop, make_compare,
    make_boolop, make_ifexp,
    # 规范化器
    normalize_expression, normalize_ast,
    # 自省
    describe_expression,
    # 规格
    grammar_spec_for_llm, grammar_spec_compact,
    AST_GRAMMAR_SPEC,
)

# ── 向后兼容 (registry.py) ──
from .registry import *  # noqa: F401,F403

__all__ = [
    # dsl — 语法层
    'parse_expression', 'evaluate', 'get_variables', 'get_functions',
    'normalize_data_keys',
    'eval_colwise', 'cross_sectional_rank',
    'ast_depth', 'ast_node_count',
    'DSLSecurityError', 'DSLSyntaxError',

    # base — AST 表达式基类
    'AstExpression',

    # functions — 原语层
    'FUNC_REGISTRY', 'SAFE_CONSTANTS',
    'register_function', 'unregister_function',
    'FUNC_CATEGORIES', 'get_func_category',

    # variables — 变量层
    'VALID_VAR_PREFIXES', 'is_valid_variable',
    'register_variable', 'unregister_variable',
    'VAR_CATEGORIES', 'get_var_category',

    # resolver — 编排层
    'CsResolver', '_get_cs_functions',
    '_has_any_cs', '_is_outer_cs_rank_call',

    # spec — 规格层
    'make_var', 'make_const', 'make_call',
    'make_binop', 'make_unaryop', 'make_compare',
    'make_boolop', 'make_ifexp',
    'normalize_expression', 'normalize_ast',
    'describe_expression',
    'grammar_spec_for_llm', 'grammar_spec_compact',
    'AST_GRAMMAR_SPEC',
]
