"""
utils/ast — 公共 AST 基础设施 (signals 和 factor 模块共享)

═══════════════════════════════════════════════════════════════
四层架构:

  1. 语法层  (dsl.py)       → 定义"能写什么"
     parse_expression()   — Python AST 解析 (白名单/黑名单安全校验)
     evaluate()           — 递归求值 AST 节点
     normalize_data_keys() — 数据键 ALL_CAPS 规范化

  2. 原语层  (registry.py)  → 定义"能算什么"
     FUNC_REGISTRY         — 67+ 时序/截面/数学/特征函数
     VALID_VAR_PREFIXES    — 70 个合法变量前缀

  3. 变量层  (registry.py)  → 定义"能引用什么"
     CLOSE, OPEN, RSI, ATR      — 原始/预计算单品种数据
     SECTOR_UP, DISP, IND_CORR — 行业广度/市场级数据
     用途: 性能缓存 (避免重复计算), 跨品种聚合 (无法从单 OHLCV 算出)

  4. 编排层  (resolver.py)  → 截面函数嵌套解算
     CsResolver.resolve()  — 单遍 bottom-up AST 变换
     自动发现 cs_* 前缀函数, 处理任意深度嵌套/组合

  依赖方向: 语法 ← 原语 + 变量 ← 编排
═══════════════════════════════════════════════════════════════

命名约定 (对齐 WQ101 行业标准):
  变量:   ALL_CAPS          CLOSE, REL_CLOSE, SECTOR_UP
  函数:   prefix_snake      ts_roc, cs_rank, expanding_std
  窗口:   参数名 d (day)      ts_mean(x, d)  ← 不对齐 WQ101 的 w
  统计:   样本 ddof=1         ts_std, ts_skew, cs_zscore

[重构] 2026-06-22 从 signals/v4 和 factor/v4 提取到 utils 公共层
"""

from .dsl import (
    parse_expression, evaluate, get_variables, get_functions,
    normalize_data_keys,
    DSLSecurityError, DSLSyntaxError,
)
from .registry import (
    FUNC_REGISTRY, is_valid_variable,
    register_function, register_variable,
    unregister_function, unregister_variable,
)
from .resolver import (
    CsResolver,
    _get_cs_functions,
    _has_any_cs,
    _is_outer_cs_rank_call,
    _eval_colwise,
)

__all__ = [
    # dsl
    'parse_expression', 'evaluate', 'get_variables', 'get_functions',
    'normalize_data_keys',
    'DSLSecurityError', 'DSLSyntaxError',
    # registry
    'FUNC_REGISTRY', 'is_valid_variable',
    'register_function', 'register_variable',
    'unregister_function', 'unregister_variable',
    # resolver
    'CsResolver', '_get_cs_functions',
    '_has_any_cs', '_is_outer_cs_rank_call', '_eval_colwise',
]
