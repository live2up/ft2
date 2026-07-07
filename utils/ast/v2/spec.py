# -*- coding: utf-8 -*-
"""
utils/ast/spec.py — AST 规格与构建器 (公共基础设施)
=============================================================================

在五层架构中的位置: 规格层 — 定义 AST 的结构约束与安全构建

用途:
  1. AST 节点构建器 — GP 引擎/LLM 用类型安全的方式构造 AST 树
  2. 表达式规范化器 — 统一表达式格式, 消除等价表达式的差异
  3. 语法规格 — LLM prompt 中引用, 明确定义"能写什么"

设计原则:
  - 构建器自带校验: 拒绝不合法的参数组合
  - 规范化器纯函数: 输入 AST → 输出规范化 AST, 不依赖全局状态
  - 规格文档化: 可直接作为 LLM system prompt 的一部分

[新增] 2026-06-22 AST 规范化
=============================================================================
"""
import ast
import operator
from typing import Optional, List, Union

from .registry import FUNC_REGISTRY, is_valid_variable, SAFE_CONSTANTS


# ============================================================
# AST 节点构建器 — 类型安全的树构造
# ============================================================

def make_var(name: str) -> ast.Name:
    """构建变量引用节点

    Args:
        name: 变量名 (大小写不敏感, 如 'CLOSE', 'RSI_14')

    Returns:
        ast.Name(id=name, ctx=Load())

    Raises:
        ValueError: 变量名未注册
    """
    if not is_valid_variable(name) and name not in SAFE_CONSTANTS:
        raise ValueError(
            f"变量 '{name}' 未注册。可用前缀见 VALID_VAR_PREFIXES"
        )
    return ast.Name(id=name, ctx=ast.Load())


def make_const(value: Union[int, float, bool]) -> ast.Constant:
    """构建常量节点"""
    if isinstance(value, bool):
        return ast.Constant(value=value)
    if isinstance(value, int):
        return ast.Constant(value=value)
    if isinstance(value, float):
        return ast.Constant(value=value)
    raise ValueError(f"不支持的常量类型: {type(value)}")


def make_call(func_name: str, *args, **kwargs) -> ast.Call:
    """构建函数调用节点

    Args:
        func_name: 注册的函数名 (如 'ts_roc', 'cs_rank')
        *args: 位置参数 (ast 节点)
        **kwargs: 关键字参数 (ast 节点)

    Returns:
        ast.Call(func=Name(func_name), args=[...], keywords=[...])

    Raises:
        ValueError: 函数未注册
    """
    if func_name not in FUNC_REGISTRY:
        raise ValueError(
            f"函数 '{func_name}' 未注册。"
            f"可用函数: {sorted(FUNC_REGISTRY.keys())}"
        )
    keywords = [ast.keyword(arg=k, value=v) for k, v in kwargs.items()]
    return ast.Call(
        func=ast.Name(id=func_name, ctx=ast.Load()),
        args=list(args),
        keywords=keywords,
    )


def make_binop(left: ast.AST, op: type, right: ast.AST) -> ast.BinOp:
    """构建二元运算节点

    Args:
        left, right: 左右子节点
        op: 运算符类型 (ast.Add, ast.Sub, ast.Mult, ast.Div, etc.)

    Usage:
        make_binop(make_var('CLOSE'), ast.Add, make_const(1.0))
    """
    return ast.BinOp(left=left, op=op(), right=right)


def make_unaryop(op: type, operand: ast.AST) -> ast.UnaryOp:
    """构建一元运算节点

    Args:
        op: 运算符类型 (ast.USub for -, ast.UAdd for +, ast.Not for not)
        operand: 操作数节点

    Usage:
        make_unaryop(ast.USub, make_var('CLOSE'))  # -CLOSE
        make_unaryop(ast.Not, make_var('CLOSE'))   # not CLOSE
    """
    return ast.UnaryOp(op=op(), operand=operand)


def make_compare(left: ast.AST, op: type, right: ast.AST) -> ast.Compare:
    """构建比较运算节点

    Args:
        left, right: 左右子节点
        op: 比较运算符类型 (ast.Gt, ast.Lt, ast.Eq, etc.)

    Usage:
        make_compare(make_var('CLOSE'), ast.Gt, make_var('OPEN'))
        # → CLOSE > OPEN
    """
    return ast.Compare(left=left, ops=[op()], comparators=[right])


def make_boolop(op: type, *values: ast.AST) -> ast.BoolOp:
    """构建布尔运算节点

    Args:
        op: 布尔运算符 (ast.And, ast.Or)
        *values: 参与运算的节点 (至少2个)

    Usage:
        make_boolop(ast.And, cond1, cond2, cond3)  # cond1 and cond2 and cond3
    """
    if len(values) < 2:
        raise ValueError("BoolOp 至少需要 2 个操作数")
    return ast.BoolOp(op=op(), values=list(values))


def make_ifexp(test: ast.AST, body: ast.AST, orelse: ast.AST) -> ast.IfExp:
    """构建三元表达式节点

    Usage:
        make_ifexp(cond, true_val, false_val)  # true_val if cond else false_val
    """
    return ast.IfExp(test=test, body=body, orelse=orelse)


# ============================================================
# 表达式规范化器 — 消除等价表达式的结构差异
# ============================================================

def normalize_expression(expr_str: str) -> str:
    """规范化表达式字符串

    执行以下变换:
      1. 统一变量名大小写 (ALL_CAPS)
      2. 统一空格 (删除多余空格)
      3. 统一括号风格 (冗余括号移除)

    Args:
        expr_str: 原始表达式字符串

    Returns:
        规范化后的表达式字符串

    Note:
        当前实现为字符串级规范化。后续可升级为 AST 级规范化
        (处理 not (a > b) → a <= b 等逻辑等价变换)。
    """
    # 解析确保语法正确
    from .dsl import parse_expression
    tree = parse_expression(expr_str)

    # 用 ast.unparse 重写 (Python 3.9+)
    normalized = ast.unparse(tree)

    return normalized


def normalize_ast(tree: ast.Expression) -> ast.Expression:
    """AST 级规范化: 重新解析 unparse 结果 (确保一致性)"""
    normalized_str = ast.unparse(tree)
    from .dsl import parse_expression
    return parse_expression(normalized_str)


# ============================================================
# 表达式自省 — 供 LLM 理解表达式结构
# ============================================================

def describe_expression(tree: ast.Expression) -> dict:
    """结构化描述表达式, 供 LLM 理解

    Returns:
        {
            'raw': str,               # 原始表达式
            'normalized': str,        # 规范化后
            'variables': [str],       # 引用的变量
            'functions': [str],       # 调用的函数
            'function_categories': {}, # 函数分类
            'complexity': {
                'depth': int,         # AST 深度
                'nodes': int,         # 节点总数
            },
            'structure': str,         # 结构类型: 'comparison', 'arithmetic', 'logic', 'ternary', 'mixed'
        }
    """
    from .dsl import get_variables, get_functions

    raw = ast.unparse(tree)
    normalized = ast.unparse(normalize_ast(tree))
    variables = get_variables(tree)
    functions = get_functions(tree)

    from .functions import get_func_category
    func_categories = {f: get_func_category(f) for f in functions}

    depth = _ast_depth(tree.body)
    nodes = sum(1 for _ in ast.walk(tree.body))

    structure = _classify_structure(tree.body)

    return {
        'raw': raw,
        'normalized': normalized,
        'variables': variables,
        'functions': functions,
        'function_categories': func_categories,
        'complexity': {'depth': depth, 'nodes': nodes},
        'structure': structure,
    }


def _ast_depth(node: ast.AST) -> int:
    """计算 AST 最大深度"""
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_ast_depth(c) for c in children)


def _classify_structure(node: ast.AST) -> str:
    """分类表达式结构类型"""
    has_compare = has_op = has_logic = has_ternary = has_arithmetic = False

    for n in ast.walk(node):
        if isinstance(n, ast.Compare):
            has_compare = True
        elif isinstance(n, ast.BoolOp):
            has_logic = True
        elif isinstance(n, ast.IfExp):
            has_ternary = True
        elif isinstance(n, ast.BinOp):
            has_arithmetic = True
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            has_op = True

    if has_ternary:
        return 'ternary'
    if has_logic and has_compare:
        return 'logic_comparison'
    if has_logic:
        return 'logic'
    if has_compare:
        return 'comparison'
    if has_arithmetic:
        return 'arithmetic'
    return 'simple'


# ============================================================
# 语法规格 — LLM Prompt 可直接引用
# ============================================================

AST_GRAMMAR_SPEC = """
## AST DSL 语法规格

### 基础规则
- 只允许数学表达式和函数调用
- 禁止: import, lambda, 属性访问(.), 下标([]), 循环, try/except
- 变量名: ALL_CAPS (如 CLOSE, RSI_14, BREADTH_S)
- 函数名: snake_case (如 ts_roc, cs_rank, expanding_mean)
- 窗口参数: 命名为 d (如 ts_mean(CLOSE, 20))
- 统计量: 样本标准差 ddof=1

### 支持的 Python 语法
```
# 算术: + - * / // % **
# 比较: > < >= <= == !=
# 逻辑: and or not
# 三元: a if cond else b
# 函数调用: func(arg1, arg2, ...)
# 括号分组: (a + b) * c
```

### 操作符语义 (量化专用)
- `and` → 逐元素 minimum (两者都 >0 才为 1)
- `or`  → 逐元素 maximum (任一 >0 即为 1)
- `not` → 逐元素取反 (0→1, >0→0)
- `/`   → 安全除法 (分母 ≈0 返回 0)
- `a if cond else b` → 逐元素选择

### 函数分类
1. **时序统计** (ts_*): ts_mean, ts_std, ts_sum, ts_max, ts_min,
   ts_median, ts_delta, ts_delay, ts_rank, ts_corr, ts_cov,
   ts_skew, ts_kurt, ts_argmax, ts_argmin, ts_roc, ts_zscore,
   ts_scale, ts_quantile, ts_av_diff, ts_decay_linear, ts_product,
   ts_var, ts_logret, ts_regression, ts_resid

2. **扩张统计** (expanding_*): expanding_mean, expanding_median,
   expanding_std, expanding_percentile

3. **截面算子** (cs_*): cs_rank, cs_zscore, cs_scale, cs_winsorize,
   cs_quantile
   注意: 截面函数需要完整 2D 面板数据, 不能在单品种择时中使用

4. **特征计算**: rsi, atr, atr_sma, macd, adx, cci, bb_width,
   stddev, ema, tsf, kama, trima, wma, dema, hv, natr, var,
   linearreg, vol_ratio, amt_ratio, wilder_smooth

5. **数学运算**: abs, log, sqrt, sign, exp, tanh, sigmoid, relu,
   signed_power, safe_max, safe_min

6. **信号确认**: persist(expr, n) — 连续 n 日同向才触发

### 典型表达式模式
- 趋势跟踪: ema(CLOSE, 20) > ts_mean(CLOSE, 50)
- 均值回归: rsi(CLOSE, 14) < -0.3 and ts_roc(CLOSE, 5) > 0
- 量价共振: vol_ratio(CLOSE, VOLUME, 5, 20) > 1.5 and ts_roc(CLOSE, 10) > 0
- 多因子融合: 0.5*((CLOSE/ts_mean(CLOSE,50)-1)*100) + 0.3*rsi(CLOSE,14) + 0.2*amt_ratio(AMOUNT,5,20)
- 状态切换: ema(CLOSE,20) if adx(HIGH,LOW,CLOSE,14)>25 else rsi(CLOSE,14)
- 信号确认: persist(atr(HIGH,LOW,CLOSE,7) > ts_mean(atr(HIGH,LOW,CLOSE,7),30), 2)
"""


def grammar_spec_for_llm() -> str:
    """返回可直接注入 LLM prompt 的语法规格"""
    return AST_GRAMMAR_SPEC.strip()


def grammar_spec_compact() -> str:
    """返回紧凑版语法规格 (减少 token)"""
    from .functions import FUNC_CATEGORIES
    from .variables import VAR_CATEGORIES

    lines = [
        "# AST DSL Compact Grammar",
        "",
        "## Syntax",
        "Python infix: + - * / > < >= <= and or not, a if cond else b",
        "Functions: func(arg1, arg2, kw=val)",
        "Variables: ALL_CAPS, window param: d",
        "",
        "## Functions (72 total)",
    ]
    for cat, names in FUNC_CATEGORIES.items():
        lines.append(f"  {cat}: {', '.join(names)}")

    lines.append("")
    lines.append("## Variables (70+ prefixes)")
    for cat, prefixes in VAR_CATEGORIES.items():
        lines.append(f"  {cat}: {', '.join(prefixes)}")

    return '\n'.join(lines)
