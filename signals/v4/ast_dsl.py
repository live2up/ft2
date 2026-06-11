"""
signals/v4/ast_dsl.py — Python AST DSL 引擎
=============================================================================
用 Python 内置 ast 模块做安全解析和求值，无需自研 Parser。

架构：三层
  1. parse()    — ast.parse() 解析表达式字符串
  2. validate() — 白名单检查（函数名、变量名、节点类型）
  3. evaluate() — 递归求值 AST → numpy 数组

用法：
  >>> from signals.v4.ast_dsl import parse_expression
  >>> tree = parse_expression("CLOSE / ts_mean(CLOSE, 50) - 1")
  >>> signal = evaluate(tree, data_dict)  # -> np.ndarray
=============================================================================
"""
import ast
import operator
import numpy as np
from typing import Dict, Any, Union

from .registry import (
    FUNC_REGISTRY, SAFE_CONSTANTS,
    is_valid_variable, VALID_VAR_PREFIXES,
)

# ============================================================
# 安全白名单 — 允许的 AST 节点类型
# ============================================================

ALLOWED_NODE_TYPES = {
    ast.Expression,     # 顶层
    ast.BinOp,          # + - * / // % **
    ast.UnaryOp,        # -x, +x, not x
    ast.BoolOp,         # and, or
    ast.Compare,        # > < >= <= == !=
    ast.IfExp,          # a if cond else b
    ast.Call,           # 函数调用
    ast.Name,           # 变量引用
    ast.Constant,       # 常量
    ast.Load,           # 加载上下文
    ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot,
    ast.keyword,        # 关键字参数
}

# 禁止的节点类型（如有则直接拒绝）
_FORBIDDEN_TYPES = {
    ast.Import, ast.ImportFrom,
    ast.Attribute,
    ast.Subscript,
    ast.Lambda,
    ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
    ast.Yield, ast.YieldFrom,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    ast.Global, ast.Nonlocal,
    ast.Await, ast.AsyncFor, ast.AsyncWith,
    ast.Raise, ast.Try, ast.Assert,
    ast.Delete, ast.Pass, ast.Break, ast.Continue,
    ast.Return,
    ast.With, ast.AsyncWith,
    ast.For, ast.AsyncFor, ast.While,
    ast.If,  # 只允许 IfExp（三元），不允许 If 语句块
    ast.JoinedStr, ast.FormattedValue,
}

# Python 3.10+ 的 match-case
if hasattr(ast, 'Match'):
    _FORBIDDEN_TYPES.add(ast.Match)

# Python 3.12 之前有 Exec
if hasattr(ast, 'Exec'):
    _FORBIDDEN_TYPES.add(ast.Exec)

FORBIDDEN_NODE_TYPES = frozenset(_FORBIDDEN_TYPES)

# ============================================================
# 运算符映射
# ============================================================

BINOP_MAP = {
    ast.Add:      operator.add,
    ast.Sub:      operator.sub,
    ast.Mult:     operator.mul,
    ast.Div:      lambda a, b: np.where(np.abs(b) > 1e-10, a / b, 0.0),
    ast.FloorDiv: lambda a, b: np.floor(np.where(np.abs(b) > 1e-10, a / b, 0.0)),
    ast.Mod:      lambda a, b: np.where(np.abs(b) > 1e-10, a % b, 0.0),
    ast.Pow:      lambda a, b: np.power(np.clip(a, -1e6, 1e6), np.clip(b, -10, 10)),
}

UNARYOP_MAP = {
    ast.USub:   operator.neg,
    ast.UAdd:   operator.pos,
    ast.Not:    lambda x: np.where(x == 0, 1.0, 0.0),
    ast.Invert: operator.inv,
}

CMPOP_MAP = {
    ast.Eq:    lambda a, b: np.where(np.abs(a - b) < 1e-10, 1.0, 0.0),
    ast.NotEq: lambda a, b: np.where(np.abs(a - b) >= 1e-10, 1.0, 0.0),
    ast.Lt:    lambda a, b: np.where(a < b, 1.0, 0.0),
    ast.LtE:   lambda a, b: np.where(a <= b, 1.0, 0.0),
    ast.Gt:    lambda a, b: np.where(a > b, 1.0, 0.0),
    ast.GtE:   lambda a, b: np.where(a >= b, 1.0, 0.0),
    ast.Is:    lambda a, b: np.where(np.abs(a - b) < 1e-10, 1.0, 0.0),
    ast.IsNot: lambda a, b: np.where(np.abs(a - b) >= 1e-10, 1.0, 0.0),
}

# ============================================================
# 解析 + 校验
# ============================================================

class DSLSecurityError(Exception):
    """安全校验失败"""
    pass

class DSLSyntaxError(Exception):
    """语法错误"""
    pass


def _check_node_safety(node: ast.AST):
    """递归检查 AST 节点安全性"""
    # 禁止节点检查
    for forbidden in FORBIDDEN_NODE_TYPES:
        if isinstance(node, forbidden):
            raise DSLSecurityError(
                f"禁止的节点类型: {type(node).__name__}。"
                f"只允许数学表达式和函数调用。"
            )
    
    # 白名单检查
    if not any(isinstance(node, allowed) for allowed in ALLOWED_NODE_TYPES):
        raise DSLSecurityError(
            f"不允许的节点类型: {type(node).__name__}"
        )
    
    # 额外检查：函数调用必须是白名单
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in FUNC_REGISTRY:
                raise DSLSecurityError(
                    f"未注册的函数: '{func_name}'。"
                    f"可用函数: {sorted(FUNC_REGISTRY.keys())}"
                )
        else:
            raise DSLSecurityError("只允许直接函数调用，不支持属性/下标调用")
    
    # 变量名检查（跳过函数名——它们在 Call 节点中已校验）
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        name = node.id
        if name in SAFE_CONSTANTS or name in FUNC_REGISTRY:
            return  # 安全常量 / 注册函数名通过
        if not is_valid_variable(name):
            raise DSLSecurityError(
                f"未注册的变量: '{name}'。"
                f"变量名必须匹配前缀: {VALID_VAR_PREFIXES}"
            )
    
    # 递归检查子节点
    for child in ast.iter_child_nodes(node):
        _check_node_safety(child)


def _check_complexity(node: ast.AST, max_depth: int = 15, max_nodes: int = 200):
    """检查表达式复杂度"""
    # 深度
    depth = _ast_depth(node)
    if depth > max_depth:
        raise DSLSecurityError(f"表达式深度 {depth} 超过上限 {max_depth}")
    
    # 节点数
    node_count = sum(1 for _ in ast.walk(node))
    if node_count > max_nodes:
        raise DSLSecurityError(f"表达式节点数 {node_count} 超过上限 {max_nodes}")


def _ast_depth(node: ast.AST) -> int:
    """计算 AST 最大深度"""
    if not list(ast.iter_child_nodes(node)):
        return 1
    return 1 + max(_ast_depth(child) for child in ast.iter_child_nodes(node))


def parse_expression(expr_str: str, 
                     max_depth: int = 15,
                     max_nodes: int = 200) -> ast.Expression:
    """
    解析并校验表达式字符串
    
    Returns:
        ast.Expression 节点
    
    Raises:
        DSLSyntaxError: 语法错误
        DSLSecurityError: 安全/语义校验失败
    """
    # 清理
    expr_str = expr_str.strip()
    if not expr_str:
        raise DSLSyntaxError("表达式为空")
    
    # 解析
    try:
        tree = ast.parse(expr_str, mode='eval')
    except SyntaxError as e:
        raise DSLSyntaxError(f"Python 语法错误: {e.msg} (位置: 行{e.lineno}, 列{e.offset})")
    
    # 三层校验
    _check_node_safety(tree.body)
    _check_complexity(tree.body, max_depth, max_nodes)
    
    return tree


# ============================================================
# 求值器
# ============================================================

def _unwrap_scalar(val: np.ndarray):
    """将标量数组解包为 Python 原生类型（int 或 float）"""
    if val.size != 1:
        return val
    v = val.item()
    # 检查原始 AST Constant 是否整数，保持 int 类型
    if isinstance(v, float) and v == int(v) and abs(v) < 1e12:
        return int(v)
    return v

def evaluate(tree: ast.Expression, 
             data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    求值 AST 表达式
    
    Args:
        tree: parse_expression() 返回的 AST
        data: 数据字典 {变量名: np.ndarray}
              包含原始 OHLCV (CLOSE, OPEN, HIGH, LOW, VOLUME)
              和预计算特征 (RSI_14, ATR_7, EMA_20, ...)
    
    Returns:
        np.ndarray，形状与数据长度一致
    """
    return _eval_node(tree.body, data)


def _eval_node(node: ast.AST, data: Dict[str, np.ndarray]) -> np.ndarray:
    """递归求值 AST 节点"""
    
    # 常量
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None:
            return np.array([0.0])
        if isinstance(value, bool):
            return np.array([1.0 if value else 0.0])
        if isinstance(value, (int, float)):
            return np.array([float(value)])
        raise DSLSecurityError(f"不支持的常量类型: {type(value)}")
    
    # 变量引用
    if isinstance(node, ast.Name):
        name = node.id
        # 安全常量（True/False/None/pi/e）
        if name in SAFE_CONSTANTS:
            return np.array([SAFE_CONSTANTS[name]])
        # 数据变量
        name_upper = name.upper()
        if name_upper in data:
            return np.asarray(data[name_upper], dtype=float).copy()
        if name in data:
            return np.asarray(data[name], dtype=float).copy()
        # 模糊匹配（允许大小写容错）
        for key in data:
            if key.upper() == name_upper:
                return np.asarray(data[key], dtype=float).copy()
        raise KeyError(
            f"变量 '{name}' 不在数据字典中。"
            f"可用变量: {sorted(data.keys())}"
        )
    
    # 二元运算: a + b, a * b, etc.
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, data)
        right = _eval_node(node.right, data)
        op_func = BINOP_MAP.get(type(node.op))
        if op_func is None:
            raise DSLSecurityError(f"不支持的二元运算: {type(node.op).__name__}")
        return op_func(left, right)
    
    # 一元运算: -x, +x, not x
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, data)
        op_func = UNARYOP_MAP.get(type(node.op))
        if op_func is None:
            raise DSLSecurityError(f"不支持的一元运算: {type(node.op).__name__}")
        return op_func(operand)
    
    # 比较运算: a > b, a < b, a == b, etc.
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise DSLSecurityError("只支持单一比较运算（如 a > b，不支持 a > b > c）")
        left = _eval_node(node.left, data)
        right = _eval_node(node.comparators[0], data)
        op_func = CMPOP_MAP.get(type(node.ops[0]))
        if op_func is None:
            raise DSLSecurityError(f"不支持的比较运算: {type(node.ops[0]).__name__}")
        return op_func(left, right)
    
    # 布尔运算: a and b, a or b
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, data) for v in node.values]
        if isinstance(node.op, ast.And):
            result = np.ones_like(values[0])
            for v in values:
                result = np.minimum(result, v)  # AND: 两者都 >0
            return result
        elif isinstance(node.op, ast.Or):
            result = np.zeros_like(values[0])
            for v in values:
                result = np.maximum(result, v)  # OR: 任一 >0
            return result
    
    # 三元表达式: a if cond else b
    if isinstance(node, ast.IfExp):
        cond = _eval_node(node.test, data)
        a_val = _eval_node(node.body, data)
        b_val = _eval_node(node.orelse, data)
        return np.where(cond > 0, a_val, b_val)
    
    # 函数调用
    if isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name is None or func_name not in FUNC_REGISTRY:
            raise DSLSecurityError(f"未注册的函数: {func_name}")
        
        func = FUNC_REGISTRY[func_name]
        # 位置参数（自动解包标量数组为 Python 原生类型）
        args = []
        for arg_node in node.args:
            val = _eval_node(arg_node, data)
            args.append(_unwrap_scalar(val))
        # 关键字参数
        kwargs = {}
        for kw in node.keywords:
            val = _eval_node(kw.value, data)
            kwargs[kw.arg] = _unwrap_scalar(val)
        
        return func(*args, **kwargs)
    
    raise DSLSecurityError(f"未处理的节点类型: {type(node).__name__}")


# ============================================================
# 工具函数
# ============================================================

def get_variables(tree: ast.Expression) -> list:
    """提取表达式中引用的所有变量名"""
    vars_set = set()
    for node in ast.walk(tree.body):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = node.id
            if name not in SAFE_CONSTANTS and is_valid_variable(name):
                vars_set.add(name.upper())
    return sorted(vars_set)


def get_functions(tree: ast.Expression) -> list:
    """提取表达式中调用的所有函数名"""
    funcs = set()
    for node in ast.walk(tree.body):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            funcs.add(node.func.id)
    return sorted(funcs)
