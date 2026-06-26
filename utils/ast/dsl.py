"""
utils/ast/dsl.py — 语法层 (公共基础设施)
=============================================================================

在四层架构中的位置: 第1层(语法) — 定义"能写什么"
  parse_expression()    — Python ast 解析 + 白名单安全校验
  evaluate()            — 递归求值 AST 节点
  normalize_data_keys() — 数据键 ALL_CAPS 规范化 (对齐 WQ101 行业标准)
  get_variables()       — 提取表达式中的变量名
  get_functions()       — 提取表达式中的函数名

安全策略:
  ALLOWED_NODE_TYPES    — 白名单 (BinOp, Compare, Call, Name, etc.)
  FORBIDDEN_NODE_TYPES  — 黑名单 (import, exec, lambda, loops, etc.)
  is_valid_variable()   — 变量名校验 (仅允许注册前缀)
  FUNC_REGISTRY lookup  — 函数名校验 (仅允许注册函数)

上游依赖:
  registry.py::FUNC_REGISTRY      — 函数注册表
  registry.py::VALID_VAR_PREFIXES — 变量前缀白名单

[重构] 2026-06-22 从 signals/v4 提取到 utils/ast 公共层
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
# 数据键规范化 — 应用端统一入口 (ALL_CAPS 约定)
# [新增] 2026-06-22 全大写规范化, 对齐 WQ101 行业标准
# ============================================================

def normalize_data_keys(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """[新增] 2026-06-22 规范化数据键名, 统一大小写

    规则:
      1. 所有键统一转为大写 (ALL_CAPS 约定, 对齐 WQ/聚宽 等行业标准)
      2. 所有值转为 float ndarray
      3. 冲突处理: 同名键 (如同时有 close 和 CLOSE) 后者覆盖前者

    业内规范:
      WorldQuant (WQ101/GT191)、聚宽、RiceQuant 等量化平台均采用
      ALL_CAPS 作为公式变量命名约定。数据源 (Yahoo/d2) 通常小写,
      在进入 AST 引擎前统一归一化。

    Args:
        data: 原始数据字典, 键名大小写不敏感

    Returns:
        规范化后的字典 {全部大写键: ndarray}
    """
    result = {}
    for k, v in data.items():
        arr = np.asarray(v, dtype=float)
        result[k.upper()] = arr
    return result

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


def _check_complexity(node: ast.AST, max_depth: int = 30, max_nodes: int = 500):
    """检查表达式复杂度
    [修正] 2026-06-25 max_depth 15→30, max_nodes 200→500.
    原限制 15 深度不足以支撑分层组合(多组 cs_rank 嵌套+算术)."""
    depth = _ast_depth(node)
    if depth > max_depth:
        raise DSLSecurityError(f"表达式深度 {depth} 超过上限 {max_depth}")
    
    node_count = sum(1 for _ in ast.walk(node))
    if node_count > max_nodes:
        raise DSLSecurityError(f"表达式节点数 {node_count} 超过上限 {max_nodes}")


def _ast_depth(node: ast.AST) -> int:
    """计算 AST 最大深度"""
    if not list(ast.iter_child_nodes(node)):
        return 1
    return 1 + max(_ast_depth(child) for child in ast.iter_child_nodes(node))


def parse_expression(expr_str: str, 
                     max_depth: int = 30,
                     max_nodes: int = 500) -> ast.Expression:
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
    """提取表达式中引用的所有变量名

    排除安全常量 (True/False/None/pi/e) 和已注册函数名
    (避免 vol_ratio/atr 等同名变量被误判)。
    """
    vars_set = set()
    for node in ast.walk(tree.body):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = node.id
            if name in SAFE_CONSTANTS or name.lower() in FUNC_REGISTRY:
                continue  # 跳过常量和函数名
            if is_valid_variable(name):
                vars_set.add(name.upper())
    return sorted(vars_set)


def get_functions(tree: ast.Expression) -> list:
    """提取表达式中调用的所有函数名"""
    funcs = set()
    for node in ast.walk(tree.body):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            funcs.add(node.func.id)
    return sorted(funcs)


def validate_expression(expr_str: str, available_vars: list = None) -> dict:
    """LLM 前置校验: 表达式语法 + 变量存在性

    不依赖数据求值, 只做 parse + 变量对比。
    适合 LLM 生成表达式后立即调用, 避免等到 evaluate() 才发现 KeyError。

    Args:
        expr_str: 表达式字符串
        available_vars: 可用变量名列表 (大小写不敏感), None 则跳过变量存在性检查

    Returns:
        {'valid': bool, 'errors': [str], 'variables': [str], 'missing_vars': [str]}
    """
    result = {'valid': True, 'errors': [], 'variables': [], 'missing_vars': []}
    try:
        tree = parse_expression(expr_str)
        result['variables'] = get_variables(tree)
    except (DSLSyntaxError, DSLSecurityError) as e:
        result['valid'] = False
        result['errors'].append(str(e))
        return result

    if available_vars is not None:
        avail_upper = {v.upper() for v in available_vars}
        for var in result['variables']:
            if var.upper() not in avail_upper:
                result['missing_vars'].append(var)
                result['valid'] = False
                result['errors'].append(
                    f"变量 '{var}' 不在可用数据中。"
                    f"可用: {sorted(avail_upper)}"
                )
    return result


# ============================================================
# 面板逐列求值器 — 时序函数在 1D 上安全求值
# ============================================================

def eval_colwise(tree: ast.Expression, data: Dict[str, np.ndarray],
                 T: int, N: int, strict: bool = False) -> np.ndarray:
    """逐列求值 AST — 时序函数在 1D 上安全求值

    设计理由:
      _rolling / _expanding / _persist 只处理 1D 数组,
      因此 2D 面板必须逐列调用 evaluate()。

    Args:
        strict: False=静默返回NaN (批量回测/搜索模式),
                True=异常立即抛出 (调试模式, 带列号定位)

    [重构] 2026-06-22 从 resolver.py 移动到 dsl.py (通用工具)
    """
    result = np.full((T, N), np.nan)
    for j in range(N):
        col_data = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                col_data[k] = v[:, j]
            else:
                col_data[k] = v
        try:
            col_result = evaluate(tree, col_data)
            if isinstance(col_result, np.ndarray):
                result[:, j] = col_result[:T].ravel()
            elif isinstance(col_result, (int, float, np.integer, np.floating)):
                result[:, j] = float(col_result)
        except Exception as e:
            if strict:
                raise RuntimeError(
                    f"eval_colwise 第 {j} 列求值失败: {e}"
                ) from e
            result[:, j] = np.nan
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def cross_sectional_rank(vals: np.ndarray) -> np.ndarray:
    """每日截面排名 → 0~1 (与 signals/v4 cs_rank 逻辑一致)

    [修正] 2026-06-25 改为 method='min', 对齐 WQ/DolphinDB 行业标准.
    [重构] 2026-06-22 从 resolver.py 移动到 dsl.py (通用工具)
    """
    from scipy.stats import rankdata
    vals = np.asarray(vals, dtype=float)
    if vals.ndim != 2:
        return vals
    T, N = vals.shape
    ranked = np.full((T, N), 0.0)
    for t in range(T):
        row = vals[t]
        valid = ~np.isnan(row)
        if valid.sum() > 0:
            rk = rankdata(row[valid], method='min')
            ranked[t, valid] = np.nan_to_num(
                rk / valid.sum(), nan=0.5, posinf=0.5, neginf=0.5)
    return ranked


# ============================================================
# 表达式自省扩展 — 结构化描述
# ============================================================

def ast_depth(tree: ast.Expression) -> int:
    """计算 AST 最大深度"""
    def _depth(node):
        children = list(ast.iter_child_nodes(node))
        if not children:
            return 1
        return 1 + max(_depth(c) for c in children)
    return _depth(tree.body)


def ast_node_count(tree: ast.Expression) -> int:
    """计算 AST 节点总数"""
    return sum(1 for _ in ast.walk(tree.body))
