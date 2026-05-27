"""
factor/v2/expression.py — 因子表达式引擎
=============================================================================

将因子逻辑表达为字符串公式，编译为 AST 后在面板数据上求值。

核心抽象：
  因子表达式字符串 → Tokenizer → Parser → AST → 因子值 Panel (T天 × N行业)

与 signals/v2/expression.py 的关系：
  - signals 的 Expression 操作单标的日线 Series
  - factor 的 FactorExpression 操作面板数据 (T, N) ndarray
  - 语法设计上保持一致的函数调用风格

============================================================================
                         表达式语法
============================================================================

终端变量:
  open, high, low, close, volume, amount
  直接引用面板中的 OHLCV 列，每个返回 (T, N) ndarray

常量: 0, -1, 20, 0.5, ...

一元函数 (1 个参数):
  abs(x), sqrt(x), log(x), neg(x)

二元函数 (2 个参数):
  add(x, y), sub(x, y), mul(x, y), div(x, y), max(x, y), min(x, y)

时序原语 (含 period 超参数):
  ts_rank(x, period)      时序排名
  ts_zscore(x, period)    时序标准化
  delay(x, period)        N 日前值
  decay_linear(x, period) 线性衰减加权平均
  cs_zscore(x, period)    截面 Z-score（去均值除标准差）

截面原语:
  cs_rank(x)              横截面排名（每行内排名）
  signed_power(x, exp)    有符号幂

使用示例:
  >>> expr = FactorExpression("ts_rank(sub(close, delay(close, 20)), 10)")
  >>> panel = expr.evaluate(data_dict)
  >>> # panel.shape = (1455, 31)
  >>> expr.to_str()  → "ts_rank(sub(close, delay(close, 20)), 10)"

============================================================================
"""

import re
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# 复用 gp_primitives 中的时序原语
from .gp_primitives import (
    ts_rank, ts_zscore, delay, decay_linear,
    cs_rank, signed_power,
)


# ============================================================
# AST 节点定义
# ============================================================

class NodeType(Enum):
    VARIABLE = 'variable'      # 终端变量: close, volume, ...
    CONSTANT = 'constant'       # 数值常量: 20, 0.5, -1
    FUNCTION = 'function'       # 函数调用: add(x,y), ts_rank(x,10)
    UNARY = 'unary'            # 一元函数: abs(x), sqrt(x)


class ASTNode:
    """AST 节点

    Attributes:
        node_type: 节点类型
        value: 节点值
            - VARIABLE: str, 变量名 'close'
            - CONSTANT: float, 数值
            - FUNCTION: str, 函数名 'ts_rank'
            - UNARY: str, 函数名 'abs'
        children: 子节点列表
        params: 额外参数字典（如 period=10）
    """

    def __init__(self, node_type: NodeType, value: Any,
                 children: List['ASTNode'] = None,
                 params: Dict[str, Any] = None):
        self.node_type = node_type
        self.value = value
        self.children = children or []
        self.params = params or {}

    def add_child(self, child: 'ASTNode'):
        self.children.append(child)

    def __repr__(self):
        if self.node_type == NodeType.VARIABLE:
            return f"VAR({self.value})"
        elif self.node_type == NodeType.CONSTANT:
            return f"CONST({self.value})"
        elif self.node_type == NodeType.UNARY:
            return f"UNARY({self.value})"
        else:
            args = ', '.join(repr(c) for c in self.children)
            extras = f", {self.params}" if self.params else ""
            return f"FUNC({self.value}, [{args}]{extras})"

    def to_str(self) -> str:
        """将 AST 节点转回表达式字符串"""
        if self.node_type == NodeType.VARIABLE:
            return str(self.value)
        elif self.node_type == NodeType.CONSTANT:
            return str(self.value)
        elif self.node_type == NodeType.UNARY:
            return f"{self.value}({self.children[0].to_str()})"
        else:
            args = ', '.join(c.to_str() for c in self.children)
            extras = []
            for k, v in self.params.items():
                extras.append(f"{k}={v}" if not isinstance(v, (int, float)) or v >= 0 else str(v))
            all_args = [args] + [str(v) for v in self.params.values()]
            return f"{self.value}({', '.join(all_args)})"


# ============================================================
# 求值引擎
# ============================================================

# 终端变量 → 面板数据 col 名映射
VARIABLE_MAP = {
    'open': 'open', 'high': 'high', 'low': 'low',
    'close': 'close', 'volume': 'volume', 'amount': 'amount',
}

# 一元函数实现（纯 numpy，操作 (T,N) 数组）
UNARY_FUNCTIONS = {
    'abs': lambda x: np.abs(x),
    'sqrt': lambda x: np.sqrt(np.maximum(x, 0)),
    'log': lambda x: np.log(np.maximum(x, 1e-10)),
    'neg': lambda x: -x,
    'sign': lambda x: np.sign(x),
}

# 二元算术函数
BINARY_FUNCTIONS = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: np.where(np.abs(y) > 1e-10, x / y, 0.0),
    'max': lambda x, y: np.maximum(x, y),
    'min': lambda x, y: np.minimum(x, y),
}

# 时序/截面原语（带超参数）
PRIMITIVE_FUNCTIONS = {
    'ts_rank': ts_rank,
    'ts_zscore': ts_zscore,
    'delay': delay,
    'decay_linear': decay_linear,
    'cs_rank': cs_rank,
    'signed_power': signed_power,
}


def cs_zscore(x: np.ndarray, period: int = None) -> np.ndarray:
    """截面 Z-score 标准化：每行 (行业间) 做 (x - mean) / std

    [新增] 2026-05-28  因子表达式引擎截面原语
    std=0 时不做变换（返回全 0）。
    period 参数预留用于滚动截面 zscore，当前未使用。
    """
    mean = np.nanmean(x, axis=1, keepdims=True)
    std = np.nanstd(x, axis=1, keepdims=True)
    std_safe = np.where(std < 1e-10, 1.0, std)
    return (x - mean) / std_safe


PRIMITIVE_FUNCTIONS['cs_zscore'] = cs_zscore


def evaluate_node(node: ASTNode, data: Dict[str, np.ndarray]) -> np.ndarray:
    """递归求值 AST 节点，返回 (T, N) ndarray

    Args:
        node: AST 节点
        data: 面板数据字典 {col_name: ndarray(T, N)}

    Returns:
        np.ndarray: shape (T, N) 的计算结果
    """
    if node.node_type == NodeType.VARIABLE:
        col = VARIABLE_MAP.get(node.value.lower())
        if col is None or col not in data:
            raise KeyError(f"终端变量 '{node.value}' 不存在于数据中，"
                           f"可用: {list(data.keys())}")
        return np.asarray(data[col], dtype=float)

    elif node.node_type == NodeType.CONSTANT:
        return np.full(
            _infer_shape(data), float(node.value), dtype=float
        )

    elif node.node_type == NodeType.UNARY:
        child_val = evaluate_node(node.children[0], data)
        fn = UNARY_FUNCTIONS.get(node.value)
        if fn is None:
            raise ValueError(f"未知一元函数: {node.value}")
        return fn(child_val)

    elif node.node_type == NodeType.FUNCTION:
        fn_name = node.value

        # 优先查内置函数
        if fn_name in BINARY_FUNCTIONS:
            vals = [evaluate_node(c, data) for c in node.children]
            return BINARY_FUNCTIONS[fn_name](*vals)

        # 时序/截面原语
        if fn_name in PRIMITIVE_FUNCTIONS:
            prim_fn = PRIMITIVE_FUNCTIONS[fn_name]
            vals = [evaluate_node(c, data) for c in node.children]

            # 注入超参数（如 period, exponent）
            if node.params:
                return prim_fn(*vals, **node.params)

            # signed_power 需要 exponent 参数
            if fn_name == 'signed_power':
                return prim_fn(vals[0], exponent=2.0)

            return prim_fn(*vals)

        raise ValueError(f"未知函数: {fn_name}")

    raise ValueError(f"未知节点类型: {node.node_type}")


def _infer_shape(data: Dict[str, np.ndarray]) -> Tuple[int, int]:
    """从数据字典推断面板形状 (T, N)"""
    for arr in data.values():
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return arr.shape
    return (1, 1)


# ============================================================
# Tokenizer（词法分析）
# ============================================================

TOKEN_SPEC = [
    ('NUMBER',   r'\d+\.?\d*'),
    ('NAME',     r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('LPAREN',   r'\('),
    ('RPAREN',   r'\)'),
    ('COMMA',    r','),
    ('EQUALS',   r'='),
    ('WHITESPACE', r'\s+'),
]

TOKEN_RE = re.compile(
    '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPEC)
)


def tokenize(expr_str: str) -> List[Tuple[str, str]]:
    """词法分析：表达式字符串 → token 列表"""
    tokens = []
    for m in TOKEN_RE.finditer(expr_str):
        kind = m.lastgroup
        value = m.group()
        if kind == 'WHITESPACE':
            continue
        tokens.append((kind, value))
    return tokens


# ============================================================
# Parser（递归下降语法分析）
# ============================================================

class Parser:
    """递归下降语法分析器

    语法规则:
        expr      → function_call | variable | constant | unary_call
        func_call → NAME '(' args ')'
        unary_call→ NAME '(' expr ')'
        args      → expr (',' expr)*
        params    → NAME '=' NUMBER (',' NAME '=' NUMBER)*
    """

    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Tuple[str, str]]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected_kind: str = None) -> Tuple[str, str]:
        if self.pos >= len(self.tokens):
            raise SyntaxError(f"意外结束，期望 {expected_kind or 'token'}")
        token = self.tokens[self.pos]
        if expected_kind and token[0] != expected_kind:
            raise SyntaxError(
                f"期望 {expected_kind}，实际 {token[0]}({token[1]})，pos={self.pos}"
            )
        self.pos += 1
        return token

    def parse(self) -> ASTNode:
        node = self.parse_expr()
        if self.pos < len(self.tokens):
            raise SyntaxError(
                f"多余 token: {self.tokens[self.pos]}，pos={self.pos}"
            )
        return node

    def parse_expr(self) -> ASTNode:
        token = self.peek()
        if token is None:
            raise SyntaxError("空表达式")

        kind, value = token

        # 函数调用: name(args, params)
        if kind == 'NAME':
            # lookahead: 下一个 token 是 '(' 吗？
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1][0] == 'LPAREN':
                return self.parse_function_call()
            # 否则是终端变量
            self.consume('NAME')
            return ASTNode(NodeType.VARIABLE, value)

        # 数值常量
        if kind == 'NUMBER':
            self.consume('NUMBER')
            val = float(value) if '.' in value else int(value)
            return ASTNode(NodeType.CONSTANT, val)

        raise SyntaxError(f"意外 token: {token}，pos={self.pos}")

    def parse_function_call(self) -> ASTNode:
        """解析函数调用: name(expr, expr, ...)"""
        name = self.consume('NAME')[1]
        self.consume('LPAREN')

        children = []
        params = {}

        # 解析参数列表
        if self.peek() and self.peek()[0] != 'RPAREN':
            while True:
                # 检查是否为关键字参数 (name=value)
                if (self.pos + 2 < len(self.tokens)
                        and self.tokens[self.pos][0] == 'NAME'
                        and self.tokens[self.pos + 1][0] == 'EQUALS'
                        and self.tokens[self.pos + 2][0] == 'NUMBER'):
                    pname = self.consume('NAME')[1]
                    self.consume('EQUALS')
                    pval_str = self.consume('NUMBER')[1]
                    pval = float(pval_str) if '.' in pval_str else int(pval_str)
                    params[pname] = pval
                else:
                    children.append(self.parse_expr())

                if self.peek() and self.peek()[0] == 'COMMA':
                    self.consume('COMMA')
                else:
                    break

        self.consume('RPAREN')

        # 一元函数
        if name in UNARY_FUNCTIONS and len(children) == 1:
            return ASTNode(NodeType.UNARY, name, children)

        # ndarray 命名特殊处理
        arr_name_map = {'ndarray': 'ndarray', 'array': 'ndarray'}

        # 带超参数的 pritimive 函数
        node = ASTNode(NodeType.FUNCTION, name, children, params)
        return node


# ============================================================
# FactorExpression — 顶层接口
# ============================================================

class FactorExpression:
    """因子表达式 — 符号回归输出的一等公民

    字符串表达式 → 编译 → 面板因子值

    Example:
        >>> expr = FactorExpression(
        ...     "ts_rank(sub(close, delay(close, 20)), 10)"
        ... )
        >>> # expr.terminals  → {'close'}
        >>> # expr.functions  → {'ts_rank', 'sub', 'delay'}
        >>> # expr.depth      → 4
        >>> panel = expr.evaluate(data_dict)  # → (T, N) ndarray

    Attributes:
        source: 原始表达式字符串
        ast: 编译后的 AST 根节点
        terminals: 使用的终端变量集合
        functions: 使用的函数名集合
        depth: AST 树深度
        node_count: AST 节点总数（衡量复杂度）
    """

    def __init__(self, expression_str: str):
        """编译表达式字符串

        Args:
            expression_str: 因子表达式，如
                "ts_rank(sub(close, delay(close, 20)), 10)"

        Raises:
            SyntaxError: 表达式语法错误
        """
        self.source = expression_str
        tokens = tokenize(expression_str)
        parser = Parser(tokens)
        self.ast = parser.parse()

        # 自省信息
        self.terminals = self._collect_terminals(self.ast)
        self.functions = self._collect_functions(self.ast)
        self.depth = self._calc_depth(self.ast)
        self.node_count = self._count_nodes(self.ast)

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """在面板数据上求值表达式

        Args:
            data: {col_name: ndarray(T, N)} 的面板数据字典
                  必须包含 terminals 中引用的所有列

        Returns:
            np.ndarray: shape (T, N) 的因子值面板
        """
        result = evaluate_node(self.ast, data)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def to_dataframe(self, data: Dict[str, np.ndarray],
                     index=None, columns=None) -> 'pd.DataFrame':
        """求值并包装为 DataFrame"""
        import pandas as pd
        values = self.evaluate(data)
        return pd.DataFrame(values, index=index, columns=columns)

    def to_str(self) -> str:
        """转回表达式字符串"""
        return self.ast.to_str()

    def __repr__(self):
        return f"FactorExpression({self.source!r})"

    def __str__(self):
        return self.source

    # ---- 自省辅助 ----

    @staticmethod
    def _collect_terminals(node: ASTNode) -> set:
        result = set()
        if node.node_type == NodeType.VARIABLE:
            result.add(node.value)
        for child in node.children:
            result |= FactorExpression._collect_terminals(child)
        return result

    @staticmethod
    def _collect_functions(node: ASTNode) -> set:
        result = set()
        if node.node_type in (NodeType.FUNCTION, NodeType.UNARY):
            result.add(node.value)
        for child in node.children:
            result |= FactorExpression._collect_functions(child)
        return result

    @staticmethod
    def _calc_depth(node: ASTNode) -> int:
        if not node.children:
            return 1
        return 1 + max(FactorExpression._calc_depth(c) for c in node.children)

    @staticmethod
    def _count_nodes(node: ASTNode) -> int:
        return 1 + sum(FactorExpression._count_nodes(c) for c in node.children)


# ============================================================
# 便捷函数
# ============================================================

def parse_expression(expr_str: str) -> FactorExpression:
    """解析因子表达式字符串"""
    return FactorExpression(expr_str)
