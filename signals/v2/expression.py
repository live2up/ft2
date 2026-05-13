"""
signals/v2/expression.py — 统一表达式引擎
=============================================================================

核心抽象：Expression 是信号的一等公民。
传统指标和 GP 表达式用同一套语言描述，编译为可执行的信号生成器。


============================================================================
                         架构层级（竖式）
============================================================================

 ┌─────────────────────────────────────────────────────────────────────┐
 │  输入：表达式字符串                                                   │
 │  "thr_mean(ATR(7)) & thr_mean(TRIMA(60))"                           │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Tokenizer（词法分析）                                                │
 │                                                                      │
 │  "thr_mean(ATR(7))" → [NAME("thr_mean"), LPAREN,                     │
 │                        NAME("ATR"), LPAREN, NUMBER("7"), RPAREN, RPAREN] │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Parser（递归下降语法分析）→ AST 树                                   │
 │                                                                      │
 │  THR_MEAN                                                           │
 │    └── FEATURE("ATR", [7])  ─→ 查找 DataFrame 列: ATR_7             │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  AST 节点类型（7 种）                                                 │
 │                                                                      │
 │  ┌──────────────┬──────────────────────────────────────────────┐    │
 │  │   节点类型    │    说明 / 语法                                │    │
 │  ├──────────────┼──────────────────────────────────────────────┤    │
 │  │ FEATURE      │ 特征引用    ATR(7) → 查找列 ATR_7              │    │
 │  │ CONSTANT     │ 常量        -1.0, 0.5, 2.0                     │    │
 │  │ OPERATOR     │ 二元运算    +, -, *, /, &, |, >, <            │    │
 │  │ FUNCTION     │ 一元函数    abs, sqrt, sigmoid, tanh, relu    │    │
 │  │ THRESHOLD    │ 阈值化      thr_0 / thr_mean / thr_med        │    │
 │  │ PERSIST      │ 信号确认    persist(expr, n) 连续N日          │    │
 │  │ SWITCH       │ 条件分支    if_then(cond, a, b)               │    │
 │  └──────────────┴──────────────────────────────────────────────┘    │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  TreeNode.evaluate(feature_data) → 信号序列 (np.ndarray)             │
 │                                                                      │
 │  递归求值：叶子节点取特征值 → 逐层向上计算 → 输出信号数组              │
└────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Expression.generate(data) → 最终信号 (pd.Series)                    │
 │                                                                      │
 │  1. 调用 FeatureSpace.fit_transform(data) → 获取特征矩阵             │
 │  2. 调用 tree.evaluate(feature_data) → 生成信号数组                  │
 │  3. 返回 pd.Series(signals, index=data.index)                        │
 └─────────────────────────────────────────────────────────────────────┘


============================================================================
                         参数规范速查
============================================================================

 ┌──────────────────────────┬───────────────────────────────────────┐
 │       语法元素            │       说明 / 示例                      │
 ├──────────────────────────┼───────────────────────────────────────┤
 │ 特征引用 (函数调用式)     │ ATR(7)  → 列名 ATR_7                  │
 │                          │ MA(5,20) → 列名 MA_5_20               │
 │                          │                                       │
 │                          │ 为何列名是 ATR_7 而不是 ATR(7)?         │
 │                          │   括号 () 在 Expression 中是语法标记    │
 │                          │   (参数边界)，若列名含括号会导致:       │
 │                          │   "thr_mean(ATR(7))" 解析时，          │
 │                          │   Parser 无法区分 () 是列名字符还是     │
 │                          │   函数调用语法 → 语义冲突。            │
 │                          │   故用 _ 替代: ATR(7) → ATR_7          │
 │                          │                                       │
 │                          │   类比: URL 空格 → %20 编码，           │
 │                          │   语法层的特殊字符到了存储层必须转义    │
 │                          │                                       │
 │                          │   TreeNode._feature_key() 负责转换:     │
 │                          │   FEATURE("ATR",[7]) → 拼接 → "ATR_7"  │
 │                          │                                       │
 │                          │   与 Config 层对照:                     │
 │                          │   Config.features: "ATR(7,14)" (函数式) │
 │                          │   Expression:  "thr_mean(ATR(7))" (同上)│
 │                          │   差异在 differences 配置用了列名式     │
 ├──────────────────────────┼───────────────────────────────────────┤
 │ 阈值函数                  │ thr_0(x)    : x > 0 → 1.0 / 0.0      │
 │                          │ thr_mean(x) : x > mean(x) → 1.0 / 0.0 │
 │                          │ thr_med(x)  : x > median(x) → 1.0/0.0 │
 ├──────────────────────────┼───────────────────────────────────────┤
 │ 二元运算                  │ expr1 & expr2 : AND (np.minimum)      │
 │                          │ expr1 | expr2 : OR  (np.maximum)       │
 │                          │ expr1 sub expr2 : 减法                  │
 ├──────────────────────────┼───────────────────────────────────────┤
 │ 信号确认                  │ persist(expr, 3) : 连续3日同向才触发  │
 ├──────────────────────────┼───────────────────────────────────────┤
 │ 运算符重载 (Python 语法)  │ expr1 & expr2   : AND 组合             │
 │                          │ expr1 | expr2   : OR 组合              │
 │                          │ -expr           : REVERSE 反转          │
 │                          │ expr1 + expr2   : 加权求和              │
 └──────────────────────────┴───────────────────────────────────────┘


============================================================================
                         架构原则（来自 AIdev 探索结论）
============================================================================

 1. AST 是信号的正确表达形式 — GP V4/V5 已验证
 2. 安全计算内置 — div 防零、sqrt 防负、sigmoid 防溢出
 3. 自省能力 — complexity / features_used / depth
 4. 序列化 — Expression 可 save/load JSON，可持久化复用
 5. 运算符重载 — &(AND) / |(OR) / -(REVERSE) / +(加权)

============================================================================

语法示例：
    简单:    "RSI(14)"
    变换:    "thr_mean(RSI(14))"
    阈值:    "thr_roll_mean(BBWIDTH(30), 60)"
    二元:    "thr_mean(BBWIDTH(30)) sub thr_mean(TSF(7))"
    条件:    "if_then(ADX(14) > 25, trend_expr, range_expr)"
    确认:    "persist(thr_mean(BBWIDTH(30)), 3)"
    组合:    "thr_mean(ATR(7)) & thr_mean(TRIMA(60))"

============================================================================
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Callable, Any, Union

from .features import FeatureSpace, _rolling_mean, NORMALIZE_MAP


# ============================================================
# AST 节点定义
# ============================================================

class NodeType(Enum):
    FEATURE = 'feature'
    CONSTANT = 'constant'
    OPERATOR = 'operator'
    FUNCTION = 'function'
    THRESHOLD = 'threshold'
    PERSIST = 'persist'
    SWITCH = 'switch'


class TreeNode:
    """表达式树节点 — 移植自AIdev GP V5"""

    BINARY_OPS = {
        'add': np.add,
        'sub': np.subtract,
        'mul': np.multiply,
        'div': lambda x, y: np.where(np.abs(y) > 1e-10, x / y, 0.0),
        'max': np.maximum,
        'min': np.minimum,
        '>': lambda x, y: np.where(x > y, 1.0, 0.0),
        '<': lambda x, y: np.where(x < y, 1.0, 0.0),
        '&': lambda x, y: np.minimum(x, y),
        '|': lambda x, y: np.maximum(x, y),
    }

    UNARY_FUNCS = {
        'abs': np.abs,
        'sign': np.sign,
        'sqrt': lambda x: np.sqrt(np.abs(x)),
        'square': np.square,
        'neg': lambda x: -x,
        'relu': lambda x: np.maximum(x, 0),
        'tanh': np.tanh,
        'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
        'log': lambda x: np.log(np.maximum(np.abs(x), 1e-10)),
        'exp': lambda x: np.exp(np.clip(x, -50, 50)),
    }

    # [修复] thr_mean/thr_med 不再在此处用全序列均值实现，改为 expanding 方式(无前瞻偏差)
    # 旧实现 'thr_mean': lambda x: np.where(x > np.nanmean(x), 1.0, 0.0) 用了全序列均值含未来数据
    # 新实现移到 evaluate() 中，调用 _expanding_mean / _expanding_median
    # thr_0 保持不变：指标>0做多，对中心化后的RSI/MFI等含义是"高于中性线"
    THRESHOLD_FUNCS = {
        'thr_0': lambda x: np.where(x > 0, 1.0, 0.0),
        'thr_mean': None,
        'thr_med': None,
    }

    def __init__(self, node_type: NodeType, value: str,
                 children: Optional[List['TreeNode']] = None,
                 param: Optional[Any] = None):
        self.node_type = node_type
        self.value = value
        self.children = children if children else []
        self.param = param

    def _feature_key(self) -> str:
        """FEATURE节点: 从 value + params 构造 DataFrame 列名

        FEATURE("ATR", [7]) → "ATR_7"
        FEATURE("KDJ", [9, 3, 3]) → "KDJ_9_3_3"
        FEATURE("ATR_7") → "ATR_7"
        """
        if self.param is not None:
            if isinstance(self.param, list):
                return f"{self.value}_{'_'.join(str(p) for p in self.param)}"
            return f"{self.value}_{self.param}"
        return self.value

    def evaluate(self, feature_data: Dict[str, np.ndarray]) -> np.ndarray:
        if self.node_type == NodeType.FEATURE:
            col_name = self._feature_key()
            if col_name in feature_data:
                return feature_data[col_name].copy()
            return np.zeros(1)
        elif self.node_type == NodeType.CONSTANT:
            return float(self.value)
        elif self.node_type == NodeType.OPERATOR:
            if len(self.children) < 2:
                return self.children[0].evaluate(feature_data) if self.children else np.zeros(1)
            left = self.children[0].evaluate(feature_data)
            right = self.children[1].evaluate(feature_data)
            return self.BINARY_OPS[self.value](left, right)
        elif self.node_type == NodeType.FUNCTION:
            arg = self.children[0].evaluate(feature_data) if self.children else np.zeros(1)
            return self.UNARY_FUNCS[self.value](arg)
        elif self.node_type == NodeType.THRESHOLD:
            arg = self.children[0].evaluate(feature_data) if self.children else np.zeros(1)
            if self.value == 'thr_roll_mean':
                w = self.param if self.param is not None else 30
                rolling_val = _rolling_mean(arg, w)
                return np.where(arg > rolling_val, 1.0, 0.0)
            # [修复] thr_roll_med 原来错误调用了 _rolling_mean，现改为 _rolling_median
            if self.value == 'thr_roll_med':
                w = self.param if self.param is not None else 30
                rolling_val = _rolling_median(arg, w)
                return np.where(arg > rolling_val, 1.0, 0.0)
            # [修复] thr_mean 从全序列均值改为 expanding mean，消除前瞻偏差
            # 旧: np.where(x > np.nanmean(x), ...) 用了未来数据
            # 新: expanding_mean 只用 [0:t+1] 的历史数据，实盘可复现
            if self.value == 'thr_mean':
                expanding_val = _expanding_mean(arg)
                return np.where(arg > expanding_val, 1.0, 0.0)
            # [修复] thr_med 同理，从全序列中位数改为 expanding median
            if self.value == 'thr_med':
                expanding_val = _expanding_median(arg)
                return np.where(arg > expanding_val, 1.0, 0.0)
            if self.value in self.THRESHOLD_FUNCS:
                return self.THRESHOLD_FUNCS[self.value](arg)
            return np.where(arg > 0, 1.0, 0.0)
        elif self.node_type == NodeType.PERSIST:
            arg = self.children[0].evaluate(feature_data) if self.children else np.zeros(1)
            n = self.param if self.param is not None else 3
            return np_persist(arg, n)
        elif self.node_type == NodeType.SWITCH:
            if len(self.children) < 3:
                return self.children[0].evaluate(feature_data) if self.children else np.zeros(1)
            cond_val = self.children[0].evaluate(feature_data)
            branch_a = self.children[1].evaluate(feature_data)
            branch_b = self.children[2].evaluate(feature_data)
            return np.where(cond_val > 0, branch_a, branch_b)
        raise ValueError(f"未知节点类型: {self.node_type}")

    def get_features(self) -> Set[str]:
        feats = set()
        if self.node_type == NodeType.FEATURE:
            feats.add(self._feature_key())
        for child in self.children:
            feats.update(child.get_features())
        return feats

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def size(self) -> int:
        return 1 + sum(child.size() for child in self.children)

    def to_string(self) -> str:
        if self.node_type == NodeType.FEATURE:
            if self.param is not None:
                if isinstance(self.param, list):
                    return f"{self.value}({', '.join(str(p) for p in self.param)})"
                return f"{self.value}({self.param})"
            return self.value
        elif self.node_type == NodeType.CONSTANT:
            return self.value
        elif self.node_type == NodeType.OPERATOR:
            left = self.children[0].to_string()
            right = self.children[1].to_string()
            return f"({left} {self.value} {right})"
        elif self.node_type == NodeType.FUNCTION:
            arg = self.children[0].to_string()
            return f"{self.value}({arg})"
        elif self.node_type == NodeType.THRESHOLD:
            arg = self.children[0].to_string()
            if self.value in ('thr_roll_mean', 'thr_roll_med'):
                return f"{self.value}({arg}, {self.param})"
            return f"{self.value}({arg})"
        elif self.node_type == NodeType.PERSIST:
            arg = self.children[0].to_string()
            return f"persist({arg}, {self.param})"
        elif self.node_type == NodeType.SWITCH:
            cond = self.children[0].to_string()
            a = self.children[1].to_string()
            b = self.children[2].to_string()
            return f"switch({cond}, {a}, {b})"
        return str(self.value)

    def copy(self) -> 'TreeNode':
        return TreeNode(
            self.node_type, self.value,
            [child.copy() for child in self.children],
            self.param
        )

    def to_dict(self) -> Dict:
        result = {
            'type': self.node_type.value,
            'value': self.value,
        }
        if self.param is not None:
            result['param'] = self.param
        if self.children:
            result['children'] = [child.to_dict() for child in self.children]
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'TreeNode':
        node_type = NodeType(data['type'])
        children = [cls.from_dict(c) for c in data.get('children', [])]
        param = data.get('param')
        return cls(node_type, data['value'], children, param)


def np_persist(arr: np.ndarray, n: int) -> np.ndarray:
    """信号确认：连续N日同向才触发"""
    result = np.zeros_like(arr)
    for i in range(n - 1, len(arr)):
        segment = arr[i - n + 1:i + 1]
        if np.all(segment > 0):
            result[i] = 1.0
        elif np.all(segment < 0):
            result[i] = -1.0
    return result


def _expanding_mean(arr: np.ndarray) -> np.ndarray:
    """扩展窗口均值：第t天的均值只用到[0:t+1]的数据，无前瞻偏差
    [修复说明] 替代原有的 np.nanmean(x) 全序列均值实现。
    原实现在第t天判断时用了t+1到末尾的未来数据(前瞻偏差)，
    回测虚增收益且实盘不可复现。expanding mean 严格只用历史数据。"""
    result = np.full_like(arr, np.nan, dtype=float)
    cumsum = np.nancumsum(arr)
    count = np.cumsum(~np.isnan(arr))
    for i in range(len(arr)):
        if count[i] > 0:
            result[i] = cumsum[i] / count[i]
    return result


def _expanding_median(arr: np.ndarray) -> np.ndarray:
    """扩展窗口中位数：第t天的中位数只用到[0:t+1]的数据，无前瞻偏差
    [修复说明] 替代原有的 np.nanmedian(x) 全序列中位数实现，同理消除前瞻偏差。"""
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(len(arr)):
        segment = arr[:i + 1]
        valid = segment[~np.isnan(segment)]
        if len(valid) > 0:
            result[i] = np.median(valid)
    return result


def _rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动窗口中位数
    [修复说明] thr_roll_med 原来错误调用了 _rolling_mean(计算的是均值而非中位数)，
    现独立实现正确的中位数计算。"""
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        segment = arr[i - window + 1:i + 1]
        valid = segment[~np.isnan(segment)]
        if len(valid) > 0:
            result[i] = np.median(valid)
    return result


# ============================================================
# 表达式解析器
# ============================================================

class Token:
    LPAREN, RPAREN, COMMA, NAME, NUMBER, OP, EOF = (
        'LPAREN', 'RPAREN', 'COMMA', 'NAME', 'NUMBER', 'OP', 'EOF'
    )

    def __init__(self, type_: str, value: str):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, '{self.value}')"


OPS = {'add', 'sub', 'mul', 'div', 'max', 'min', '>', '<', '&', '|', '+'}


class Tokenizer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch.isspace():
                self.pos += 1
                continue
            if ch == '(':
                self.tokens.append(Token(Token.LPAREN, '('))
                self.pos += 1
            elif ch == ')':
                self.tokens.append(Token(Token.RPAREN, ')'))
                self.pos += 1
            elif ch == ',':
                self.tokens.append(Token(Token.COMMA, ','))
                self.pos += 1
            elif ch.isdigit() or ch == '.' or (ch == '-' and self.pos + 1 < len(self.source) and self.source[self.pos + 1].isdigit()):
                self._number()
            elif ch.isalpha() or ch == '_':
                self._name()
            elif ch in '><+-&|':
                self._op()
            else:
                self.pos += 1
        self.tokens.append(Token(Token.EOF, ''))
        return self.tokens

    def _number(self):
        start = self.pos
        if self.source[self.pos] == '-':
            self.pos += 1
        while self.pos < len(self.source) and (self.source[self.pos].isdigit() or self.source[self.pos] == '.'):
            self.pos += 1
        val = self.source[start:self.pos]
        self.tokens.append(Token(Token.NUMBER, val))

    def _name(self):
        start = self.pos
        while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
            self.pos += 1
        val = self.source[start:self.pos]
        self.tokens.append(Token(Token.NAME, val))

    def _op(self):
        val = self.source[self.pos]
        if self.pos + 1 < len(self.source):
            two = val + self.source[self.pos + 1]
            if two in OPS:
                self.tokens.append(Token(Token.OP, two))
                self.pos += 2
                return
        self.tokens.append(Token(Token.OP, val))
        self.pos += 1


class Parser:
    """递归下降解析器"""

    def __init__(self, tokens: List[Token], feature_names: Optional[Set[str]] = None):
        self.tokens = tokens
        self.pos = 0
        self.feature_names = feature_names or set()

    def peek(self) -> Token:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token(Token.EOF, '')

    def advance(self) -> Token:
        t = self.peek()
        self.pos += 1
        return t

    def match(self, type_: str) -> bool:
        return self.peek().type == type_

    def expect(self, type_: str, value: Optional[str] = None):
        t = self.advance()
        if t.type != type_:
            raise SyntaxError(f"期望 {type_}，得到 {t.type}('{t.value}') at pos {self.pos}")
        if value is not None and t.value != value:
            raise SyntaxError(f"期望 '{value}'，得到 '{t.value}'")

    def parse(self) -> TreeNode:
        tree = self._parse_expr()
        if self.peek().type != Token.EOF:
            # 可能有剩余的二元运算
            pass
        return tree

    def _parse_expr(self) -> TreeNode:
        """解析表达式，处理二元运算的优先级"""
        left = self._parse_primary()
        while self.peek().type == Token.OP:
            op_token = self.advance()
            op_str = self._normalize_op(op_token.value)
            right = self._parse_primary()
            left = TreeNode(NodeType.OPERATOR, op_str, [left, right])
        return left

    def _normalize_op(self, op_val: str) -> str:
        op_map = {'+': 'add', '&&': '&', '||': '|'}
        return op_map.get(op_val, op_val)

    def _parse_primary(self) -> TreeNode:
        token = self.peek()
        if token.type == Token.NUMBER:
            return self._parse_number()
        elif token.type == Token.NAME:
            return self._parse_name_or_call()
        elif token.type == Token.LPAREN:
            self.expect(Token.LPAREN)
            inner = self._parse_expr()
            self.expect(Token.RPAREN)
            return inner
        else:
            raise SyntaxError(f"意外的 token: {token}")

    def _parse_number(self) -> TreeNode:
        t = self.advance()
        return TreeNode(NodeType.CONSTANT, t.value)

    def _parse_name_or_call(self) -> TreeNode:
        name = self.advance().value
        if self.peek().type == Token.LPAREN:
            # 函数调用: name(args...)
            return self._parse_call(name)
        # 普通名称: 可能是特征名
        if name in self.feature_names or not self.feature_names:
            return TreeNode(NodeType.FEATURE, name)
        return TreeNode(NodeType.FEATURE, name)

    def _parse_call(self, func_name: str) -> TreeNode:
        self.expect(Token.LPAREN)
        args = []
        while self.peek().type != Token.RPAREN:
            arg_node = self._parse_expr()
            args.append(arg_node)
            if self.peek().type == Token.COMMA:
                self.advance()
            elif self.peek().type == Token.RPAREN:
                break
        self.expect(Token.RPAREN)
        return self._build_call(func_name, args)

    def _build_call(self, func_name: str, args: List[TreeNode]) -> TreeNode:
        """根据函数名构建对应的AST节点"""
        name = func_name.lower()

        # persist(expr, n)
        if name == 'persist':
            n = 3
            if len(args) >= 2 and args[1].node_type == NodeType.CONSTANT:
                n = int(float(args[1].value))
            return TreeNode(NodeType.PERSIST, 'persist', [args[0]], param=n)

        # switch(cond, a, b) — 条件分支
        if name == 'switch' or name == 'if_then':
            if len(args) >= 3:
                return TreeNode(NodeType.SWITCH, 'switch', [args[0], args[1], args[2]])
            return args[0] if args else TreeNode(NodeType.CONSTANT, '0')

        # 阈值函数: thr_0, thr_mean, thr_med, thr_roll_mean, thr_roll_med
        if name in ('thr_0', 'thr_mean', 'thr_med', 'thr_roll_mean', 'thr_roll_med'):
            param = None
            if name in ('thr_roll_mean', 'thr_roll_med') and len(args) >= 2:
                if args[1].node_type == NodeType.CONSTANT:
                    param = int(float(args[1].value))
            return TreeNode(NodeType.THRESHOLD, name, [args[0]], param=param)

        # 一元函数
        if name in TreeNode.UNARY_FUNCS:
            return TreeNode(NodeType.FUNCTION, name, [args[0]])

        # 默认为特征函数: RSI(14) → FEATURE("RSI", param=[14])
        # KDJ(9, 3, 3) → FEATURE("KDJ", param=[9, 3, 3])
        feature_params = []
        for a in args:
            if a.node_type == NodeType.CONSTANT:
                feature_params.append(int(float(a.value)))
        if feature_params:
            return TreeNode(NodeType.FEATURE, func_name, param=feature_params)
        return TreeNode(NodeType.FEATURE, func_name)


def parse_expression(expr_str: str, feature_names: Optional[Set[str]] = None) -> TreeNode:
    """解析表达式字符串 → AST树"""
    tokenizer = Tokenizer(expr_str)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens, feature_names)
    return parser.parse()


# ============================================================
# Expression 类 — 信号的一等公民
# ============================================================

class Expression:
    """
    统一表达式信号

    用法:
        # 简单表达式
        expr = Expression("RSI(14)", feature_space=fs)
        signals = expr.generate(data)

        # 复杂表达式（GP结果）
        expr = Expression("thr_mean(BBWIDTH(30)) sub thr_mean(TSF(7))", feature_space=fs)

        # 运算符组合
        expr = Expression("ATR(7)", fs) & Expression("TRIMA(60)", fs)

        # 自省
        print(expr.complexity)    # 节点数
        print(expr.features_used) # ['BBWIDTH_30', 'TSF_7']

        # 序列化
        expr.save("best.json")
        loaded = Expression.load("best.json")
    """

    def __init__(self, source: Union[str, TreeNode],
                 feature_space: Optional[FeatureSpace] = None,
                 feature_df: Optional[pd.DataFrame] = None,
                 name: Optional[str] = None):
        """
        Args:
            source: 表达式字符串 或 TreeNode
            feature_space: FeatureSpace实例（用于自动计算特征）
            feature_df: 预计算的特征矩阵（替代feature_space）
            name: 可选的信号名称
        """
        if isinstance(source, TreeNode):
            self._tree = source
            self.source = source.to_string()
        else:
            self.source = source
            feature_names = set()
            if feature_space:
                feature_names = set(feature_space.get_feature_names())
            elif feature_df is not None:
                feature_names = set(feature_df.columns)
            self._tree = parse_expression(self.source, feature_names)

        self._feature_space = feature_space
        self._feature_df = feature_df
        self._name = name or f"Expr_{self.source[:30]}"
        self._compiled = None

        # 自省属性
        self.complexity = self._tree.size()
        self.depth = self._tree.depth()
        self.features_used = sorted(list(self._tree.get_features()))
        self.expression = self._tree.to_string()

    def generate(self, data: pd.DataFrame) -> pd.Series:
        """生成信号序列"""
        if self._feature_df is not None:
            feature_data = {col: self._feature_df[col].values for col in self._feature_df.columns}
        elif self._feature_space is not None:
            features = self._feature_space.fit_transform(data)
            self._feature_df = features
            feature_data = {col: features[col].values for col in features.columns}
        else:
            raise RuntimeError("需要提供 feature_space 或 feature_df")

        signals = self._tree.evaluate(feature_data)
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

        index = data.index if self._feature_df is None else self._feature_df.index
        return pd.Series(signals, index=index, name=self._name)

    def generate_from_features(self, feature_df: pd.DataFrame) -> pd.Series:
        """直接从特征DataFrame生成信号（GPU风格求值）"""
        feature_data = {col: feature_df[col].values for col in feature_df.columns}
        signals = self._tree.evaluate(feature_data)
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.Series(signals, index=feature_df.index, name=self._name)

    # ---- 运算符重载 ----

    def __and__(self, other: 'Expression') -> 'Expression':
        return self._binary_op(other, '&', 'AND')

    def __or__(self, other: 'Expression') -> 'Expression':
        return self._binary_op(other, '|', 'OR')

    def __neg__(self) -> 'Expression':
        tree = TreeNode(NodeType.FUNCTION, 'neg', [self._tree.copy()])
        expr = Expression(tree, self._feature_space, self._feature_df,
                          name=f'REV_{self._name}')
        expr.source = f'neg({self.source})'
        return expr

    def __add__(self, other: 'Expression') -> 'Expression':
        return self._binary_op(other, 'add', 'ADD')

    def __sub__(self, other: 'Expression') -> 'Expression':
        return self._binary_op(other, 'sub', 'SUB')

    def __mul__(self, other: 'Expression') -> 'Expression':
        return self._binary_op(other, 'mul', 'MUL')

    def _binary_op(self, other: 'Expression', op_name: str, label: str) -> 'Expression':
        tree = TreeNode(NodeType.OPERATOR, op_name, [self._tree.copy(), other._tree.copy()])
        expr = Expression(tree, self._feature_space or other._feature_space,
                          self._feature_df or other._feature_df,
                          name=f'{self._name}_{label}_{other._name}')
        expr.source = f'({self.source}) {op_name} ({other.source})'
        return expr

    # ---- 序列化 ----

    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'expression': self.expression,
            'name': self._name,
            'tree': self._tree.to_dict(),
            'complexity': self.complexity,
            'depth': self.depth,
            'features_used': self.features_used,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict, feature_space: Optional[FeatureSpace] = None,
                  feature_df: Optional[pd.DataFrame] = None) -> 'Expression':
        tree = TreeNode.from_dict(data.get('tree', data.get('ast', {})))
        expr = cls(tree, feature_space, feature_df, name=data.get('name'))
        expr.source = data.get('source', expr.source)
        return expr

    @classmethod
    def from_json(cls, json_str: str, feature_space: Optional[FeatureSpace] = None,
                  feature_df: Optional[pd.DataFrame] = None) -> 'Expression':
        data = json.loads(json_str)
        return cls.from_dict(data, feature_space, feature_df)

    def save(self, path: str):
        """保存表达式到JSON文件"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str, feature_space: Optional[FeatureSpace] = None,
             feature_df: Optional[pd.DataFrame] = None) -> 'Expression':
        """从JSON文件加载表达式"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data, feature_space, feature_df)

    def __repr__(self):
        return f"Expression('{self.source[:60]}', complexity={self.complexity}, features={self.features_used})"

    def __str__(self):
        return self.source


# ============================================================
# Persist / RegimeSwitch 变换器
# ============================================================

def persist(expr: Expression, n: int = 3) -> Expression:
    """创建信号确认包装器"""
    tree = TreeNode(NodeType.PERSIST, 'persist', [expr._tree.copy()], param=n)
    return Expression(tree, expr._feature_space, expr._feature_df,
                      name=f'Persist({n})_{expr._name}')


def regime_switch(condition: Expression, expr_a: Expression, expr_b: Expression) -> Expression:
    """创建条件分支信号"""
    tree = TreeNode(NodeType.SWITCH, 'switch',
                     [condition._tree.copy(), expr_a._tree.copy(), expr_b._tree.copy()])
    fs = condition._feature_space or expr_a._feature_space or expr_b._feature_space
    fd = condition._feature_df or expr_a._feature_df or expr_b._feature_df
    return Expression(tree, fs, fd, name=f'Switch_{condition._name}')


def parse_and_build(expr_str: str,
                    feature_space: Optional[FeatureSpace] = None,
                    feature_df: Optional[pd.DataFrame] = None) -> Expression:
    """便捷函数：从字符串创建Expression"""
    return Expression(expr_str, feature_space, feature_df)
