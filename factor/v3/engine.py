"""
factor/v3/engine.py — 因子表达式引擎
=============================================================================

合并 v2 的 expression.py + expression_factor.py。
从表达式字符串到因子值 DataFrame 的一站式入口。

模块结构：
  Part 1: AST 定义 (NodeType / ASTNode / evaluate_node)
  Part 2: Tokenizer + Parser (tokenize / Parser / FactorExpression)
  Part 3: ExpressionFactor 适配器 (Factor 子类，可接入 backtest)
  Part 4: AlphaExplorer + AlphaResult (批量探索器)

[重构] 2026-06-01 从 v2 两个文件合并为 v3/engine.py
=============================================================================
"""

import re
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass

from .base import Factor, FactorMetadata, FactorCategory, FactorFrequency
from .primitives import (
    ts_rank, ts_zscore, delay, decay_linear,
    cs_rank, signed_power, correlation,
    ts_sum, ts_mean, ts_std, ts_max, ts_min,
    sma, covariance, regbeta,
    ts_argmin, ts_argmax, ifelse,
    cs_mean, ts_skew, delta, winsorize,     # [新增] 2026-06-01
)
from .primitives import cs_zscore as _prim_cs_zscore


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: AST 定义
# ═══════════════════════════════════════════════════════════════════════════

class NodeType(Enum):
    VARIABLE = 'variable'
    CONSTANT = 'constant'
    FUNCTION = 'function'
    UNARY = 'unary'


class ASTNode:
    """AST 节点"""

    def __init__(self, node_type: NodeType, value: Any,
                 children: List['ASTNode'] = None,
                 params: Dict[str, Any] = None):
        self.node_type = node_type
        self.value = value
        self.children = children or []
        self.params = params or {}

    def add_child(self, child: 'ASTNode'):
        self.children.append(child)

    def to_str(self) -> str:
        if self.node_type == NodeType.VARIABLE:
            return str(self.value)
        elif self.node_type == NodeType.CONSTANT:
            return str(self.value)
        elif self.node_type == NodeType.UNARY:
            return f"{self.value}({self.children[0].to_str()})"
        else:
            args = ', '.join(c.to_str() for c in self.children)
            all_args = [args] + [str(v) for v in self.params.values()]
            return f"{self.value}({', '.join(all_args)})"

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


# 终端变量映射
VARIABLE_MAP = {
    'open': 'open', 'high': 'high', 'low': 'low',
    'close': 'close', 'volume': 'volume', 'amount': 'amount',
    'returns': 'returns', 'vwap': 'vwap',
}

# 一元函数
UNARY_FUNCTIONS = {
    'abs': lambda x: np.abs(x),
    'sqrt': lambda x: np.sqrt(np.maximum(x, 0)),
    'log': lambda x: np.log(np.maximum(x, 1e-10)),
    'exp': lambda x: np.exp(x),                       # [新增] 2026-06-01 指数变换
    'neg': lambda x: -x,
    'sign': lambda x: np.sign(x),
    'tanh': lambda x: np.tanh(x),                     # [新增] 2026-06-01 双曲正切软压缩
}

# 二元函数
BINARY_FUNCTIONS = {
    'add': lambda x, y: x + y,
    'sub': lambda x, y: x - y,
    'mul': lambda x, y: x * y,
    'div': lambda x, y: np.where(np.abs(y) > 1e-10, x / y, 0.0),
    'max': lambda x, y: np.maximum(x, y),
    'min': lambda x, y: np.minimum(x, y),
}

# 时序/截面原语
PRIMITIVE_FUNCTIONS = {
    'ts_rank': ts_rank, 'ts_zscore': ts_zscore, 'delay': delay,
    'correlation': correlation, 'decay_linear': decay_linear,
    'cs_rank': cs_rank, 'signed_power': signed_power,
    'ts_sum': ts_sum, 'ts_mean': ts_mean, 'ts_std': ts_std,
    'ts_max': ts_max, 'ts_min': ts_min,
    'sma': sma, 'covariance': covariance, 'regbeta': regbeta,
    'ts_argmin': ts_argmin, 'ts_argmax': ts_argmax, 'ifelse': ifelse,
    'cs_zscore': _prim_cs_zscore,
    'cs_mean': cs_mean, 'ts_skew': ts_skew, 'delta': delta,  # [新增] 2026-06-01
    'winsorize': winsorize,  # [新增] 2026-06-01
    'ret': None, 'adv': None, 'intra_ret': None,  # [新增] 零参数语法糖
}


def evaluate_node(node: ASTNode, data: Dict[str, np.ndarray]) -> np.ndarray:
    """递归求值 AST 节点"""
    if node.node_type == NodeType.VARIABLE:
        # [新增] 2026-06-01 零参数语法糖直接求值
        if node.value == 'intra_ret':
            close, opn = data['close'], data['open']
            diff = close - opn
            return np.where(np.abs(opn) > 1e-10, diff / opn, 0.0)
        col = VARIABLE_MAP.get(node.value.lower())
        if col is None or col not in data:
            raise KeyError(f"字段 '{node.value}' 未提供，当前可用: {list(data.keys())}")
        return np.asarray(data[col], dtype=float)

    elif node.node_type == NodeType.CONSTANT:
        return np.full(_infer_shape(data), float(node.value), dtype=float)

    elif node.node_type == NodeType.UNARY:
        child_val = evaluate_node(node.children[0], data)
        fn = UNARY_FUNCTIONS.get(node.value)
        if fn is None:
            raise ValueError(f"未知一元函数: {node.value}")
        return fn(child_val)

    elif node.node_type == NodeType.FUNCTION:
        fn_name = node.value
        # [新增] 2026-06-01 零参数语法糖，省 1 层 AST 深度
        if fn_name == 'ret':
            period = int(node.params.get('period', 1))
            return delta(data.get('close'), period)
        if fn_name == 'adv':
            period = int(node.params.get('period', 10))
            return ts_mean(data.get('volume'), period)
        if fn_name == 'intra_ret':
            close, opn = data['close'], data['open']
            return np.where(np.abs(opn) > 1e-10, (close - opn) / opn, 0.0)
        if fn_name in BINARY_FUNCTIONS:
            vals = [evaluate_node(c, data) for c in node.children]
            return BINARY_FUNCTIONS[fn_name](*vals)
        if fn_name in PRIMITIVE_FUNCTIONS:
            prim_fn = PRIMITIVE_FUNCTIONS[fn_name]
            vals = [evaluate_node(c, data) for c in node.children]
            def _to_scalar_if_const(v, idx):
                if idx > 0 and isinstance(v, np.ndarray):
                    try:
                        flat = float(v.flat[0])
                        if np.allclose(v, flat):
                            return flat if flat == int(flat) else flat
                    except Exception:
                        pass
                return v
            vals = [_to_scalar_if_const(v, i) for i, v in enumerate(vals)]
            if node.params:
                return prim_fn(*vals, **node.params)
            if fn_name == 'signed_power':
                return prim_fn(vals[0], exponent=2.0)
            return prim_fn(*vals)
        raise ValueError(f"未知函数: {fn_name}")
    raise ValueError(f"未知节点类型: {node.node_type}")


def _infer_shape(data: Dict[str, np.ndarray]) -> Tuple[int, int]:
    for arr in data.values():
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return arr.shape
    return (1, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Tokenizer + Parser
# ═══════════════════════════════════════════════════════════════════════════

TOKEN_SPEC = [
    ('NUMBER', r'\d+\.?\d*'), ('NAME', r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('LPAREN', r'\('), ('RPAREN', r'\)'), ('COMMA', r','),
    ('EQUALS', r'='), ('WHITESPACE', r'\s+'),
]

TOKEN_RE = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPEC))


def tokenize(expr_str: str) -> List[Tuple[str, str]]:
    tokens = []
    for m in TOKEN_RE.finditer(expr_str):
        kind = m.lastgroup
        value = m.group()
        if kind == 'WHITESPACE':
            continue
        tokens.append((kind, value))
    return tokens


class Parser:
    """递归下降语法分析器"""

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
            raise SyntaxError(f"期望 {expected_kind}，实际 {token[0]}({token[1]})，pos={self.pos}")
        self.pos += 1
        return token

    def parse(self) -> ASTNode:
        node = self.parse_expr()
        if self.pos < len(self.tokens):
            raise SyntaxError(f"多余 token: {self.tokens[self.pos]}，pos={self.pos}")
        return node

    def parse_expr(self) -> ASTNode:
        token = self.peek()
        if token is None:
            raise SyntaxError("空表达式")
        kind, value = token
        if kind == 'NAME':
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1][0] == 'LPAREN':
                return self.parse_function_call()
            self.consume('NAME')
            return ASTNode(NodeType.VARIABLE, value)
        if kind == 'NUMBER':
            self.consume('NUMBER')
            val = float(value) if '.' in value else int(value)
            return ASTNode(NodeType.CONSTANT, val)
        raise SyntaxError(f"意外 token: {token}，pos={self.pos}")

    def parse_function_call(self) -> ASTNode:
        name = self.consume('NAME')[1]
        self.consume('LPAREN')
        children, params = [], {}
        if self.peek() and self.peek()[0] != 'RPAREN':
            while True:
                if (self.pos + 2 < len(self.tokens)
                        and self.tokens[self.pos][0] == 'NAME'
                        and self.tokens[self.pos + 1][0] == 'EQUALS'
                        and self.tokens[self.pos + 2][0] == 'NUMBER'):
                    pname = self.consume('NAME')[1]
                    self.consume('EQUALS')
                    pval_str = self.consume('NUMBER')[1]
                    params[pname] = float(pval_str) if '.' in pval_str else int(pval_str)
                else:
                    children.append(self.parse_expr())
                if self.peek() and self.peek()[0] == 'COMMA':
                    self.consume('COMMA')
                else:
                    break
        self.consume('RPAREN')
        if name in UNARY_FUNCTIONS and len(children) == 1:
            return ASTNode(NodeType.UNARY, name, children)
        return ASTNode(NodeType.FUNCTION, name, children, params)


class FactorExpression:
    """因子表达式 — 字符串公式 → AST → (T,N) ndarray"""

    def __init__(self, expression_str: str):
        self.source = expression_str
        tokens = tokenize(expression_str)
        parser = Parser(tokens)
        self.ast = parser.parse()
        self.terminals = self._collect_terminals(self.ast)
        self.functions = self._collect_functions(self.ast)
        self.depth = self._calc_depth(self.ast)
        self.node_count = self._count_nodes(self.ast)

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        result = evaluate_node(self.ast, data)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def to_dataframe(self, data, index=None, columns=None) -> pd.DataFrame:
        values = self.evaluate(data)
        return pd.DataFrame(values, index=index, columns=columns)

    def to_str(self) -> str:
        return self.ast.to_str()

    def __repr__(self):
        return f"FactorExpression({self.source!r})"

    def __str__(self):
        return self.source

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


def parse_expression(expr_str: str) -> FactorExpression:
    return FactorExpression(expr_str)


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: ExpressionFactor 适配器
# ═══════════════════════════════════════════════════════════════════════════

class ExpressionFactor(Factor):
    """将 FactorExpression 字符串包装为 Factor 子类

    使表达式因子可以接入 backtest、validator 等所有 v3 基础设施。

    Example:
        >>> ef = ExpressionFactor("ts_rank(sub(close, delay(close, 20)), 10)")
        >>> fv = ef.calculate({'close': close_df}, symbols, dates)
    """

    def __init__(self, expression_str: str, name: str = 'expr_factor',
                 category: FactorCategory = FactorCategory.CUSTOM,
                 description: str = ''):
        metadata = FactorMetadata(
            name=name,
            description=description or f'表达式因子: {expression_str[:60]}',
            category=category,
            frequency=FactorFrequency.DAILY,
            parameters={'expression': expression_str},
        )
        super().__init__(metadata)
        self.expression_str = expression_str
        self.expr = FactorExpression(expression_str)

    def calculate(self, data: Dict[str, pd.DataFrame],
                  symbols: List[str], dates: List[Any]) -> pd.DataFrame:
        ndarray_data = {}
        for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if field in data:
                ndarray_data[field] = np.asarray(data[field].values, dtype=float)
        if not ndarray_data:
            raise ValueError("数据字典为空，至少需要 close")

        close_arr = ndarray_data.get('close')
        vol_arr = ndarray_data.get('volume')
        if close_arr is not None:
            close_delayed = delay(close_arr, 1)
            safe_delayed = np.where(np.abs(close_delayed) > 1e-10, close_delayed, np.nan)
            ndarray_data['returns'] = close_arr / safe_delayed - 1.0
            ndarray_data['returns'] = np.nan_to_num(ndarray_data['returns'], nan=0.0, posinf=0.0, neginf=0.0)
            if vol_arr is not None:
                amount_arr = close_arr * vol_arr
                ndarray_data['vwap'] = np.where(np.abs(vol_arr) > 1e-10, amount_arr / vol_arr, close_arr)

        result_ndarray = self.expr.evaluate(ndarray_data)
        result = pd.DataFrame(result_ndarray, index=dates[:result_ndarray.shape[0]],
                              columns=symbols[:result_ndarray.shape[1]])
        return result

    def __repr__(self):
        return f"ExpressionFactor({self.expression_str[:60]}...)"


def expression_factor(alpha_id: str, formulas: Dict[str, str] = None, **kwargs) -> ExpressionFactor:
    """根据 alpha ID 创建 ExpressionFactor"""
    if formulas is None:
        raise ValueError("formulas 字典不能为 None，请传入公式字典")
    formula = formulas.get(alpha_id)
    if formula is None:
        raise KeyError(f"未知因子: {alpha_id}，可用: {list(formulas.keys())[:10]}...")
    return ExpressionFactor(formula, name=alpha_id, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Part 4: AlphaExplorer
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AlphaResult:
    """单个 Alpha 因子的检验 + 回测结果"""
    alpha_id: str
    ic_mean: float = 0.0
    ic_ir: float = 0.0
    sharpe: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    expression: str = ''
    error: str = ''

    def __repr__(self):
        return (f"AlphaResult({self.alpha_id}, IC={self.ic_mean:.4f}, "
                f"IR={self.ic_ir:.2f}, Sharpe={self.sharpe:.2f})")


class AlphaExplorer:
    """Alpha 公式批量探索器

    对任意公式字典执行：表达式求值 → IC 检验 → Pipeline 回测 → 汇总排序。

    Example:
        >>> explorer = AlphaExplorer(data, returns, formulas=ALPHA101)
        >>> results = explorer.run(['alpha001', 'alpha006'])
        >>> best = explorer.best(min_ic=0.02, min_sharpe=0.5)
    """

    def __init__(self, data: Dict[str, pd.DataFrame], returns: pd.DataFrame,
                 formulas: Dict[str, str] = None, cost_rate: float = 0.0,
                 freq: str = 'ME', top_n: int = 3):
        self.data = data
        self.returns = returns
        self.cost_rate = cost_rate
        self.freq = freq
        self.top_n = top_n
        self.formulas = formulas or {}
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns = self.returns.copy()
            self.returns.index = pd.to_datetime(self.returns.index)
        self.symbols = list(self.returns.columns)
        self.dates = list(self.returns.index)
        self.future_returns = self.returns.shift(-1)
        self.results: List[AlphaResult] = []

    def run(self, alpha_ids: List[str] = None, verbose: bool = True) -> List[AlphaResult]:
        """批量探索因子"""
        # Lazy import to avoid circular
        from .validator import FactorValidator
        from .backtest import FactorPipeline, FixedScheduler, TopNEqualWeight

        if alpha_ids is None:
            alpha_ids = list(self.formulas.keys())
        self.results = []
        scheduler = FixedScheduler(self.freq)
        allocator = TopNEqualWeight(self.top_n)

        for i, aid in enumerate(alpha_ids):
            if verbose:
                print(f"[{i+1}/{len(alpha_ids)}] {aid}...", end=' ')
            try:
                formula = self.formulas.get(aid)
                if formula is None:
                    if verbose:
                        print("跳过(未定义)")
                    continue
                ef = ExpressionFactor(formula, name=aid)
                fv = ef.calculate(self.data, self.symbols, self.dates)
                if fv.isna().all().all():
                    if verbose:
                        print("全NaN")
                    continue
                validator = FactorValidator(fv, self.future_returns)
                ic_result = validator.information_coefficient()
                ic_mean = ic_result.get('mean', 0.0)
                ic_ir = ic_result.get('ir', 0.0)
                hr = validator.hit_rate()
                if ic_mean is None or np.isnan(ic_mean):
                    if verbose:
                        print("IC=NaN")
                    continue
                pipeline = FactorPipeline(returns=self.returns, scheduler=scheduler,
                                          allocator=allocator, cost_rate=self.cost_rate)
                bt = pipeline.evaluate(fv)
                result = AlphaResult(alpha_id=aid, ic_mean=float(ic_mean),
                                     ic_ir=float(ic_ir) if not np.isnan(ic_ir) else 0.0,
                                     sharpe=bt.sharpe_ratio, annual_return=bt.annual_return,
                                     max_drawdown=bt.max_drawdown,
                                     hit_rate=float(hr) if not np.isnan(hr) else 0.0,
                                     expression=formula)
                self.results.append(result)
                if verbose:
                    print(f"IC={ic_mean:.4f} IR={ic_ir:.2f} Sharpe={bt.sharpe_ratio:.2f}")
            except Exception as e:
                if verbose:
                    print(f"失败: {e}")
                self.results.append(AlphaResult(alpha_id=aid, error=str(e)))

        self.results.sort(key=lambda r: abs(r.ic_mean) * max(r.ic_ir, 0), reverse=True)
        return self.results

    def best(self, min_ic: float = 0.0, min_sharpe: float = 0.0, top_n: int = 20) -> List[AlphaResult]:
        filtered = [r for r in self.results
                    if abs(r.ic_mean) >= min_ic and r.sharpe >= min_sharpe and not r.error]
        return filtered[:top_n]

    def to_dataframe(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        rows = []
        for r in self.results:
            rows.append({
                'Alpha': r.alpha_id, '|IC|': round(abs(r.ic_mean), 4),
                'IC': round(r.ic_mean, 4), 'IR': round(r.ic_ir, 2),
                'Sharpe': round(r.sharpe, 2), '年化收益': f"{r.annual_return:.1%}",
                '最大回撤': f"{r.max_drawdown:.1%}", '命中率': f"{r.hit_rate:.1%}",
                '错误': r.error[:40] if r.error else '',
            })
        return pd.DataFrame(rows)

    def __repr__(self):
        n = len(self.results)
        valid = sum(1 for r in self.results if not r.error)
        return f"AlphaExplorer(total={n}, valid={valid})"
