# [新增] 2026-06-17 v3前缀表达式→v4 Python DSL 翻译器
"""
factor/v4/translate_v3.py — v3 前缀表达式 → v4 Python DSL 翻译器

将 v3 的 Lisp 风格前缀表达式（如 sub(close, delay(close, 5))）
翻译为 v4 的 Python infix DSL（如 (CLOSE - ts_delay(CLOSE, 5))）。

核心转换:
  - 算术: sub(a,b)→(a-b), add(a,b)→(a+b), mul(a,b)→(a*b), div(a,b)→(a/b), neg(a)→(-a)
  - 条件: ifelse(cond,a,b)→(a if cond else b)  ← AST IfExp
  - 函数重命名: delay→ts_delay, correlation→ts_corr, covariance→ts_cov, decay_linear→ts_decay_linear
  - 变量大写: close→CLOSE, volume→VOLUME, vwap→VWAP, returns→RETURNS
  - sma(x,w,k) 保持不变（需在 v4 registry 注册）

用法:
    >>> from factor.v4.translate_v3 import translate_expr, translate_dict
    >>> v4_expr = translate_expr('sub(cs_rank(delay(close, 5)), 0.5)')
    >>> # → '(cs_rank(ts_delay(CLOSE, 5)) - 0.5)'
    >>>
    >>> from factor.v3.formulas.wq101 import ALPHA101
    >>> v4_dict = translate_dict(ALPHA101)
"""
import ast
import re
from typing import Dict, Tuple, List

# ============================================================
# 函数名映射: v3 → v4
# ============================================================
_FUNC_MAP = {
    'delay':           'ts_delay',
    'delta':           'ts_delta',           # v3 delta = sub(x, delay(x,w)), 但某些地方直接用
    'correlation':     'ts_corr',
    'covariance':      'ts_cov',
    'decay_linear':    'ts_decay_linear',
    # 以下函数名不变，但列出以备查
    'cs_rank':         'cs_rank',
    'cs_zscore':       'cs_zscore',
    'ts_rank':         'ts_rank',
    'ts_std':          'ts_std',
    'ts_mean':         'ts_mean',
    'ts_sum':          'ts_sum',
    'ts_max':          'ts_max',
    'ts_min':          'ts_min',
    'ts_zscore':       'ts_zscore',
    'ts_argmax':       'ts_argmax',
    'ts_argmin':       'ts_argmin',
    'sma':             'sma',               # 3参数SMA, 需在v4注册
    'regbeta':         'ts_regression',      # regbeta(x,y,w) → ts_regression(x,y,w,0)
    'signed_power':    'signed_power',
    'abs':             'abs',
    'log':             'log',
    'sqrt':            'sqrt',
    'sign':            'sign',
    'exp':             'exp',
}

# 变量名映射: v3 小写 → v4 大写
_VAR_MAP = {
    'close':   'CLOSE',
    'open':    'OPEN',
    'high':    'HIGH',
    'low':     'LOW',
    'volume':  'VOLUME',
    'amount':  'AMOUNT',
    'vwap':    'VWAP',
    'returns': 'RETURNS',
}

# 算术前缀 → 中缀运算符
_ARITH_OPS = {
    'sub': '-',
    'add': '+',
    'mul': '*',
    'div': '/',
}


class _V3ToV4Translator(ast.NodeVisitor):
    """AST 递归翻译器: v3 前缀 → v4 Python DSL 字符串"""

    def __init__(self):
        self.errors = []

    def translate(self, expr_str: str) -> str:
        """翻译一个 v3 表达式字符串"""
        # v3 表达式是合法的 Python 函数调用语法，可以直接 ast.parse
        try:
            tree = ast.parse(expr_str, mode='eval')
        except SyntaxError as e:
            self.errors.append(f"语法错误: {e}")
            return expr_str  # 返回原文

        result = self.visit(tree.body)
        return result

    def visit(self, node):
        """分派到对应的 visit 方法"""
        method = f'_visit_{type(node).__name__}'
        visitor = getattr(self, method, None)
        if visitor:
            return visitor(node)
        # 未识别的节点类型
        self.errors.append(f"未识别节点: {type(node).__name__}")
        return ast.dump(node)

    def _visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_map = {
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
            ast.Mod: '%', ast.Pow: '**',
        }
        op = op_map.get(type(node.op), '?')
        return f'({left} {op} {right})'

    def _visit_UnaryOp(self, node):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return f'(-{operand})'
        elif isinstance(node.op, ast.UAdd):
            return f'(+{operand})'
        elif isinstance(node.op, ast.Not):
            return f'(not {operand})'
        return f'({ast.dump(node.op)} {operand})'

    def _visit_Call(self, node):
        """函数调用: 核心翻译逻辑"""
        # 获取函数名
        if isinstance(node.func, ast.Name):
            fname = node.func.id
        else:
            # 不支持的调用方式
            self.errors.append(f"不支持的函数调用: {ast.dump(node.func)}")
            return ast.dump(node)

        args = [self.visit(a) for a in node.args]

        # ── 算术前缀 → 中缀 ──
        if fname in _ARITH_OPS:
            op = _ARITH_OPS[fname]
            if len(args) == 2:
                return f'({args[0]} {op} {args[1]})'
            elif len(args) == 1 and fname == 'sub':
                # sub(x) 可能不存在，但防御性处理
                return f'(-{args[0]})'

        # ── neg → 取负 ──
        if fname == 'neg':
            return f'(-{args[0]})'

        # ── ifelse → Python 三元 if ──
        if fname == 'ifelse':
            if len(args) == 3:
                # ifelse(cond, a, b) → (a if cond else b)
                return f'({args[1]} if {args[0]} else {args[2]})'
            else:
                self.errors.append(f"ifelse 需要3个参数, 收到{len(args)}个")
                return f'ifelse({", ".join(args)})'

        # ── regbeta → ts_regression (需要加 rettype=0) ──
        if fname == 'regbeta':
            v4_fname = _FUNC_MAP.get(fname, fname)
            if len(args) == 3:
                return f'{v4_fname}({args[0]}, {args[1]}, {args[2]}, 0)'
            return f'{v4_fname}({", ".join(args)})'

        # ── 通用函数重命名 ──
        v4_fname = _FUNC_MAP.get(fname, fname)
        return f'{v4_fname}({", ".join(args)})'

    def _visit_Name(self, node):
        """变量名: 小写 → 大写"""
        name = node.id
        if name in _VAR_MAP:
            return _VAR_MAP[name]
        # 数字常量如 'nan' 等保持
        if name.lower() in ('nan', 'inf', 'true', 'false', 'none'):
            return name
        # 可能是已大写的变量名
        if name.isupper():
            return name
        # 未识别变量，保持原样并记录
        return name

    def _visit_Constant(self, node):
        """常量: 数字/字符串"""
        if isinstance(node.value, (int, float)):
            # 避免如 -1 显示为 (- 1)
            return str(node.value)
        if isinstance(node.value, str):
            return f'"{node.value}"'
        return str(node.value)

    def _visit_BoolOp(self, node):
        """布尔运算: and/or"""
        op = 'and' if isinstance(node.op, ast.And) else 'or'
        parts = [self.visit(v) for v in node.values]
        return f'({" op ".join(parts)})'.replace(' op ', f' {op} ')

    def _visit_Compare(self, node):
        """比较运算"""
        left = self.visit(node.left)
        parts = [left]
        ops = {ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=',
               ast.Eq: '==', ast.NotEq: '!='}
        for op, comp in zip(node.ops, node.comparators):
            op_str = ops.get(type(op), '?')
            parts.append(op_str)
            parts.append(self.visit(comp))
        return f'({" ".join(parts)})'

    def _visit_IfExp(self, node):
        """三元 if 表达式（v3 不太会用，但防御性支持）"""
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return f'({body} if {test} else {orelse})'


def translate_expr(expr_str: str) -> str:
    """翻译单个 v3 前缀表达式为 v4 DSL

    Args:
        expr_str: v3 表达式, 如 'sub(cs_rank(delay(close, 5)), 0.5)'

    Returns:
        v4 表达式, 如 '(cs_rank(ts_delay(CLOSE, 5)) - 0.5)'
    """
    translator = _V3ToV4Translator()
    result = translator.translate(expr_str)
    if translator.errors:
        # 附加错误信息到注释
        result += f'  # ERRORS: {"; ".join(translator.errors)}'
    return result


def translate_dict(formulas: Dict[str, str]) -> Dict[str, str]:
    """批量翻译 v3 因子字典

    Args:
        formulas: {因子名: v3表达式}, 如 ALPHA101

    Returns:
        {因子名: v4表达式}
    """
    result = {}
    errors = {}
    for name, expr in formulas.items():
        translator = _V3ToV4Translator()
        v4_expr = translator.translate(expr)
        if translator.errors:
            errors[name] = translator.errors
        result[name] = v4_expr

    if errors:
        print(f"[translate_v3] 翻译完成: {len(result)}个成功, {len(errors)}个有警告")
        for name, errs in errors.items():
            print(f"  ⚠ {name}: {'; '.join(errs)}")
    else:
        print(f"[translate_v3] 翻译完成: 全部 {len(result)} 个成功")

    return result


def validate_translation(v3_expr: str, v4_expr: str) -> Tuple[bool, str]:
    """验证翻译结果: 尝试用 v4 DSL 解析

    Returns:
        (success, error_message)
    """
    try:
        from utils.ast.dsl import parse_expression
        tree = parse_expression(v4_expr)
        if tree is None:
            return False, "解析返回 None"
        return True, ""
    except Exception as e:
        return False, str(e)


def batch_translate_and_validate(formulas: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """批量翻译并验证

    Returns:
        (成功字典, 失败字典)
    """
    translated = translate_dict(formulas)
    success = {}
    failed = {}

    for name, v4_expr in translated.items():
        # 跳过带 ERROR 注释的
        if '# ERRORS:' in v4_expr:
            failed[name] = v4_expr
            continue
        ok, err = validate_translation(formulas[name], v4_expr)
        if ok:
            success[name] = v4_expr
        else:
            failed[name] = f'{v4_expr}  # PARSE_ERROR: {err}'

    print(f"[translate_v3] 验证完成: {len(success)}个通过, {len(failed)}个失败")
    if failed:
        for name, expr in list(failed.items())[:10]:
            print(f"  ✗ {name}: {expr[:120]}...")

    return success, failed


# ============================================================
# SMA 函数注册 (v3 的 sma(x, w, k) 在 v4 中需要注册)
# ============================================================

_sma_registered = False

def register_sma():
    """注册 sma(x, window, k) 函数到 v4 registry

    SMA(x, w, k) = 前一日SMA*(1-k/w) + x*k/w
    等价于 EMA 变体, k 控制平滑速度。
    """
    global _sma_registered
    if _sma_registered:
        return

    import numpy as np
    from utils.ast import register_function

    def sma_v4(x, window, k=1):
        """SMA(x, window, k): 前值SMA*(1-k/window) + x*k/window
        兼容 v3 的 sma(x, w, k) 定义。
        """
        x = np.asarray(x, float)
        window = int(window)
        k = float(k)
        n = len(x)
        r = np.full(n, np.nan)
        if n < window:
            return r
        # 种子: 前 window 个的均值
        r[window - 1] = np.mean(x[:window])
        alpha = k / window
        for i in range(window, n):
            if np.isnan(r[i - 1]):
                r[i] = np.nan
            else:
                r[i] = r[i - 1] * (1 - alpha) + x[i] * alpha
        return r

    register_function('sma', sma_v4)
    _sma_registered = True
    print("[translate_v3] 已注册 sma(x, window, k) 到 v4 registry")


# ============================================================
# 便捷: 从 v3 公式库直接导出 v4 因子字典
# ============================================================

def get_wq101_v4() -> Dict[str, str]:
    """翻译 WQ101 因子库为 v4 DSL"""
    from factor.v3.formulas.wq101 import ALPHA101
    success, failed = batch_translate_and_validate(ALPHA101)
    return success

def get_gt191_v4() -> Dict[str, str]:
    """翻译 GT191 因子库为 v4 DSL"""
    from factor.v3.formulas.gt191 import ALPHA191
    success, failed = batch_translate_and_validate(ALPHA191)
    return success


if __name__ == '__main__':
    # 快速测试
    test_cases = {
        'alpha001': 'sub(cs_rank(ts_argmax(signed_power(ifelse(neg(returns), ts_std(returns, 20), close), 2), 5)), 0.5)',
        'alpha007': 'ifelse(sub(volume, ts_mean(volume, 20)), mul(neg(ts_rank(abs(sub(close, delay(close, 7))), 60)), sign(sub(close, delay(close, 7)))), -1)',
        'alpha101': 'div(sub(close, open), add(sub(high, low), 0.001))',
        'gtja_042': 'div(ts_mean(ifelse(sub(close, delay(close, 1)), sub(close, delay(close, 1)), 0.0), 14), add(ts_mean(abs(sub(close, delay(close, 1))), 14), 0.001))',
        'gtja_067': 'neg(ts_std(ifelse(sub(close, delay(close, 1)), 0.0, sub(close, delay(close, 1))), 20))',
    }

    print("=" * 60)
    print("v3 → v4 翻译测试")
    print("=" * 60)

    for name, v3_expr in test_cases.items():
        v4_expr = translate_expr(v3_expr)
        ok, err = validate_translation(v3_expr, v4_expr)
        status = "✓" if ok else f"✗ {err}"
        print(f"\n{name}:")
        print(f"  v3: {v3_expr[:100]}...")
        print(f"  v4: {v4_expr[:100]}...")
        print(f"  {status}")
