# -*- coding: utf-8 -*-
"""
utils/ast/v2/registry.py — 注册管理 + 宏引擎 + 变量注册
=============================================================================

[重构] 2026-07-15 方案E: 统一 register 入口 + 架构分离

融合原 _common.py + functions.py 模块9-11 + macros.py 为一个文件。

职责:
  1. 共享数据类 (VarSpec, ParamRange) + 工具函数 (_normalize_data_args, _count_pool_args)
  2. 函数注册 (FunctionSpec, FUNC_REGISTRY, register, register_function, register_macro)
  3. 宏编译 (_infer_params, _parse_macro_body, _compile_macro) + 查询/注销
  4. 变量注册 (_VAR_REGISTRY, is_valid_variable, register_variable)
  5. 内置注册调用 (~90 原语 + 5 宏)

依赖方向 (无循环):
  functions.py (纯原语) ← registry.py ← dsl.py
  registry.py 单向依赖 functions.py (取原语做 exec namespace + _register 调用)
  _parse_macro_body 延迟导入 dsl.py 的 parse_expression (避免循环)
=============================================================================
"""
import ast
import inspect
import re
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# 取原语函数做 exec namespace + _register 调用
from . import functions as _functions_mod
from .functions import *  # noqa: F401,F403  导入原语函数名 (ts_mean/ts_cov/...)
# [修复] 2026-07-15 import * 不导入 _ 开头的名字, 需显式导入特征函数
from .functions import (
    _feature_rsi, _feature_atr, _feature_atr_sma, _feature_bbwidth,
    _feature_stddev, _feature_adx, _feature_cci, _feature_macd,
    _feature_trima, _feature_ema, _wilder_smooth, _feature_tsf,
    _feature_kama, _feature_wma, _feature_dema, _feature_hv,
    _feature_natr, _feature_var, _feature_linearreg,
    _feature_vol_ratio, _feature_amt_ratio,
)


# ============================================================
# 共享数据类 (原 _common.py)
# ============================================================

@dataclass
class VarSpec:
    """变量规格 — 描述变量名、所属类别和匹配模式。"""
    name: str
    category: str = '自定义'
    is_prefix: bool = False
    description: str = ''


@dataclass
class ParamRange:
    """参数值域约束 — 描述 param_pool 无法覆盖的带范围参数。"""
    name: str
    dtype: str = 'float'
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    pool: Optional[List[Any]] = None


# ============================================================
# 共享工具函数 (原 _common.py)
# ============================================================

def _normalize_data_args(data_args: Optional[int],
                         data_vars: Optional[List[str]]) -> int:
    """统一 data_args 默认值推导 (data_vars 优先)"""
    if data_vars is not None:
        return len(data_vars)
    return data_args if data_args is not None else 1


def _count_pool_args(param_pool: Optional[List[Any]]) -> int:
    """param_pool 展开后的参数数"""
    if not param_pool:
        return 0
    first = param_pool[0]
    return len(first) if isinstance(first, tuple) else 1


# ============================================================
# 函数注册表
# ============================================================

@dataclass
class FunctionSpec:
    """函数注册项：实现 + 表达式/GP 元数据。"""
    func: Callable
    category: str
    data_args: int = 1
    param_pool: Optional[List[Any]] = None
    param_ranges: Optional[List[ParamRange]] = None
    data_vars: Optional[List[str]] = None
    description: str = ''
    macro_body: Optional[str] = None  # 非 None 即宏

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.func, name)


FUNC_REGISTRY: Dict[str, FunctionSpec] = {}
FUNC_CATEGORIES: Dict[str, List[str]] = {}
VALID_FUNC_CATEGORIES = frozenset([
    'ts_function', 'cs_function', 'math_function',
    'ta_function', 'feature_function',
])


def get_func_category(name: str) -> str:
    """查询函数所属 GP 大类"""
    spec = FUNC_REGISTRY.get(name.lower())
    return spec.category if spec is not None else 'math_function'


def _check_param_arity(name: str, func: Callable,
                       data_args: int,
                       param_pool: Optional[List[Any]],
                       param_ranges: Optional[List[ParamRange]]) -> None:
    """校验函数签名与 FunctionSpec 声明的参数数一致性"""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return
    params = list(sig.parameters.values())
    if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params):
        return
    n_required = sum(1 for p in params if p.default is p.empty)
    n_total = len(params)
    n_declared = data_args + _count_pool_args(param_pool)
    if param_ranges:
        n_declared += len(param_ranges)
    if n_declared < n_required:
        raise ValueError(
            f"register: '{name}' 签名需要 {n_required} 个位置参数, "
            f"但 spec 只声明了 {n_declared} 个. GP 会生成缺参数的调用."
        )
    elif n_declared > n_total:
        raise ValueError(
            f"register: '{name}' 签名只有 {n_total} 个参数, "
            f"但 spec 声明了 {n_declared} 个. GP 会生成多余参数."
        )


# ============================================================
# register — 统一注册入口
# ============================================================

def register(name: str, impl, category: str, *,
             data_args: Optional[int] = None,
             param_pool: Optional[List[Any]] = None,
             param_ranges: Optional[List[ParamRange]] = None,
             data_vars: Optional[List[str]] = None,
             description: str = '') -> None:
    """统一注册函数 — impl 为 Callable 是原语, 为 str 是宏

    [重构] 2026-07-16 宏路径对齐原语:
      - data_args 提前规范化 (两条路径一致, 修复 data_vars 在宏路径被忽略的 bug)
      - data_vars 参与宏形参推导 (形参名 = data_vars[i].lower(), 与原语签名同构)
      - _validate_macro_body 替代 _parse_macro_body (线程安全, 不修改全局 _VAR_REGISTRY)
      - _compile_macro 用独立 _MACRO_NAMESPACE (不污染 functions 模块)
    """
    name_lower = name.lower()

    # [修复] 2026-07-16 统一提前规范化 data_args, 保障宏/原语两条路径一致
    data_args = _normalize_data_args(data_args, data_vars)

    if isinstance(impl, str):
        # 宏路径: str → Callable (编译后与原语无差别)
        params = _infer_params(data_args, param_pool, param_ranges, data_vars)
        _validate_macro_body(impl, params)
        func = _compile_macro(name_lower, params, impl)
        macro_body = impl
    else:
        # 原语路径: Callable 直接用
        func = impl
        macro_body = None

    if category not in VALID_FUNC_CATEGORIES:
        raise ValueError(
            f"无效函数分类 '{category}' (函数 '{name_lower}')。"
            f"有效分类: {sorted(VALID_FUNC_CATEGORIES)}"
        )
    # 统一签名校验 (宏/原语走同一个 _check_param_arity)
    _check_param_arity(name_lower, func, data_args, param_pool, param_ranges)
    FUNC_REGISTRY[name_lower] = FunctionSpec(
        func=func, category=category, data_args=data_args,
        param_pool=param_pool, param_ranges=param_ranges,
        data_vars=data_vars, description=description,
        macro_body=macro_body,
    )
    if category not in FUNC_CATEGORIES:
        FUNC_CATEGORIES[category] = []
    if name_lower not in FUNC_CATEGORIES[category]:
        FUNC_CATEGORIES[category].append(name_lower)


# 向后兼容别名
_register = register


# ============================================================
# 内置原语注册 (~90 个)
# ============================================================

# ── 时序 (ts_) ──
_register('ts_mean', ts_mean, 'ts_function', param_pool=[5, 10, 20, 60])
_register('ts_std', ts_std, 'ts_function', param_pool=[10, 20, 60])
_register('ts_sum', ts_sum, 'ts_function', param_pool=[5, 10, 20])
_register('ts_max', ts_max, 'ts_function', param_pool=[10, 20])
_register('ts_min', ts_min, 'ts_function', param_pool=[10, 20])
_register('ts_median', ts_median, 'ts_function', param_pool=[10, 20])
_register('ts_delta', ts_delta, 'ts_function', param_pool=[1, 5, 10, 20])
_register('ts_delay', ts_delay, 'ts_function', param_pool=[1, 5, 10, 20])
_register('ts_rank', ts_rank, 'ts_function', param_pool=[5, 10, 20, 60])
_register('ts_corr', ts_corr, 'ts_function', data_args=2, param_pool=[10, 20, 60])
_register('ts_skew', ts_skew, 'ts_function', param_pool=[20, 60])
_register('ts_kurt', ts_kurt, 'ts_function', param_pool=[20, 60])
_register('ts_argmax', ts_argmax, 'ts_function', param_pool=[10, 20])
_register('ts_argmin', ts_argmin, 'ts_function', param_pool=[10, 20])
_register('ts_roc', ts_roc, 'ts_function', param_pool=[5, 10, 20])
_register('ts_cov', ts_cov, 'ts_function', data_args=2, param_pool=[5, 10, 20])
_register('ts_var', lambda x, d: ts_std(x, d) ** 2, 'ts_function', param_pool=[10, 20])
_register('logret', lambda x: safe_log(x / ts_delay(x, 1)), 'math_function')
_register('ts_zscore', ts_zscore, 'ts_function', param_pool=[10, 20, 60])
_register('ts_autocorr', ts_autocorr, 'ts_function', param_pool=[(1, 10), (5, 20), (10, 60)])
_register('ts_step', ts_step, 'ts_function', param_pool=[5, 10, 20])
_register('ts_hump', ts_hump, 'ts_function', param_pool=[10, 20])
_register('ts_scale', ts_scale, 'ts_function', param_pool=[10, 20])
_register('ts_quantile', ts_quantile, 'ts_function', param_pool=[5, 10, 20],
          param_ranges=[ParamRange('p', 'float', 0.0, 1.0)])
_register('ts_av_diff', ts_av_diff, 'ts_function', param_pool=[10, 20])
_register('ts_decay_linear', ts_decay_linear, 'ts_function', param_pool=[5, 10, 20])
_register('ts_product', ts_product, 'ts_function', param_pool=[10, 20])
# ── 双变量回归 reg_ ──
_register('reg_slope', lambda y, x, d: regression(y, x, d, 0), 'ts_function', data_args=2, param_pool=[5, 10])
_register('reg_intercept', lambda y, x, d: regression(y, x, d, 1), 'ts_function', data_args=2, param_pool=[5, 10])
_register('reg_resid', lambda y, x, d: regression(y, x, d, 2), 'ts_function', data_args=2, param_pool=[5, 10])
_register('reg_predict', lambda y, x, d: regression(y, x, d, 3), 'ts_function', data_args=2, param_pool=[5, 10])
_register('reg_rsq', lambda y, x, d: regression(y, x, d, 4), 'ts_function', data_args=2, param_pool=[5, 10])
_register('ts_slope', ts_slope, 'ts_function', param_pool=[10, 20])
_register('ts_intercept', ts_intercept, 'ts_function', param_pool=[10, 20])
_register('ts_resid', ts_resid, 'ts_function', param_pool=[10, 20])
_register('ts_predict', ts_predict, 'ts_function', param_pool=[10, 20])
_register('ts_rsq', ts_rsq, 'ts_function', param_pool=[10, 20])
_register('ts_ar_resid', ts_ar_resid, 'ts_function', param_pool=[3, 5, 10])

# ── 扩张统计 (expanding_) ──
_register('expanding_mean', expanding_mean, 'ts_function', param_pool=[20, 60])
_register('expanding_median', expanding_median, 'ts_function', param_pool=[20, 60])
_register('expanding_std', expanding_std, 'ts_function', param_pool=[20, 60])
_register('expanding_percentile', expanding_percentile, 'ts_function', param_pool=[(0.1, 20), (0.5, 20), (0.9, 20)])

# ── 截面 (cs_) ──
_register('cs_rank', cs_rank, 'cs_function')
_register('cs_zscore', cs_zscore, 'cs_function')
_register('cs_scale', cs_scale, 'cs_function',
          param_ranges=[ParamRange('scale', 'float', 0.5, 2.0)])
_register('cs_winsorize', cs_winsorize, 'cs_function',
          param_ranges=[ParamRange('std', 'float', 2.0, 5.0)])
_register('cs_quantile', cs_quantile, 'cs_function')
_register('cs_normalize', cs_normalize, 'cs_function')

# ── 数学 ──
_register('abs', safe_abs, 'math_function')
_register('log', safe_log, 'math_function')
_register('sqrt', safe_sqrt, 'math_function')
_register('sign', safe_sign, 'math_function')
_register('exp', safe_exp, 'math_function')
_register('tanh', safe_tanh, 'math_function')
_register('sigmoid', safe_sigmoid, 'math_function')
_register('relu', safe_relu, 'math_function')
_register('softsign', safe_softsign, 'math_function')
_register('sin', lambda x: np.sin(x), 'math_function')
_register('cos', lambda x: np.cos(x), 'math_function')
_register('gauss', safe_gauss, 'math_function')
_register('p4', safe_p4, 'math_function')
_register('neg', safe_neg, 'math_function')
_register('square_sigmoid', safe_square_sigmoid, 'math_function')
_register('signed_power', signed_power, 'math_function',
          param_ranges=[ParamRange('exponent', 'float', 0.5, 4.0)])
_register('max', safe_max, 'math_function', data_args=2)
_register('min', safe_min, 'math_function', data_args=2)

# ── 信号 ──
_register('persist', persist, 'ts_function', param_pool=[3, 5, 10])

# ── 特征计算 ──
_register('rsi', _feature_rsi, 'ta_function', param_pool=[14, 20])
_register('atr', _feature_atr, 'ta_function', data_args=3, param_pool=[14], data_vars=['HIGH', 'LOW', 'CLOSE'])
_register('atr_sma', _feature_atr_sma, 'ta_function', data_args=3, param_pool=[14], data_vars=['HIGH', 'LOW', 'CLOSE'])
_register('bb_width', _feature_bbwidth, 'ta_function', param_pool=[20])
_register('stddev', _feature_stddev, 'ta_function', param_pool=[10, 20])
_register('adx', _feature_adx, 'ta_function', data_args=3, param_pool=[14, 20], data_vars=['HIGH', 'LOW', 'CLOSE'])
_register('cci', _feature_cci, 'ta_function', data_args=3, param_pool=[14, 20], data_vars=['HIGH', 'LOW', 'CLOSE'])
_register('macd', _feature_macd, 'ta_function', param_pool=[(12, 26, 9)])
_register('trima', _feature_trima, 'ta_function', param_pool=[40])
_register('ema', _feature_ema, 'ta_function', param_pool=[5, 10, 20, 60])
_register('wilder_smooth', _wilder_smooth, 'feature_function', param_pool=[10, 20])
_register('tsf', _feature_tsf, 'ta_function', param_pool=[10, 20])
_register('kama', _feature_kama, 'ta_function', param_pool=[30])
_register('wma', _feature_wma, 'ta_function', param_pool=[5, 10, 20, 60])
_register('dema', _feature_dema, 'ta_function', param_pool=[10, 20])
_register('hv', _feature_hv, 'ta_function', param_pool=[20, 60])
_register('natr', _feature_natr, 'ta_function', data_args=3, param_pool=[5, 14], data_vars=['HIGH', 'LOW', 'CLOSE'])
_register('var', _feature_var, 'ta_function', param_pool=[10, 20])
_register('linearreg', _feature_linearreg, 'ta_function', param_pool=[10, 20])
_register('vol_ratio', _feature_vol_ratio, 'ta_function', param_pool=[(5, 20)], data_vars=['CLOSE', 'VOLUME'])
_register('amt_ratio', _feature_amt_ratio, 'ta_function', param_pool=[(5, 20)], data_vars=['AMOUNT'])

# ── 旧名别名 ──
_register('ts_resi', ts_resid, 'ts_function', param_pool=[10, 20])
_register('ts_regression_residual', lambda y, x, d: regression(y, x, d, 2), 'ts_function', data_args=2, param_pool=[5, 10])
_register('ts_rsquare', ts_rsq, 'ts_function', param_pool=[10, 20])
_register('ts_logret', lambda x: safe_log(x / ts_delay(x, 1)), 'math_function')


# ============================================================
# 内置宏注册
# ============================================================

_BUILTIN_MACROS_REGISTERED = False


def _register_builtin_macros() -> None:
    """注册内置宏定义 (延迟调用, 避免循环导入)"""
    global _BUILTIN_MACROS_REGISTERED
    if _BUILTIN_MACROS_REGISTERED:
        return
    _BUILTIN_MACROS_REGISTERED = True

    # ── 风险指标 ──
    register('beta', 'ts_cov(x, y, d) / (ts_std(y, d) ** 2)', 'ts_function',
             data_args=2, param_pool=[20, 60],
             description='Beta系数: Cov(资产,市场) / Var(市场)')
    register('sharpe', 'ts_mean(x, d) / ts_std(x, d)', 'ts_function',
             data_args=1, param_pool=[20, 60],
             description='时序夏普比率: 均值 / 标准差')
    register('info_ratio', 'ts_mean(x - y, d) / ts_std(x - y, d)', 'ts_function',
             data_args=2, param_pool=[20, 60],
             description='信息比率: 超额收益均值 / 跟踪误差')
    register('ts_deviate', '(x - ts_mean(x, d)) / ts_std(x, d)', 'ts_function',
             data_args=1, param_pool=[20, 60],
             description='偏离度: (当前值 - 均值) / 标准差')
    # ── 量价复合 ──
    register('vol_price_corr', 'ts_corr(ts_roc(x, d), ts_roc(y, d), d)', 'ts_function',
             data_args=2, param_pool=[10, 20],
             description='量价相关: ROC(x) 与 ROC(y) 的相关系数')


# ============================================================
# 宏编译命名空间 (独立于 functions 模块, 支持隔离注销)
# [新增] 2026-07-16 宏函数不再注入 _functions_mod.__dict__,
#        避免污染原语模块; 注销时从独立 namespace 清理
# ============================================================

_MACRO_NAMESPACE: Dict[str, Any] = {}


def _get_macro_namespace() -> Dict[str, Any]:
    """获取宏编译命名空间 (延迟初始化, 包含所有原语函数)

    宏体编译时需要访问原语函数 (如 ts_mean/ts_corr),
    通过复制 functions 模块的可调用对象到独立 namespace 实现。
    首次调用时初始化, 后续宏注册共享同一 namespace (支持宏调用宏)。
    """
    if not _MACRO_NAMESPACE:
        # 复制 functions 模块的原语函数, 不包含 dunder 和模块属性
        for k, v in vars(_functions_mod).items():
            if callable(v) and not k.startswith('__'):
                _MACRO_NAMESPACE[k] = v
        # 补充 numpy (宏体可能用 np.where 等)
        _MACRO_NAMESPACE['np'] = np
    return _MACRO_NAMESPACE


# ============================================================
# 宏编译工具 (同文件, 无跨文件调用)
# ============================================================

_MACRO_NAME_RE = re.compile(r'^[a-z][a-z0-9_]*$')


def _infer_params(data_args: Optional[int],
                  param_pool: Optional[List],
                  param_ranges: Optional[List[ParamRange]],
                  data_vars: Optional[List[str]] = None) -> List[str]:
    """推导宏的形参名 (对齐原语函数签名风格: snake_case)

    优先级:
      data_vars[i].lower()  → 语义形参名 (如 high/low/close, 与 _feature_atr 同构)
      x/y/z/x4/...          → 自动推导 (无 data_vars 时)
      d/d2/d3/...           → 窗口参数
      ParamRange.name       → 范围参数 (原样)

    [重构] 2026-07-16 接受 data_vars, 形参名与原语函数签名对齐
    """
    da = _normalize_data_args(data_args, data_vars)
    params = []
    for i in range(da):
        if data_vars and i < len(data_vars):
            params.append(data_vars[i].lower())  # HIGH → high
        elif i < 3:
            params.append(['x', 'y', 'z'][i])
        else:
            params.append(f'x{i+1}')
    n_pool = _count_pool_args(param_pool)
    for i in range(n_pool):
        params.append('d' if i == 0 else f'd{i+1}')
    if param_ranges:
        for pr in param_ranges:
            params.append(pr.name)
    return params


def _validate_macro_body(body: str, params: List[str]) -> None:
    """校验宏体 (线程安全, 不修改全局 _VAR_REGISTRY)

    校验规则:
      1. Python 语法 (ast.parse mode='eval')
      2. AST 安全白名单 (复用 dsl 的节点校验, 跳过变量名校验)
      3. 函数调用必须在 FUNC_REGISTRY
      4. 变量引用必须 ⊆ params (形参) 或 SAFE_CONSTANTS

    与原语的区别:
      原语函数体是 Python 代码, 形参由函数签名决定, 自由变量是 import 的名字。
      宏体是 DSL 表达式, 所有变量必须是形参 (不能引用数据变量或全局名)。
      这保证宏的"参数-体"契约与原语一致: 调用方传入的实参唯一决定宏体行为。

    [重构] 2026-07-16 替代 _parse_macro_body, 线程安全:
      - 不再临时修改全局 _VAR_REGISTRY (消除竞态)
      - 不再调用 _sync_var_backward_compat() (避免重复全表重建)
      - 变量校验改为局部 param_set 查询
    """
    from .dsl import (FORBIDDEN_NODE_TYPES, ALLOWED_NODE_TYPES,
                      DSLSecurityError, DSLSyntaxError)

    try:
        tree = ast.parse(body, mode='eval')
    except SyntaxError as e:
        raise DSLSyntaxError(
            f"宏体语法错误: {e.msg} (行{e.lineno}, 列{e.offset})"
        )

    param_set = {p.lower() for p in params}

    def _check(node):
        # 节点白名单/黑名单 (复用 dsl 的常量)
        for fb in FORBIDDEN_NODE_TYPES:
            if isinstance(node, fb):
                raise DSLSecurityError(
                    f"宏体禁止节点: {type(node).__name__}。"
                    f"只允许数学表达式和函数调用。"
                )
        if not any(isinstance(node, a) for a in ALLOWED_NODE_TYPES):
            raise DSLSecurityError(
                f"宏体不允许节点: {type(node).__name__}"
            )

        # 函数调用校验
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in FUNC_REGISTRY:
                    raise DSLSecurityError(
                        f"宏体引用未注册函数: '{node.func.id}'。"
                        f"可用: {sorted(FUNC_REGISTRY.keys())}"
                    )
            else:
                raise DSLSecurityError("宏体只允许直接函数调用")

        # 变量引用校验: 必须是形参或安全常量
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            name = node.id
            if name in SAFE_CONSTANTS:
                return  # True/False/None/pi/e
            if name.lower() in FUNC_REGISTRY:
                return  # 函数名出现在 Name 节点 (罕见, 放行)
            if name.lower() not in param_set:
                raise DSLSecurityError(
                    f"宏体引用未声明变量 '{name}'。"
                    f"声明的形参: {sorted(param_set)}。"
                    f"宏体只能引用形参, 不能引用数据变量 (由调用方传入)。"
                )

        for child in ast.iter_child_nodes(node):
            _check(child)

    _check(tree.body)


def _compile_macro(name: str, params: List[str], body: str) -> Callable:
    """把宏体编译为 Python 函数, 注入独立 _MACRO_NAMESPACE

    [重构] 2026-07-16 不再注入 _functions_mod.__dict__, 避免污染原语模块。
           宏函数驻留 _MACRO_NAMESPACE, 注销时独立清理。
           宏体可调用所有原语函数 (namespace 初始化时复制) 和已注册宏。
    """
    if not _MACRO_NAME_RE.match(name):
        raise ValueError(
            f"宏名 '{name}' 不合法, 必须是 snake_case "
            f"(小写字母开头, 仅含 a-z0-9_)"
        )
    ns = _get_macro_namespace()
    param_list = ', '.join(params)
    func_src = f"def {name}({param_list}):\n    return {body}"
    exec(compile(func_src, f'<macro:{name}>', 'exec'), ns)
    func = ns[name]
    func.__doc__ = f"宏: {name}({param_list}) = {body}"
    return func


# ============================================================
# 宏查询/注销
# ============================================================

def is_macro(name: str) -> bool:
    """判断函数是否为宏 (macro_body is not None)"""
    spec = FUNC_REGISTRY.get(name.lower())
    return spec is not None and spec.macro_body is not None


def list_macros() -> Dict[str, FunctionSpec]:
    """列出所有已注册的宏"""
    return {k: v for k, v in FUNC_REGISTRY.items() if v.macro_body is not None}


def macro_to_str(name: str) -> str:
    """返回宏定义的人类可读字符串"""
    name_lower = name.lower()
    spec = FUNC_REGISTRY.get(name_lower)
    if spec is None or spec.macro_body is None:
        raise KeyError(f"'{name}' 不是宏")
    # [修复] 2026-07-16 传入 data_vars, 形参名与编译时一致
    params = _infer_params(spec.data_args, spec.param_pool, spec.param_ranges,
                           spec.data_vars)
    return f"{name_lower}({', '.join(params)}) = {spec.macro_body}"


def unregister_macro(name: str) -> bool:
    """注销宏定义, 返回是否成功"""
    name_lower = name.lower()
    spec = FUNC_REGISTRY.get(name_lower)
    if spec is None or spec.macro_body is None:
        return False
    unregister_function(name_lower)
    # [修复] 2026-07-16 从独立 _MACRO_NAMESPACE 清理, 而非 _functions_mod
    _MACRO_NAMESPACE.pop(name_lower, None)
    return True


# ============================================================
# 变量注册
# ============================================================

SAFE_CONSTANTS = {'True': 1.0, 'False': 0.0, 'None': 0.0, 'pi': np.pi, 'e': np.e}

_VAR_REGISTRY: Dict[str, VarSpec] = {
    'OPEN':    VarSpec('OPEN', '原始OHLCV', description='开盘价'),
    'HIGH':    VarSpec('HIGH', '原始OHLCV', description='最高价'),
    'LOW':     VarSpec('LOW', '原始OHLCV', description='最低价'),
    'CLOSE':   VarSpec('CLOSE', '原始OHLCV', description='收盘价'),
    'VOLUME':  VarSpec('VOLUME', '原始OHLCV', description='成交量'),
    'AMOUNT':  VarSpec('AMOUNT', '原始OHLCV', description='成交额'),
    'VWAP':    VarSpec('VWAP', '原始OHLCV', description='均价 (WQ065)'),
    'RETURNS': VarSpec('RETURNS', '原始OHLCV', description='收益率'),
    'REL':   VarSpec('REL', '相对基准', is_prefix=True, description='REL_CLOSE/REL_AMOUNT/REL_VOLUME'),
    'BENCH': VarSpec('BENCH', '相对基准', is_prefix=True, description='BENCH_CLOSE/BENCH_RETURNS'),
    'SHARE': VarSpec('SHARE', '市场份额', description='跨品种成交额占比'),
    'DOWNSIDE_VOL': VarSpec('DOWNSIDE_VOL', '下行风险', description='下行标准差'),
    'PE_TTM_INDEX':  VarSpec('PE_TTM_INDEX', '基本面', description='滚动市盈率'),
    'PB_MRQ':        VarSpec('PB_MRQ', '基本面', description='市净率'),
    'TURNOVERRATIO': VarSpec('TURNOVERRATIO', '基本面', description='换手率'),
    'TOTALCAPITAL':  VarSpec('TOTALCAPITAL', '基本面', description='总市值'),
    'ATR':    VarSpec('ATR', '波动率'),
    'STDDEV': VarSpec('STDDEV', '波动率'),
    'HV':     VarSpec('HV', '波动率'),
    'NATR':   VarSpec('NATR', '波动率'),
    'BBWIDTH': VarSpec('BBWIDTH', '通道指标'),
    'TRIMA': VarSpec('TRIMA', '趋势指标'), 'SMA': VarSpec('SMA', '趋势指标'),
    'MA':    VarSpec('MA', '趋势指标'), 'EMA': VarSpec('EMA', '趋势指标'),
    'TSF':   VarSpec('TSF', '趋势指标'), 'WMA': VarSpec('WMA', '趋势指标'),
    'DEMA':  VarSpec('DEMA', '趋势指标'), 'KAMA': VarSpec('KAMA', '趋势指标'),
    'ADX':   VarSpec('ADX', '趋势指标'),
    'RSI':   VarSpec('RSI', '动量指标'), 'CCI': VarSpec('CCI', '动量指标'),
    'MACD':  VarSpec('MACD', '动量指标'), 'MFI': VarSpec('MFI', '动量指标'),
    'ULTOSC': VarSpec('ULTOSC', '动量指标'), 'ROC': VarSpec('ROC', '动量指标'),
    'LINEARREG': VarSpec('LINEARREG', '统计指标'),
    'VAR':   VarSpec('VAR', '统计指标'), 'CORREL': VarSpec('CORREL', '统计指标'),
    'VOL_RATIO': VarSpec('VOL_RATIO', '量价指标'), 'VOL_CHG': VarSpec('VOL_CHG', '量价指标'),
    'OBV':   VarSpec('OBV', '量价指标'), 'UP_RATIO': VarSpec('UP_RATIO', '量价指标'),
    'AVGPRICE': VarSpec('AVGPRICE', '价格水平'), 'WCLPRICE': VarSpec('WCLPRICE', '价格水平'),
    'SECTOR_UP': VarSpec('SECTOR_UP', '市场宽度'), 'SECTOR_MOM': VarSpec('SECTOR_MOM', '市场宽度'),
    'SECTOR_AD': VarSpec('SECTOR_AD', '市场宽度'),
    'BREADTH_S': VarSpec('BREADTH_S', '市场宽度'), 'BREADTH_M': VarSpec('BREADTH_M', '市场宽度'),
    'BREADTH_L': VarSpec('BREADTH_L', '市场宽度'), 'BREADTH_AMT': VarSpec('BREADTH_AMT', '市场宽度'),
    'DISP':   VarSpec('DISP', '市场结构'), 'ROTSPD': VarSpec('ROTSPD', '市场结构'),
    'NHL':    VarSpec('NHL', '市场结构'), 'SKEW': VarSpec('SKEW', '市场结构'),
    'IND_CORR': VarSpec('IND_CORR', '市场结构'),
    'VMED':   VarSpec('VMED', '资金结构'), 'VDISP': VarSpec('VDISP', '资金结构'),
    'VSKEW':  VarSpec('VSKEW', '资金结构'),
    'TAILUP': VarSpec('TAILUP', '尾部风险'), 'TAILDOWN': VarSpec('TAILDOWN', '尾部风险'),
    'TAILNET': VarSpec('TAILNET', '尾部风险'),
}

VALID_VAR_PREFIXES: List[str] = list(_VAR_REGISTRY.keys())
VAR_CATEGORIES: Dict[str, list] = {}
for _spec in _VAR_REGISTRY.values():
    VAR_CATEGORIES.setdefault(_spec.category, []).append(_spec.name)


def get_var_category(name: str) -> str:
    """查询变量所属分类"""
    upper = name.upper()
    spec = _VAR_REGISTRY.get(upper)
    if spec is not None:
        return spec.category
    for spec in _VAR_REGISTRY.values():
        if spec.is_prefix and upper.startswith(spec.name + '_'):
            return spec.category
    return '自定义'


def is_valid_variable(name: str) -> bool:
    """检查变量名是否合法（匹配已注册变量）"""
    upper = name.upper()
    if upper in _VAR_REGISTRY:
        return True
    for spec in _VAR_REGISTRY.values():
        if spec.is_prefix and upper.startswith(spec.name + '_'):
            rest = upper[len(spec.name) + 1:]
            if rest and all(c.isascii() and (c.isalnum() or c == '_') for c in rest):
                return True
    return False


def register_variable(name: str, category: str = '自定义',
                      is_prefix: bool = False,
                      description: str = '') -> None:
    """临时注册自定义变量到表达式引擎。"""
    upper = name.upper()
    if upper not in _VAR_REGISTRY:
        _VAR_REGISTRY[upper] = VarSpec(upper, category=category, is_prefix=is_prefix, description=description)
        _sync_var_backward_compat()


def unregister_variable(name: str) -> bool:
    """注销自定义变量，返回是否成功。"""
    upper = name.upper()
    if upper in _VAR_REGISTRY:
        del _VAR_REGISTRY[upper]
        _sync_var_backward_compat()
        return True
    return False


def _sync_var_backward_compat():
    """同步 VALID_VAR_PREFIXES 和 VAR_CATEGORIES 与 _VAR_REGISTRY 一致"""
    global VALID_VAR_PREFIXES, VAR_CATEGORIES
    VALID_VAR_PREFIXES = list(_VAR_REGISTRY.keys())
    cats = {}
    for spec in _VAR_REGISTRY.values():
        cats.setdefault(spec.category, []).append(spec.name)
    VAR_CATEGORIES = cats


# ============================================================
# 用户便捷注册 API (覆盖警告)
# ============================================================

def register_function(
    name: str,
    func: Callable,
    category: str = 'math_function',
    data_args: Optional[int] = None,
    param_pool: Optional[List[Any]] = None,
    param_ranges: Optional[List[ParamRange]] = None,
    data_vars: Optional[List[str]] = None,
    description: str = '',
) -> None:
    """临时注册自定义函数到表达式引擎 (覆盖时警告)"""
    name_lower = name.lower()
    if name_lower in FUNC_REGISTRY:
        import warnings
        warnings.warn(
            f"register_function: '{name}' 已存在，将被覆盖。"
            f"原函数: {FUNC_REGISTRY[name_lower].__name__}"
        )
    register(name_lower, func, category, data_args=data_args,
             param_pool=param_pool, param_ranges=param_ranges,
             data_vars=data_vars, description=description)


def register_macro(
    name: str,
    body: str,
    category: str = 'math_function',
    data_args: Optional[int] = None,
    param_pool: Optional[List[Any]] = None,
    param_ranges: Optional[List[ParamRange]] = None,
    data_vars: Optional[List[str]] = None,
    description: str = '',
) -> None:
    """临时注册宏函数 (DSL 短语封装, 覆盖时警告)"""
    name_lower = name.lower()
    if name_lower in FUNC_REGISTRY:
        import warnings
        warnings.warn(
            f"register_macro: '{name}' 已存在，将被覆盖。"
            f"原函数: {FUNC_REGISTRY[name_lower].__name__}"
        )
    register(name_lower, body, category, data_args=data_args,
             param_pool=param_pool, param_ranges=param_ranges,
             data_vars=data_vars, description=description)


def unregister_function(name: str) -> bool:
    """注销自定义函数，返回是否成功。"""
    name_lower = name.lower()
    removed = FUNC_REGISTRY.pop(name_lower, None) is not None
    if removed:
        for cat_names in FUNC_CATEGORIES.values():
            if name_lower in cat_names:
                cat_names.remove(name_lower)
                break
    return removed
