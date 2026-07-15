# -*- coding: utf-8 -*-
"""
utils/ast/v2/macros.py — 宏定义引擎 (AST 短语封装函数)
=============================================================================

在五层架构中的位置: 宏层 — 用 DSL 表达式字符串封装可复用的小函数

核心思想:
  把"复合表达式"封装成一个新函数名, 调用时等价于表达式模板编译。
  避免为每个复合算子写完整 Python def, 一行 DSL 短语即可封装。

用法:
  >>> from utils.ast.v2 import register_macro
  >>> register_macro('beta', 'ts_cov(x, y, d) / (ts_std(y, d) ** 2)',
  ...                category='ts_function', data_args=2, param_pool=[20, 60])
  >>> # 之后在表达式中可直接用:
  >>> expr = "beta(RETURNS, BENCH_RETURNS, 20) > 1.2"

参数命名规范 (自动推导, 遵循原语约定):
  数据参数: x, y, z (按 data_args 顺序)
  窗口参数: d, d2, d3 (按 param_pool 顺序)
  范围参数: 用 ParamRange.name
  宏体中必须使用这些参数名 (parse_expression 会校验)

实现机制 (路径A: exec 编译):
  1. _infer_params: 按 data_args/param_pool/param_ranges 推导参数名
  2. _parse_macro_body: 临时放行 params, 用 parse_expression 校验宏体
  3. _compile_macro: exec 编译为 Python 函数, 放入 functions 模块 globals()
  4. register_function: 注册到 FUNC_REGISTRY (对求值器/GP 透明)

依赖顺序:
  无。exec 用 functions 模块 globals() 作为 namespace, 所有原语函数
  (ts_mean/ts_cov/...) 和已编译的宏函数都在其中, 后注册的对先注册的
  也可见 (调用时才查找名字, 与 Python def 一致)。

  内置宏定义注册留在 functions.py (_register_builtin_macros), 由
  __init__.py 延迟调用, 避免循环导入。

[新增] 2026-07-15 宏定义引擎, 路径A: exec 编译为 Python 函数
[重构] 2026-07-15 从 functions.py 模块十二拆出, 引擎逻辑独立
=============================================================================
"""
import ast
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# [新增] 2026-07-15 宏名合法性校验: snake_case (小写字母开头, 仅含 a-z0-9_)
_MACRO_NAME_RE = re.compile(r'^[a-z][a-z0-9_]*$')

# 引用 functions 模块 (用于 _compile_macro 的 exec namespace)
# functions.py 的 globals() 包含所有原语函数 (ts_mean/ts_cov/...) 和 numpy
from . import functions as _functions_mod
from ._common import (
    ParamRange, VarSpec,
    _normalize_data_args, _count_pool_args,
)
from .functions import (
    register_function, unregister_function,
    _VAR_REGISTRY, _sync_var_backward_compat,
)
from .dsl import parse_expression


# ============================================================
# MacroSpec — 宏定义规格
# ============================================================

@dataclass
class MacroSpec:
    """宏定义规格 — 描述一个用 AST 短语封装的函数

    Attributes:
        name: 宏名 (snake_case, 表达式中用)
        params: 参数名列表 (自动推导, 遵循 x/y/d 规范)
        body_str: 宏体原始字符串 (DSL 表达式)
        body_tree: 宏体 AST 树 (解析后缓存, 用于自省)
        category: GP 大类
        data_args: 数据序列参数个数
        param_pool: 配置参数候选池
        param_ranges: 配置参数值域约束
        data_vars: 固定变量名列表
        description: 人类可读描述
    """
    name: str
    params: List[str]
    body_str: str
    body_tree: ast.Expression
    category: str
    data_args: int
    param_pool: Optional[List[Any]] = None
    param_ranges: Optional[List[ParamRange]] = None
    data_vars: Optional[List[str]] = None
    description: str = ''


# 宏注册表 — 全局单例
MACRO_REGISTRY: Dict[str, MacroSpec] = {}


# ============================================================
# 内部工具: 参数自动推导
# ============================================================

def _infer_params(data_args: Optional[int],
                  param_pool: Optional[List[Any]],
                  param_ranges: Optional[List[ParamRange]]) -> List[str]:
    """按 x/y/d 规范自动推导参数名

    推导规则:
      - 数据参数 (data_args 个): x, y, z (按顺序, 超过3个用 x4, x5...)
      - param_pool 参数: d, d2, d3 (tuple 展开为多个)
      - param_ranges 参数: 用 ParamRange.name

    Examples:
        data_args=1, param_pool=[5,10,20] → ['x', 'd']
        data_args=2, param_pool=[20,60] → ['x', 'y', 'd']
        data_args=2, param_pool=[(5,20)] → ['x', 'y', 'd', 'd2']
        data_args=1, param_ranges=[ParamRange('p')] → ['x', 'p']
    """
    params = []
    # 数据参数: x, y, z
    # [重构] 2026-07-15 提取 _normalize_data_args 消除重复
    da = _normalize_data_args(data_args, None)
    for i in range(da):
        if i < 3:
            params.append(['x', 'y', 'z'][i])
        else:
            params.append(f'x{i+1}')
    # param_pool 参数
    # [重构] 2026-07-15 提取 _count_pool_args 消除重复
    n_pool = _count_pool_args(param_pool)
    for i in range(n_pool):
        params.append('d' if i == 0 else f'd{i+1}')
    # param_ranges 参数
    if param_ranges:
        for pr in param_ranges:
            params.append(pr.name)
    return params


# ============================================================
# 内部工具: 解析宏体 (临时放行参数名)
# ============================================================

def _parse_macro_body(body: str, params: List[str]) -> ast.Expression:
    """解析宏体表达式, 临时放行 params 中的参数名

    普通 parse_expression 会拒绝 'x', 'd' 等未注册变量名,
    宏定义需放行这些参数占位符。

    实现: 临时把 params 注册到 _VAR_REGISTRY, 解析后立即注销。
    try/finally 保证即使解析失败也能清理, 不污染全局状态。
    """
    added = []
    try:
        for p in params:
            upper = p.upper()
            if upper not in _VAR_REGISTRY:
                _VAR_REGISTRY[upper] = VarSpec(upper, category='宏参数',
                                               description='宏参数占位符')
                added.append(upper)
        if added:
            _sync_var_backward_compat()
        return parse_expression(body)
    finally:
        for upper in added:
            _VAR_REGISTRY.pop(upper, None)
        if added:
            _sync_var_backward_compat()


# ============================================================
# 内部工具: 编译宏体为 Python 函数
# ============================================================

def _compile_macro(name: str, params: List[str], body: str) -> Callable:
    """把宏体编译为 Python 函数, 放入 functions 模块 globals()

    通过 exec 构造函数定义:
        def {name}({params}):
            return {body}

    用 functions 模块的 globals() 作为 namespace, 所有原语函数
    (ts_mean/ts_cov/...) 和已编译的宏函数都在其中, 后注册的对先注册
    的也可见 (调用时才查找名字, 与 Python def 一致, 无依赖顺序问题)。

    编译后的函数签名与 params 一致, 调用时参数自然代入。
    """
    param_list = ', '.join(params)
    func_src = f"def {name}({param_list}):\n    return {body}"

    # 用 functions 模块 globals() 作为 namespace
    # 宏函数放入 functions 模块级 (与 ts_mean 等原语同级)
    # [重构] 2026-07-15 从 functions.py 拆出, 改用 _functions_mod.__dict__
    exec(compile(func_src, f'<macro:{name}>', 'exec'), _functions_mod.__dict__)
    func = _functions_mod.__dict__[name]
    func.__doc__ = f"宏定义: {name}({param_list}) = {body}"
    return func


# ============================================================
# 公共 API
# ============================================================

def register_macro(name: str, body: str,
                   category: str = 'math_function',
                   data_args: Optional[int] = None,
                   param_pool: Optional[List[Any]] = None,
                   param_ranges: Optional[List[ParamRange]] = None,
                   data_vars: Optional[List[str]] = None,
                   description: str = '') -> None:
    """通过 AST 短语注册自定义函数 (宏定义)

    把一个 DSL 表达式封装成可调用的函数, 注册到 FUNC_REGISTRY。
    注册后可在表达式中直接使用, 对求值器/GP 完全透明。

    参数名自动推导, 遵循原语规范 (x/y/d), 无需手动指定。
    宏体中必须使用这些参数名 (parse_expression 会校验)。

    Args:
        name: 宏名 (snake_case, 表达式中用)
        body: 宏体表达式字符串 (DSL 语法, 如 'ts_cov(x, y, d) / ts_std(y, d)**2')
        category: GP 大类 (ts_function/cs_function/math_function/ta_function/feature_function)
        data_args: 数据序列参数个数 (GP 生成子树数), 默认1
        param_pool: 配置参数候选池 (如 [20, 60])
        param_ranges: 配置参数值域约束列表
        data_vars: 固定变量名列表 (如 ['HIGH', 'LOW', 'CLOSE'])
        description: 描述信息

    Examples:
        # Beta 系数
        register_macro('beta', 'ts_cov(x, y, d) / (ts_std(y, d) ** 2)',
                       category='ts_function', data_args=2, param_pool=[20, 60])
        # 调用: beta(RETURNS, BENCH_RETURNS, 20) > 1.2

        # 夏普比率 (时序)
        register_macro('sharpe', 'ts_mean(x, d) / ts_std(x, d)',
                       category='ts_function', data_args=1, param_pool=[20, 60])
        # 调用: sharpe(CLOSE, 20) > 0.5
    """
    name_lower = name.lower()

    # [新增] 2026-07-15 宏名合法性校验 (防 func_src 注入)
    if not _MACRO_NAME_RE.match(name_lower):
        raise ValueError(
            f"宏名 '{name}' 不合法, 必须是 snake_case "
            f"(小写字母开头, 仅含 a-z0-9_)"
        )

    # 1. 自动推导参数名 (x/y/d 规范)
    params = _infer_params(data_args, param_pool, param_ranges)

    # 2. 解析宏体 (临时放行 params, 用 parse_expression 校验安全性)
    body_tree = _parse_macro_body(body, params)

    # 3. 编译为 Python 函数 (exec 到 functions 模块 globals(), 无依赖顺序问题)
    func = _compile_macro(name_lower, params, body)

    # 4. 注册到 FUNC_REGISTRY (对求值器/GP 透明)
    register_function(
        name_lower, func, category,
        data_args=data_args,
        param_pool=param_pool,
        param_ranges=param_ranges,
        data_vars=data_vars,
    )

    # 5. 存 MacroSpec (用于自省/文档/注销)
    # [重构] 2026-07-15 提取 _normalize_data_args 消除重复
    MACRO_REGISTRY[name_lower] = MacroSpec(
        name=name_lower, params=params, body_str=body, body_tree=body_tree,
        category=category, data_args=_normalize_data_args(data_args, data_vars),
        param_pool=param_pool, param_ranges=param_ranges,
        data_vars=data_vars, description=description,
    )


def unregister_macro(name: str) -> bool:
    """注销宏定义, 返回是否成功

    同时从 MACRO_REGISTRY、FUNC_REGISTRY 和 functions 模块 globals() 中清除。
    """
    name_lower = name.lower()
    if name_lower not in MACRO_REGISTRY:
        return False
    del MACRO_REGISTRY[name_lower]
    # 从 FUNC_REGISTRY 注销
    unregister_function(name_lower)
    # 从 functions 模块 globals() 清除 (exec 放入的函数对象)
    _functions_mod.__dict__.pop(name_lower, None)
    return True


def list_macros() -> Dict[str, MacroSpec]:
    """列出所有已注册的宏定义"""
    return dict(MACRO_REGISTRY)


def macro_to_str(name: str) -> str:
    """返回宏定义的人类可读字符串"""
    name_lower = name.lower()
    spec = MACRO_REGISTRY.get(name_lower)
    if spec is None:
        raise KeyError(f"宏 '{name}' 未注册")
    params_str = ', '.join(spec.params)
    return f"{spec.name}({params_str}) = {spec.body_str}"
