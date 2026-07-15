# -*- coding: utf-8 -*-
"""
utils/ast/v2/_common.py — 共享数据类与工具函数
=============================================================================

functions.py 和 macros.py 的公共依赖, 消除循环耦合。

内容:
  数据类: VarSpec, ParamRange
  工具函数: _normalize_data_args, _count_pool_args

设计原则:
  - 纯数据类 + 纯函数, 不依赖任何全局状态 (FUNC_REGISTRY 等)
  - 不依赖 functions.py / dsl.py / macros.py, 零循环风险
  - functions.py 和 macros.py 平级引用, 无层级依赖

[新增] 2026-07-15 从 functions.py 抽离共享数据类与工具函数
=============================================================================
"""
from dataclasses import dataclass
from typing import Any, List, Optional


# ============================================================
# 共享数据类
# ============================================================

@dataclass
class VarSpec:
    """变量规格 — 描述变量名、所属类别和匹配模式。

    Args:
        name: 变量名 (ALL_CAPS, 如 'CLOSE', 'REL')
        category: 所属类别 (如 '原始OHLCV', '相对基准')
        is_prefix: True=前缀通配 (REL_CLOSE 通过), False=精确匹配 (只认 CLOSE)
        description: 描述信息 (仅文档/调试)
    """
    name: str
    category: str = '自定义'
    is_prefix: bool = False
    description: str = ''


@dataclass
class ParamRange:
    """参数值域约束 — 描述 param_pool 无法覆盖的带范围参数。

    用于函数签名中 param_pool 之外的额外参数 (如 ts_quantile 的 p, cs_scale 的 scale)。

    Args:
        name: 参数名 (仅用于文档/调试, 不参与生成逻辑)
        dtype: 'int' 或 'float', 默认 'float'
        min_val: 最小值 (含), None 表示无下界
        max_val: 最大值 (含), None 表示无上界
        pool: 离散候选列表。非 None 时优先于 min/max 采样
    """
    name: str
    dtype: str = 'float'
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    pool: Optional[List[Any]] = None


# ============================================================
# 共享工具函数
# ============================================================

def _normalize_data_args(data_args: Optional[int],
                         data_vars: Optional[List[str]]) -> int:
    """统一 data_args 默认值推导 (data_vars 优先)

    消除 3 处重复: _register / register_macro / _infer_params

    规则:
      - data_vars 非空 → len(data_vars) (固定变量, GP 不生成子树)
      - data_args is None → 1 (默认 1 个数据参数)
      - data_args=0 → 0 (无数据参数, 有效值)
    """
    if data_vars is not None:
        return len(data_vars)
    return data_args if data_args is not None else 1


def _count_pool_args(param_pool: Optional[List[Any]]) -> int:
    """param_pool 展开后的参数数

    标量列表 [5,10,20] → 1 个参数
    tuple 列表 [(5,20)] → 2 个参数
    None / 空 → 0 个参数

    消除 2 处重复: _check_param_arity / _infer_params
    """
    if not param_pool:
        return 0
    first = param_pool[0]
    return len(first) if isinstance(first, tuple) else 1
