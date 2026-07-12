"""
signals/v5 — 择时信号模块 (直接适配 utils.ast.v2)

[v5] 2026-07-10 新建，独立直连 utils.ast.v2，移除 v4 兼容重导出壳。
仍保留 register_function / register_variable 等常用注册接口，方便迁移。
"""
from .expression import Expression, stateful_signal
from .engine import SigEngine
from .registry import register_function, register_variable, unregister_function, unregister_variable

__all__ = [
    'Expression',
    'stateful_signal',
    'SigEngine',
    'register_function', 'register_variable',
    'unregister_function', 'unregister_variable',
]
