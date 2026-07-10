"""
signals/v5 — 择时信号模块 (直接适配 ast.v2)

[v5] 2026-07-10 新建，独立直连 utils.ast.v2，移除 v4 兼容层。
"""
from .expression import Expression, stateful_signal
from .engine import SigEngine

__all__ = [
    'Expression',
    'stateful_signal',
    'SigEngine',
]
