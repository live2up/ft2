"""
signals/v5/search/__init__.py — 搜索层 (v5 GP + 网格)
=============================================================================
基于 utils/gp/v5 核心的 GP 遗传算法 (权重聚焦/方向追踪/缓存) + 参数网格搜索。
v5 搜索全部使用 Expression + SigEngine（ft2.core 驱动）。

用法:
  from signals.v5.search import SigGPEngine, GridSearch
=============================================================================
"""

from .gp import SigGPEngine
from .grid import GridSearch

__all__ = ['SigGPEngine', 'GridSearch']
