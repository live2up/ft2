"""
signals/v4/search/__init__.py — 搜索层 (GP + 网格)
=============================================================================
基于 Python ast 的 GP 遗传算法 + 参数网格搜索。
v4 搜索全部使用 Expression + SigEngine（ft2.core 驱动）。

用法:
  from signals.v4.search import GPSearch, GridSearch
=============================================================================
"""

from .gp import GPSearch
from .grid import GridSearch

__all__ = ['GPSearch', 'GridSearch']
