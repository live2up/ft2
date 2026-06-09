"""
signals/v3/search/__init__.py — 搜索层 (GP + 网格)
=============================================================================
保留 v2 的算法 (gp_optimizer / grid_search), 替换回测引擎为 v3.EngineV3。

接口对齐: v3.search.GPSearch ≈ v2.GPOptimizer, v3.search.GridSearch ≈ v2.GridSearch
"""

from .gp import GPSearch
from .grid import GridSearch

__all__ = ['GPSearch', 'GridSearch']
