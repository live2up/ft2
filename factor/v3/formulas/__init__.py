"""
factor/v3/formulas/ — 公式数据库
=============================================================================

子模块：
  wq101.py   — WorldQuant 101 Alpha (101 公式)
  gt191.py   — GTJA 191 Alpha (171 公式)
  basic.py   — 因子原子基元 (20 公式)

兼容旧导入：from factor.v3.formulas import ALPHA101 仍可用。
"""

from .wq101 import ALPHA101, ALPHA101_CATEGORIES
from .gt191 import ALPHA191, ALPHA191_CATEGORIES
from .basic import BASIC_FACTORS, BASIC_FACTORS_CATEGORIES

__all__ = [
    'ALPHA101', 'ALPHA101_CATEGORIES',
    'ALPHA191', 'ALPHA191_CATEGORIES',
    'BASIC_FACTORS', 'BASIC_FACTORS_CATEGORIES',
]
