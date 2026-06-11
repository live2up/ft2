"""
factor/v4/formulas/ — V4 AST 语法公式库

子模块：
  wq101.py     — WorldQuant 101 Alpha (101 公式)
  gt191.py     — GTJA 191 Alpha (171 公式)
  basic.py     — 因子原子基元 (20 公式)
  industry.py  — 行业轮动专有因子 (16 公式)

所有公式已转换为 Python infix 语法，可直接被 V4FactorExpression 解析。
"""

from .wq101 import ALPHA101
from .gt191 import ALPHA191
from .basic import BASIC_FACTORS
from .industry import INDUSTRY_ALPHA

__all__ = [
    'ALPHA101',
    'ALPHA191',
    'BASIC_FACTORS',
    'INDUSTRY_ALPHA',
]
