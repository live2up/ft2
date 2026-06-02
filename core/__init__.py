"""
ft2.core - 核心回测模块

包含：
- engine: 回测引擎
- storage: 数据存储和上下文
- account: 账户管理
- analyzer: 账户分析
"""

from .engine import engine, Engine
from .storage import context
from .account import (
    account, AccountManager,
    OrderSide, PositionEffect, PositionSide, OrderType,
    BenchHolder,
)
from .analyzer import AccountAnalyzer

__all__ = [
    'engine', 'Engine',
    'context',
    'account', 'AccountManager',
    'OrderSide',       # 买卖方向: OrderSide.Buy, OrderSide.Sell
    'PositionEffect',  # 开平标志: PositionEffect.Open, PositionEffect.Close
    'PositionSide',    # 持仓方向: PositionSide.Long, PositionSide.Short
    'OrderType',       # 委托类型: OrderType.Limit, OrderType.Market
    'BenchHolder',     # 内置买入持有基准策略
    'AccountAnalyzer',
]
