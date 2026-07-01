"""
ft2.core - 核心回测模块

包含：
- engine: 回测引擎 (时间轴驱动, 事件驱动)
- backtest: 简化回测入口 (Engine 二次封装, Backtester + PositionsStrategy)
- storage: 数据存储和上下文
- account: 账户管理
- analyzer: 账户分析

v2.4 (2026-06-09): 移除全局 account/engine 实例，每 Engine() 自带独立账户
v2.5 (2026-06-30): 新增 Backtester — Engine 二次封装, 简化调用不丢灵活性
v2.6 (2026-07-01): Backtester.add_datas(assets) — 一步传入全量数据, 消除构造+add_data 冗余拆分
"""

from .engine import Engine
from .storage import context
from .account import (
    AccountManager,
    OrderSide, PositionEffect, PositionSide, OrderType,
    BenchHolder,
)
from .analyzer import AccountAnalyzer
from .backtest import Backtester, PositionsStrategy

__all__ = [
    'Engine',
    'context',
    'AccountManager',
    'OrderSide',       # 买卖方向: OrderSide.Buy, OrderSide.Sell
    'PositionEffect',  # 开平标志: PositionEffect.Open, PositionEffect.Close
    'PositionSide',    # 持仓方向: PositionSide.Long, PositionSide.Short
    'OrderType',       # 委托类型: OrderType.Limit, OrderType.Market
    'BenchHolder',     # 内置买入持有基准策略
    'AccountAnalyzer',
    'Backtester',      # positions 驱动的简化回测入口
    'PositionsStrategy',
]
