#这个类是带东八时区的，逐一其他数据要时区一致
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from .storage import context


# ============================================================================
# 枚举常量 - 参考掘金SDK规范
# ============================================================================

class OrderSide:
    """买卖方向"""
    Unknown = 0
    Buy = 1      # 买入
    Sell = 2     # 卖出

    @staticmethod
    def to_str(side: int) -> str:
        """转换为字符串"""
        return 'buy' if side == OrderSide.Buy else 'sell'


class PositionEffect:
    """开平标志"""
    Unknown = 0
    Open = 1           # 开仓
    Close = 2          # 平仓
    CloseToday = 3      # 平今仓
    CloseYesterday = 4  # 平昨仓


class PositionSide:
    """持仓方向"""
    Unknown = 0
    Long = 1     # 多方向
    Short = 2    # 空方向


class OrderType:
    """委托类型"""
    Unknown = 0
    Limit = 1          # 限价委托
    Market = 2         # 市价委托


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class PositionSnapshot:
    """持仓快照数据类"""
    symbol: str
    volume: float
    cost_price: float
    price: float
    created_at: datetime


@dataclass
class AccountSnapshot:
    """账户快照数据类"""
    cash: float
    nav: float
    created_at: datetime
    positions: Dict[str, PositionSnapshot] = field(default_factory=dict)


@dataclass
class TradeRecord:
    """成交记录数据类 - 兼容掘金规范"""
    created_at: datetime
    symbol: str
    price: float
    volume: float
    side: int                        # 买卖方向: 1=买入, 2=卖出
    position_effect: int             # 开平标志: 1=开仓, 2=平仓
    position_side: int = PositionSide.Long  # 持仓方向: 1=多, 2=空
    order_type: int = OrderType.Limit  # 委托类型: 1=限价, 2=市价
    fee: float = 0.0
    order_id: str = ''
    filled_volume: float = 0.0        # 已成交数量
    amount: float = 0.0              # 成交金额
    # [新增] 2026-05-30 信号备注，可追溯每笔交易触发原因（如 "温度计75度买入"）
    note: str = ''


# ============================================================================
# 账户管理类
# ============================================================================

class AccountManager:
    """
    账户管理器 — 时间轴驱动的资金、持仓、交易记录和快照管理

    snapshots[0] 恒为初始盘前快照（Engine.run() 循环前创建），后续快照由每根 bar 驱动追加。

    AccountManager
    ├── __init__()              # 初始化：资金/持仓/trade_records/snapshots/费用配置
    │
    ├── [快照操作]              # 时间轴状态切片
    │   ├── init_snapshot()     #  创建初始盘前快照（Engine.run() 循环前调用一次）
    │   ├── take_snapshot()     #  创建快照 → snapshots.append()（纯追加，不排序/不去重）
    │   └── load_snapshot()     #  从快照恢复账户状态（资金+持仓）
    │
    ├── [交易操作]              # 策略 on_bar 中调用，成交后自动触发 take_snapshot()
    │   ├── order_percent()     #  按净值比例下单 → 金额→数量 → 调用 order_volume()
    │   ├── order_volume()      #  按指定数量下单 → 调用 _process_order()
    │   ├── _process_order()    #  [私有] 费用计算 → 验资/验券 → 扣款/收款 → 记录 TradeRecord
    │   └── _update_position()  #  [私有] 均价法更新持仓（买入加权均价，卖出扣减数量）
    │
    ├── [查询操作]              # 策略中获取实时状态，优先从最新快照反查
    │   ├── get_account()       #  查询账户现金+净值（从 snapshots 反查 ≤ query_time 的快照）
    │   ├── get_position()      #  查询持仓（单品种返回 dict，全部返回 Dict[str,dict]）
    │   └── get_orders()        #  查询成交记录（支持时间区间过滤）
    │
    └── [底层支撑]
        └── _get_price()        #  [私有] 从 context 缓存获取当前价格（多频率逐次查找）
    """
    
    FREQ_ORDER = ['1m', '60s', '5m', '300s', '15m', '900s', '30m', '1800s', '60m', '3600s', '1d']

    def __init__(
        self,
        init_cash: float = 1e6,
        fee_config: Dict = None,
    ):
        """
        初始化账户管理器
        
        Args:
            init_cash: 初始资金，默认100万
            fee_config: 费用配置，包含佣金率、印花税率、最低佣金
        """
        self.cash = round(init_cash, 2)
        self.positions: Dict[str, Dict] = {}
        self.trade_records: List[TradeRecord] = []
        self.snapshots: List[AccountSnapshot] = []
        self.fee_config = fee_config or {
            'commission_rate': 0.0003,
            'stamp_tax_rate': 0.001,
            'min_commission': 5.0
        }

    # ------------------------------------------------------------------------
    # 快照操作
    # ------------------------------------------------------------------------

    def init_snapshot(self, created_at: datetime):
        """
        创建初始盘前快照（仅在回测开始前调用一次）
        
        snapshots[0] 语义固定：表示首笔交易前的账户状态（初始资金，零持仓）
        时间锚定在首根 bar 的前一日，确保基准日期独立于任何交易日
        
        Args:
            created_at: 快照时间，由引擎传入（通常 = 首根 bar 的 eob - 1 天）
        """
        snapshot = AccountSnapshot(
            cash=self.cash,
            nav=self.cash,
            created_at=created_at,
            positions={}
        )
        self.snapshots.append(snapshot)

    def take_snapshot(self, created_at: datetime = None) -> AccountSnapshot:
        """
        创建账户快照
        
        Args:
            created_at: 快照时间，默认为当前上下文时间
            
        Returns:
            AccountSnapshot: 账户快照对象
        """
        if created_at is None:
            created_at = context.now

        pos_snapshots = {}
        total_assets = self.cash

        for symbol, pos in self.positions.items():
            price = self._get_price(symbol)
            pos_snap = PositionSnapshot(
                symbol=symbol,
                volume=pos['volume'],
                cost_price=round(pos['cost_price'], 3),
                price=price,
                created_at=created_at
            )
            pos_snapshots[symbol] = pos_snap
            total_assets += pos['volume'] * price

        total_assets = round(total_assets, 2)
        snapshot = AccountSnapshot(
            cash=self.cash,
            nav=total_assets,
            created_at=created_at,
            positions=pos_snapshots
        )
        self.snapshots.append(snapshot)

        return snapshot

    def load_snapshot(self, snapshot: AccountSnapshot):
        """
        从快照恢复账户状态
        
        Args:
            snapshot: 账户快照对象
        """
        self.cash = round(snapshot.cash, 2)
        self.positions = {
            sym: {'volume': pos.volume, 'cost_price': round(pos.cost_price, 3)}
            for sym, pos in snapshot.positions.items()
        }
        self.current_time = snapshot.created_at

    # ------------------------------------------------------------------------
    # 交易操作
    # ------------------------------------------------------------------------

    def order_percent(
        self,
        symbol: str,
        percent: float,
        side: int,
        position_effect: int = PositionEffect.Open,
        order_type: int = OrderType.Limit,
        price: float = None,
        note: str = '',
    ) -> str:
        """
        按账户净值比例委托

        Args:
            symbol: 交易品种代码
            percent: 委托比例，0-1之间（正数）
            side: 买卖方向，OrderSide.Buy=1买入, OrderSide.Sell=2卖出
            position_effect: 开平标志，PositionEffect.Open=开仓, PositionEffect.Close=平仓
            order_type: 委托类型，OrderType.Limit=限价, OrderType.Market=市价
            price: 委托价格，默认为当前价格

        Returns:
            str: 订单ID

        Raises:
            ValueError: 比例超出范围，或计算出的数量为0
        """
        if not 0 < abs(percent) <= 1:
            raise ValueError("Percent must be between -1 and 1 (non-zero)")

        account_info = self.get_account()
        nav = account_info['nav']

        order_amount = nav * abs(percent)

        if side == OrderSide.Buy:
            available_amount = self.cash
            if order_amount > available_amount:
                order_amount = available_amount

        price = price or self._get_price(symbol)
        if price <= 0:
            raise ValueError(f"Invalid price {price} for {symbol}")

        if side == OrderSide.Buy:
            commission = max(
                round(order_amount * self.fee_config['commission_rate'], 2),
                self.fee_config['min_commission']
            )
            available_amount = order_amount - commission
            volume = int(available_amount / price)
        else:
            current_pos = self.positions.get(symbol, {'volume': 0})
            volume = int(current_pos['volume'] * abs(percent))

        if volume == 0:
            raise ValueError("Calculated order volume is zero")

        return self.order_volume(symbol, volume, side, position_effect, order_type, price, note)

    def order_volume(
        self,
        symbol: str,
        volume: int,
        side: int,
        position_effect: int = PositionEffect.Open,
        order_type: int = OrderType.Limit,
        price: float = None,
        note: str = '',
    ) -> str:
        """
        按指定数量委托

        Args:
            symbol: 交易品种代码
            volume: 委托数量（正数）
            side: 买卖方向，OrderSide.Buy=1买入, OrderSide.Sell=2卖出
            position_effect: 开平标志，PositionEffect.Open=开仓, PositionEffect.Close=平仓
            order_type: 委托类型，OrderType.Limit=限价, OrderType.Market=市价（回测中实际无区别）
            price: 委托价格，默认为当前价格

        Returns:
            str: 订单ID

        Raises:
            ValueError: 数量为0或价格无效
        """
        if volume == 0:
            raise ValueError("Order volume cannot be zero")

        if side not in (OrderSide.Buy, OrderSide.Sell):
            raise ValueError(f"Invalid side value: {side}, must be OrderSide.Buy or OrderSide.Sell")

        price = price or self._get_price(symbol)
        if price <= 0:
            raise ValueError(f"Invalid price {price} for {symbol}")

        order_id = f"order_{len(self.trade_records)+1}"

        executed_volume = self._process_order(
            symbol, volume, side, position_effect, order_type, price, order_id, note
        )

        if executed_volume != 0:
            self.take_snapshot()
        return order_id

    def _process_order(
        self,
        symbol: str,
        volume: int,
        side: int,
        position_effect: int,
        order_type: int,
        price: float,
        order_id: str,
        note: str = '',
    ) -> int:
        """
        处理订单执行

        Args:
            symbol: 交易品种代码
            volume: 委托数量（正数）
            side: 买卖方向，OrderSide.Buy=1, OrderSide.Sell=2
            position_effect: 开平标志
            order_type: 委托类型
            price: 委托价格
            order_id: 订单ID

        Returns:
            int: 实际成交数量，0表示未成交
        """
        commission = max(
            round(price * volume * self.fee_config['commission_rate'], 2),
            self.fee_config['min_commission']
        )
        stamp_tax = round(price * volume * self.fee_config['stamp_tax_rate'], 2) if side == OrderSide.Sell else 0
        total_fee = round(commission + stamp_tax, 2)

        if side == OrderSide.Buy:
            total_cost = round(volume * price + total_fee, 2)
            if self.cash < total_cost:
                print(f"订单 {order_id} 买入 {symbol} 失败，资金不足。需要 {total_cost}，可用资金 {self.cash}")
                return 0
            self.cash = round(self.cash - total_cost, 2)
            self._update_position(symbol, volume, price, total_fee, OrderSide.Buy)
        else:
            current_pos = self.positions.get(symbol, {'volume': 0})
            if current_pos['volume'] < volume:
                print(f"订单 {order_id} 卖出 {symbol} 失败，持仓不足。需要 {volume}，当前持仓 {current_pos['volume']}")
                return 0
            self.cash = round(self.cash + volume * price - total_fee, 2)
            self._update_position(symbol, volume, price, total_fee, OrderSide.Sell)

        self.trade_records.append(TradeRecord(
            created_at=context.now,
            symbol=symbol,
            price=price,
            volume=volume,
            side=side,
            position_effect=position_effect,
            position_side=PositionSide.Long,
            order_type=order_type,
            fee=total_fee,
            order_id=order_id,
            filled_volume=volume,
            amount=price * volume,
            note=note,
        ))
        return volume

    def _update_position(self, symbol: str, volume: int, price: float, total_fee: float, side: int):
        """
        更新持仓信息

        Args:
            symbol: 交易品种代码
            volume: 成交数量（正数）
            price: 成交价格
            total_fee: 总费用
            side: 买卖方向，OrderSide.Buy=1, OrderSide.Sell=2
        """
        pos = self.positions.get(symbol, {'volume': 0, 'cost_price': 0})
        if side == OrderSide.Buy:
            new_volume = pos['volume'] + volume
            total_purchase_cost = pos['volume'] * pos['cost_price'] + volume * price + total_fee
            new_cost = total_purchase_cost / new_volume
            pos['volume'] = new_volume
            pos['cost_price'] = round(new_cost, 3)
        else:
            pos['volume'] -= volume
            if pos['volume'] == 0:
                del self.positions[symbol]
                return
        self.positions[symbol] = pos

    # ------------------------------------------------------------------------
    # 查询操作
    # ------------------------------------------------------------------------

    def get_account(self, query_time: datetime = None) -> Dict:
        """
        获取账户信息
        
        Args:
            query_time: 查询时间，默认为当前上下文时间
            
        Returns:
            Dict: 包含cash、nav、created_at的字典
        """
        query_time = context.now if query_time is None else query_time

        if not self.snapshots:
            return {
                'cash': self.cash,
                'nav': self.cash,
                'created_at': query_time
            }

        snapshot = next(
            (s for s in reversed(self.snapshots) if s.created_at <= query_time),
            None
        )

        if snapshot is None:
            return {
                'cash': self.cash,
                'nav': self.cash,
                'created_at': query_time
            }

        return {
            'cash': snapshot.cash,
            'nav': snapshot.nav,
            'created_at': snapshot.created_at
        }

    def get_position(self, symbol: str = None) -> Dict:
        """
        获取持仓信息
        
        Args:
            symbol: 交易品种代码，为None时返回所有持仓
            
        Returns:
            Dict: 单个品种返回{'volume', 'cost_price'}，所有品种返回字典
        """
        if not self.snapshots:
            positions = self.positions.copy()
        else:
            last_snapshot = self.snapshots[-1]
            positions = {
                sym: {'volume': pos.volume, 'cost_price': pos.cost_price}
                for sym, pos in last_snapshot.positions.items()
            }

        if symbol:
            pos = positions.get(symbol, {'volume': 0, 'cost_price': 0})
            pos['cost_price'] = round(pos['cost_price'], 3)
            return pos
        for pos in positions.values():
            pos['cost_price'] = round(pos['cost_price'], 3)
        return positions

    def get_orders(
        self,
        start_query_time: datetime = None,
        end_query_time: datetime = None
    ) -> List[TradeRecord]:
        """
        获取成交记录
        
        Args:
            start_query_time: 查询起始时间
            end_query_time: 查询结束时间
            
        Returns:
            List[TradeRecord]: 成交记录列表
        """
        trades = self.trade_records

        if start_query_time:
            trades = [t for t in trades if t.created_at >= start_query_time]
        if end_query_time:
            trades = [t for t in trades if t.created_at <= end_query_time]

        return trades.copy()

    # ------------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------------

    def _get_price(self, symbol: str) -> float:
        """
        获取品种当前价格
        
        Args:
            symbol: 交易品种代码
            
        Returns:
            float: 当前价格
            
        Raises:
            ValueError: 品种未订阅或无法获取有效价格
        """
        action_time = context.now
        subscribed_freqs = {freq for (s, freq) in context._subscribed if s == symbol}
        if not subscribed_freqs:
            raise ValueError(f"品种 {symbol} 未订阅任何频率数据")

        frequencies = [f for f in self.FREQ_ORDER if f in subscribed_freqs]
        # [修复] 2026-05-30 补充不在 FREQ_ORDER 中的自定义频率（如 m10）
        frequencies += [f for f in subscribed_freqs if f not in self.FREQ_ORDER]

        for freq in frequencies:
            try:
                raw_data = context.data(
                    symbol=symbol,
                    frequency=freq,
                    count=3,
                    fields='close,eob',
                )

                if isinstance(raw_data, pd.DataFrame):
                    data = raw_data.to_dict('records')
                else:
                    data = raw_data

                for d in reversed(data):
                    if d['eob'] <= action_time:
                        price = d['close']
                        if not isinstance(price, (float, int)) or price <= 0:
                            raise ValueError(f"Invalid price {price} for {symbol} at {action_time}")
                        return float(price)
            except Exception:
                continue

        raise ValueError(f"No valid price found for {symbol} at {action_time}")


# ============================================================================
# 全局实例
# ============================================================================

account = AccountManager(init_cash=1e6)
