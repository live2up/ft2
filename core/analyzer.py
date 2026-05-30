#通过 account 的快照，分析结果
"""
账户分析器模块

设计思路:
---------
1. 数据初始化
   - 传入 AccountManager 实例，自动聚合快照为日数据(_daily_assets)
   - 计算交易盈亏记录(_trade_profits)
   - 数据在初始化时固化，后续计算基于此数据

2. 时间区间切片
   - sliced_data: 缓存当前区间的切片数据
   - getTimeRange(period): 设置时间区间，更新缓存
   - _ensure_sliced_data(): 确保切片数据已初始化

3. 使用方式
   - 默认使用全数据: 直接调用指标方法，自动初始化
   - 指定区间: 先调用 getTimeRange('3m')，再调用指标方法
   - 切换区间: 再次调用 getTimeRange() 即可

   示例:
   >>> analyzer = AccountAnalyzer(account=my_account)
   >>> analyzer.volatility()           # 使用全数据
   >>> analyzer.getTimeRange('3m')     # 设置近3月区间
   >>> analyzer.volatility()           # 使用近3月数据
   >>> analyzer.sharpe_ratio()         # 复用缓存的近3月数据

4. 指标分类
   - 资产类指标: 支持区间切片（收益率、波动率、夏普比率等）
   - 交易类指标: 暂不支持区间切片（胜率、盈亏比等）
"""
from collections import defaultdict
from dateutil.relativedelta import relativedelta
import math
import numpy as np
import os
from datetime import datetime, date
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import inspect
from functools import wraps
# [修复] 2026-05-30 _calculate_profit 需要用 OrderSide 枚举做比较
from .account import OrderSide




# ============================================================================
# 指标装饰器
# ============================================================================

def metric(name: str = None, group: str = '', desc: str = '', 
           fmt: str = '.2f', order: int = 99):
    """
    指标装饰器，用于标记分析方法
    
    Args:
        name: 指标名称，默认使用函数名
        group: 指标分组，如 '收益'、'风险'、'交易'，用于自动分组展示
        desc: 指标描述（可选），输入到 notebook metric-desc 标签
        fmt: 输出格式，默认 '.1%'（百分比）。可选 '.2f'、'.1f'
            仅影响 to_notebook / to_excel 的输出，不改变方法返回值
        order: 排序号，用于报告中的顺序
    
    Usage:
        @metric(name='夏普比率', group='风险', fmt='.2f', order=30)
        @metric(name='年化波动率', group='风险', order=20)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        
        wrapper._is_metric = True
        wrapper._metric_name = name or func.__name__
        wrapper._metric_group = group
        wrapper._metric_desc = desc
        wrapper._metric_fmt = fmt
        wrapper._metric_order = order
        return wrapper
    return decorator


# ============================================================================
# 账户分析类
# ============================================================================

"""
账户分析器模块

输出规范:
---------
JSON输出结构:
{
    "title": "回测报告",
    "createdAt": "2024-01-01 12:00:00",
    "metrics": [{"name": "指标名", "value": 值, "order": 排序号}, ...],
    "dailyAssets": [{"date": "2024-01-01", "assets": 100000}, ...],
    "trades": [...],
    "topProfits": [...],
    "topLosses": [...]
}

指标分类排序规范:
----------------
order范围    | 类别              | 指标
------------|------------------|------------------
1-9         | 基础信息          | 回测区间、初始资金、最终资产
10-19       | 收益指标          | 累计收益率、年化收益率
20-29       | 风险指标          | 年化波动率、最大回撤、VaR、CVaR、Ulcer Index
30-39       | 风险调整收益指标   | 夏普比率、索提诺比率、UPI
40-49       | 交易分析指标       | 胜率、平均盈亏比、平均持仓时间
50-59       | 仓位建议          | 凯利公式最优仓位、半凯利仓位

新增指标规范:
------------
1. 用 @metric 装饰器声明元数据（name/group/fmt/desc/order）
2. 方法返回纯数字，格式化由输出层处理
3. to_notebook() / to_excel() 通过 self.metrics() 自动收集
"""

class AccountAnalyzer:
    """账户分析器，负责计算各类风险收益指标和生成分析报告"""

    def __init__(self, account=None, daily_assets=None):
        """
        初始化账户分析器
        
        支持两种数据输入方式：
        1. 传入 account 实例：自动聚合快照为日数据，并计算交易盈亏
        2. 传入 daily_assets：直接使用外部提供的日数据（不计算交易盈亏）
        
        Args:
            account: AccountManager 实例，提供原始快照数据
                   如果提供，将自动调用 _aggregate_daily_assets 聚合日数据
            daily_assets: 外部每日资产数据，支持两种格式：
                         - List[Dict]: [{'date': date(2024,1,1), 'assets': 100000}, ...]
                         - Dict[date, float]: {date(2024,1,1): 100000, ...}
                        如果提供，将优先使用此数据而非 account 的快照
            
        Example:
            >>> # 方式 1：使用 account 实例
            >>> analyzer = AccountAnalyzer(account=my_account)
            >>> 
            >>> # 方式 2：使用外部日数据（List 格式）
            >>> daily_data = [
            ...     {'date': date(2024, 1, 1), 'assets': 100000},
            ...     {'date': date(2024, 1, 2), 'assets': 101000},
            ... ]
            >>> analyzer = AccountAnalyzer(daily_assets=daily_data)
            >>> 
            >>> # 方式 3：使用外部日数据（Dict 格式）
            >>> daily_dict = {date(2024,1,1): 100000, date(2024,1,2): 101000}
            >>> analyzer = AccountAnalyzer(daily_assets=daily_dict)
        """
        self.account = account
        self._metrics = {}
        self.risk_free_rate = 0.02
        self.sliced_data = None
        
        if daily_assets is not None:
            if isinstance(daily_assets, dict):
                self._daily_assets = daily_assets
            else:
                self._daily_assets = {item['date']: item['assets'] for item in daily_assets}
            self._trade_profits = []
        elif account:
            self._daily_assets = self._aggregate_daily_assets(account.snapshots)
            self._trade_profits = self._calculate_profit(account.trade_records)
        else:
            self._daily_assets = {}
            self._trade_profits = []

        self.base_dir = self._get_caller_dir()

    @property
    def daily_assets(self) -> Dict:
        """
        获取每日资产净值的副本
        
        返回内部存储的每日资产数据的浅拷贝，防止外部修改
        数据格式：{date: nav}
        
        Returns:
            Dict[date, float]: 每日资产净值字典的副本
            
        Example:
            >>> assets = analyzer.daily_assets
            >>> print(f"最新资产：{assets[max(assets.keys())]}")
        """
        return self._daily_assets.copy()

    @property
    def trade_profits(self) -> List:
        """
        获取交易盈亏记录的副本
        
        返回内部存储的交易盈亏数据的浅拷贝，防止外部修改
        每笔交易记录包含：symbol, profit, open_time, close_time 等信息
        
        Returns:
            List[Dict]: 交易盈亏记录列表的副本
            
        Example:
            >>> profits = analyzer.trade_profits
            >>> total_profit = sum(t['profit'] for t in profits)
            >>> print(f"总盈利：{total_profit}")
        """
        return self._trade_profits.copy()

    # ------------------------------------------------------------------------
    # 统一指标获取方法
    # ------------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Dict]:
        """
        获取所有已计算的指标
        
        返回所有以 calc_ 开头的方法计算并存储的指标结果
        
        Returns:
            Dict[str, Dict]: 指标字典，格式为：
                {
                    'calc_return_rate': {
                        'name': '累计收益率',
                        'value': 0.15,
                        'order': 10,
                        'desc': '统计期间内的总收益率'
                    },
                    ...
                }
            
        Example:
            >>> analyzer.calc_return_rate()
            >>> metrics = analyzer.get_metrics()
            >>> print(metrics['calc_return_rate']['value'])
        """
        return self._metrics.copy()

    # ------------------------------------------------------------------------
    # 链式调用方法（支持 @metric 装饰器）
    # ------------------------------------------------------------------------

    _periods_map = {
        '1m': relativedelta(months=1),
        '3m': relativedelta(months=3),
        '6m': relativedelta(months=6),
        '1y': relativedelta(years=1),
        '2y': relativedelta(years=2),
        '3y': relativedelta(years=3),
        '5y': relativedelta(years=5),
        'all': None
    }

    def _ensure_sliced_data(self) -> Optional[Dict]:
        """
        确保切片数据已初始化
        
        如果 sliced_data 为 None，自动调用 getTimeRange() 初始化全数据
        
        Returns:
            Dict: 切片数据，格式为：
                {
                    'startDate': date,
                    'endDate': date,
                    'baseDate': date,
                    'daily_assets': Dict
                }
                数据不足时返回 None
        """
        if self.sliced_data is None:
            self.getTimeRange()
        return self.sliced_data

    def getTimeRange(self, period_or_start=None, end=None) -> Optional[Dict]:
        """
        获取时间区间信息并缓存切片数据
        
        Args:
            period_or_start: 
                - None: 使用全部数据
                - str: 周期字符串，如 '1m', '3m', '6m', '1y', 'all'
                - date: 自定义开始日期
            end: 结束日期（当 period_or_start 为 date 时使用）
        
        Returns:
            Dict: 时间区间信息，格式为：
                {
                    'startDate': date,      # 区间起始日期
                    'endDate': date,        # 区间结束日期
                    'baseDate': date,       # 基准日日期
                    'daily_assets': Dict    # 切片后的日资产数据
                }
                数据不足时返回 None
            
        Example:
            >>> timeRange = analyzer.getTimeRange('3m')
            >>> print(timeRange['startDate'], timeRange['endDate'])
            >>> timeRange = analyzer.getTimeRange(date(2024,1,1), date(2024,6,30))
        """
        if not self._daily_assets:
            return None
        
        dates = sorted(self._daily_assets.keys())
        if len(dates) < 2:
            return None
        
        all_benchmark = dates[0]
        all_start = dates[1]
        all_end = dates[-1]
        
        if period_or_start is None:
            start_date = all_start
            end_date = all_end
            benchmark_date = all_benchmark
        elif isinstance(period_or_start, str):
            if period_or_start == 'all':
                start_date = all_start
                end_date = all_end
                benchmark_date = all_benchmark
            else:
                delta = self._periods_map.get(period_or_start)
                if delta is None:
                    start_date = all_start
                    end_date = all_end
                    benchmark_date = all_benchmark
                else:
                    calculated_start = dates[-1] - delta
                    valid_dates = [d for d in dates if d <= calculated_start]
                    raw_start = max(valid_dates) if valid_dates else all_start
                    
                    if raw_start == all_start:
                        start_date = all_start
                        end_date = all_end
                        benchmark_date = all_benchmark
                    else:
                        prev_dates = [d for d in dates if d < raw_start]
                        benchmark_date = max(prev_dates) if prev_dates else all_benchmark
                        start_date = raw_start
                        end_date = all_end
        else:
            raw_start = period_or_start or all_start
            raw_end = end or all_end
            
            if raw_start < all_start:
                raw_start = all_start
            
            prev_dates = [d for d in dates if d < raw_start]
            if prev_dates:
                benchmark_date = max(prev_dates)
            else:
                benchmark_date = all_benchmark
            
            start_date = raw_start
            end_date = raw_end
        
        sliced_assets = {
            d: v for d, v in self._daily_assets.items()
            if benchmark_date <= d <= end_date
        }
        
        # ====================================================================
        # 性能优化：预计算基础数据
        # ====================================================================
        # 在切片数据时一次性计算所有指标需要的基础数据，避免重复计算
        # 
        # 优化效果：
        # 1. 日收益率只需计算一次，volatility/var/cvar/sortino_ratio 等方法共享
        # 2. 使用 NumPy 向量化计算，比循环快 10-50 倍
        # 3. 后续指标方法直接使用预计算结果，无需重复排序和计算
        # ====================================================================
        
        # 排序日期列表（用于后续索引）
        dates = sorted(sliced_assets.keys())
        
        # 资产值数组（NumPy 向量化计算的基础）
        values = np.array([sliced_assets[d] for d in dates])
        
        # 日收益率数组（向量化计算：今日/昨日 - 1）
        # 公式：(values[1:] - values[:-1]) / values[:-1]
        # 等价于：[values[i]/values[i-1] - 1 for i in range(1, len(values))]
        # 但 NumPy 向量化版本快 20-50 倍
        daily_returns = (values[1:] - values[:-1]) / values[:-1] if len(values) > 1 else np.array([])
        
        self.sliced_data = {
            'startDate': start_date,
            'endDate': end_date,
            'baseDate': benchmark_date,
            'daily_assets': sliced_assets,  # 原始数据（兼容旧代码）
            'dates': dates,                  # 排序后的日期列表
            'values': values,                # 资产值数组（用于最大回撤、Ulcer Index 等）
            'returns': daily_returns         # 日收益率数组（用于波动率、VaR、CVaR、索提诺比率等）
        }
        
        return self.sliced_data

    @metric(name='累计收益率', group='收益', fmt='.1%', desc='统计期间内的总收益率', order=10)
    def return_rate(self) -> Optional[float]:
        """计算累计收益率（链式调用版本）"""
        if not self._daily_assets:
            return None
        
        sliced_data_info = self._ensure_sliced_data()
        if sliced_data_info is None:
            return None
        
        sliced_data = sliced_data_info['daily_assets']
        
        if len(sliced_data) < 2:
            return None
        
        dates = sorted(sliced_data.keys())
        benchmark_value = sliced_data[dates[0]]
        end_value = sliced_data[dates[-1]]
        
        if benchmark_value == 0:
            return None
        
        return (end_value - benchmark_value) / benchmark_value

    @metric(name='年化收益率', group='收益', fmt='.1%', desc='将收益率换算为年化基准', order=11)
    def annualized_return(self) -> Optional[float]:
        """计算年化收益率"""
        interval_return = self.return_rate()
        if interval_return is None:
            return None
        
        sliced_data_info = self._ensure_sliced_data()
        if sliced_data_info is None:
            return None
        
        sliced_data = sliced_data_info['daily_assets']
        
        dates = sorted(sliced_data.keys())
        days = (dates[-1] - dates[0]).days
        
        if days == 0:
            return 0
        if interval_return <= -1:
            return None
        
        return ((1 + interval_return) ** (365 / days)) - 1

    @metric(name='年化波动率', group='风险', fmt='.1%', desc='衡量资产价格的波动程度', order=20)
    def volatility(self) -> Optional[float]:
        """计算年化波动率"""
        sliced_data_info = self._ensure_sliced_data()
        if sliced_data_info is None:
            return None
        
        returns = sliced_data_info.get('returns')
        if returns is None or len(returns) < 2:
            return None
        
        return np.std(returns) * np.sqrt(252)

    @metric(name='夏普比率', group='风险', fmt='.2f', desc='每承担一单位风险获得的超额收益', order=30)
    def sharpe_ratio(self) -> Optional[float]:
        """计算夏普比率"""
        annualized = self.annualized_return()
        vol = self.volatility()
        
        if annualized is None or vol is None or vol == 0:
            return None
        
        return (annualized - self.risk_free_rate) / vol

    @metric(name='最大回撤', group='风险', fmt='.1%', desc='历史最大亏损幅度', order=21)
    def max_drawdown(self) -> Optional[Tuple[float, date, date]]:
        """计算最大回撤"""
        sliced_data_info = self._ensure_sliced_data()
        if sliced_data_info is None:
            return None
        
        values = sliced_data_info.get('values')
        dates = sliced_data_info.get('dates')
        
        if values is None or len(values) < 2:
            return None
        
        cumulative = values / values[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        
        max_dd_idx = np.argmax(drawdown)
        max_dd = drawdown[max_dd_idx]
        
        if max_dd == 0:
            return None
        
        peak_idx = np.argmax(cumulative[:max_dd_idx + 1])
        
        return max_dd, dates[peak_idx], dates[max_dd_idx]

    @metric(name='VaR(95%) / 风险价值', group='风险', fmt='.1%', desc='95% 置信度下的最大可能损失', order=22)
    def var(self, confidence: float = 0.95) -> Optional[float]:
        """计算风险价值"""
        sliced_data_info = self._ensure_sliced_data()
        if sliced_data_info is None:
            return None
        
        returns = sliced_data_info.get('returns')
        if returns is None or len(returns) < 2:
            return None
        
        return -np.percentile(returns, (1 - confidence) * 100)

    @metric(name='CVaR(95%) / 条件风险价值', group='风险', fmt='.1%', desc='超过 VaR 阈值的平均损失', order=23)
    def cvar(self, confidence: float = 0.95) -> Optional[float]:
        """计算条件风险价值"""
        sliced_data_info = self._ensure_sliced_data()
        if sliced_data_info is None:
            return None
        
        returns = sliced_data_info.get('returns')
        if returns is None or len(returns) < 2:
            return None
        
        index = int((1 - confidence) * len(returns))
        if index < 1:
            index = 1
        
        sorted_returns = np.sort(returns)
        return -np.mean(sorted_returns[:index])

    @metric(name='Ulcer Index / 溃疡指数', group='风险', fmt='.2f', desc='衡量回撤深度和持续时间的综合指标', order=24)
    def ulcer_index(self) -> Optional[float]:
        """计算溃疡指数"""
        sliced_data_info = self._ensure_sliced_data()
        if sliced_data_info is None:
            return None
        
        values = sliced_data_info.get('values')
        
        if values is None or len(values) < 2:
            return None
        
        cumulative = values / values[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown_pct = ((running_max - cumulative) / running_max) * 100
        
        return np.sqrt(np.mean(drawdown_pct ** 2))

    @metric(name='索提诺比率', group='风险', fmt='.2f', desc='只考虑下行风险的夏普比率改进版', order=31)
    def sortino_ratio(self, risk_free_rate: float = 0.02) -> Optional[float]:
        """计算索提诺比率"""
        annualized_return = self.annualized_return()
        if annualized_return is None:
            return None
        
        sliced_data_info = self._ensure_sliced_data()
        if sliced_data_info is None:
            return None
        
        returns = sliced_data_info.get('returns')
        if returns is None or len(returns) < 2:
            return None
        
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_variance = np.mean(returns ** 2 * (returns < 0))
        downside_deviation = np.sqrt(downside_variance)
        annualized_downside_deviation = downside_deviation * np.sqrt(252)
        
        if annualized_downside_deviation == 0:
            return float('inf')
        
        return (annualized_return - risk_free_rate) / annualized_downside_deviation

    @metric(name='UPI / 溃疡绩效指数', group='风险', fmt='.2f', desc='用溃疡指数调整的风险收益比', order=32)
    def upi(self, risk_free_rate: float = 0.02) -> Optional[float]:
        """计算 Ulcer Performance Index"""
        annualized_return = self.annualized_return()
        ulcer_idx = self.ulcer_index()
        
        if annualized_return is None or ulcer_idx is None or ulcer_idx == 0:
            return None
        
        return (annualized_return - risk_free_rate) / (ulcer_idx / 100)

    @metric(name='胜率', group='交易', fmt='.1%', desc='盈利交易次数占总交易次数的比例', order=40)
    def win_rate(self) -> Optional[float]:
        """计算胜率"""
        if not self._trade_profits:
            return None
        
        wins = sum(1 for t in self._trade_profits if t['profit'] > 0)
        return wins / len(self._trade_profits)

    def avg_profit(self, mode: str = 'amount') -> Optional[float]:
        """
        计算平均盈利
        
        Args:
            mode: 计算模式
                  - 'amount': 金额模式，计算平均盈利金额
                  - 'percentage': 百分比模式，计算平均盈利占本金的比例
            
        Returns:
            float: 平均盈利
                  无盈利交易时返回 None
        """
        profitable_trades = [t for t in self._trade_profits if t['profit'] > 0]
        if not profitable_trades:
            return None
        
        if mode == 'amount':
            profits = [t['profit'] for t in profitable_trades]
        elif mode == 'percentage':
            profits = [
                t['profit'] / (abs(t['volume']) * t['open_price'])
                for t in profitable_trades
            ]
        else:
            raise ValueError("mode 必须是 'amount' 或 'percentage'")
        
        return sum(profits) / len(profits)

    def avg_loss(self, mode: str = 'amount') -> Optional[float]:
        """
        计算平均亏损
        
        Args:
            mode: 计算模式
                  - 'amount': 金额模式，计算平均亏损金额
                  - 'percentage': 百分比模式，计算平均亏损占本金的比例
            
        Returns:
            float: 平均亏损（负数）
                  无亏损交易时返回 None
        """
        loss_trades = [t for t in self._trade_profits if t['profit'] < 0]
        if not loss_trades:
            return None
        
        if mode == 'amount':
            losses = [t['profit'] for t in loss_trades]
        elif mode == 'percentage':
            losses = [
                t['profit'] / (abs(t['volume']) * t['open_price'])
                for t in loss_trades
            ]
        else:
            raise ValueError("mode 必须是 'amount' 或 'percentage'")
        
        return sum(losses) / len(losses)

    @metric(name='平均盈亏比', group='交易', fmt='.2f', desc='平均盈利与平均亏损的比值', order=41)
    def avg_profit_loss_ratio(self) -> Optional[float]:
        """计算平均盈亏比"""
        avg_profit = self.avg_profit()
        avg_loss = self.avg_loss()
        
        if avg_profit is None or avg_loss is None or avg_loss == 0:
            return None
        
        return abs(avg_profit / avg_loss)

    @metric(name='平均持仓时间', group='交易', fmt='.1f', desc='所有交易的平均持仓天数', order=42)
    def avg_holding_period(self) -> Optional[float]:
        """计算平均持仓时间"""
        if not self._trade_profits:
            return None
        
        total_days = sum((t['close_time'] - t['open_time']).days for t in self._trade_profits)
        return total_days / len(self._trade_profits)

    @metric(name='凯利公式最优仓位', group='交易', fmt='.1%', desc='根据胜率和盈亏比计算的最优仓位比例', order=50)
    def kelly_criterion(self) -> Optional[float]:
        """计算凯利公式最优仓位"""
        if not self._trade_profits:
            return None
        
        win_rate_val = self.win_rate()
        if win_rate_val is None:
            return None
        
        avg_profit = self.avg_profit()
        avg_loss = self.avg_loss()
        
        if avg_profit is None or avg_loss is None or avg_loss == 0:
            return None
        
        profit_loss_ratio = abs(avg_profit / avg_loss)
        return win_rate_val - (1 - win_rate_val) / profit_loss_ratio

    @metric(name='半凯利仓位', group='交易', fmt='.1%', desc='凯利公式最优仓位的50%', order=51)
    def kelly_fraction(self, fraction: float = 0.5) -> Optional[float]:
        """计算凯利分数仓位"""
        kelly = self.kelly_criterion()
        if kelly is None:
            return None
        return kelly * fraction

    def _collect_metrics(self) -> Dict:
        """收集所有 @metric 装饰器标记的方法"""
        metrics = {}
        for name in dir(self):
            if name.startswith('_'):
                continue
            attr = getattr(self, name, None)
            if callable(attr) and hasattr(attr, '_is_metric'):
                try:
                    value = attr()
                    metrics[attr._metric_name] = {
                        'name': attr._metric_name,
                        'value': value,
                        'group': getattr(attr, '_metric_group', ''),
                        'desc': attr._metric_desc,
                        'fmt': getattr(attr, '_metric_fmt', '.1%'),
                        'order': attr._metric_order,
                        'method': name
                    }
                except:
                    pass
        return metrics

    def metrics(self) -> Dict:
        """获取当前区间的所有指标"""
        return self._collect_metrics()

    def returns(self, periods: str = None) -> Dict:
        """批量计算多区间收益率"""
        if periods is None:
            return {'default': self.return_rate()}
        
        period_list = [p.strip() for p in periods.split(',')]
        result = {}
        
        for p in period_list:
            timeRange = self.getTimeRange(p)
            if timeRange:
                sliced_data = timeRange['daily_assets']
                if len(sliced_data) >= 2:
                    dates = sorted(sliced_data.keys())
                    benchmark_value = sliced_data[dates[0]]
                    end_value = sliced_data[dates[-1]]
                    if benchmark_value != 0:
                        result[p] = (end_value - benchmark_value) / benchmark_value
                    else:
                        result[p] = None
                else:
                    result[p] = None
            else:
                result[p] = None
        
        return result

    # ------------------------------------------------------------------------
    # 查询方法
    # ------------------------------------------------------------------------

    def get_daily_total_assets(self) -> Dict:
        """
        获取每日资产净值
        
        返回聚合后的每日资产数据，可用于绘制净值曲线
        
        Returns:
            Dict[date, float]: {日期：资产净值} 字典
                              日期为 date 对象，净值为 float
            
        Example:
            >>> assets = analyzer.get_daily_total_assets()
            >>> for date, value in sorted(assets.items()):
            ...     print(f"{date}: {value}")
        """
        return self._daily_assets

    def get_largest_profit_trades(self, n: int) -> List:
        """
        获取盈利最大的 N 笔交易
        
        按盈利金额降序排序，返回前 N 笔交易记录
        
        Args:
            n: 返回的交易数量
            
        Returns:
            List[Dict]: 交易记录列表，每笔交易包含：
                       - symbol: 标的代码
                       - profit: 盈亏金额
                       - open_time, close_time: 开平仓时间
                       - open_price, close_price: 开平仓价格
                       - volume: 交易数量
                       按盈利降序排序
            
        Example:
            >>> top5 = analyzer.get_largest_profit_trades(5)
            >>> for trade in top5:
            ...     print(f"{trade['symbol']}: 盈利 {trade['profit']}")
        """
        if not self._trade_profits or n <= 0:
            return []
        return sorted(self._trade_profits, key=lambda t: t['profit'], reverse=True)[:n]

    def get_largest_loss_trades(self, n: int) -> List:
        """
        获取亏损最大的 N 笔交易
        
        按亏损金额升序排序（亏损最多的在前），返回前 N 笔交易记录
        用于分析策略的失败案例，找出需要改进的地方
        
        Args:
            n: 返回的交易数量
            
        Returns:
            List[Dict]: 交易记录列表，按亏损升序排序
                       （亏损最多的交易在最前面）
            
        Example:
            >>> worst5 = analyzer.get_largest_loss_trades(5)
            >>> for trade in worst5:
            ...     print(f"{trade['symbol']}: 亏损 {abs(trade['profit'])}")
        """
        if not self._trade_profits or n <= 0:
            return []
        return sorted(self._trade_profits, key=lambda t: t['profit'])[:n]

    # ------------------------------------------------------------------------
    # 导出方法
    # ------------------------------------------------------------------------

    def to_notebook(self, title: str = "回测报告"):
        """导出为 Notebook 交互式报告

        遵循 notebook 推荐的 section 层次结构：
        - 顶层 KPI 卡片（无 section 包裹，一眼掌握全局）
        - 基础信息 / 指标汇总 / 收益分析(图表) / 交易明细 四大章节

        Args:
            title: 报告标题（同时作为默认输出文件名）

        Returns:
            Notebook 对象（链式调用）

        Example:
            analyzer.to_notebook("动量策略").export_html()
        """
        # [重构] 2026-05-30 按 notebook 推荐层次结构重组：
        #   顶层KPI → 基础信息(日期/资金) → 指标汇总(收益+风险+交易) →
        #   收益分析(净值+回撤图表) → 交易明细(表格,折叠)
        from notebook import Notebook
        import numpy as np

        nb = Notebook(title)
        # [修复] 2026-05-30 Notebook 在 analyzer.py 内部创建，
        #   base_dir 会指向 core/ 而非调用者目录，需手动纠正
        nb.base_dir = self.base_dir
        assets = self._daily_assets
        dates = sorted(assets.keys()) if assets else []

        def _fmt(val, t='.1%'):
            """格式化指标值，遵循 Python format 规范 ('.1%', '.2%', '.2f', '.1f')"""
            if val is None:
                return 'N/A'
            if isinstance(val, tuple):
                val = val[0]
            if t.endswith('%'):
                d = int(t[1:-1])
                return f"{val*100:.{d}f}%"
            if t.endswith('f'):
                d = int(t[1:-1])
                return f"{val:.{d}f}"
            return str(val)



        has_records = self.account and self.account.trade_records

        # 预计算（供 section 内使用）
        initial_cash = self.account.snapshots[0].cash if self.account and self.account.snapshots else 0
        final_nav = self.account.snapshots[-1].nav if self.account and self.account.snapshots else 0

        # 净值 + 回撤数据（图表共用）
        nav_values = []
        dd_pct = []
        if dates and len(dates) >= 2:
            nav_values = [assets[d] for d in dates]
            vals = np.array(nav_values)
            cumulative = vals / vals[0]
            running_max = np.maximum.accumulate(cumulative)
            dd_pct = (-(running_max - cumulative) / running_max * 100).tolist()

        # [重构] 2026-05-30 指标改为 @metric(group=...) 自动分组驱动
        #   新增指标只需加 @metric(group='xxx')，无需修改此处
        all_metrics = self.metrics()

        # 基础信息（非 @metric，单独处理）
        info_m = {}
        if dates and len(dates) >= 2:
            info_m['开始日期'] = dates[1].strftime('%Y-%m-%d')
            info_m['结束日期'] = dates[-1].strftime('%Y-%m-%d')
        if initial_cash > 0:
            info_m['初始资金'] = f"{initial_cash:,.0f}"
        if final_nav > 0:
            info_m['最终资产'] = f"{final_nav:,.0f}"

        # 按 group 分组指标（组内按 order 排序，附带 desc，根据指标类型判断格式）
        grouped = {}
        for m in all_metrics.values():
            g = m.get('group', '')
            if not g:
                continue
            val = m['value']
            if isinstance(val, tuple):
                val = val[0]
            if val is not None and not (isinstance(val, float) and val == float('inf')):
                # 使用 @metric 声明的 fmt 格式（默认 .1%）
                fmt_val = _fmt(val, m.get('fmt', '.1%'))
                grouped.setdefault(g, []).append(
                    (m['order'], m['name'], fmt_val, m.get('desc', ''))
                )
        for items in grouped.values():
            items.sort(key=lambda x: x[0])

        # ═══════════════════════════════════════════
        # ① Section：回测指标（@metric 驱动 + 基础信息）
        # ═══════════════════════════════════════════
        with nb.section("回测指标"):
            nb.metrics(info_m, title="基础指标", columns=4)

            # 按预设顺序输出各组指标
            for group_name in ('收益', '风险', '交易'):
                if group_name not in grouped:
                    continue
                # 交易组只在有成交记录时展示
                if group_name == '交易' and not has_records:
                    continue
                metrics_list = []
                for _, name, val, desc in grouped[group_name]:
                    item = {'name': name, 'value': val}
                    if desc:
                        item['desc'] = desc
                    metrics_list.append(item)
                # 交易组补充非 @metric 的平均盈亏
                if group_name == '交易':
                    avg_p = self.avg_profit(mode='amount')
                    avg_l = self.avg_loss(mode='amount')
                    if avg_p is not None:
                        metrics_list.append({'name': '平均盈利', 'value': f"{avg_p:,.0f}"})
                    if avg_l is not None:
                        metrics_list.append({'name': '平均亏损', 'value': f"{avg_l:,.0f}"})
                nb.metrics(metrics_list, title=f"{group_name}指标", columns=4)

        # ═══════════════════════════════════════════
        # ② Section：收益分析 — 净值 + 回撤图表
        # ═══════════════════════════════════════════
        with nb.section("收益分析"):
            # 净值曲线
            if nav_values:
                nb.chart('line', {
                    'xAxis': [d.strftime('%Y-%m-%d') for d in dates],
                    'series': [{'name': '策略净值', 'data': nav_values}],
                }, title='净值曲线', height='350px',
                    series_opts={'is_smooth': True},
                    yaxis_opts={'min_': nav_values[0] * 0.85},
                    datazoom_opts=[{'type_': 'slider', 'range_start': 0, 'range_end': 100}])

            # 回撤序列（负值表示下跌幅度）
            if dd_pct:
                max_dd_val = float(min(dd_pct))
                y_min = min(max_dd_val * 1.2, -5)
                nb.chart('area', {
                    'xAxis': [d.strftime('%Y-%m-%d') for d in dates],
                    'series': [{'name': '回撤%', 'data': dd_pct}],
                }, title='回撤序列', height='250px',
                    yaxis_opts={'min_': y_min, 'max_': 0})

        # ═══════════════════════════════════════════
        # ⑤ Section：交易明细（有成交记录时默认折叠）
        # ═══════════════════════════════════════════
        if has_records:
            trades = []
            # [新增] 2026-05-30 如果存在信号备注则展示备注列
            has_notes = any(getattr(t, 'note', '') for t in self.account.trade_records)
            for t in self.account.trade_records:
                row = {
                    '日期': t.created_at.strftime('%Y-%m-%d %H:%M'),
                    '标的': t.symbol,
                    '方向': '买入' if t.side == 1 else '卖出',
                    '价格': t.price,
                    '数量': t.volume,
                    '金额': round(t.amount, 2),
                    '手续费': round(t.fee, 2),
                }
                if has_notes:
                    row['备注'] = getattr(t, 'note', '')
                trades.append(row)
            with nb.section("交易明细", collapsed=True):
                nb.table(trades, page={'size': 20})

        return nb

    def to_excel(self, report_name: str = "回测报告", output_dir: str = "."):
        """导出 Excel 文件，与 notebook 报告结构对应

        Sheet 结构:
            回测指标 — 所有 @metric 指标汇总
            每日资产 — 日期 + 现金 + 持仓市值 + 总净值
            交易记录 — 全部成交明细（日期/标的/方向/价格/数量/金额/手续费/备注）

        Args:
            report_name: 报告名称，用于生成文件名
                       格式：{report_name}_{YYYYMMDD_HHMM}.xlsx
            output_dir: 输出目录（相对路径，基于调用者所在目录）
        """
        import openpyxl
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

        # ---------- Sheet 1: 回测指标 ----------
        metrics_data = self.metrics()
        ordered = sorted(metrics_data.values(), key=lambda x: x.get('order', 99))

        # ---------- Sheet 2: 每日资产 ----------
        assets = self._daily_assets
        dates_sorted = sorted(assets.keys())
        # 从 snapshots 反推每日 cash（取当日最后一个快照）
        if self.account and self.account.snapshots:
            daily_last_snap = {}
            for s in self.account.snapshots:
                d = s.created_at.date()
                daily_last_snap[d] = s
            asset_rows = []
            for d in dates_sorted:
                nav = assets[d]
                snap = daily_last_snap.get(d)
                cash = snap.cash if snap else 0
                pos_val = nav - cash
                asset_rows.append({
                    '日期': d.strftime('%Y-%m-%d'),
                    '现金': round(cash, 2),
                    '持仓市值': round(pos_val, 2),
                    '总净值': round(nav, 2),
                })
        else:
            asset_rows = [{'日期': d.strftime('%Y-%m-%d'), '总净值': round(v, 2)}
                         for d, v in sorted(assets.items())]

        # ---------- Sheet 3: 交易记录 ----------
        trade_rows = []
        has_notes = self.account and any(getattr(t, 'note', '') for t in self.account.trade_records)
        if self.account:
            for t in self.account.trade_records:
                row = {
                    '日期': t.created_at.strftime('%Y-%m-%d %H:%M'),
                    '标的': t.symbol,
                    '方向': '买入' if t.side == 1 else '卖出',
                    '价格': t.price,
                    '数量': t.volume,
                    '金额': round(t.amount, 2),
                    '手续费': round(t.fee, 2),
                }
                if has_notes:
                    row['备注'] = getattr(t, 'note', '')
                trade_rows.append(row)

        # ---------- 写入 Excel ----------
        wb = openpyxl.Workbook()

        # 样式定义
        header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        header_font_white = Font(bold=True, size=11, color='FFFFFF')
        thin_border = Border(
            left=Side(style='thin'), right=Side(style='thin'),
            top=Side(style='thin'), bottom=Side(style='thin')
        )

        def write_sheet(ws, title, headers, rows):
            """写入一个 sheet，带标题和表头"""
            # 标题行
            ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(headers))
            ws.cell(row=1, column=1, value=title).font = Font(bold=True, size=14)
            # 表头
            for ci, h in enumerate(headers, 1):
                cell = ws.cell(row=2, column=ci, value=h)
                cell.font = header_font_white
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center')
                cell.border = thin_border
            # 数据行
            for ri, row in enumerate(rows, 3):
                for ci, h in enumerate(headers, 1):
                    val = row.get(h, '')
                    cell = ws.cell(row=ri, column=ci, value=val)
                    cell.border = thin_border
                    # 数值列右对齐
                    if isinstance(val, (int, float)):
                        cell.alignment = Alignment(horizontal='right')
            # 列宽自适应
            for ci, h in enumerate(headers, 1):
                max_len = max(len(str(h)), max((len(str(r.get(h, ''))) for r in rows), default=0)) + 2
                ws.column_dimensions[openpyxl.utils.get_column_letter(ci)].width = min(max_len, 30)

        # Sheet 1: 回测指标
        ws1 = wb.active
        ws1.title = '回测指标'
        m_headers = ['指标名称', '数值', '说明']
        m_rows = []
        for m in ordered:
            val = m['value']
            if isinstance(val, tuple):
                val = val[0]
            # 使用 @metric 声明的 fmt 格式输出
            val_str = val
            if isinstance(val, (int, float)):
                fmt = m.get('fmt', '.1%')
                if fmt.endswith('%'):
                    d = int(fmt[1:-1])
                    val_str = f"{val*100:.{d}f}%"
                elif fmt.endswith('f'):
                    d = int(fmt[1:-1])
                    val_str = f"{val:.{d}f}"
            m_rows.append({'指标名称': m['name'], '数值': val_str, '说明': m.get('desc', '')})
        write_sheet(ws1, f'{report_name} — 回测指标', m_headers, m_rows)

        # Sheet 2: 每日资产
        ws2 = wb.create_sheet('每日资产')
        a_headers = list(asset_rows[0].keys()) if asset_rows else []
        write_sheet(ws2, f'{report_name} — 每日资产', a_headers, asset_rows)

        # Sheet 3: 交易记录
        ws3 = wb.create_sheet('交易记录')
        t_headers = list(trade_rows[0].keys()) if trade_rows else []
        write_sheet(ws3, f'{report_name} — 交易记录', t_headers, trade_rows)

        # 保存
        from pathlib import Path
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = Path(self.base_dir) / output_dir / f"{report_name}_{current_datetime}.xlsx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(output_path)
        print(f'Excel 已生成至: {output_path}')
        return str(output_path)

    # [移除] 2026-05-30 to_html() 已由 to_notebook() 替代

    @staticmethod
    def _aggregate_daily_assets(snapshots: List) -> Dict:
        """
        聚合账户快照为日数据（每日取最后一个快照）
        
        账户类每次交易都会生成快照，一天内可能有多次交易
        此方法将日内多个快照聚合为每日一个数据点，取最后一个快照的资产值
        
        Args:
            snapshots: AccountSnapshot 对象列表
                      每个快照包含 created_at（datetime）和 nav（资产净值）
            
        Returns:
            Dict[date, float]: {日期：资产净值} 字典
                              日期为 date 对象，资产净值为 float
            
        Example:
            >>> daily = AccountAnalyzer._aggregate_daily_assets(account.snapshots)
            >>> # 结果：{date(2024, 1, 1): 100000, date(2024, 1, 2): 101500, ...}
        """
        daily_snapshots = defaultdict(list)
        for snapshot in snapshots:
            snapshot_date = snapshot.created_at.date()
            daily_snapshots[snapshot_date].append(snapshot)

        return {
            snapshot_date: snaps[-1].nav
            for snapshot_date, snaps in daily_snapshots.items()
        }

    def _slice_data_by_range(self, time_range: Optional[Dict] = None, include_benchmark: bool = False) -> Dict[date, float]:
        """
        根据时间区间截取数据
        
        核心功能：
        1. 支持三种方式指定区间：直接设置 start/end、使用预设 period、使用全部数据
        2. 处理基准日逻辑：
           - 不传 time_range 时：使用固定的第一个交易日为基准日（date[0]），从 date[1] 开始计算
           - 传入 time_range 时：使用区间起点的前一个交易日为基准日（类似 JS 版本）
        3. 可选择是否包含基准日数据（用于计算收益率）
        
        Args:
            time_range: 时间区间配置（Dict），可选：
                       - None: 使用全部数据
                       - {'period': '3m'}: 使用预设周期
                       - {'start': date, 'end': date}: 指定日期区间
            include_benchmark: 是否包含基准日数据
                              - True: 返回包含基准日的完整数据（用于计算）
                              - False: 只返回计算区间的数据（用于展示）
            
        Returns:
            Dict[date, float]: 截取后的资产数据 {日期：资产净值}
            
        基准日逻辑说明（向 JS 版本看齐）：
            场景 1：不传 time_range（使用全部数据）
                - 基准日：date[0]（第一个交易日）
                - 计算起点：date[1]（第二个交易日）
                - 用途：生成固定报告，从成立来统计
            
            场景 2：传入 time_range（自定义区间）
                - 基准日：区间起点的前一个交易日
                - 计算起点：区间的实际起点
                - 用途：动态区间统计，确保第一天就有实际收益率
                - 示例：用户选择 2024-01-03 至 2024-03-31
                  基准日：2024-01-02（起点的前一日）
                  计算：2024-01-03 的收益率 = (01-03 净值 - 01-02 净值) / 01-02 净值
            
            为什么这样设计？
                - 如果区间从 date[0] 开始，无法找到前一个交易日，所以使用 date[0] 作为基准
                - 如果区间从 date[N] 开始（N>0），使用 date[N-1] 作为基准，确保 date[N] 有实际收益率
                - 这与 JS 版本的逻辑完全一致，支持动态区间切换
            
        Example:
            >>> # 只返回计算区间（用于展示）
            >>> data = analyzer._slice_data_by_range({'period': '3m'}, include_benchmark=False)
            >>>
            >>> # 返回包含基准日的数据（用于计算收益率）
            >>> data_with_base = analyzer._slice_data_by_range({'period': '3m'}, include_benchmark=True)
            >>> start_value = data_with_base[benchmark_date]  # 基准日的值
        """
        if not self._daily_assets:
            return {}
        
        dates = sorted(self._daily_assets.keys())
        if len(dates) < 2:
            return self._daily_assets.copy()
        
        # 全部数据的基准日和第一个交易日
        all_benchmark_date = dates[0]
        all_first_trading_date = dates[1]
        
        # 确定区间的起止日期和基准日
        if time_range is None:
            # 不传参数：使用全部数据，从 date[1] 开始
            benchmark_date = all_benchmark_date
            start_date = all_first_trading_date
            end_date = dates[-1]
        elif time_range.period:
            # 使用预设周期
            periods = {
                '1m': relativedelta(months=1),
                '3m': relativedelta(months=3),
                '6m': relativedelta(months=6),
                '1y': relativedelta(years=1),
                '2y': relativedelta(years=2),
                '3y': relativedelta(years=3),
                '5y': relativedelta(years=5),
                'all': None
            }
            
            if time_range.period == 'all':
                benchmark_date = all_benchmark_date
                start_date = all_first_trading_date
                end_date = dates[-1]
            else:
                delta = periods.get(time_range.period)
                if delta is None:
                    benchmark_date = all_benchmark_date
                    start_date = all_first_trading_date
                    end_date = dates[-1]
                else:
                    # 从结束日往前推 delta，找到最接近的交易日
                    calculated_start = dates[-1] - delta
                    # 找到 <= calculated_start 的最大日期
                    valid_dates = [d for d in dates if d <= calculated_start]
                    raw_start_date = max(valid_dates) if valid_dates else all_first_trading_date
                    
                    # 确定基准日：起点的前一个交易日
                    if raw_start_date == all_first_trading_date:
                        benchmark_date = all_benchmark_date
                        start_date = all_first_trading_date
                    else:
                        # 找到 raw_start_date 的前一个交易日
                        prev_dates = [d for d in dates if d < raw_start_date]
                        benchmark_date = max(prev_dates) if prev_dates else all_benchmark_date
                        start_date = raw_start_date
                    
                    end_date = dates[-1]
        else:
            # 使用自定义 start/end
            raw_start_date = time_range.start if time_range.start else all_first_trading_date
            
            # 确保不早于 all_first_trading_date
            if raw_start_date < all_first_trading_date:
                benchmark_date = all_benchmark_date
                start_date = all_first_trading_date
            else:
                # 找到 raw_start_date 的前一个交易日作为基准日
                prev_dates = [d for d in dates if d < raw_start_date]
                if prev_dates:
                    benchmark_date = max(prev_dates)
                    start_date = raw_start_date
                else:
                    benchmark_date = all_benchmark_date
                    start_date = all_first_trading_date
            
            end_date = time_range.end if time_range.end else dates[-1]
        
        # 截取数据
        if include_benchmark:
            # 包含基准日：从 benchmark_date 到 end_date
            sliced_data = {
                d: v for d, v in self._daily_assets.items()
                if benchmark_date <= d <= end_date
            }
        else:
            # 不包含基准日：从 start_date 到 end_date
            sliced_data = {
                d: v for d, v in self._daily_assets.items()
                if start_date <= d <= end_date
            }
        
        return sliced_data

    def _calculate_daily_returns(self, daily_assets: Dict) -> List:
        """
        计算日收益率序列
        
        根据每日资产数据计算相邻交易日之间的收益率
        用于计算波动率、VaR、CVaR 等风险指标
        
        计算公式：(今日资产 - 昨日资产) / 昨日资产
        
        Args:
            daily_assets: 每日资产字典 {date: nav}
            
        Returns:
            List[float]: 日收益率列表（小数形式）
                        长度为 N-1（N 为数据点数量）
            
        Example:
            >>> assets = {date(2024,1,1): 100000, date(2024,1,2): 101000}
            >>> returns = analyzer._calculate_daily_returns(assets)
            >>> # 结果：[0.01] 表示 1% 的日收益率
        """
        dates = sorted(daily_assets.keys())
        returns = []
        for i in range(1, len(dates)):
            prev = daily_assets[dates[i - 1]]
            curr = daily_assets[dates[i]]
            returns.append((curr - prev) / prev)
        return returns



    def _calculate_profit(self, trade_records: List) -> List:
        """
        计算每笔交易的盈亏
        
        使用 FIFO（先进先出）原则匹配开平仓交易：
        - 买入：累积持仓，记录平均开仓价格和手续费
        - 卖出：按持仓比例计算成本和盈亏，平仓后重置持仓信息
        
        盈亏计算公式：
        盈亏 = 卖出金额 - 成本 - 手续费
        成本 = (卖出数量 / 持仓数量) × 总成本
        
        Args:
            trade_records: 成交记录列表
                         每条记录包含：symbol, volume, price, side, fee, created_at
            
        Returns:
            List[Dict]: 盈亏记录列表，每条记录包含：
                       - symbol: 标的代码
                       - profit: 盈亏金额（正为盈利，负为亏损）
                       - open_time, close_time: 开平仓时间
                       - open_price, close_price: 开平仓价格
                       - volume: 交易数量（负数表示卖出）
                       - open_fee, close_fee: 开平仓手续费
                       - original_trade: 原始交易记录对象
            
        Example:
            >>> profits = analyzer._calculate_profit(trade_records)
            >>> for trade in profits:
            ...     print(f"{trade['symbol']}: 盈亏 {trade['profit']}")
        """
        positions = defaultdict(lambda: {'volume': 0, 'cost': 0, 'open_time': None, 'open_price': 0, 'open_fee': 0})
        processed_trades = []

        for trade in trade_records:
            if trade.volume == 0 or math.isnan(trade.price):
                continue
            symbol = trade.symbol
            volume = trade.volume
            abs_volume = abs(volume)
            price = trade.price
            side = trade.side
            created_at = trade.created_at
            fee = trade.fee

            # [修复] 2026-05-30 TradeRecord.side 是 int (OrderSide.Buy=1)，
            #   原用字符串 'buy'/'sell' 比较导致永久为 False，交易盈亏始终为空
            if side == OrderSide.Buy:
                positions[symbol]['volume'] += abs_volume
                positions[symbol]['cost'] += abs_volume * price + fee
                if positions[symbol]['open_time'] is None:
                    positions[symbol]['open_time'] = created_at
                    positions[symbol]['open_price'] = price
                    positions[symbol]['open_fee'] += fee
            elif side == OrderSide.Sell:
                if positions[symbol]['volume'] == 0:
                    continue

                sell_amount = abs_volume * price
                cost = (abs_volume / positions[symbol]['volume']) * positions[symbol]['cost']
                profit = sell_amount - cost - fee

                open_fee_portion = (abs_volume / positions[symbol]['volume']) * positions[symbol]['open_fee']

                processed_trades.append({
                    'symbol': symbol,
                    'profit': profit,
                    'open_time': positions[symbol]['open_time'],
                    'close_time': created_at,
                    'open_price': positions[symbol]['open_price'],
                    'open_fee': open_fee_portion,
                    'close_fee': fee,
                    'close_price': price,
                    'volume': volume,
                    'original_trade': trade
                })

                positions[symbol]['volume'] -= abs_volume
                positions[symbol]['cost'] -= cost
                positions[symbol]['open_fee'] -= open_fee_portion

                if positions[symbol]['volume'] == 0:
                    positions[symbol]['open_time'] = None
                    positions[symbol]['open_price'] = 0
                    positions[symbol]['open_fee'] = 0

        return processed_trades

    def _get_caller_dir(self) -> str:
        """
        获取调用者所在目录
        
        使用 inspect 模块获取调用当前方法的代码所在的目录
        用于确定 HTML 报告的输出路径基准
        
        Returns:
            str: 调用者脚本的绝对目录路径
            
        Example:
            # 如果在 d:\\project\\backtest\\main.py 中调用
            # analyzer._get_caller_dir() 返回 "d:\\project\\backtest"
        """
        frame = inspect.currentframe()
        try:
            caller_frame = None
            for frame_info in inspect.stack():
                if frame_info.filename != __file__:
                    caller_frame = frame_info
                    break

            if caller_frame:
                return os.path.dirname(os.path.abspath(caller_frame.filename))
            else:
                return os.path.dirname(os.path.abspath(__file__))
        finally:
            del frame
