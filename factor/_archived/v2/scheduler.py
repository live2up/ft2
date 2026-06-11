"""
调度器抽象层 - 调仓日生成器

设计思路：
--------
1. RebalanceScheduler(ABC)：调仓日生成抽象基类，定义 generate(dates) 接口
2. FixedScheduler：固定频率调度器，支持 'ME'(月末) / 'W'(周末) / 'M'(月初)
3. IntervalScheduler：固定间隔调度器，每 N 个交易日调仓一次
4. recommend_scheduler_from_decay()：根据 IC 半衰期自动推荐调度器参数

使用方式：
--------
>>> scheduler = FixedScheduler('ME')           # 月末调仓
>>> rebalance_dates = scheduler.generate(dates)  # 生成调仓日列表
>>> scheduler = recommend_scheduler_from_decay(8.5)  # 半衰期8.5天 → 推荐周频

与 V1 的关系：
------------
V1 中调仓频率作为全局配置锁死在脚本参数中，无法逐因子切换。
V2 将调度器抽象为独立可插拔模块，每个因子可使用不同调度器进行回测评估。

依赖说明：
--------
纯日期运算，仅依赖 pandas，不依赖 ft2 其他模块。
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd


class RebalanceScheduler(ABC):
    """调仓日生成器抽象基类
    
    所有调度器必须实现 generate() 方法，输入交易日序列，输出调仓日列表。
    子类可扩展支持自适应/信号驱动等高级调度策略。
    """

    @abstractmethod
    def generate(self, dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """从交易日序列中生成调仓日列表
        
        Args:
            dates: 交易日序列（DatetimeIndex）
            
        Returns:
            List[pd.Timestamp]: 调仓日列表，按时间升序排列
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class FixedScheduler(RebalanceScheduler):
    """固定频率调度器
    
    支持三种模式：
    - 'ME': 每月最后一个交易日（Month End）
    - 'W' : 每周最后一个交易日（Week End）
    - 'M' : 每月第一个交易日（Month Start）
    
    注意：'ME'/'M' 基于交易日历（dates）而非自然日历，
    即取每个自然月内在 dates 中出现的最后/第一个交易日。
    """

    # 支持的模式及其描述
    VALID_MODES = {'ME', 'W', 'M'}

    def __init__(self, mode: str = 'ME'):
        """初始化固定频率调度器
        
        Args:
            mode: 调度模式，可选 'ME'(月末) / 'W'(周末) / 'M'(月初)
            
        Raises:
            ValueError: 如果 mode 不在 VALID_MODES 中
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"不支持的模式 '{mode}'，可选: {self.VALID_MODES}")
        self.mode = mode

    def generate(self, dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """生成固定频率调仓日列表
        
        Args:
            dates: 交易日序列
            
        Returns:
            List[pd.Timestamp]: 调仓日列表
        """
        if len(dates) == 0:
            return []

        df = pd.DataFrame({'date': dates})
        df['year'] = dates.year
        df['month'] = dates.month

        if self.mode == 'W':
            # 周频：取每周最后一个交易日
            # [修复] 2026-05-20 使用 isocalendar().year 处理跨年周
            # 旧实现用 dates.year 拼接，跨年周会分组错误（如 2023-12-31 属 ISO 2024-W01）
            # .values 避免 UInt32 索引对齐问题（iso 的 index 是 DatetimeIndex，df 是 RangeIndex）
            iso = dates.isocalendar()
            iso_year = iso['year'].to_numpy(dtype=int)
            iso_week = iso['week'].to_numpy(dtype=int)
            df['year_week'] = [
                f"{y}-{w:02d}" for y, w in zip(iso_year, iso_week)
            ]
            rebalance = df.groupby('year_week')['date'].last()
        elif self.mode == 'ME':
            # 月频：取每月最后一个交易日
            df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
            rebalance = df.groupby('year_month')['date'].last()
        elif self.mode == 'M':
            # 月初：取每月第一个交易日
            df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
            rebalance = df.groupby('year_month')['date'].first()

        result = sorted(rebalance.tolist())
        # 过滤掉第一个周期（数据不足，无法调仓）
        # 第一个日期通常是起始观察点，而非调仓点
        if len(result) > 1:
            return result[1:]
        return result

    def __repr__(self) -> str:
        return f"FixedScheduler(mode='{self.mode}')"


class IntervalScheduler(RebalanceScheduler):
    """固定间隔调度器
    
    每隔 N 个交易日调仓一次。
    首个调仓日在第 N 个交易日（给因子计算留出 warm-up 期）。
    """

    def __init__(self, interval_days: int = 5):
        """初始化间隔调度器
        
        Args:
            interval_days: 调仓间隔（交易日数），默认 5
            
        Raises:
            ValueError: 如果 interval_days < 1
        """
        if interval_days < 1:
            raise ValueError(f"interval_days 必须 >= 1，当前值: {interval_days}")
        self.interval_days = interval_days

    def generate(self, dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """生成固定间隔调仓日列表
        
        Args:
            dates: 交易日序列
            
        Returns:
            List[pd.Timestamp]: 调仓日列表
        """
        n = len(dates)
        if n <= self.interval_days:
            return []

        # 从第 interval_days 个位置开始（0-indexed），每 interval_days 步取一个
        result = []
        for i in range(self.interval_days - 1, n, self.interval_days):
            result.append(dates[i])

        return result

    def __repr__(self) -> str:
        return f"IntervalScheduler(interval_days={self.interval_days})"


def recommend_scheduler_from_decay(half_life: float) -> RebalanceScheduler:
    """根据 IC 半衰期自动推荐调度器
    
    推荐规则：
    - half_life <= 5   → IntervalScheduler(5)，每 5 天调仓（因子衰减极快）
    - half_life <= 10  → FixedScheduler('W')，周频调仓（因子衰减较快）
    - half_life <= 22  → IntervalScheduler(10)，双周调仓（因子衰减中等）
    - half_life > 22   → FixedScheduler('ME')，月频调仓（因子衰减慢，长周期有效）
    
    Args:
        half_life: IC 半衰期（交易日数），由 validator.decay_rate() 计算得到
        
    Returns:
        RebalanceScheduler: 推荐的调度器实例
        
    Raises:
        ValueError: 如果 half_life <= 0
    """
    if half_life <= 0:
        raise ValueError(f"半衰期必须 > 0，当前值: {half_life}")

    if half_life <= 5:
        return IntervalScheduler(5)
    elif half_life <= 10:
        return FixedScheduler('W')
    elif half_life <= 22:
        return IntervalScheduler(10)
    else:
        return FixedScheduler('ME')
