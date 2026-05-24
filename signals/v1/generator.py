# signal/generator.py - 信号生成器
"""
信号生成器：指标计算 → 信号

TA-Lib 分类：
1. Overlap Studies (重叠研究/趋势指标)
2. Momentum Indicators (动量指标)
3. Volume Indicators (成交量指标)
4. Volatility Indicators (波动率指标)
5. Price Transform (价格转换)
6. Cycle Indicators (周期指标)
7. Pattern Recognition (形态识别)
8. Statistic Functions (统计函数)

支持：
1. 内置信号生成器（按 TA-Lib 分类组织）
2. 自定义信号生成器（函数式）
3. 可组合信号（多个生成器组合）
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import pandas as pd
import numpy as np
import talib

from .base import Signal, SignalType, SignalDirection, SignalSeries


class SignalGenerator(ABC):
    """
    信号生成器基类
    
    输入：原始 K 线数据
    输出：Signal / SignalSeries
    
    支持：
    1. 传统技术指标（无需训练，直接计算）
    2. 机器学习模型（需要训练/拟合）
    3. 遗传编程（需要进化优化）
    """
    
    def __init__(self, name: str):
        self.name = name
        self.params: Dict[str, Any] = {}
        self.is_fitted = True  # 默认已拟合（传统指标不需要训练）
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """
        从配置字典创建信号生成器实例
        
        Args:
            config: 配置字典，包含 'period', 'nbdev' 等参数
            
        Returns:
            SignalGenerator: 信号生成器实例
        """
        # 子类可以重写此方法以支持自定义参数映射
        return cls(**config)
    
    def fit(self, data: pd.DataFrame, target: pd.Series = None, **kwargs):
        """
        训练/拟合信号生成器（可选）
        
        对于传统技术指标：不需要实现，保持 is_fitted=True
        对于机器学习/遗传编程：需要重写此方法实现训练逻辑
        
        Args:
            data: K 线数据（训练集）
            target: 目标变量（如未来收益率），可选
            **kwargs: 其他训练参数
            
        Returns:
            self: 返回自身，支持链式调用
        """
        return self
    
    @abstractmethod
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """
        生成信号序列
        
        Args:
            data: K 线数据，必须包含 open/high/low/close/volume
            
        Returns:
            pd.Series: 信号值序列
        """
        pass
    
    def generate_latest(self, data: pd.DataFrame) -> Signal:
        """
        生成最新信号
        
        Args:
            data: K 线数据
            
        Returns:
            Signal: 最新信号
            
        Raises:
            ValueError: 如果模型未拟合（is_fitted=False）
        """
        if not self.is_fitted:
            raise ValueError(
                f"Signal generator '{self.name}' is not fitted. "
                f"Call fit() method before generating signals."
            )
        
        series = self.generate(data)
        latest_value = series.iloc[-1] if not series.empty else 0
        
        if latest_value > 0:
            direction = SignalDirection.LONG
        elif latest_value < 0:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL
        
        return Signal(
            name=self.name,
            value=latest_value,
            direction=direction,
            timestamp=datetime.now(),
            metadata={
                'params': self.params,
                'is_fitted': self.is_fitted
            }
        )
    
    def save(self, path: str):
        """保存模型到文件"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'SignalGenerator':
        """从文件加载模型"""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def __str__(self):
        params_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"


# =============================================================================
# 1. Overlap Studies (重叠研究/趋势指标)
# =============================================================================

class MASignal(SignalGenerator):
    """均线交叉信号
    
    信号值：
    - > 0: 短期均线 > 长期均线（金叉）
    - < 0: 短期均线 < 长期均线（死叉）
    """
    
    def __init__(self, short_period: int = 5, long_period: int = 20):
        super().__init__(f"MA{short_period}_{long_period}")
        self.short_period = short_period
        self.long_period = long_period
        self.params = {'short_period': short_period, 'long_period': long_period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        short_ma = close.rolling(self.short_period).mean()
        long_ma = close.rolling(self.long_period).mean()
        signal = (short_ma - long_ma) / long_ma
        return signal


class BOLLSignal(SignalGenerator):
    """布林带信号
    
    信号值：
    - 价格 < 下轨: 超卖，做多信号
    - 价格 > 上轨: 超买，做空信号
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(f"BOLL{period}_{std_dev}")
        self.period = period
        self.std_dev = std_dev
        self.params = {'period': period, 'std_dev': std_dev}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        mid = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = mid + self.std_dev * std
        lower = mid - self.std_dev * std
        signal = (close - lower) / (upper - lower)
        signal = signal.fillna(0.5)
        return signal - 0.5


class SARSignal(SignalGenerator):
    """SAR 抛物线转向指标信号
    
    信号值：价格与 SAR 的差距
    - 正值: SAR 在价格下方（多头）
    - 负值: SAR 在价格上方（空头）
    """
    
    def __init__(self, acceleration: float = 0.02, maximum: float = 0.2):
        super().__init__(f"SAR")
        self.acceleration = acceleration
        self.maximum = maximum
        self.params = {'acceleration': acceleration, 'maximum': maximum}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        sar = talib.SAR(high, low, acceleration=self.acceleration, maximum=self.maximum)
        signal = pd.Series((close - sar) / close * 100, index=data.index)
        return signal.fillna(0)


class MIDPOINTSignal(SignalGenerator):
    """MIDPOINT 中点指标信号
    
    信号值：价格偏离中点的百分比
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"MIDPOINT{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        midpoint = talib.MIDPOINT(close, timeperiod=self.period)
        signal = pd.Series((close - midpoint) / midpoint * 100, index=data.index)
        return signal.fillna(0)


class DMISignal(SignalGenerator):
    """DMI 趋向指标信号
    
    信号值：
    - +DI > -DI: 多头趋势
    - +DI < -DI: 空头趋势
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"DMI{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = np.where((high_diff > low_diff.diff().abs()) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff.abs() > high_diff) & (low_diff < 0), low_diff.abs(), 0)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        plus_dm_smooth = pd.Series(plus_dm, index=data.index).rolling(self.period).sum()
        minus_dm_smooth = pd.Series(minus_dm, index=data.index).rolling(self.period).sum()
        tr_smooth = pd.Series(tr).rolling(self.period).sum()
        
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        signal = plus_di - minus_di
        signal = signal.fillna(0)
        
        return signal


class AROONSignal(SignalGenerator):
    """AROON 阿隆指标信号
    
    信号值：Aroon_Up - Aroon_Down
    - 正值: 上升趋势
    - 负值: 下降趋势
    """
    
    def __init__(self, period: int = 25):
        super().__init__(f"AROON{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        
        aroon_up = high.rolling(self.period + 1).apply(lambda x: x.argmax(), raw=True)
        aroon_up = 100 * (self.period - aroon_up) / self.period
        
        aroon_down = low.rolling(self.period + 1).apply(lambda x: x.argmin(), raw=True)
        aroon_down = 100 * (self.period - aroon_down) / self.period
        
        signal = aroon_up - aroon_down
        signal = signal.fillna(0)
        
        return signal


class SMASignal(SignalGenerator):
    """SMA 简单移动平均信号
    
    信号值：价格偏离 SMA 的百分比
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"SMA{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        sma = talib.SMA(close, timeperiod=self.period)
        signal = pd.Series((close - sma) / sma * 100, index=data.index)
        return signal.fillna(0)


class EMASignal(SignalGenerator):
    """EMA 指数移动平均信号
    
    信号值：价格偏离 EMA 的百分比
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"EMA{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        ema = talib.EMA(close, timeperiod=self.period)
        signal = pd.Series((close - ema) / ema * 100, index=data.index)
        return signal.fillna(0)


class WMASignal(SignalGenerator):
    """WMA 加权移动平均信号
    
    信号值：价格偏离 WMA 的百分比
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"WMA{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        wma = talib.WMA(close, timeperiod=self.period)
        signal = pd.Series((close - wma) / wma * 100, index=data.index)
        return signal.fillna(0)


class DEMASignal(SignalGenerator):
    """DEMA 双指数移动平均信号
    
    信号值：价格偏离 DEMA 的百分比
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"DEMA{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        dema = talib.DEMA(close, timeperiod=self.period)
        signal = pd.Series((close - dema) / dema * 100, index=data.index)
        return signal.fillna(0)


class TEMASignal(SignalGenerator):
    """TEMA 三重指数移动平均信号
    
    信号值：价格偏离 TEMA 的百分比
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"TEMA{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        tema = talib.TEMA(close, timeperiod=self.period)
        signal = pd.Series((close - tema) / tema * 100, index=data.index)
        return signal.fillna(0)


class KAMASignal(SignalGenerator):
    """KAMA 考夫曼自适应移动平均信号
    
    信号值：价格偏离 KAMA 的百分比
    """
    
    def __init__(self, period: int = 30):
        super().__init__(f"KAMA{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        kama = talib.KAMA(close, timeperiod=self.period)
        signal = pd.Series((close - kama) / kama * 100, index=data.index)
        return signal.fillna(0)


class T3Signal(SignalGenerator):
    """T3 三重指数平滑移动平均信号
    
    信号值：价格偏离 T3 的百分比
    """
    
    def __init__(self, period: int = 5, vfactor: float = 0.7):
        super().__init__(f"T3{period}")
        self.period = period
        self.vfactor = vfactor
        self.params = {'period': period, 'vfactor': vfactor}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        t3 = talib.T3(close, timeperiod=self.period, vfactor=self.vfactor)
        signal = pd.Series((close - t3) / t3 * 100, index=data.index)
        return signal.fillna(0)


class TRIMASignal(SignalGenerator):
    """TRIMA 三角移动平均信号
    
    信号值：价格偏离 TRIMA 的百分比
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"TRIMA{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        trima = talib.TRIMA(close, timeperiod=self.period)
        signal = pd.Series((close - trima) / trima * 100, index=data.index)
        return signal.fillna(0)


class MAMASignal(SignalGenerator):
    """MAMA 梅斯自适应移动平均信号
    
    信号值：价格偏离 MAMA 的百分比
    """
    
    def __init__(self, fastlimit: float = 0.5, slowlimit: float = 0.05):
        super().__init__("MAMA")
        self.fastlimit = fastlimit
        self.slowlimit = slowlimit
        self.params = {'fastlimit': fastlimit, 'slowlimit': slowlimit}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        mama, fama = talib.MAMA(close, fastlimit=self.fastlimit, slowlimit=self.slowlimit)
        signal = pd.Series((close - mama) / mama * 100, index=data.index)
        return signal.fillna(0)


class MAVPSignal(SignalGenerator):
    """MAVP 可变周期移动平均信号
    
    信号值：价格偏离 MAVP 的百分比
    """
    
    def __init__(self, minperiod: int = 2, maxperiod: int = 30, matype: int = 0):
        super().__init__("MAVP")
        self.minperiod = minperiod
        self.maxperiod = maxperiod
        self.matype = matype
        self.params = {'minperiod': minperiod, 'maxperiod': maxperiod, 'matype': matype}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        periods = np.linspace(self.minperiod, self.maxperiod, len(close))
        mavp = talib.MAVP(close, periods, minperiod=self.minperiod, maxperiod=self.maxperiod, matype=self.matype)
        signal = pd.Series((close - mavp) / mavp * 100, index=data.index)
        return signal.fillna(0)


class AVGPRICESignal(SignalGenerator):
    """AVGPRICE 平均价格信号
    
    信号值：(开盘+最高+最低+收盘)/4 与实际收盘价的差值百分比
    """
    
    def __init__(self):
        super().__init__("AVGPRICE")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        open_p = data['open'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        avgprice = talib.AVGPRICE(open_p, high, low, close)
        signal = pd.Series((close - avgprice) / avgprice * 100, index=data.index)
        return signal.fillna(0)


class MEDPRICESignal(SignalGenerator):
    """MEDPRICE 中间价格信号
    
    信号值：(最高+最低)/2 与实际收盘价的差值百分比
    """
    
    def __init__(self):
        super().__init__("MEDPRICE")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        medprice = talib.MEDPRICE(high, low)
        signal = pd.Series((close - medprice) / medprice * 100, index=data.index)
        return signal.fillna(0)


class TYPPRICESignal(SignalGenerator):
    """TYPPRICE 典型价格信号
    
    信号值：(最高+最低+收盘)/3 与实际收盘价的差值百分比
    """
    
    def __init__(self):
        super().__init__("TYPPRICE")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        typprice = talib.TYPPRICE(high, low, close)
        signal = pd.Series((close - typprice) / typprice * 100, index=data.index)
        return signal.fillna(0)


class WCLPRICESignal(SignalGenerator):
    """WCLPRICE 加权收盘价信号
    
    信号值：(最高+最低+2*收盘)/4 与实际收盘价的差值百分比
    """
    
    def __init__(self):
        super().__init__("WCLPRICE")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        wclprice = talib.WCLPRICE(high, low, close)
        signal = pd.Series((close - wclprice) / wclprice * 100, index=data.index)
        return signal.fillna(0)


class HTTRENDLINESignal(SignalGenerator):
    """HT_TRENDLINE 希尔伯特瞬时趋势线信号
    
    信号值：价格偏离趋势线的百分比
    """
    
    def __init__(self):
        super().__init__("HT_TRENDLINE")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        ht_trendline = talib.HT_TRENDLINE(close)
        signal = pd.Series((close - ht_trendline) / ht_trendline * 100, index=data.index)
        return signal.fillna(0)


class MIDPRICESignal(SignalGenerator):
    """MIDPRICE 中间价信号
    
    信号值：价格偏离中间价的百分比
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"MIDPRICE{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        midprice = talib.MIDPRICE(high, low, timeperiod=self.period)
        signal = pd.Series((close - midprice) / midprice * 100, index=data.index)
        return signal.fillna(0)


class SAREXTSignal(SignalGenerator):
    """SAREXT SAR 扩展版信号
    
    信号值：价格与 SAR 的差距
    """
    
    def __init__(self, start: float = 0.0, offset_on_reverse: float = 0.0, 
                 acceleration_init_long: float = 0.02, acceleration_long: float = 0.02, 
                 acceleration_max_long: float = 0.2, acceleration_init_short: float = 0.02, 
                 acceleration_short: float = 0.02, acceleration_max_short: float = 0.2):
        super().__init__("SAREXT")
        self.start = start
        self.offset_on_reverse = offset_on_reverse
        self.acceleration_init_long = acceleration_init_long
        self.acceleration_long = acceleration_long
        self.acceleration_max_long = acceleration_max_long
        self.acceleration_init_short = acceleration_init_short
        self.acceleration_short = acceleration_short
        self.acceleration_max_short = acceleration_max_short
        self.params = {'start': start, 'offset_on_reverse': offset_on_reverse}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        sarext = talib.SAREXT(high, low, 
                              startvalue=self.start, 
                              offsetonreverse=self.offset_on_reverse,
                              accelerationinitlong=self.acceleration_init_long,
                              accelerationlong=self.acceleration_long,
                              accelerationmaxlong=self.acceleration_max_long,
                              accelerationinitshort=self.acceleration_init_short,
                              accelerationshort=self.acceleration_short,
                              accelerationmaxshort=self.acceleration_max_short)
        signal = pd.Series((close - sarext) / close * 100, index=data.index)
        return signal.fillna(0)


class ACCBANDSSignal(SignalGenerator):
    """ACCBANDS 加速度带信号
    
    信号值：价格在带中的位置（0-1 归一化）
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"ACCBANDS{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        upper, mid, lower = talib.ACCBANDS(high, low, close, timeperiod=self.period)
        signal = pd.Series((close - lower) / (upper - lower), index=data.index)
        return signal.fillna(0.5) - 0.5


# =============================================================================
# 2. Momentum Indicators (动量指标)
# =============================================================================

class MACDSignal(SignalGenerator):
    """MACD 信号
    
    信号值：
    - DIF > 0 且 DIF > DEA: 做多
    - DIF < 0 且 DIF < DEA: 做空
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(f"MACD{fast}_{slow}_{signal}")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.params = {'fast': fast, 'slow': slow, 'signal': signal}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        ema_fast = close.ewm(span=self.fast).mean()
        ema_slow = close.ewm(span=self.slow).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal_period).mean()
        return dif - dea


class RSISignal(SignalGenerator):
    """RSI 信号
    
    信号值：
    - > 70: 超买，做空信号
    - < 30: 超卖，做多信号
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"RSI{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(self.period).mean()
        avg_loss = loss.rolling(self.period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        return rsi - 50


class KDJSignal(SignalGenerator):
    """KDJ 信号
    
    信号值：
    - K > D 且 K < 30: 金叉，做多
    - K < D 且 K > 70: 死叉，做空
    """
    
    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3):
        super().__init__(f"KDJ{n}_{m1}_{m2}")
        self.n = n
        self.m1 = m1
        self.m2 = m2
        self.params = {'n': n, 'm1': m1, 'm2': m2}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        low = data['low']
        high = data['high']
        close = data['close']
        
        lowest_low = low.rolling(self.n).min()
        highest_high = high.rolling(self.n).max()
        
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        rsv = rsv.fillna(50)
        
        k = rsv.ewm(com=self.m1 - 1).mean()
        d = k.ewm(com=self.m2 - 1).mean()
        
        return k - d


class CCISignal(SignalGenerator):
    """CCI 商品通道指数信号
    
    信号值：
    - > 100: 超买，做空信号
    - < -100: 超卖，做多信号
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"CCI{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        typical_price = (high + low + close) / 3
        ma = typical_price.rolling(self.period).mean()
        mad = typical_price.rolling(self.period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (typical_price - ma) / (0.015 * mad)
        cci = cci.fillna(0)
        
        return cci


class WRSignal(SignalGenerator):
    """威廉指标信号
    
    信号值：
    - > 80: 超卖，做多信号
    - < 20: 超买，做空信号
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"WR{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        highest_high = high.rolling(self.period).max()
        lowest_low = low.rolling(self.period).min()
        
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        wr = wr.fillna(-50)
        
        return wr + 50


class ROCSignal(SignalGenerator):
    """ROC 变化率指标信号
    
    信号值：
    - > 0: 价格上涨，做多信号
    - < 0: 价格下跌，做空信号
    """
    
    def __init__(self, period: int = 12):
        super().__init__(f"ROC{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        roc = (close / close.shift(self.period) - 1) * 100
        roc = roc.fillna(0)
        return roc


class STOCHFSignal(SignalGenerator):
    """STOCHF 快速随机指标信号
    
    信号值：%K 值（归一化到 -50 ~ 50）
    - < -30: 超卖
    - > 30: 超买
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"STOCHF{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        
        fastk, fastd = talib.STOCHF(high, low, close,
                                     fastk_period=self.period,
                                     fastd_period=1,
                                     fastd_matype=0)
        
        signal = pd.Series(fastk - 50, index=data.index)
        return signal.fillna(0)


class CMOSignal(SignalGenerator):
    """CMO 钱德动量指标信号
    
    信号值：CMO（归一化到 -100 ~ 100）
    - > 50: 超买
    - < -50: 超卖
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"CMO{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        cmo = talib.CMO(close, timeperiod=self.period)
        return pd.Series(cmo, index=data.index).fillna(0)


class MFISignal(SignalGenerator):
    """MFI 资金流量指标信号
    
    信号值：MFI - 50
    - > 30: 超买
    - < -30: 超卖
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"MFI{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        volume = data['volume'].values.astype(float)
        mfi = talib.MFI(high, low, close, volume, timeperiod=self.period)
        return pd.Series(mfi - 50, index=data.index).fillna(0)


class MOMSignal(SignalGenerator):
    """MOM 动量指标信号
    
    信号值：价格变化率（百分比）
    - 正值: 上升动量
    - 负值: 下降动量
    """
    
    def __init__(self, period: int = 10):
        super().__init__(f"MOM{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        mom = talib.MOM(close, timeperiod=self.period)
        signal = pd.Series(mom / close, index=data.index) * 100
        return signal.fillna(0)


class PPOSignal(SignalGenerator):
    """PPO 比例 MACD 信号
    
    信号值：PPO 线（百分比 MACD）
    - 正值: 多头趋势
    - 负值: 空头趋势
    
    【与 TA-Lib 差异说明】
    TA-Lib PPO 函数只返回 1 个值（PPO 线），不像 MACD 返回 3 个值。
    原始调用方式：ppo = talib.PPO(close, fastperiod, slowperiod, matype)
    注意：signal 参数仅用于命名，不参与 TA-Lib 计算。
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(f"PPO{fast}_{slow}_{signal}")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.params = {'fast': fast, 'slow': slow, 'signal': signal}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        ppo = talib.PPO(close, fastperiod=self.fast, slowperiod=self.slow, matype=0)
        return pd.Series(ppo, index=data.index).fillna(0)


class STOCHRSISignal(SignalGenerator):
    """STOCHRSI 随机 RSI 信号
    
    信号值：%K 值（归一化到 -50 ~ 50）
    - < -30: 超卖
    - > 30: 超买
    """
    
    def __init__(self, period: int = 14, fastk: int = 5, fastd: int = 3):
        super().__init__(f"STOCHRSI{period}")
        self.period = period
        self.fastk = fastk
        self.fastd = fastd
        self.params = {'period': period, 'fastk': fastk, 'fastd': fastd}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        fastk, fastd = talib.STOCHRSI(close, timeperiod=self.period, 
                                       fastk_period=self.fastk, fastd_period=self.fastd)
        return pd.Series(fastk - 50, index=data.index).fillna(0)


class TRIXSignal(SignalGenerator):
    """TRIX 三重指数平滑信号
    
    信号值：TRIX 变化率（放大 1000 倍）
    - 正值: 上升动量
    - 负值: 下降动量
    """
    
    def __init__(self, period: int = 30):
        super().__init__(f"TRIX{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        trix = talib.TRIX(close, timeperiod=self.period)
        return pd.Series(trix, index=data.index).fillna(0) * 1000


class ULTOSCSignal(SignalGenerator):
    """ULTOSC 终极振荡器信号
    
    信号值：ULTOSC - 50
    - < -20: 超卖
    - > 20: 超买
    """
    
    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28):
        super().__init__(f"ULTOSC{period1}_{period2}_{period3}")
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.params = {'period1': period1, 'period2': period2, 'period3': period3}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        ultosc = talib.ULTOSC(high, low, close, 
                              timeperiod1=self.period1, 
                              timeperiod2=self.period2, 
                              timeperiod3=self.period3)
        return pd.Series(ultosc - 50, index=data.index).fillna(0)


class WILLRSignal(SignalGenerator):
    """WILLR 威廉指标信号（TA-Lib 版本）
    
    信号值：W%R 转换后（-50 ~ 50）
    - 原始 W%R 范围：-100 ~ 0
    - 转换后：W%R + 50，范围变为 -50 ~ 50
    - < -30（原始 < -80）: 超卖，做多信号
    - > 30（原始 > -20）: 超买，做空信号
    
    【与 TA-Lib 差异说明】
    TA-Lib WILLR 返回值范围：-100 ~ 0（负值）
    本类做了转换：willr + 50，使范围变为 -50 ~ 50
    原因：回测逻辑判断 signal > 0 才持仓，原始值永远 ≤ 0 导致无法交易
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"WILLR{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        willr = talib.WILLR(high, low, close, timeperiod=self.period)
        return pd.Series(willr + 50, index=data.index).fillna(0)


class ADXSignal(SignalGenerator):
    """ADX 平均趋向指数信号

    信号值：ADX偏离历史均值的百分比
    - > 0: 趋势强度高于均值，做多
    - < 0: 趋势强度低于均值，空仓
    """

    def __init__(self, period: int = 14):
        super().__init__(f"ADX{period}")
        self.period = period
        self.params = {'period': period}

    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        adx = talib.ADX(high, low, close, timeperiod=self.period)
        adx_ma = talib.SMA(adx, timeperiod=self.period)
        signal = np.where(adx_ma > 0, (adx - adx_ma) / adx_ma * 100, 0)
        return pd.Series(signal, index=data.index).fillna(0)


class ADXRSignal(SignalGenerator):
    """ADXR 平均趋向指数评估信号
    
    信号值：ADXR（0-100）
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"ADXR{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        adxr = talib.ADXR(high, low, close, timeperiod=self.period)
        return pd.Series(adxr - 50, index=data.index).fillna(0)


class AROONOSCSignal(SignalGenerator):
    """AROONOSC 阿隆振荡器信号
    
    信号值：Aroon Oscillator（-100 ~ +100）
    - 正值: 上升趋势
    - 负值: 下降趋势
    """
    
    def __init__(self, period: int = 25):
        super().__init__(f"AROONOSC{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        aroonosc = talib.AROONOSC(high, low, timeperiod=self.period)
        return pd.Series(aroonosc, index=data.index).fillna(0)


class APOSignal(SignalGenerator):
    """APO 绝对价格振荡器信号
    
    信号值：快线 - 慢线（百分比）
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, matype: int = 0):
        super().__init__(f"APO{fast}_{slow}")
        self.fast = fast
        self.slow = slow
        self.matype = matype
        self.params = {'fast': fast, 'slow': slow, 'matype': matype}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        apo = talib.APO(close, fastperiod=self.fast, slowperiod=self.slow, matype=self.matype)
        return pd.Series(apo / close * 100, index=data.index).fillna(0)


class DXSignal(SignalGenerator):
    """DX 趋向指数信号
    
    信号值：DX（0-100）
    - > 25: 强趋势
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"DX{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        dx = talib.DX(high, low, close, timeperiod=self.period)
        return pd.Series(dx - 50, index=data.index).fillna(0)


class IMISignal(SignalGenerator):
    """IMI 日内动量指数信号
    
    信号值：IMI - 50
    - > 20: 多头动量
    - < -20: 空头动量
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"IMI{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        open_p = data['open'].values.astype(float)
        close = data['close'].values.astype(float)
        imi = talib.IMI(open_p, close, timeperiod=self.period)
        return pd.Series(imi - 50, index=data.index).fillna(0)


class MACDEXTSignal(SignalGenerator):
    """MACDEXT MACD 扩展信号
    
    信号值：MACD 柱状图
    """
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, 
                 fast_matype: int = 0, slow_matype: int = 0, signal_matype: int = 0):
        super().__init__(f"MACDEXT{fast}_{slow}_{signal}")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.fast_matype = fast_matype
        self.slow_matype = slow_matype
        self.signal_matype = signal_matype
        self.params = {'fast': fast, 'slow': slow, 'signal': signal}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        macd, macdsignal, macdhist = talib.MACDEXT(
            close, fastperiod=self.fast, slowperiod=self.slow, signalperiod=self.signal_period,
            fastmatype=self.fast_matype, slowmatype=self.slow_matype, signalmatype=self.signal_matype
        )
        return pd.Series(macdhist, index=data.index).fillna(0)


class MACDFIXSignal(SignalGenerator):
    """MACDFIX MACD 固定参数信号
    
    信号值：MACD 柱状图
    """
    
    def __init__(self, period: int = 30):
        super().__init__(f"MACDFIX{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        macd, macdsignal, macdhist = talib.MACDFIX(close, timeperiod=self.period)
        return pd.Series(macdhist, index=data.index).fillna(0)


class PLUSDISignal(SignalGenerator):
    """PLUS_DI 正向趋向指标信号
    
    信号值：+DI - 50
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"PLUS_DI{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.period)
        return pd.Series(plus_di - 50, index=data.index).fillna(0)


class MINUSDISignal(SignalGenerator):
    """MINUS_DI 负向趋向指标信号
    
    信号值：-DI - 50
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"MINUS_DI{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.period)
        return pd.Series(minus_di - 50, index=data.index).fillna(0)


class PLUSDMSignal(SignalGenerator):
    """PLUS_DM 正向趋向动量信号
    
    信号值：+DM
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"PLUS_DM{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        plus_dm = talib.PLUS_DM(high, low, timeperiod=self.period)
        return pd.Series(plus_dm, index=data.index).fillna(0)


class MINUSDMSignal(SignalGenerator):
    """MINUS_DM 负向趋向动量信号
    
    信号值：-DM
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"MINUS_DM{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        minus_dm = talib.MINUS_DM(high, low, timeperiod=self.period)
        return pd.Series(minus_dm, index=data.index).fillna(0)


class ROCPSignal(SignalGenerator):
    """ROCP 价格变化率信号（百分比）
    
    信号值：(价格 - N日前价格) / N日前价格 * 100
    """
    
    def __init__(self, period: int = 10):
        super().__init__(f"ROCP{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        rocp = talib.ROCP(close, timeperiod=self.period)
        return pd.Series(rocp * 100, index=data.index).fillna(0)


class ROCRSignal(SignalGenerator):
    """ROCR 价格变化比率信号
    
    信号值：价格 / N日前价格
    """
    
    def __init__(self, period: int = 10):
        super().__init__(f"ROCR{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        rocr = talib.ROCR(close, timeperiod=self.period)
        return pd.Series((rocr - 1) * 100, index=data.index).fillna(0)


class ROCR100Signal(SignalGenerator):
    """ROCR100 价格变化比率信号（100 基准）
    
    信号值：价格 / N日前价格 * 100
    """
    
    def __init__(self, period: int = 10):
        super().__init__(f"ROCR100{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        rocr100 = talib.ROCR100(close, timeperiod=self.period)
        return pd.Series(rocr100 - 100, index=data.index).fillna(0)


class STOCHSignal(SignalGenerator):
    """STOCH 慢速随机指标信号
    
    信号值：%K - 50
    - < -30: 超卖
    - > 30: 超买
    """
    
    def __init__(self, fastk: int = 5, slowk: int = 3, slowd: int = 3, 
                 slowk_matype: int = 0, slowd_matype: int = 0):
        super().__init__(f"STOCH{fastk}_{slowk}_{slowd}")
        self.fastk = fastk
        self.slowk = slowk
        self.slowd = slowd
        self.slowk_matype = slowk_matype
        self.slowd_matype = slowd_matype
        self.params = {'fastk': fastk, 'slowk': slowk, 'slowd': slowd}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        slowk, slowd = talib.STOCH(
            high, low, close,
            fastk_period=self.fastk, slowk_period=self.slowk, slowk_matype=self.slowk_matype,
            slowd_period=self.slowd, slowd_matype=self.slowd_matype
        )
        return pd.Series(slowk - 50, index=data.index).fillna(0)


class BOPSignal(SignalGenerator):
    """BOP 均势信号
    
    信号值：均势（-1 ~ 1）
    - > 0: 多方占优
    - < 0: 空方占优
    """
    
    def __init__(self):
        super().__init__("BOP")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        open_p = data['open'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        bop = talib.BOP(open_p, high, low, close)
        return pd.Series(bop, index=data.index).fillna(0)


# =============================================================================
# 3. Volume Indicators (成交量指标)
# =============================================================================

class OBVSignal(SignalGenerator):
    """OBV 能量潮指标信号
    
    信号值：OBV 的移动平均变化率
    - 正值: 资金流入
    - 负值: 资金流出
    """
    
    def __init__(self, signal_period: int = 10):
        super().__init__(f"OBV{signal_period}")
        self.signal_period = signal_period
        self.params = {'signal_period': signal_period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        direction = np.sign(close.diff())
        obv = (volume * direction).cumsum()
        
        obv_ma = obv.rolling(self.signal_period).mean()
        signal = obv_ma.pct_change() * 100
        return signal.fillna(0)


class VOLSignal(SignalGenerator):
    """量能信号
    
    信号值：
    - 量比 > 1: 放量，趋势可能延续
    - 量比 < 1: 缩量，趋势可能反转
    """
    
    def __init__(self, period: int = 5):
        super().__init__(f"VOL{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        volume = data['volume']
        avg_volume = volume.rolling(self.period).mean()
        vol_ratio = volume / avg_volume
        return vol_ratio - 1


class ADSignal(SignalGenerator):
    """AD 蔡金线信号
    
    信号值：AD 线的变化率
    """
    
    def __init__(self):
        super().__init__("AD")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        volume = data['volume'].values.astype(float)
        ad = talib.AD(high, low, close, volume)
        return pd.Series(ad, index=data.index).fillna(0)


class ADOSCSignal(SignalGenerator):
    """ADOSC 蔡金振荡器信号
    
    信号值：AD 线的短期 EMA - 长期 EMA
    - 正值: 资金流入
    - 负值: 资金流出
    """
    
    def __init__(self, fast: int = 3, slow: int = 10):
        super().__init__(f"ADOSC{fast}_{slow}")
        self.fast = fast
        self.slow = slow
        self.params = {'fast': fast, 'slow': slow}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        volume = data['volume'].values.astype(float)
        adosc = talib.ADOSC(high, low, close, volume, 
                            fastperiod=self.fast, slowperiod=self.slow)
        signal = pd.Series(adosc / volume, index=data.index) * 100
        return signal.fillna(0)


# =============================================================================
# 4. Volatility Indicators (波动率指标)
# =============================================================================

class ATRSignal(SignalGenerator):
    """ATR 平均真实波幅信号

    信号值：ATR偏离历史均值的百分比
    - > 0: 波动率高于均值（异常扩张），做多
    - < 0: 波动率低于均值（收缩），空仓
    """

    def __init__(self, period: int = 14):
        super().__init__(f"ATR{period}")
        self.period = period
        self.params = {'period': period}

    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        atr = talib.ATR(high, low, close, timeperiod=self.period)
        atr_ma = talib.SMA(atr, timeperiod=self.period)
        signal = np.where(atr_ma > 0, (atr - atr_ma) / atr_ma * 100, 0)
        return pd.Series(signal, index=data.index).fillna(0)


class STDDEVSignal(SignalGenerator):
    """STDDEV 标准差信号
    
    信号值：当前标准差偏离历史均值的百分比
    - > 0: 波动率高于均值
    - < 0: 波动率低于均值
    """
    
    def __init__(self, period: int = 20, nbdev: float = 1.0):
        super().__init__(f"STDDEV{period}")
        self.period = period
        self.nbdev = nbdev
        self.params = {'period': period, 'nbdev': nbdev}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        stddev = talib.STDDEV(close, timeperiod=self.period, nbdev=self.nbdev)
        stddev_ma = talib.SMA(stddev, timeperiod=self.period)
        signal = np.where(stddev_ma > 0, (stddev - stddev_ma) / stddev_ma * 100, 0)
        return pd.Series(signal, index=data.index).fillna(0)


class NATRSignal(SignalGenerator):
    """NATR 归一化平均真实波幅信号

    信号值：NATR偏离历史均值的百分比
    - > 0: 波动率高于均值（异常扩张），做多
    - < 0: 波动率低于均值（收缩），空仓
    """

    def __init__(self, period: int = 14):
        super().__init__(f"NATR{period}")
        self.period = period
        self.params = {'period': period}

    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        natr = talib.NATR(high, low, close, timeperiod=self.period)
        natr_ma = talib.SMA(natr, timeperiod=self.period)
        signal = np.where(natr_ma > 0, (natr - natr_ma) / natr_ma * 100, 0)
        return pd.Series(signal, index=data.index).fillna(0)


class TRANGESignal(SignalGenerator):
    """TRANGE 真实波幅信号
    
    信号值：真实波幅（相对价格百分比）
    """
    
    def __init__(self):
        super().__init__("TRANGE")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        trange = talib.TRANGE(high, low, close)
        return pd.Series(trange / close * 100, index=data.index).fillna(0)


class AVGDEVSignal(SignalGenerator):
    """AVGDEV 平均偏差信号
    
    信号值：平均偏差（相对价格百分比）
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"AVGDEV{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        avgdev = talib.AVGDEV(close, timeperiod=self.period)
        return pd.Series(avgdev / close * 100, index=data.index).fillna(0)


class TrueVolSignal(SignalGenerator):
    """TrueVol 真实波动率信号
    
    信号值：ATR / Close × 100（百分比）
    - 高值: 高波动率
    - 低值: 低波动率
    
    计算方式：基于 ATR 计算真实波动率百分比
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"TrueVol{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        atr = talib.ATR(high, low, close, timeperiod=self.period)
        return pd.Series(atr / close * 100, index=data.index).fillna(0)


class ATRPercentSignal(SignalGenerator):
    """ATR% 平均真实波幅百分比信号
    
    信号值：ATR / Close（小数形式）
    - 高值: 波动率大
    - 低值: 波动率小
    
    与 TrueVol 的区别：输出小数而非百分比
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"ATR%{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        atr = talib.ATR(high, low, close, timeperiod=self.period)
        return pd.Series(atr / close, index=data.index).fillna(0)


class BBPercentSignal(SignalGenerator):
    """BB% 布林带位置百分比信号
    
    信号值：(Close - 下轨) / (上轨 - 下轨)
    - > 1: 价格突破上轨
    - 0.5: 价格在中轨
    - < 0: 价格突破下轨
    
    计算方式：基于布林带计算价格在通道中的相对位置
    """
    
    def __init__(self, period: int = 20, nbdev: float = 2.0):
        super().__init__(f"BB%{period}")
        self.period = period
        self.nbdev = nbdev
        self.params = {'period': period, 'nbdev': nbdev}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        upper, middle, lower = talib.BBANDS(
            close, timeperiod=self.period, 
            nbdevup=self.nbdev, nbdevdn=self.nbdev
        )
        bandwidth = upper - lower
        bbp = (close - lower) / bandwidth
        return pd.Series(bbp, index=data.index).fillna(0.5)


class KeltnerWidthSignal(SignalGenerator):
    """KeltnerWidth 肯特纳通道宽度信号
    
    信号值：(上轨 - 下轨) / 中轨 × 100
    - 高值: 波动率高（通道扩张）
    - 低值: 波动率低（通道收窄）
    
    计算方式：基于 ATR 的肯特纳通道宽度
    """
    
    def __init__(self, period: int = 20, atr_period: int = 20, multiplier: float = 2.0):
        super().__init__(f"KeltnerWidth{period}")
        self.period = period
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.params = {'period': period, 'atr_period': atr_period, 'multiplier': multiplier}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        
        mid = talib.EMA(close, timeperiod=self.period)
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
        
        upper = mid + self.multiplier * atr
        lower = mid - self.multiplier * atr
        
        keltner_width = (upper - lower) / mid * 100
        return pd.Series(keltner_width, index=data.index).fillna(0)


class DonchianWidthSignal(SignalGenerator):
    """DonchianWidth 唐奇安通道宽度信号
    
    信号值：(Highest High - Lowest Low) / Midpoint × 100
    - 高值: 波动率高（通道宽）
    - 低值: 波动率低（通道窄）
    
    计算方式：纯 pandas 计算唐奇安通道宽度
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"DonchianWidth{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        
        highest_high = high.rolling(self.period).max()
        lowest_low = low.rolling(self.period).min()
        midpoint = (highest_high + lowest_low) / 2
        
        donchian_width = (highest_high - lowest_low) / midpoint * 100
        return pd.Series(donchian_width, index=data.index).fillna(0)


class ParkinsonVolSignal(SignalGenerator):
    """ParkinsonVol 帕金森波动率信号
    
    信号值：年化帕金森波动率（百分比）
    - 高值: 高波动率
    - 低值: 低波动率
    
    计算方式：基于 High-Low 范围估计波动率
    公式：σ = sqrt(1/(4*ln(2)) * avg(ln(H/L)^2)) * sqrt(252)
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"ParkinsonVol{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        
        log_hl = np.log(high / low)
        variance = (log_hl ** 2) / (4 * np.log(2))
        
        parkinson_vol = variance.rolling(self.period).mean() ** 0.5 * np.sqrt(252) * 100
        return pd.Series(parkinson_vol, index=data.index).fillna(0)


class GarmanKlassVolSignal(SignalGenerator):
    """GarmanKlassVol 加曼-克莱斯波动率信号
    
    信号值：年化 Garman-Klass 波动率（百分比）
    - 高值: 高波动率
    - 低值: 低波动率
    
    计算方式：基于 OHLC 四价计算波动率
    公式：σ = sqrt(0.5 * avg(ln(H/L)^2) - (2*ln(2) - 1) * avg(ln(C/O)^2)) * sqrt(252)
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"GarmanKlassVol{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        open_p = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        log_hl = np.log(high / low)
        log_co = np.log(close / open_p)
        
        variance = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        
        gk_vol = variance.rolling(self.period).mean() ** 0.5 * np.sqrt(252) * 100
        return pd.Series(gk_vol, index=data.index).fillna(0)


class RVISignal(SignalGenerator):
    """RVI 相对波动率指标信号
    
    信号值：log(StdDev(上涨日) / StdDev(下跌日))
    - > 0: 上涨日波动率 > 下跌日波动率
    - < 0: 下跌日波动率 > 上涨日波动率
    
    计算方式：分别计算上涨日和下跌日的波动率，然后求比值
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"RVI{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        returns = close.pct_change()
        
        up_days = returns.where(returns > 0)
        down_days = returns.where(returns < 0)
        
        up_std = up_days.rolling(self.period).std()
        down_std = down_days.rolling(self.period).std()
        
        rvi = np.log(up_std / down_std)
        return pd.Series(rvi, index=data.index).fillna(0)


class UlcerIndexSignal(SignalGenerator):
    """UlcerIndex 溃疡指数信号
    
    信号值：溃疡指数（百分比）
    - 高值: 深度回撤，高风险
    - 低值: 浅度回撤，低风险
    
    计算方式：基于 N 日内收盘价相对于最高价的回撤计算
    公式：UI = sqrt(avg(回撤百分比^2))
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"UlcerIndex{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        
        highest_close = close.rolling(self.period).max()
        drawdown_pct = (close - highest_close) / highest_close * 100
        
        ulcer_index = (drawdown_pct ** 2).rolling(self.period).mean() ** 0.5
        return pd.Series(ulcer_index, index=data.index).fillna(0)


class YangZhangVolSignal(SignalGenerator):
    """YangZhangVol 杨张波动率信号
    
    信号值：年化杨张波动率（百分比）
    - 高值: 高波动率
    - 低值: 低波动率
    
    计算方式：最稳健的OHLC波动率估计量
    公式：σ² = σ²_overnight + k * σ²_open + (1-k) * σ²_rogers_satchell
    特点：考虑隔夜跳空、开盘价漂移和日内波动
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"YangZhangVol{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        open_p = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 隔夜收益率（收盘到开盘）
        log_oc = np.log(open_p / close.shift(1))
        sigma_overnight = log_oc.rolling(self.period).var()
        
        # 开盘价波动率
        log_o_c = np.log(open_p / close)
        sigma_open = log_o_c.rolling(self.period).var()
        
        # Rogers-Satchell 波动率
        log_ho = np.log(high / open_p)
        log_lo = np.log(low / open_p)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)
        
        rs_var = log_ho * log_hc + log_lo * log_lc
        sigma_rs = rs_var.rolling(self.period).mean()
        
        # 最优权重 k = 0.34 / (1.34 + (period + 1) / (period - 1))
        k = 0.34 / (1.34 + (self.period + 1) / (self.period - 1))
        
        # 综合波动率
        variance = sigma_overnight + k * sigma_open + (1 - k) * sigma_rs
        yz_vol = variance ** 0.5 * np.sqrt(252) * 100
        
        return pd.Series(yz_vol, index=data.index).fillna(0)


class RogersSatchellVolSignal(SignalGenerator):
    """RogersSatchellVol 罗杰斯-萨切尔波动率信号
    
    信号值：年化 Rogers-Satchell 波动率（百分比）
    - 高值: 高波动率
    - 低值: 低波动率
    
    计算方式：带漂移校正的OHLC波动率估计
    公式：σ² = avg(ln(H/O)·ln(H/C) + ln(L/O)·ln(L/C))
    特点：对趋势市场中的波动率估计更准确
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"RogersSatchellVol{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        open_p = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Rogers-Satchell 方差估计
        log_ho = np.log(high / open_p)
        log_lo = np.log(low / open_p)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)
        
        rs_variance = (log_ho * log_hc + log_lo * log_lc).rolling(self.period).mean()
        
        rs_vol = rs_variance ** 0.5 * np.sqrt(252) * 100
        return pd.Series(rs_vol, index=data.index).fillna(0)


class ChaikinVolatilitySignal(SignalGenerator):
    """ChaikinVolatility 蔡金波动率信号
    
    信号值：蔡金波动率变化率（百分比）
    - 正值: 波动率扩张
    - 负值: 波动率收缩
    
    计算方式：基于 High-Low 差值的 EMA 变化率
    公式：(EMA(HL, N) - EMA(HL, N)[N天前]) / EMA(HL, N)[N天前] * 100
    """
    
    def __init__(self, period: int = 10, roc_period: int = 10):
        super().__init__(f"ChaikinVol{period}")
        self.period = period
        self.roc_period = roc_period
        self.params = {'period': period, 'roc_period': roc_period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        
        # High-Low 差值
        hl = high - low
        
        # EMA 平滑
        hl_ema = hl.ewm(span=self.period, adjust=False).mean()
        
        # 变化率
        chaikin_vol = hl_ema.pct_change(self.roc_period) * 100
        return pd.Series(chaikin_vol, index=data.index).fillna(0)


class VolatilityRatioSignal(SignalGenerator):
    """VolatilityRatio 波动率比率信号
    
    信号值：短期波动率 / 长期波动率
    - > 1: 短期波动率高于长期（波动率扩张）
    - < 1: 短期波动率低于长期（波动率收缩）
    
    计算方式：比较不同时间窗口的波动率
    """
    
    def __init__(self, short_period: int = 10, long_period: int = 20):
        super().__init__(f"VolRatio{short_period}_{long_period}")
        self.short_period = short_period
        self.long_period = long_period
        self.params = {'short_period': short_period, 'long_period': long_period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        log_return = np.log(close / close.shift(1))
        
        short_vol = log_return.rolling(self.short_period).std() * np.sqrt(252)
        long_vol = log_return.rolling(self.long_period).std() * np.sqrt(252)
        
        vol_ratio = short_vol / long_vol
        return pd.Series(vol_ratio, index=data.index).fillna(1)


class BBBreakoutSignal(SignalGenerator):
    """BBBreakout 布林带突破信号
    
    信号值：突破强度（百分比）
    - > 0: 突破上轨
    - < 0: 突破下轨
    - 0: 在布林带内部
    
    计算方式：计算价格突破布林带的百分比距离
    """
    
    def __init__(self, period: int = 20, nbdev: float = 2.0):
        super().__init__(f"BBBreakout{period}")
        self.period = period
        self.nbdev = nbdev
        self.params = {'period': period, 'nbdev': nbdev}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        mid = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        
        upper = mid + self.nbdev * std
        lower = mid - self.nbdev * std
        
        # 突破上轨：正值，突破下轨：负值
        breakout_upper = (close - upper) / upper * 100
        breakout_lower = (close - lower) / lower * 100
        
        # 合并：只保留突破方向
        breakout = pd.Series(0.0, index=data.index)
        breakout[close > upper] = breakout_upper[close > upper]
        breakout[close < lower] = breakout_lower[close < lower]
        
        return breakout


class VaRSignal(SignalGenerator):
    """VaR 风险价值信号
    
    信号值：N% 置信水平下的风险价值（百分比）
    - 负值越大：潜在亏损越大
    - 用于衡量下行风险
    
    计算方式：基于历史模拟法计算 VaR
    """
    
    def __init__(self, period: int = 20, confidence: float = 0.05):
        super().__init__(f"VaR{period}")
        self.period = period
        self.confidence = confidence
        self.params = {'period': period, 'confidence': confidence}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        returns = close.pct_change()
        
        def calc_var(x):
            return np.percentile(x, self.confidence * 100)
        
        var = returns.rolling(self.period).apply(calc_var, raw=True) * 100
        return pd.Series(var, index=data.index).fillna(0)


class CVaRSignal(SignalGenerator):
    """CVaR 条件风险价值信号（Expected Shortfall）
    
    信号值：N% 置信水平下的条件风险价值（百分比）
    - 负值越大：尾部风险越大
    - 比 VaR 更保守，考虑极端亏损
    
    计算方式：计算低于 VaR 的平均收益率
    """
    
    def __init__(self, period: int = 20, confidence: float = 0.05):
        super().__init__(f"CVaR{period}")
        self.period = period
        self.confidence = confidence
        self.params = {'period': period, 'confidence': confidence}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        returns = close.pct_change()
        
        def calc_cvar(x):
            var = np.percentile(x, self.confidence * 100)
            return x[x <= var].mean() if len(x[x <= var]) > 0 else var
        
        cvar = returns.rolling(self.period).apply(calc_cvar, raw=True) * 100
        return pd.Series(cvar, index=data.index).fillna(0)


# =============================================================================
# 4. Volatility Indicators (波动率指标) - 补充
# =============================================================================

class BBWidthSignal(SignalGenerator):
    """BBWidth 布林带宽信号

    信号值：布林带宽偏离历史均值的百分比
    - > 0: 带宽高于均值（波动率扩张），做多
    - < 0: 带宽低于均值（波动率收缩），空仓
    """

    def __init__(self, period: int = 20, nbdev: float = 2.0):
        super().__init__(f"BBWidth{period}")
        self.period = period
        self.nbdev = nbdev
        self.params = {'period': period, 'nbdev': nbdev}

    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        mid = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()

        upper = mid + self.nbdev * std
        lower = mid - self.nbdev * std
        bandwidth = (upper - lower) / mid * 100

        bw_ma = bandwidth.rolling(self.period).mean()
        signal = np.where(bw_ma > 0, (bandwidth - bw_ma) / bw_ma * 100, 0)

        return pd.Series(signal, index=data.index).fillna(0)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """从配置字典创建 BBWidthSignal 实例"""
        period = config.get('period', 20)
        nbdev = config.get('std_dev', config.get('nbdev', 2.0))
        return cls(period=period, nbdev=nbdev)


class RealizedVolatilitySignal(SignalGenerator):
    """RealizedVolatility 真实波动率信号

    信号值：HV偏离历史均值的百分比
    - > 0: 波动率高于均值（异常扩张），做多
    - < 0: 波动率低于均值（收缩），空仓
    """

    def __init__(self, period: int = 20):
        super().__init__(f"RealizedVol{period}")
        self.period = period
        self.params = {'period': period}

    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']

        log_return = np.log(close / close.shift(1))

        hv = log_return.rolling(self.period).std() * np.sqrt(252) * 100
        hv_ma = hv.rolling(self.period).mean()
        signal = np.where(hv_ma > 0, (hv - hv_ma) / hv_ma * 100, 0)

        return pd.Series(signal, index=data.index).fillna(0)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """从配置字典创建 RealizedVolatilitySignal 实例"""
        period = config.get('period', 20)
        return cls(period=period)


# =============================================================================
# 5. Price Transform (价格转换)
# =============================================================================

# 价格转换指标通常不生成交易信号，而是作为其他指标的输入
# 已在 Overlap Studies 中添加了 AVGPRICE、MEDPRICE、TYPPRICE、WCLPRICE


# =============================================================================
# 6. Cycle Indicators (周期指标)
# =============================================================================

class HTDCCPERIODSignal(SignalGenerator):
    """HT_DCPERIOD 希尔伯特变换主导周期信号
    
    信号值：当前市场周期长度（天数）
    - 高值: 长周期
    - 低值: 短周期
    """
    
    def __init__(self):
        super().__init__("HT_DCPERIOD")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        ht_dcperiod = talib.HT_DCPERIOD(close)
        return pd.Series(ht_dcperiod, index=data.index).fillna(0)


class HTDCPHASESignal(SignalGenerator):
    """HT_DCPHASE 希尔伯特变换主导相位信号
    
    信号值：主导相位角度
    """
    
    def __init__(self):
        super().__init__("HT_DCPHASE")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        ht_dcphase = talib.HT_DCPHASE(close)
        return pd.Series(ht_dcphase, index=data.index).fillna(0)


class HTPHASORSignal(SignalGenerator):
    """HT_PHASOR 希尔伯特变换相量信号
    
    信号值：同相分量 - 正交分量
    """
    
    def __init__(self):
        super().__init__("HT_PHASOR")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        inphase, quadrature = talib.HT_PHASOR(close)
        return pd.Series(inphase - quadrature, index=data.index).fillna(0)


class HTSINESignal(SignalGenerator):
    """HT_SINE 希尔伯特变换正弦信号
    
    信号值：正弦线 - 导引线
    """
    
    def __init__(self):
        super().__init__("HT_SINE")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        sine, leadsine = talib.HT_SINE(close)
        return pd.Series(sine - leadsine, index=data.index).fillna(0)


class HTTRENDMODESignal(SignalGenerator):
    """HT_TRENDMODE 希尔伯特变换趋势/周期模式信号
    
    信号值：模式标记（1=趋势，0=周期）- 0.5
    """
    
    def __init__(self):
        super().__init__("HT_TRENDMODE")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        ht_trendmode = talib.HT_TRENDMODE(close)
        return pd.Series(ht_trendmode - 0.5, index=data.index).fillna(0)


# =============================================================================
# 7. Pattern Recognition (形态识别)
# =============================================================================

class PatternSignal(SignalGenerator):
    """形态识别信号基类
    
    形态识别指标通常生成离散信号：
    - 100: 看涨形态
    - -100: 看跌形态
    - 0: 无形态
    
    信号值归一化为：
    - 1.0: 看涨信号
    - -1.0: 看跌信号
    - 0.0: 无信号
    """
    
    # 子类需要定义的类属性
    talib_func = None  # TA-Lib 形态识别函数
    
    def __init__(self):
        super().__init__(self.__class__.__name__.replace('Signal', ''))
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        """生成形态识别信号
        
        Args:
            data: K线数据，必须包含 open/high/low/close
            
        Returns:
            pd.Series: 信号值序列，值为 -1.0, 0.0, 1.0
        """
        open_price = data['open'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        
        # 调用 TA-Lib 形态识别函数
        pattern = self.talib_func(open_price, high, low, close)
        
        # 归一化为 -1.0, 0.0, 1.0
        signal = pattern.astype(float) / 100.0
        
        return pd.Series(signal, index=data.index)


class CDLDOJISignal(PatternSignal):
    """十字星形态信号
    
    特征：开盘价≈收盘价，上下影线较长
    含义：市场犹豫不决，可能反转
    """
    talib_func = talib.CDLDOJI


class CDLHAMMERSignal(PatternSignal):
    """锤子线形态信号
    
    特征：实体小，下影线长（至少是实体2倍），上影线短或无
    含义：下跌后的反转信号（看涨）
    """
    talib_func = talib.CDLHAMMER


class CDLENGULFINGSignal(PatternSignal):
    """吞没形态信号
    
    特征：一根K线完全吞没前一根K线
    含义：看涨吞没（阳包阴）或看跌吞没（阴包阳），强反转信号
    """
    talib_func = talib.CDLENGULFING


class CDLMORNINGSTARSignal(PatternSignal):
    """启明星形态信号
    
    特征：三根K线组成，下跌趋势中：大阴线→小实体→大阳线
    含义：底部反转信号（看涨）
    """
    talib_func = talib.CDLMORNINGSTAR


class CDLEVENINGSTARSignal(PatternSignal):
    """黄昏星形态信号
    
    特征：三根K线组成，上涨趋势中：大阳线→小实体→大阴线
    含义：顶部反转信号（看跌）
    """
    talib_func = talib.CDLEVENINGSTAR


class CDL3BLACKCROWSSignal(PatternSignal):
    """三只乌鸦形态信号
    
    特征：连续三根下跌阴线，每根收盘价低于前一根
    含义：顶部反转信号（看跌）
    """
    talib_func = talib.CDL3BLACKCROWS


class CDL3WHITESOLDIERSSignal(PatternSignal):
    """三白兵形态信号
    
    特征：连续三根上涨阳线，每根收盘价高于前一根
    含义：底部反转信号（看涨）
    """
    talib_func = talib.CDL3WHITESOLDIERS


class CDLSHOOTINGSTARSignal(PatternSignal):
    """流星线形态信号
    
    特征：实体小，上影线长（至少是实体2倍），下影线短或无
    含义：上涨后的反转信号（看跌）
    """
    talib_func = talib.CDLSHOOTINGSTAR


class CDLHARAMISignal(PatternSignal):
    """孕线形态信号
    
    特征：第二根K线实体完全在第一根实体范围内
    含义：趋势减弱，可能反转
    """
    talib_func = talib.CDLHARAMI


# =============================================================================
# 8. Statistic Functions (统计函数)
# =============================================================================

class BETASignal(SignalGenerator):
    """BETA 贝塔系数信号
    
    信号值：收益率的自回归系数（动量beta）
    - > 0: 动量效应（收益延续），做多
    - < 0: 均值回归（收益反转），空仓
    """
    
    def __init__(self, period: int = 5):
        super().__init__(f"BETA{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        returns = np.concatenate([[0], np.diff(close) / close[:-1]])
        lagged_returns = np.zeros_like(returns)
        lagged_returns[1:] = returns[:-1]
        beta = talib.BETA(returns, lagged_returns, timeperiod=self.period)
        return pd.Series(beta, index=data.index).fillna(0)


class RSRSMSignal(SignalGenerator):
    """RSRS 信号（阻力支撑相对强度 - 上海证券版本）
    
    计算方法：
    1. 取前 N 日最高价、最低价
    2. 最高价对最低价做线性回归，得到斜率 β
    3. 计算前 M 日斜率的标准化分 z
    4. RSRS = z × R²
    
    信号值：
    - > 0: 支撑 > 阻力，多头
    - < 0: 支撑 < 阻力，空头
    """
    
    def __init__(self, n: int = 18, m: int = 600):
        super().__init__(f"RSRS{n}_{m}")
        self.n = n
        self.m = m
        self.params = {'n': n, 'm': m}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        
        slopes = []
        for i in range(len(high)):
            if i < self.n - 1:
                slopes.append(np.nan)
                continue
            
            window_high = high.iloc[i - self.n + 1:i + 1]
            window_low = low.iloc[i - self.n + 1:i + 1]
            
            if window_low.std() == 0:
                slopes.append(np.nan)
            else:
                low_values = window_low.values
                high_values = window_high.values
                cov = np.cov(low_values, high_values)[0, 1]
                var = np.var(low_values)
                beta = cov / var
                slopes.append(beta)
        
        slope_series = pd.Series(slopes, index=data.index)
        
        if len(slope_series.dropna()) < self.m:
            return slope_series.fillna(0)
        
        rolling_mean = slope_series.rolling(self.m).mean()
        rolling_std = slope_series.rolling(self.m).std()
        
        z_score = (slope_series - rolling_mean) / rolling_std
        return z_score.fillna(0)


class CORRELSignal(SignalGenerator):
    """CORREL 相关系数信号
    
    信号值：收盘价与时间序列的相关系数（趋势方向）
    - > 0: 正相关，上升趋势
    - < 0: 负相关，下降趋势
    """
    
    def __init__(self, period: int = 30):
        super().__init__(f"CORREL{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        time_seq = np.arange(len(close), dtype=float)
        correl = talib.CORREL(close, time_seq, timeperiod=self.period)
        return pd.Series(correl, index=data.index).fillna(0)


class LINEARREGSignal(SignalGenerator):
    """LINEARREG 线性回归信号
    
    信号值：价格偏离线性回归线的百分比
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"LINEARREG{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        linearreg = talib.LINEARREG(close, timeperiod=self.period)
        return pd.Series((close - linearreg) / linearreg * 100, index=data.index).fillna(0)


class LINEARREGANGLESignal(SignalGenerator):
    """LINEARREG_ANGLE 线性回归角度信号
    
    信号值：线性回归角度
    - 正值: 上升角度
    - 负值: 下降角度
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"LINEARREG_ANGLE{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        angle = talib.LINEARREG_ANGLE(close, timeperiod=self.period)
        return pd.Series(angle, index=data.index).fillna(0)


class LINEARREGINTERCEPTSignal(SignalGenerator):
    """LINEARREG_INTERCEPT 线性回归截距信号
    
    信号值：截距的变化率
    - > 0: 截距上升，趋势增强
    - < 0: 截距下降，趋势减弱
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"LINEARREG_INTERCEPT{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        intercept = talib.LINEARREG_INTERCEPT(close, timeperiod=self.period)
        intercept_prev = np.roll(intercept, 1)
        intercept_prev[0] = intercept[0]
        signal = np.where(np.abs(intercept_prev) > 1e-10,
                          (intercept - intercept_prev) / np.abs(intercept_prev) * 100,
                          0)
        return pd.Series(signal, index=data.index).fillna(0)


class LINEARREGSLOPESignal(SignalGenerator):
    """LINEARREG_SLOPE 线性回归斜率信号
    
    信号值：线性回归斜率
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"LINEARREG_SLOPE{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        slope = talib.LINEARREG_SLOPE(close, timeperiod=self.period)
        return pd.Series(slope, index=data.index).fillna(0)


class TSFSignal(SignalGenerator):
    """TSF 时间序列预测信号
    
    信号值：价格偏离 TSF 的百分比
    """
    
    def __init__(self, period: int = 14):
        super().__init__(f"TSF{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        tsf = talib.TSF(close, timeperiod=self.period)
        return pd.Series((close - tsf) / tsf * 100, index=data.index).fillna(0)


class VARSignal(SignalGenerator):
    """VAR 方差信号
    
    信号值：当前方差偏离历史均值的百分比
    - > 0: 方差高于均值
    - < 0: 方差低于均值
    """
    
    def __init__(self, period: int = 5, nbdev: float = 1.0):
        super().__init__(f"VAR{period}")
        self.period = period
        self.nbdev = nbdev
        self.params = {'period': period, 'nbdev': nbdev}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        var = talib.VAR(close, timeperiod=self.period, nbdev=self.nbdev)
        var_ma = talib.SMA(var, timeperiod=self.period)
        signal = np.where(var_ma > 0, (var - var_ma) / var_ma * 100, 0)
        return pd.Series(signal, index=data.index).fillna(0)


# =============================================================================
# 9. Math Operators (数学运算函数)
# =============================================================================

class ADDSignal(SignalGenerator):
    """ADD 加法信号
    
    信号值：最高价 + 最低价
    """
    
    def __init__(self):
        super().__init__("ADD")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        result = talib.ADD(high, low)
        return pd.Series(result, index=data.index).fillna(0)


class SUBSignal(SignalGenerator):
    """SUB 减法信号
    
    信号值：最高价 - 最低价（价格区间）
    """
    
    def __init__(self):
        super().__init__("SUB")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        result = talib.SUB(high, low)
        return pd.Series(result, index=data.index).fillna(0)


class MULTSignal(SignalGenerator):
    """MULT 乘法信号
    
    信号值：收盘价 * 成交量（标准化）
    """
    
    def __init__(self):
        super().__init__("MULT")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        volume = data['volume'].values.astype(float)
        result = talib.MULT(close, volume)
        return pd.Series(result / result.mean(), index=data.index).fillna(0) if result.mean() != 0 else pd.Series(result, index=data.index).fillna(0)


class DIVSignal(SignalGenerator):
    """DIV 除法信号
    
    信号值：最高价 / 最低价
    """
    
    def __init__(self):
        super().__init__("DIV")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        result = talib.DIV(high, low)
        return pd.Series(result - 1, index=data.index).fillna(0)


class SUMSignal(SignalGenerator):
    """SUM 求和信号
    
    信号值：收盘价的 N 日累计
    """
    
    def __init__(self, period: int = 10):
        super().__init__(f"SUM{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.SUM(close, timeperiod=self.period)
        return pd.Series(result / self.period, index=data.index).fillna(0)


class MAXSignal(SignalGenerator):
    """MAX 最大值信号
    
    信号值：价格偏离 N 日最高价的百分比
    """
    
    def __init__(self, period: int = 10):
        super().__init__(f"MAX{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.MAX(close, timeperiod=self.period)
        return pd.Series((close - result) / result * 100, index=data.index).fillna(0)


class MINSignal(SignalGenerator):
    """MIN 最小值信号
    
    信号值：价格偏离 N 日最低价的百分比
    """
    
    def __init__(self, period: int = 10):
        super().__init__(f"MIN{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.MIN(close, timeperiod=self.period)
        return pd.Series((close - result) / result * 100, index=data.index).fillna(0)


class MAXINDEXSignal(SignalGenerator):
    """MAXINDEX 最大值索引信号
    
    信号值：N 日内最高价的索引位置（天数）
    """
    
    def __init__(self, period: int = 10):
        super().__init__(f"MAXINDEX{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.MAXINDEX(close, timeperiod=self.period)
        return pd.Series(result / self.period - 0.5, index=data.index).fillna(0)


class MININDEXSignal(SignalGenerator):
    """MININDEX 最小值索引信号
    
    信号值：N 日内最低价的索引位置（天数）
    """
    
    def __init__(self, period: int = 10):
        super().__init__(f"MININDEX{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.MININDEX(close, timeperiod=self.period)
        return pd.Series(result / self.period - 0.5, index=data.index).fillna(0)


class MINMAXSignal(SignalGenerator):
    """MINMAX 最大值最小值信号
    
    信号值：(价格 - 最小值) / (最大值 - 最小值) - 0.5
    """
    
    def __init__(self, period: int = 10):
        super().__init__(f"MINMAX{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        minval, maxval = talib.MINMAX(close, timeperiod=self.period)
        result = (close - minval) / (maxval - minval)
        return pd.Series(result - 0.5, index=data.index).fillna(0)


class MINMAXINDEXSignal(SignalGenerator):
    """MINMAXINDEX 最大值最小值索引信号
    
    信号值：最高索引 - 最低索引
    """
    
    def __init__(self, period: int = 10):
        super().__init__(f"MINMAXINDEX{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        minidx, maxidx = talib.MINMAXINDEX(close, timeperiod=self.period)
        return pd.Series((maxidx - minidx) / self.period, index=data.index).fillna(0)


# =============================================================================
# 10. Math Transform (数学变换函数)
# =============================================================================

class ACOSSignal(SignalGenerator):
    """ACOS 反余弦变换信号"""
    
    def __init__(self):
        super().__init__("ACOS")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.ACOS(close / close.max())
        return pd.Series(result, index=data.index).fillna(0)


class ASINSignal(SignalGenerator):
    """ASIN 反正弦变换信号"""
    
    def __init__(self):
        super().__init__("ASIN")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.ASIN(close / close.max())
        return pd.Series(result, index=data.index).fillna(0)


class ATANSignal(SignalGenerator):
    """ATAN 反正切变换信号"""
    
    def __init__(self):
        super().__init__("ATAN")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.ATAN(close.pct_change().fillna(0).values.astype(float))
        return pd.Series(result, index=data.index).fillna(0)


class CEILSignal(SignalGenerator):
    """CEIL 向上取整信号"""
    
    def __init__(self):
        super().__init__("CEIL")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.CEIL(close)
        return pd.Series(result, index=data.index).fillna(0)


class COSSignal(SignalGenerator):
    """COS 余弦变换信号"""
    
    def __init__(self):
        super().__init__("COS")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.COS(close.pct_change().fillna(0).values.astype(float))
        return pd.Series(result, index=data.index).fillna(0)


class COSHSignal(SignalGenerator):
    """COSH 双曲余弦变换信号"""
    
    def __init__(self):
        super().__init__("COSH")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.COSH(close.pct_change().fillna(0).values.astype(float))
        return pd.Series(result, index=data.index).fillna(0)


class EXPSignal(SignalGenerator):
    """EXP 指数变换信号"""
    
    def __init__(self):
        super().__init__("EXP")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.EXP(close.pct_change().fillna(0).values.astype(float))
        return pd.Series(result, index=data.index).fillna(0)


class FLOORSignal(SignalGenerator):
    """FLOOR 向下取整信号"""
    
    def __init__(self):
        super().__init__("FLOOR")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.FLOOR(close)
        return pd.Series(result, index=data.index).fillna(0)


class LNSignal(SignalGenerator):
    """LN 自然对数变换信号"""
    
    def __init__(self):
        super().__init__("LN")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.LN(close)
        return pd.Series(result, index=data.index).fillna(0)


class LOG10Signal(SignalGenerator):
    """LOG10 以 10 为底对数变换信号"""
    
    def __init__(self):
        super().__init__("LOG10")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.LOG10(close)
        return pd.Series(result, index=data.index).fillna(0)


class SINSignal(SignalGenerator):
    """SIN 正弦变换信号"""
    
    def __init__(self):
        super().__init__("SIN")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.SIN(close.pct_change().fillna(0).values.astype(float))
        return pd.Series(result, index=data.index).fillna(0)


class SINHSignal(SignalGenerator):
    """SINH 双曲正弦变换信号"""
    
    def __init__(self):
        super().__init__("SINH")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.SINH(close.pct_change().fillna(0).values.astype(float))
        return pd.Series(result, index=data.index).fillna(0)


class SQRTSignal(SignalGenerator):
    """SQRT 平方根变换信号"""
    
    def __init__(self):
        super().__init__("SQRT")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.SQRT(close)
        return pd.Series(result, index=data.index).fillna(0)


class TANSignal(SignalGenerator):
    """TAN 正切变换信号"""
    
    def __init__(self):
        super().__init__("TAN")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.TAN(close.pct_change().fillna(0).values.astype(float))
        return pd.Series(result, index=data.index).fillna(0)


class TANHSignal(SignalGenerator):
    """TANH 双曲正切变换信号"""
    
    def __init__(self):
        super().__init__("TANH")
        self.params = {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close'].values.astype(float)
        result = talib.TANH(close.pct_change().fillna(0).values.astype(float))
        return pd.Series(result, index=data.index).fillna(0)


# =============================================================================
# 11. 组合信号与函数式信号
# =============================================================================
class CompositeSignal(SignalGenerator):
    """组合信号生成器"""
    
    def __init__(self, generators: List[SignalGenerator], name: str = None):
        self.generators = generators
        self._name = name or '_'.join([g.name for g in generators])
        super().__init__(self._name)
        self.params = {'generators': [g.name for g in generators]}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        all_signals = []
        for gen in self.generators:
            signal = gen.generate(data)
            all_signals.append(signal)
        combined = pd.concat(all_signals, axis=1).mean(axis=1)
        return combined
    
    def generate_latest(self, data: pd.DataFrame) -> Signal:
        series = self.generate(data)
        latest_value = series.iloc[-1] if not series.empty else 0
        signals = [gen.generate_latest(data) for gen in self.generators]
        
        if latest_value > 0:
            direction = SignalDirection.LONG
        elif latest_value < 0:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL
        
        return Signal(
            name=self.name,
            value=latest_value,
            direction=direction,
            timestamp=datetime.now(),
            metadata={'params': self.params, 'components': [s.to_dict() for s in signals]}
        )


class FunctionSignal(SignalGenerator):
    """函数式信号生成器"""
    
    def __init__(
        self,
        name: str,
        func: Callable[[pd.DataFrame], pd.Series],
        params: Dict[str, Any] = None
    ):
        super().__init__(name)
        self.func = func
        self.params = params or {}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        return self.func(data)


class IndicatorSignal(SignalGenerator):
    """
    通用指标信号生成器
    
    通过注入计算函数，支持任意指标。
    无需为每个指标编写完整类。
    """
    
    def __init__(
        self,
        name: str,
        calc_func: Callable[[pd.DataFrame, Dict[str, Any]], pd.Series],
        params: Dict[str, Any] = None,
        normalize: bool = True,
        threshold: float = 0.0
    ):
        super().__init__(name)
        self.calc_func = calc_func
        self.params = params or {}
        self.normalize = normalize
        self.threshold = threshold
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        signal = self.calc_func(data, self.params)
        if self.normalize:
            signal = self._normalize(signal)
        return signal
    
    def _normalize(self, signal: pd.Series) -> pd.Series:
        median = signal.median()
        return signal - median


def ma_cross_signal(data: pd.DataFrame, short: int = 5, long: int = 20) -> pd.Series:
    """均线交叉信号（函数形式）"""
    close = data['close']
    short_ma = close.rolling(short).mean()
    long_ma = close.rolling(long).mean()
    return (short_ma - long_ma) / long_ma


def momentum_signal(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """动量信号"""
    close = data['close']
    return close / close.shift(period) - 1


def volatility_signal(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """波动率信号（布林带宽度）"""
    close = data['close']
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    bandwidth = (2 * std) / mid
    return bandwidth


# =============================================================================
# 工厂函数
# =============================================================================

def create_signal(
    indicator: str,
    short: int = None,
    long: int = None,
    period: int = None,
    **kwargs
) -> SignalGenerator:
    """
    快速创建信号生成器
    
    Args:
        indicator: 指标名称（'ma', 'macd', 'rsi', 等）
        short: 短期参数
        long: 长期参数
        period: 单参数指标（如 RSI 的 period）
        **kwargs: 其他参数
    
    Returns:
        SignalGenerator: 信号生成器实例
    """
    from .indicators import (
        calc_ma_cross, calc_ema_cross, calc_wma_cross,
        calc_dema_cross, calc_tema_cross, calc_kama_cross,
        calc_t3_cross, calc_trima_cross, calc_accbands
    )
    
    factory = {
        # 1. Overlap Studies (重叠研究/趋势指标)
        'ma': lambda: MASignal(short_period=short, long_period=long),
        'boll': lambda: BOLLSignal(period=period or 20),
        'sar': lambda: SARSignal(),
        'midpoint': lambda: MIDPOINTSignal(period=period or 14),
        'dmi': lambda: DMISignal(period=period or 14),
        'aroon': lambda: AROONSignal(period=period or 25),
        'sma': lambda: SMASignal(period=period or 20),
        'ema': lambda: EMASignal(period=period or 20),
        'wma': lambda: WMASignal(period=period or 20),
        'dema': lambda: DEMASignal(period=period or 20),
        'tema': lambda: TEMASignal(period=period or 20),
        'kama': lambda: KAMASignal(period=period or 30),
        't3': lambda: T3Signal(period=period or 5),
        'trima': lambda: TRIMASignal(period=period or 20),
        'mama': lambda: MAMASignal(),
        'mavp': lambda: MAVPSignal(),
        'avgprice': lambda: AVGPRICESignal(),
        'medprice': lambda: MEDPRICESignal(),
        'typprice': lambda: TYPPRICESignal(),
        'wclprice': lambda: WCLPRICESignal(),
        'ht_trendline': lambda: HTTRENDLINESignal(),
        'midprice': lambda: MIDPRICESignal(period=period or 14),
        'sarext': lambda: SAREXTSignal(),
        'accbands': lambda: ACCBANDSSignal(period=period or 20),
        # 2. Momentum Indicators (动量指标)
        'macd': lambda: MACDSignal(fast=short or 12, slow=long or 26, signal=kwargs.get('signal', 9)),
        'rsi': lambda: RSISignal(period=period or 14),
        'kdj': lambda: KDJSignal(n=period or 9),
        'cci': lambda: CCISignal(period=period or 20),
        'wr': lambda: WRSignal(period=period or 14),
        'roc': lambda: ROCSignal(period=period or 12),
        'stochf': lambda: STOCHFSignal(period=period or 14),
        'cmo': lambda: CMOSignal(period=period or 14),
        'mfi': lambda: MFISignal(period=period or 14),
        'mom': lambda: MOMSignal(period=period or 10),
        'ppo': lambda: PPOSignal(fast=short or 12, slow=long or 26, signal=kwargs.get('signal', 9)),
        'stochrsi': lambda: STOCHRSISignal(period=period or 14),
        'trix': lambda: TRIXSignal(period=period or 30),
        'ultosc': lambda: ULTOSCSignal(),
        'willr': lambda: WILLRSignal(period=period or 14),
        'adx': lambda: ADXSignal(period=period or 14),
        'adxr': lambda: ADXRSignal(period=period or 14),
        'aroonosc': lambda: AROONOSCSignal(period=period or 25),
        'apo': lambda: APOSignal(fast=short or 12, slow=long or 26),
        'dx': lambda: DXSignal(period=period or 14),
        'imi': lambda: IMISignal(period=period or 14),
        'macdext': lambda: MACDEXTSignal(fast=short or 12, slow=long or 26, signal=kwargs.get('signal', 9)),
        'macdfix': lambda: MACDFIXSignal(period=period or 30),
        'plus_di': lambda: PLUSDISignal(period=period or 14),
        'minus_di': lambda: MINUSDISignal(period=period or 14),
        'plus_dm': lambda: PLUSDMSignal(period=period or 14),
        'minus_dm': lambda: MINUSDMSignal(period=period or 14),
        'rocp': lambda: ROCPSignal(period=period or 10),
        'rocr': lambda: ROCRSignal(period=period or 10),
        'rocr100': lambda: ROCR100Signal(period=period or 10),
        'stoch': lambda: STOCHSignal(),
        'bop': lambda: BOPSignal(),
        # 3. Volume Indicators (成交量指标)
        'obv': lambda: OBVSignal(signal_period=period or 10),
        'vol': lambda: VOLSignal(period=period or 5),
        'adosc': lambda: ADOSCSignal(fast=short or 3, slow=long or 10),
        'ad': lambda: ADSignal(),
        # 4. Volatility Indicators (波动率指标)
        'atr': lambda: ATRSignal(period=period or 14),
        'stddev': lambda: STDDEVSignal(period=period or 20),
        'natr': lambda: NATRSignal(period=period or 14),
        'trange': lambda: TRANGESignal(),
        'avgdev': lambda: AVGDEVSignal(period=period or 14),
        'bbwidth': lambda: BBWidthSignal(period=period or 20),
        'bbpct': lambda: BBPercentSignal(period=period or 20),
        'truevol': lambda: TrueVolSignal(period=period or 14),
        'atrpct': lambda: ATRPercentSignal(period=period or 14),
        'keltnerwidth': lambda: KeltnerWidthSignal(period=period or 20),
        'donchianwidth': lambda: DonchianWidthSignal(period=period or 20),
        'parkinsonvol': lambda: ParkinsonVolSignal(period=period or 20),
        'garklassvol': lambda: GarmanKlassVolSignal(period=period or 20),
        'rvi': lambda: RVISignal(period=period or 20),
        'ulcerindex': lambda: UlcerIndexSignal(period=period or 14),
        'realizedvol': lambda: RealizedVolatilitySignal(period=period or 20),
        'yangzhangvol': lambda: YangZhangVolSignal(period=period or 20),
        'rogerssatchellvol': lambda: RogersSatchellVolSignal(period=period or 20),
        'chaikinvol': lambda: ChaikinVolatilitySignal(period=period or 10),
        'volratio': lambda: VolatilityRatioSignal(short_period=short or 10, long_period=long or 20),
        'bbbreakout': lambda: BBBreakoutSignal(period=period or 20),
        'histvar': lambda: VaRSignal(period=period or 20),
        'histcvar': lambda: CVaRSignal(period=period or 20),
        # 6. Cycle Indicators (周期指标)
        'ht_dcperiod': lambda: HTDCCPERIODSignal(),
        'ht_dcphase': lambda: HTDCPHASESignal(),
        'ht_phasor': lambda: HTPHASORSignal(),
        'ht_sine': lambda: HTSINESignal(),
        'ht_trendmode': lambda: HTTRENDMODESignal(),
        # 8. Statistic Functions (统计函数)
        'beta': lambda: BETASignal(period=period or 5),
        'rsrs': lambda: RSRSMSignal(n=period or 18, m=kwargs.get('m', 600)),
        'correl': lambda: CORRELSignal(period=period or 30),
        'linearreg': lambda: LINEARREGSignal(period=period or 14),
        'linearreg_angle': lambda: LINEARREGANGLESignal(period=period or 14),
        'linearreg_intercept': lambda: LINEARREGINTERCEPTSignal(period=period or 14),
        'linearreg_slope': lambda: LINEARREGSLOPESignal(period=period or 14),
        'tsf': lambda: TSFSignal(period=period or 14),
        'var': lambda: VARSignal(period=period or 5),
        # 9. Math Operators (数学运算)
        'add': lambda: ADDSignal(),
        'sub': lambda: SUBSignal(),
        'mult': lambda: MULTSignal(),
        'div': lambda: DIVSignal(),
        'sum': lambda: SUMSignal(period=period or 10),
        'max': lambda: MAXSignal(period=period or 10),
        'min': lambda: MINSignal(period=period or 10),
        'maxindex': lambda: MAXINDEXSignal(period=period or 10),
        'minindex': lambda: MININDEXSignal(period=period or 10),
        'minmax': lambda: MINMAXSignal(period=period or 10),
        'minmaxindex': lambda: MINMAXINDEXSignal(period=period or 10),
        # 10. Math Transform (数学变换)
        'acos': lambda: ACOSSignal(),
        'asin': lambda: ASINSignal(),
        'atan': lambda: ATANSignal(),
        'ceil': lambda: CEILSignal(),
        'cos': lambda: COSSignal(),
        'cosh': lambda: COSHSignal(),
        'exp': lambda: EXPSignal(),
        'floor': lambda: FLOORSignal(),
        'ln': lambda: LNSignal(),
        'log10': lambda: LOG10Signal(),
        'sin': lambda: SINSignal(),
        'sinh': lambda: SINHSignal(),
        'sqrt': lambda: SQRTSignal(),
        'tan': lambda: TANSignal(),
        'tanh': lambda: TANHSignal(),
    }
    
    if indicator not in factory:
        raise ValueError(f"Unsupported indicator: {indicator}. Available: {list(factory.keys())}")
    
    return factory[indicator]()
