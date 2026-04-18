# signal/generator.py - 信号生成器
"""
信号生成器：指标计算 → 信号

支持：
1. 内置信号生成器（MA、MACD、RSI、KDJ、BOLL、VOL、RSRS）
2. 自定义信号生成器（函数式）
3. 可组合信号（多个生成器组合）
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import pandas as pd
import numpy as np

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
        # 默认实现：什么也不做，保持 is_fitted=True
        # 子类可以根据需要重写
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
        # 检查拟合状态
        if not self.is_fitted:
            raise ValueError(
                f"Signal generator '{self.name}' is not fitted. "
                f"Call fit() method before generating signals."
            )
        
        series = self.generate(data)
        latest_value = series.iloc[-1] if not series.empty else 0
        
        # 判断方向
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
        """
        保存模型到文件
        
        Args:
            path: 文件路径
        
        Usage:
            signal.save('my_model.pkl')
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'SignalGenerator':
        """
        从文件加载模型
        
        Args:
            path: 文件路径
        
        Returns:
            SignalGenerator: 加载的模型实例
        
        Usage:
            model = MASignal.load('my_model.pkl')
        """
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def __str__(self):
        params_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"


# =============================================================================
# 内置信号生成器
# =============================================================================

class MASignal(SignalGenerator):
    """均线交叉信号
    
    信号值：
    - > 0: 短期均线 > 长期均线（金叉）
    - < 0: 短期均线 < 长期均线（死叉）
    - 绝对值越大，趋势越强
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
        
        # 信号 = (短期 - 长期) / 长期（标准化）
        signal = (short_ma - long_ma) / long_ma
        return signal


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
        macd = (dif - dea) * 2
        
        # 信号 = DIF - DEA（macd 柱）
        # 也可以用标准化 DIF
        return dif - dea


class RSISignal(SignalGenerator):
    """RSI 信号
    
    信号值：
    - > 70: 超买，做空信号
    - < 30: 超卖，做多信号
    - 50: 多空分界线
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
        
        # 信号 = RSI - 50（中心化）
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
        j = 3 * k - 2 * d
        
        # 信号 = K - D
        return k - d


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
        
        # 信号 = (close - lower) / (upper - lower)
        # 接近 0: 接近下轨；接近 1: 接近上轨
        signal = (close - lower) / (upper - lower)
        signal = signal.fillna(0.5)
        
        # 中心化：-0.5，这样 < -0.3 做多，> 0.3 做空
        return signal - 0.5


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
        
        # 信号 = (量比 - 1)
        return vol_ratio - 1


class RSRSMSignal(SignalGenerator):
    """
    RSRS 信号（阻力支撑相对强度 - 上海证券版本）
    
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
        self.n = n      # 计算斜率的窗口
        self.m = m      # 计算标准分的窗口
        self.params = {'n': n, 'm': m}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        
        # 计算每日的斜率
        slopes = []
        for i in range(len(high)):
            if i < self.n - 1:
                slopes.append(np.nan)
                continue
            
            window_high = high.iloc[i - self.n + 1:i + 1]
            window_low = low.iloc[i - self.n + 1:i + 1]
            
            # 线性回归：high = α + β × low
            if window_low.std() == 0:
                slopes.append(np.nan)
            else:
                # 转换为 numpy 数组避免 pandas 兼容性问题
                low_values = window_low.values
                high_values = window_high.values
                cov = np.cov(low_values, high_values)[0, 1]
                var = np.var(low_values)
                beta = cov / var
                slopes.append(beta)
        
        slope_series = pd.Series(slopes, index=data.index)
        
        # 计算标准分
        if len(slope_series.dropna()) < self.m:
            return slope_series.fillna(0)
        
        rolling_mean = slope_series.rolling(self.m).mean()
        rolling_std = slope_series.rolling(self.m).std()
        
        z_score = (slope_series - rolling_mean) / rolling_std
        z_score = z_score.fillna(0)
        
        return z_score


class CompositeSignal(SignalGenerator):
    """
    组合信号生成器
    
    将多个信号生成器组合成一个
    """
    
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
        
        # 等权平均
        combined = pd.concat(all_signals, axis=1).mean(axis=1)
        return combined
    
    def generate_latest(self, data: pd.DataFrame) -> Signal:
        series = self.generate(data)
        latest_value = series.iloc[-1] if not series.empty else 0
        
        # 获取所有生成器的最新信号
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


# =============================================================================
# 函数式信号生成器
# =============================================================================

class FunctionSignal(SignalGenerator):
    """
    函数式信号生成器
    
    通过函数定义信号
    """
    
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


def ma_cross_signal(data: pd.DataFrame, short: int = 5, long: int = 20) -> pd.Series:
    """
    均线交叉信号（函数形式）
    """
    close = data['close']
    short_ma = close.rolling(short).mean()
    long_ma = close.rolling(long).mean()
    return (short_ma - long_ma) / long_ma


def momentum_signal(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    动量信号
    """
    close = data['close']
    return close / close.shift(period) - 1


def volatility_signal(data: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    波动率信号（布林带宽度）
    """
    close = data['close']
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    bandwidth = (2 * std) / mid
    return bandwidth


class CCISignal(SignalGenerator):
    """CCI 商品通道指数信号
    
    信号值：
    - > 100: 超买，做空信号
    - < -100: 超卖，做多信号
    - 0: 多空分界线
    """
    
    def __init__(self, period: int = 20):
        super().__init__(f"CCI{period}")
        self.period = period
        self.params = {'period': period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 计算典型价格
        typical_price = (high + low + close) / 3
        
        # 计算 CCI
        ma = typical_price.rolling(self.period).mean()
        mad = typical_price.rolling(self.period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (typical_price - ma) / (0.015 * mad)
        cci = cci.fillna(0)
        
        # 信号 = CCI（已经中心化）
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
        
        # 计算 N 日最高价和最低价
        highest_high = high.rolling(self.period).max()
        lowest_low = low.rolling(self.period).min()
        
        # 计算 WR
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        wr = wr.fillna(-50)
        
        # 信号 = WR + 50（中心化到 0 附近）
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
        
        # 计算 ROC
        roc = (close / close.shift(self.period) - 1) * 100
        roc = roc.fillna(0)
        
        # 信号 = ROC（已经中心化）
        return roc


class OBVSignal(SignalGenerator):
    """OBV 能量潮指标信号
    
    信号值：
    - OBV 上升：资金流入，做多信号
    - OBV 下降：资金流出，做空信号
    """
    
    def __init__(self, signal_period: int = 10):
        super().__init__(f"OBV{signal_period}")
        self.signal_period = signal_period
        self.params = {'signal_period': signal_period}
    
    def generate(self, data: pd.DataFrame) -> pd.Series:
        close = data['close']
        volume = data['volume']
        
        # 计算 OBV
        direction = np.sign(close.diff())
        obv = (volume * direction).cumsum()
        
        # 信号 = OBV 的移动平均变化率
        obv_ma = obv.rolling(self.signal_period).mean()
        signal = obv_ma.pct_change() * 100
        signal = signal.fillna(0)
        
        return signal


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
        
        # 计算 +DM 和 -DM
        high_diff = high.diff()
        low_diff = low.diff()
        
        plus_dm = np.where((high_diff > low_diff.diff().abs()) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff.abs() > high_diff) & (low_diff < 0), low_diff.abs(), 0)
        
        # 计算 TR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 平滑处理
        plus_dm_smooth = pd.Series(plus_dm, index=data.index).rolling(self.period).sum()
        minus_dm_smooth = pd.Series(minus_dm, index=data.index).rolling(self.period).sum()
        tr_smooth = pd.Series(tr).rolling(self.period).sum()
        
        # 计算 +DI 和 -DI
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # 信号 = +DI - -DI
        signal = plus_di - minus_di
        signal = signal.fillna(0)
        
        return signal


# =============================================================================
# 通用指标信号生成器
# =============================================================================

class IndicatorSignal(SignalGenerator):
    """
    通用指标信号生成器
    
    通过注入计算函数，支持任意指标。
    无需为每个指标编写完整类。
    
    用法:
        from ft2.signals.indicators import calc_kama_cross
        
        signal = IndicatorSignal(
            name="KAMA10_30",
            calc_func=calc_kama_cross,
            params={'short': 10, 'long': 30},
            normalize=True
        )
    """
    
    def __init__(
        self,
        name: str,
        calc_func: Callable[[pd.DataFrame, Dict[str, Any]], pd.Series],
        params: Dict[str, Any] = None,
        normalize: bool = True,
        threshold: float = 0.0
    ):
        """
        Args:
            name: 信号名称（如 "KAMA10_30"）
            calc_func: 计算函数，签名: func(data: DataFrame, params: dict) -> Series
            params: 计算参数（如 {'short': 10, 'long': 30}）
            normalize: 是否标准化信号（使 0 为多空分界线）
            threshold: 做多阈值（默认 >0 做多）
        """
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
        """标准化信号，使 0 为多空分界线"""
        median = signal.median()
        return signal - median


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
        indicator: 指标名称（'ma', 'ema', 'kama', 't3', 'trima', 等）
        short: 短期参数
        long: 长期参数
        period: 单参数指标（如 RSI 的 period）
        **kwargs: 其他参数
    
    Returns:
        SignalGenerator: 信号生成器实例
    
    Examples:
        >>> create_signal('ma', short=5, long=20)
        >>> create_signal('kama', short=10, long=30)
        >>> create_signal('rsi', period=14)
    """
    from .indicators import (
        calc_ma_cross, calc_ema_cross, calc_wma_cross,
        calc_dema_cross, calc_tema_cross, calc_kama_cross,
        calc_t3_cross, calc_trima_cross, calc_accbands
    )
    
    factory = {
        'ma': lambda: MASignal(short_period=short, long_period=long),
        'macd': lambda: MACDSignal(fast=short, slow=long, signal=kwargs.get('signal', 9)),
        'rsi': lambda: RSISignal(period=period),
        'kdj': lambda: KDJSignal(n=period or 9),
        'boll': lambda: BOLLSignal(period=period or 20),
        'cci': lambda: CCISignal(period=period or 20),
        'wr': lambda: WRSignal(period=period or 14),
        'roc': lambda: ROCSignal(period=period or 12),
        'obv': lambda: OBVSignal(signal_period=period or 10),
        'dmi': lambda: DMISignal(period=period or 14),
        'ema': lambda: IndicatorSignal(
            f"EMA{short}_{long}",
            calc_ema_cross,
            {'short': short, 'long': long}
        ),
        'wma': lambda: IndicatorSignal(
            f"WMA{short}_{long}",
            calc_wma_cross,
            {'short': short, 'long': long}
        ),
        'dema': lambda: IndicatorSignal(
            f"DEMA{short}_{long}",
            calc_dema_cross,
            {'short': short, 'long': long}
        ),
        'tema': lambda: IndicatorSignal(
            f"TEMA{short}_{long}",
            calc_tema_cross,
            {'short': short, 'long': long}
        ),
        'kama': lambda: IndicatorSignal(
            f"KAMA{short}_{long}",
            calc_kama_cross,
            {'short': short, 'long': long}
        ),
        't3': lambda: IndicatorSignal(
            f"T3{short}_{long}",
            calc_t3_cross,
            {'short': short, 'long': long, 'vf': kwargs.get('vf', 0.7)}
        ),
        'trima': lambda: IndicatorSignal(
            f"TRIMA{short}_{long}",
            calc_trima_cross,
            {'short': short, 'long': long}
        ),
        'accbands': lambda: IndicatorSignal(
            f"ACCBANDS{period or 20}",
            calc_accbands,
            {'period': period or 20}
        ),
    }
    
    if indicator not in factory:
        raise ValueError(f"Unsupported indicator: {indicator}. Available: {list(factory.keys())}")
    
    return factory[indicator]()
