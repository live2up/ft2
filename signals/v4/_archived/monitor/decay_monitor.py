"""
signals/v2/decay_monitor.py — 因子衰减监控
=============================================================================

追踪信号的 IC 随时间衰减，提供早期预警。


============================================================================
                         架构层级（竖式）
============================================================================

 ┌─────────────────────────────────────────────────────────────────────┐
 │  输入：Expression + OHLCV DataFrame + 历史信号                         │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 1  —  滚动 IC 计算                                              │
 │                                                                      │
 │  信号值(t) vs 未来收益(t+1) → 滚动窗口 IC                             │
 │  IC = corr(signal[t], forward_return[t+1])                           │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 2  —  衰减趋势拟合                                               │
 │                                                                      │
 │  对滚动 IC 时间序列做线性拟合 → 判断斜率                              │
 │  斜率 < 0: 衰减中 / 斜率 ≈ 0: 正常 / 斜率 << 0: 已失效              │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 3  —  预警输出 (AlertLevel)                                     │
 │                                                                      │
 │  ┌───────────┬─────────────────────────────────────────────────┐    │
 │  │  NORMAL   │ IC 稳定 > 0，信号正常                            │    │
 │  │  DECAYING │ IC 持续下降，但尚未归零                          │    │
 │  │  DEAD     │ IC ≈ 0，信号已失效                               │    │
 │  │  REVERSED │ IC < 0，信号已反转 (原先做多变做空更好)           │    │
 │  └───────────┴─────────────────────────────────────────────────┘    │
 └─────────────────────────────────────────────────────────────────────┘

============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from ..expression_v3 import Expression


class AlertLevel(Enum):
    NORMAL = 'normal'
    DECAYING = 'decaying'
    DEAD = 'dead'
    REVERSED = 'reversed'


@dataclass
class DecayResult:
    """因子衰减监控结果"""
    expression_name: str
    expression_str: str

    rolling_ic: pd.Series
    rolling_icir: pd.Series

    ic_mean_full: float
    ic_mean_recent_90d: float
    ic_mean_recent_180d: float

    decay_rate_per_month: float
    decay_trend_r2: float

    alert_level: AlertLevel
    alert_message: str
    days_to_zero_ic_estimate: Optional[float] = None

    recent_negative_ratio: float = 0.0

    @property
    def is_healthy(self) -> bool:
        return self.alert_level == AlertLevel.NORMAL

    @property
    def needs_attention(self) -> bool:
        return self.alert_level != AlertLevel.NORMAL

    def __repr__(self):
        return (
            f"DecayResult({self.expression_name}: "
            f"alert={self.alert_level.value}, "
            f"IC全期={self.ic_mean_full:.4f}, "
            f"IC近90日={self.ic_mean_recent_90d:.4f}, "
            f"月衰减={self.decay_rate_per_month:.4f})"
        )


class DecayMonitor:
    """
    因子衰减监控器

    追踪信号的预测力（IC）随时间的变化：
    - 计算滚动IC序列
    - 拟合线性衰减趋势
    - 判断失效程度
    """

    def __init__(self, feature_space=None):
        self._fs = feature_space

    def monitor(self, expression: str, data: pd.DataFrame,
                forward_period: int = 5,
                window: int = 60,
                feature_space=None) -> DecayResult:
        """
        监控一个信号的衰减情况

        Args:
            expression: 表达式字符串
            data: OHLCV DataFrame
            forward_period: 预测天数
            window: 滚动IC窗口大小
            feature_space: FeatureSpace

        Returns:
            DecayResult
        """
        fs = feature_space or self._fs
        if fs is None:
            raise ValueError("需要提供 FeatureSpace")

        target_col = f'forward_return_{forward_period}d'
        if target_col not in data.columns:
            data = data.copy()
            data[target_col] = data['close'].shift(-forward_period) / data['close'] - 1

        expr = Expression(expression, feature_space=fs)
        signal = expr.generate(data).reindex(data.index)

        target = data[target_col].reindex(signal.index)

        ic_rolling = signal.rolling(window).corr(target)
        ic_rolling = ic_rolling.dropna()

        if len(ic_rolling) < 30:
            return DecayResult(
                expression_name=expression, expression_str=expression,
                rolling_ic=ic_rolling, rolling_icir=pd.Series(dtype=float),
                ic_mean_full=float(ic_rolling.mean()) if len(ic_rolling) > 0 else 0.0,
                ic_mean_recent_90d=0.0, ic_mean_recent_180d=0.0,
                decay_rate_per_month=0.0, decay_trend_r2=0.0,
                alert_level=AlertLevel.NORMAL, alert_message='数据不足，无法评估衰减',
            )

        ic_full = float(ic_rolling.mean())
        ic_90d = float(ic_rolling.iloc[-90:].mean()) if len(ic_rolling) >= 90 else ic_full
        ic_180d = float(ic_rolling.iloc[-180:].mean()) if len(ic_rolling) >= 180 else ic_full

        x = np.arange(len(ic_rolling), dtype=float)
        y = ic_rolling.values
        mask = ~np.isnan(y)
        x_clean, y_clean = x[mask], y[mask]

        decay_rate, r2 = 0.0, 0.0
        days_to_zero = None
        if len(x_clean) > 20:
            coeffs = np.polyfit(x_clean, y_clean, 1)
            decay_rate = coeffs[0]
            y_pred = coeffs[0] * x_clean + coeffs[1]
            ss_res = np.sum((y_clean - y_pred) ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            if abs(decay_rate) > 1e-10:
                days_to_zero = abs(coeffs[1] / decay_rate) if coeffs[1] != 0 else None

        decay_per_month = decay_rate * 21

        recent_neg_ratio = float((ic_rolling.iloc[-90:] < 0).mean()) if len(ic_rolling) >= 90 else 0.0

        alert_level, alert_msg = self._classify_alert(
            ic_full, ic_90d, decay_per_month, r2, recent_neg_ratio
        )

        icir_rolling = ic_rolling.rolling(252).apply(
            lambda x: x.mean() / x.std() if x.std() > 0 else 0, raw=False
        )

        return DecayResult(
            expression_name=expression,
            expression_str=expression,
            rolling_ic=ic_rolling,
            rolling_icir=icir_rolling,
            ic_mean_full=ic_full,
            ic_mean_recent_90d=ic_90d,
            ic_mean_recent_180d=ic_180d,
            decay_rate_per_month=decay_per_month,
            decay_trend_r2=r2,
            alert_level=alert_level,
            alert_message=alert_msg,
            days_to_zero_ic_estimate=days_to_zero,
            recent_negative_ratio=recent_neg_ratio,
        )

    def monitor_all(self, expressions: List[str], data: pd.DataFrame,
                    feature_space=None) -> List[DecayResult]:
        """批量监控多个信号"""
        return [self.monitor(expr, data, feature_space=feature_space)
                for expr in expressions]

    def _classify_alert(self, ic_full: float, ic_recent: float,
                        decay_rate: float, r2: float,
                        negative_ratio: float) -> Tuple[AlertLevel, str]:
        """根据指标判断告警等级"""
        ic_full_abs = abs(ic_full)
        ic_recent_abs = abs(ic_recent)

        if ic_full_abs < 0.03:
            return AlertLevel.DEAD, (
                f'因子已失效: 全期IC={ic_full:.4f}，无显著预测力'
            )

        if ic_recent_abs < 0.03:
            return AlertLevel.DEAD, (
                f'因子已失效: 近90日IC降至{ic_recent:.4f}，全期IC={ic_full:.4f}'
            )

        if ic_recent_abs < ic_full_abs * 0.3:
            return AlertLevel.DEAD, (
                f'因子已失效: 近期IC({ic_recent:.4f})仅为全期({ic_full:.4f})的30%'
            )

        if r2 > 0.3 and abs(decay_rate) > 1e-4 and (
            (ic_full > 0 and decay_rate < 0) or (ic_full < 0 and decay_rate > 0)
        ):
            return AlertLevel.DECAYING, (
                f'因子正在衰减: 月衰减率={decay_rate*21:.4f}, 趋势R²={r2:.3f}'
            )

        if ic_recent_abs < ic_full_abs * 0.5:
            return AlertLevel.DECAYING, (
                f'因子正在衰减: 近期IC({ic_recent:.4f})低于全期({ic_full:.4f})的50%'
            )

        if ic_full > 0 and negative_ratio > 0.6:
            return AlertLevel.REVERSED, (
                f'因子方向反转: 近90日IC为负占{negative_ratio*100:.0f}%，全期IC为正'
            )
        if ic_full < 0 and negative_ratio < 0.4:
            return AlertLevel.REVERSED, (
                f'因子方向反转: 近90日IC为正占{(1-negative_ratio)*100:.0f}%，全期IC为负'
            )

        return AlertLevel.NORMAL, (
            f'因子正常: IC全期={ic_full:.4f} 近90日={ic_recent:.4f}'
        )


def check_decay(expression: str, data: pd.DataFrame,
                feature_space=None) -> DecayResult:
    """快捷函数: 检查一个信号的衰减状态"""
    monitor = DecayMonitor(feature_space)
    return monitor.monitor(expression, data)
