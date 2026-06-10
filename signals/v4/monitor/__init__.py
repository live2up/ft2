"""
signals/v3/monitor/__init__.py — 信号分析层
=============================================================================
提供信号的预测力评估、衰减检测和白盒解释。
=============================================================================
"""
from .ic_analyzer import ICAnalyzer
from .decay_monitor import DecayMonitor, DecayResult, AlertLevel, check_decay
from .explainer import Explainer, ExplanationReport, RegimePerformance, explain_signal

__all__ = [
    'ICAnalyzer',
    'DecayMonitor', 'DecayResult', 'AlertLevel', 'check_decay',
    'Explainer', 'ExplanationReport', 'RegimePerformance', 'explain_signal',
]
