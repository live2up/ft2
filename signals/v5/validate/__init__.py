"""
signals/v5/validate — 信号验证层 (ft2.core Engine)

[重构] 2026-07-07 从 v4 升级到 v5。
"""
from .single import validate_single
from .compare import compare_signals, signal_correlation
from .walkforward import validate_walkforward

__all__ = ['validate_single', 'compare_signals', 'signal_correlation', 'validate_walkforward']
