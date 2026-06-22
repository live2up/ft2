"""
signals/v4/validate — 信号验证层 (ft2.core Engine)

[规范化] 2026-06-22 walkforward_validate → validate_walkforward, 对齐动词在前模式。
"""
from .single import validate_single
from .compare import compare_signals, signal_correlation
from .walkforward import validate_walkforward

__all__ = ['validate_single', 'compare_signals', 'signal_correlation', 'validate_walkforward']
