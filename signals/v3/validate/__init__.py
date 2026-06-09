"""
signals/v3/validate/__init__.py — 验证层
=============================================================================
全部走 v3.EngineV3(mode='full') → AccountAnalyzer → to_notebook
"""

from .single import validate_single
from .compare import compare_signals
from .walkforward import walkforward_validate

__all__ = ['validate_single', 'compare_signals', 'walkforward_validate']
