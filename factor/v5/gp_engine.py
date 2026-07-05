"""
factor/v5/gp_engine.py — 因子 GP 引擎 (wrapper)
=============================================================================

[重构] 2026-07-05 核心算法抽取到 utils/gp/v5/engine.py，
本文件只负责注入因子端的 evaluator (_ExpressionFromAST)。

用法不变:
  >>> from factor.v5.gp_engine import GPEngine
  >>> engine = GPEngine(data=data_dict, fitness_calculator=calc, ...)
"""
from utils.gp.v5.engine import GPEngine as _BaseGPEngine
from utils.gp.v5.config import TreeGenConfig, Individual
from .expression import _ExpressionFromAST


class GPEngine(_BaseGPEngine):
    """因子端 GP 引擎 — 注入 _ExpressionFromAST evaluator"""

    def __init__(self, *args, _no_auto_eval=False, **kwargs):
        # auto-inject factor evaluator (can be overridden)
        if not _no_auto_eval and 'evaluator' not in kwargs:
            kwargs['evaluator'] = lambda data, tree: _ExpressionFromAST(tree).evaluate(data)
        super().__init__(*args, **kwargs)


# 重新导出 (保持 API 兼容)
__all__ = ['GPEngine', 'TreeGenConfig', 'Individual', '_ExpressionFromAST']
