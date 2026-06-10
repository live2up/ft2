"""
signals/v2/pipeline.py — 信号管线
=============================================================================

编排多个 Expression 的组合逻辑。SignalPipeline 本身也是信号生成器
（generate() 接口），可嵌套。


============================================================================
                         架构层级（竖式）
============================================================================

 ┌─────────────────────────────────────────────────────────────────────┐
 │  输入：阶段列表 [(stage_name, handler), ...]                          │
 │                                                                      │
 │  pipeline = SignalPipeline([                                         │
 │      ("signal",    Expression("ATR(7) & TRIMA(60)", fs)),           │
 │      ("persist",   persist(3)),         ← 连续3日确认                │
 │      ("threshold", thr_func),            ← 阈值过滤                  │
 │  ])                                                                  │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  阶段类型                                                             │
 │                                                                      │
 │  ┌──────────────┬──────────────────────────────────────────────┐    │
 │  │   阶段名      │   处理器                                     │    │
 │  ├──────────────┼──────────────────────────────────────────────┤    │
 │  │ "signal"     │ Expression 对象 / 表达式字符串 / callable     │    │
 │  │ "persist"    │ int (连续确认天数) / callable                 │    │
 │  │ "threshold"  │ callable → 二值化过滤                         │    │
 │  │ "combine"    │ 多信号合成 (预留)                             │    │
 │  │ 自定义名     │ callable → 任意信号变换                       │    │
 │  └──────────────┴──────────────────────────────────────────────┘    │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  generate(data) → 逐阶段执行 → 最终信号                               │
 │                                                                      │
 │  每个阶段的输出作为下一阶段的输入，最终返回 pd.Series 信号序列         │
 └─────────────────────────────────────────────────────────────────────┘


============================================================================
                         快捷组合函数
============================================================================

 ┌─────────────────────┬─────────────────────────────────────────────┐
 │       函数           │       说明                                  │
 ├─────────────────────┼─────────────────────────────────────────────┤
 │ pipe_and(a, b)      │ 两个信号取 AND (两者>0 才做多)               │
 │ pipe_or(a, b)       │ 两个信号取 OR (任一>0 就做多)               │
 │ pipe_vote(signals)  │ 多数投票 (3+ 信号的民主表决)                │
 │ pipe_weighted(...)  │ 加权组合 (带权重系数)                        │
 └─────────────────────┴─────────────────────────────────────────────┘


============================================================================
                         组合方式枚举
============================================================================

 ┌──────────────┬──────────────────────────────────────────────────┐
 │  AND          │ 全部信号触发才做多                                │
 │  OR           │ 任一信号触发就做多                                │
 │  VOTE         │ 多数信号触发才做多                                │
 │  WEIGHTED     │ 加权求和后阈值判断                                │
 │  ADAPTIVE     │ 动态权重 (预留)                                   │
 └──────────────┴──────────────────────────────────────────────────┘

============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum

from .expression_v3 import Expression, persist, regime_switch


class CombineMethod(Enum):
    AND = 'and'
    OR = 'or'
    VOTE = 'vote'
    WEIGHTED = 'weighted'
    ADAPTIVE = 'adaptive'


class SignalPipeline:
    """
    信号管线 — 多阶段信号处理

    声明式多阶段：
        pipeline = SignalPipeline([
            ("signal", Expression("ATR(7) & TRIMA(60)", fs)),
            ("persist", persist(3)),
            ("threshold", thr_func),
        ])

    Args:
        stages: 声明式阶段列表 [(name, Expression or callable), ...]
        最后一个阶段必须返回可生成信号的对象
    """

    def __init__(self, stages: List[tuple], name: str = None):
        self.stages = stages
        self._name = name or 'Pipeline'
        self._cached_expr = None

    def _resolve(self, stage_data, data: pd.DataFrame) -> Any:
        """解析一个阶段"""
        name, handler = stage_data
        if hasattr(handler, 'generate'):
            return handler.generate(data)
        elif callable(handler):
            if hasattr(handler, '__name__') and handler.__name__ == '<lambda>':
                return handler(data)
            else:
                return handler(data)
        elif isinstance(handler, int):
            # persist(n)
            return handler
        elif isinstance(handler, str):
            # 表达式字符串
            from .expression_v3 import Expression
            expr = Expression(handler)
            return expr.generate(data)
        return handler

    def generate(self, data: pd.DataFrame) -> pd.Series:
        """执行管线"""
        current = data
        result = None

        for i, (name, handler) in enumerate(self.stages):
            if name == "signal":
                if hasattr(handler, 'generate'):
                    signal = handler.generate(data)
                elif isinstance(handler, str):
                    from .expression_v3 import Expression
                    signal = Expression(handler).generate(data)
                else:
                    signal = handler(data) if callable(handler) else handler
                current = signal
                result = signal
            elif name == "persist":
                n = handler if isinstance(handler, int) else 3
                if hasattr(current, 'values'):
                    from .expression_v3 import np_persist
                    arr = np_persist(current.values, n)
                    result = pd.Series(arr, index=current.index)
                    current = result
            elif name == "threshold":
                if callable(handler):
                    result = handler(current)
                    current = result
            elif name == "combine":
                # 组合阶段，传入多个信号
                pass
            else:
                if callable(handler):
                    result = handler(current)
                    current = result

        if result is None:
            return pd.Series(np.zeros(len(data)), index=data.index)
        if isinstance(result, np.ndarray):
            result = pd.Series(result, index=data.index)
        return result

    def __repr__(self):
        stage_names = [s[0] for s in self.stages]
        return f"SignalPipeline({self._name}, stages={stage_names})"


# ============================================================
# 便捷组合函数
# ============================================================

def pipe_and(*expressions: Expression, name: str = None) -> SignalPipeline:
    """AND组合管线"""
    combined = expressions[0]
    for e in expressions[1:]:
        combined = combined & e
    return SignalPipeline([("signal", combined)], name=name or 'AND')


def pipe_or(*expressions: Expression, name: str = None) -> SignalPipeline:
    """OR组合管线"""
    combined = expressions[0]
    for e in expressions[1:]:
        combined = combined | e
    return SignalPipeline([("signal", combined)], name=name or 'OR')


def pipe_vote(*expressions: Expression, threshold: float = 0.5, name: str = None) -> SignalPipeline:
    """投票组合管线"""
    def vote_func(data):
        signals_list = []
        for e in expressions:
            s = e.generate(data)
            signals_list.append((s > 0).astype(int))
        vote_sum = sum(signals_list) / len(signals_list)
        return (vote_sum > threshold).astype(int)
    return SignalPipeline([("signal", vote_func)], name=name or 'VOTE')


def pipe_weighted(*pairs, name: str = None) -> SignalPipeline:
    """
    加权组合管线

    Args:
        pairs: (Expression, weight) 对
    """
    def weighted_func(data):
        total = 0
        for expr, weight in pairs:
            s = expr.generate(data)
            total += s * weight
        return total
    return SignalPipeline([("signal", weighted_func)], name=name or 'WEIGHTED')
