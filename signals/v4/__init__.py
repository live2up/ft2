"""
signals/v4/__init__.py — V4 信号模块（基于 Python AST DSL）
=============================================================================
V4 核心变革：用 Python 内置 ast 模块替代自研 Parser。

对比 V3：
  V3: 自研 Parser → 7 种 AST 节点 → 37 预计算特征 → 0/1 信号
  V4: Python ast → 全量 Python 表达式 → 按需计算 + 原始 OHLCV → 连续值信号

复用 V3 的：
  - EngineV3.backtest()   fast/full 回测
  - pipeline / scoring    信号管线、复合打分
  - presets / registry    预设策略、注册

============================================================================
快速上手
============================================================================

  from signals.v4 import Expression
  from signals.v3 import EngineV3

  # 构建表达式
  expr = Expression("(CLOSE / ts_mean(CLOSE, 50) - 1) * 100")
  signal = expr.generate(ohlcv_df)

  # 回测
  result = EngineV3.backtest(signal, ohlcv_df, mode='fast')
  print(f"Sharpe={result.sharpe:.3f}")

  # 完整报告
  analyzer = EngineV3.backtest(signal, ohlcv_df, mode='full', bench_label='399317.SZ')
  analyzer.to_notebook("V4 策略回测")

============================================================================
V3 → V4 表达式迁移
============================================================================

  V3                                  V4
  ─────────────────────────────────────────────────────
  thr_mean(ATR{7})                    ATR_7 > expanding_mean(ATR_7)
  thr_50(RSI{14})                     RSI_14 > 50
  persist(thr_zscore(EMA{20}, 60, 0.5), 2)
                                      persist(EMA_20 > ts_mean(EMA_20, 60) + 0.5 * ts_std(EMA_20, 60), 2)
  expr1 & expr2                       (expr1) and (expr2)
  if_then(cond, a, b)                 a if cond else b
============================================================================
"""
from .ast_dsl import (
    parse_expression,
    evaluate,
    get_variables,
    get_functions,
    DSLSecurityError,
    DSLSyntaxError,
)
from .expression import Expression
from .registry import (
    FUNC_REGISTRY,
    is_valid_variable,
)

__all__ = [
    'Expression',
    'parse_expression',
    'evaluate',
    'get_variables',
    'get_functions',
    'FUNC_REGISTRY',
    'is_valid_variable',
    'DSLSecurityError',
    'DSLSyntaxError',
]
