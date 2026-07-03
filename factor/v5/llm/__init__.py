"""
factor/v4/llm — LLM 辅助因子发现内核

设计原则:
  - LLM = 表达式编译器（自然语言→AST），不是研究主管
  - 人在回路中心：生成→人评估→观察→反馈→再生成
  - 完全复用 signals/v4 AST DSL 安全校验

模块结构:
  prompts.py    — Prompt 模板 (系统提示/few-shot/解释/反馈)
  generator.py  — LLMGenerator (generate/mutate/explain/generate_with_feedback)
  eval_utils.py — 快速评估 (IC/IR/回测)

快速使用:
  >>> from factor.v4.llm import LLMGenerator, quick_ic_batch
  >>> gen = LLMGenerator(provider="deepseek")
  >>> exprs = gen.generate("量价背离反转", n=10)
  >>> results = quick_ic_batch(exprs, panel_data, returns)
"""

from .generator import LLMGenerator
from .eval_utils import quick_ic, quick_ic_batch, quick_rank_panel, quick_sharpe

__all__ = [
    'LLMGenerator',
    'quick_ic', 'quick_ic_batch', 'quick_rank_panel', 'quick_sharpe',
]
