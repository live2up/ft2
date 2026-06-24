"""
utils/kb — 通用知识库管理模块

核心功能:
  - JSONL 持久化存储探索记录
  - DAG 演化链追踪
  - 正交融合推荐
  - 全貌快照 (LLM 友好)

用法:
  >>> from utils.kb import KnowledgeBase
  >>> kb = KnowledgeBase(memory_dir="./my_project")
  >>> kb.record(session_id="exp_01", category="动量", template="roc(CLOSE,20)")
  >>> kb.get_tree_state()

自定义分类:
  >>> kb = KnowledgeBase(memory_dir="./project", categories={
  ...     "动量": "趋势跟踪",
  ...     "反转": "均值回归",
  ... })
"""

from .core import (
    KnowledgeBase,
    ExplorationRecord,
    ExplorationStatus,
    KBNode,
)

__all__ = [
    'KnowledgeBase',
    'ExplorationRecord',
    'ExplorationStatus',
    'KBNode',
]
