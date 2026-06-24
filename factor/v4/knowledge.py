"""
factor/v4/knowledge.py — 因子知识库 (薄封装)

继承自 utils.kb 通用核心，传入因子专有分类。
"""

from utils.kb import KnowledgeBase, ExplorationRecord, ExplorationStatus


# ════════════════════════════════════════════════════════════════
# 因子专有分类
# ════════════════════════════════════════════════════════════════

ALPHA_SOURCES = {
    "过度反应→反转": "跌幅过大的均值回归，熊市穿越之王",
    "反应不足→趋势": "趋势延续",
    "资金流动":      "量先于价",
    "结构性低风险":  "低波动/低换手，做组合稳定器",
    "信息冲击/跳空": "隔夜跳空 + 量确认",
    "相对价值/截面": "行业间相对强弱，只有周频有效",
    "周期阶段/体制": "条件切换 / 自适应权重",
}


class FactorKnowledgeBase(KnowledgeBase):
    """因子知识树 — 因子探索专用

    继承自 utils.kb.KnowledgeBase, 自动传入 ALPHA_SOURCES 分类,
    使用旧文件名 factor_knowledge_state.json 保持向后兼容。
    """

    def __init__(self, memory_dir: str = None):
        super().__init__(memory_dir=memory_dir, categories=ALPHA_SOURCES)
        # 向后兼容: 使用旧索引文件名
        self._state_path = self._state_path.parent / "factor_knowledge_state.json"
