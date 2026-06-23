"""
factor/v4/knowledge.py — 因子知识树 (FactorKnowledgeBase)

解决不同智能体间的探索记忆共享问题。
每次 explore_*.py 跑完后调用 record() 记录，Agent 可随时 query() 查已探索空间。

用法:
  >>> from factor.v4 import FactorKnowledgeBase
  >>> kb = FactorKnowledgeBase()
  >>>
  >>> # 跑完一次探索后，记录结果
  >>> kb.record(
  ...     session_id="量化_条件切换_20260607",
  ...     category="周期阶段/体制",
  ...     template="ifelse(gt(ts_rank(bench_close, W), T), A1, M6)",
  ...     params_grid={"W": [5,15,30,50,60], "T": [0.1,0.3,0.5,0.7,0.9]},
  ...     best_params={"W": 60, "T": 0.7},
  ...     metrics={
  ...         "sharpe_ratio_d": 1.230, "sharpe_ratio_w": 0.879,
  ...         "bear_sharpe": 0.205, "bull_sharpe": 1.951, "sideways_sharpe": 1.136,
  ...         "wf_avg_sharpe": 0.964, "wf_std_sharpe": 1.197, "wf_min_sharpe": -0.547,
  ...         "max_drawdown": -21.17, "annualized_return": 25.3,
  ...     },
  ...     exhausted=True,
  ...     note="W=60 T=0.7 最优, 但熊市弱(0.205), 需搭配A1保护"
  ... )
  >>>
  >>> # 下一步方向推荐
  >>> kb.suggest_next()
  >>> # 查看某类别的探索状态
  >>> kb.query(category="资金流动")
"""

import os
import json
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# 常量 — 7 类 Alpha Source（来自 RESEARCH.md 实战归纳）
# ═══════════════════════════════════════════════════════════════════════════

ALPHA_SOURCES = {
    "过度反应→反转": "跌幅过大的均值回归，熊市穿越之王",
    "反应不足→趋势": "趋势延续，argmax 60d 唯一双频王者",
    "资金流动":      "量先于价，rel_amount 残差 = 当前天花板 1.026",
    "结构性低风险":  "低波动/低换手，做组合稳定器",
    "信息冲击/跳空": "隔夜跳空 + 量确认，与资金类互补",
    "相对价值/截面": "行业间相对强弱，只有周频有效",
    "周期阶段/体制": "条件切换 / 自适应权重，框架层最优方向",
}

# 默认记忆相对于 ft2 根目录（从 __file__ 推算），外部项目应传 memory_dir 覆盖
#   推荐: FactorKnowledgeBase(memory_dir=PROJECT_ROOT)
#   默认: ft2/memory/exploration_log.jsonl + ft2/memory/factor_knowledge_state.json
DEFAULT_LOG_NAME = "exploration_log.jsonl"
DEFAULT_STATE_NAME = "factor_knowledge_state.json"


class ExplorationStatus(Enum):
    UNTOUCHED = "untouched"    # 尚未探索
    PARTIAL = "partial"        # 已探索但参数边界未穷尽
    EXHAUSTED = "exhausted"    # 参数网格已全部跑完
    REDUNDANT = "redundant"    # 与已有因子同质（相关系数 > 0.85）


# ═══════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExplorationRecord:
    """一次探索实验的标准化记录 — 对应 JSONL 的一行

    所有字段均为 JSON 可序列化。
    """
    # ── 身份标识 ──
    session_id: str                          # 会话ID
    timestamp: str                           # ISO时间
    category: str                            # 7类 alpha source 之一
    template: str                            # 表达式模板（含参数占位符）

    # ── 搜索空间 ──
    params_grid: Dict[str, list] = field(default_factory=dict)
    """参数网格, e.g. {"W": [5,15,30,50,60], "T": [0.1,0.3,0.5,0.7,0.9]}"""

    params_explored: int = 0                 # 已探索的组合数
    params_total: int = 0                    # 网格总组合数

    # ── 最优结果 ──
    best_params: Dict[str, Any] = field(default_factory=dict)
    """最优参数组合, e.g. {"W": 60, "T": 0.7}"""

    # ── 多维度性能档案 ──
    metrics: Dict[str, float] = field(default_factory=dict)
    """
    支持的键（按维度分层，命名对齐 AccountAnalyzer @metric 的方法名，频率后缀: _d=日度, _w=周度, _m=月度):
      sharpe_ratio_d / sharpe_ratio_w         — 日度/周度 Sharpe
      bear_sharpe / bull_sharpe / sideways_sharpe — 不同市场环境下的 Sharpe
      wf_avg_sharpe / wf_std_sharpe / wf_min_sharpe — Walk-forward Sharpe 统计
      max_drawdown                       — 最大回撤 (%)
      annualized_return                  — 年化收益率 (%)
      ic_mean / ic_ir                    — IC 统计
      win_rate                           — 胜率
    """

    # ── 状态 ──
    exhausted: bool = False                  # True=参数网格已穷尽
    orthogonal_to: List[str] = field(default_factory=list)
    """与此节点正交的已有节点ID列表（相关系数 < 0.3）"""

    redundant_with: List[str] = field(default_factory=list)
    """与此节点同质的已有节点ID列表（相关系数 > 0.85）"""

    # ── 人工/Agent 标记 ──
    note: str = ""
    """探索结论、教训、下一步建议"""

    def __post_init__(self):
        # 兼容旧日志：timestamp 字段可能不存在
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat(timespec='seconds')


@dataclass
class FactorTreeNode:
    """知识树中的一个模板节点 — 从 ExplorationRecord 聚合生成"""
    category: str
    template: str
    records: List[ExplorationRecord] = field(default_factory=list)

    # 聚合指标（最优记录的各项指标）
    status: ExplorationStatus = ExplorationStatus.UNTOUCHED
    best_params: Dict = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    # 已探索参数空间
    explored_grid: Dict[str, set] = field(default_factory=dict)
    """{param_name: set of explored values}"""

    def add_record(self, record: ExplorationRecord):
        """合并一条新记录到树节点"""
        self.records.append(record)

        # 更新已探索参数
        for k, vals in record.params_grid.items():
            if k not in self.explored_grid:
                self.explored_grid[k] = set()
            self.explored_grid[k].update(vals)

        # 更新最优（按 sharpe_ratio_d 比较）
        current_best = self.metrics.get("sharpe_ratio_d", -999)
        new_val = record.metrics.get("sharpe_ratio_d", -999)
        if new_val > current_best:
            self.best_params = record.best_params
            self.metrics = record.metrics.copy()

        # 更新状态
        if record.exhausted:
            self.status = ExplorationStatus.EXHAUSTED
        elif self.records:
            self.status = ExplorationStatus.PARTIAL

    def is_redundant(self) -> bool:
        return self.status == ExplorationStatus.REDUNDANT

    def summary(self) -> Dict:
        """简洁摘要，用于 suggest_next 输出"""
        return {
            "category": self.category,
            "template": self.template,
            "status": self.status.value,
            "best_sharpe_ratio_d": self.metrics.get("sharpe_ratio_d"),
            "best_sharpe_ratio_w": self.metrics.get("sharpe_ratio_w"),
            "best_params": self.best_params,
            "explored_ratio": self._explored_ratio(),
            "records": len(self.records),
        }

    def _explored_ratio(self) -> float:
        """已探索/总组合的占比"""
        if not self.records:
            return 0.0
        total_explored = sum(r.params_explored for r in self.records)
        total_all = sum(r.params_total or r.params_explored for r in self.records)
        if total_all == 0:
            return 0.0
        return min(total_explored / total_all, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 核心类：FactorKnowledgeBase
# ═══════════════════════════════════════════════════════════════════════════

class FactorKnowledgeBase:
    """因子知识树 — 记录已探索空间，供多智能体共享查询

    设计原则:
      - 写一次: 跑完 explore 后调 record()
      - 随时查: query() / suggest_next()
      - 不自动决策: 只辅助 Agent 判断方向
    """

    def __init__(self, memory_dir: str = None):
        """
        Args:
            memory_dir: 记忆文件存放目录。
                传入后知识库文件存在 memory_dir/exploration_log.jsonl + .../factor_knowledge_state.json。
                None=默认放 ft2/memory/（ft2 根目录从 __file__ 推算）。
                外部项目（如 AI_yinzi）推荐传入 PROJECT_ROOT，记忆与项目共存。
        """
        if memory_dir is not None:
            memory_dir = Path(memory_dir) / 'memory'
        else:
            memory_dir = Path(__file__).resolve().parent.parent.parent / 'memory'
        self._log_path = memory_dir / DEFAULT_LOG_NAME
        self._state_path = memory_dir / DEFAULT_STATE_NAME

        # 树结构: {(category, template): FactorTreeNode}
        self._tree: Dict[Tuple[str, str], FactorTreeNode] = {}
        self._records: List[ExplorationRecord] = []

        # 加载已有数据
        self._load()

    # ──────────────────────────────────────────────────────────────
    # 写操作
    # ──────────────────────────────────────────────────────────────

    def record(self,
               session_id: str,
               category: str,
               template: str,
               params_grid: Dict[str, list] = None,
               best_params: Dict[str, Any] = None,
               metrics: Dict[str, float] = None,
               exhausted: bool = False,
               note: str = "",
               ) -> ExplorationRecord:
        """记录一次探索结果（追加到 JSONL，更新树）

        Args:
            session_id: 会话ID, e.g. "量化_条件切换_20260607"
            category:   7类 alpha source 之一
            template:   表达式模板（含参数占位符）
            params_grid: 此次跑的完整参数网格
            best_params: 最优参数组合
            metrics:    最优参数的性能指标字典
            exhausted:  True=参数网格已全部跑完
            note:       探索结论/教训

        Returns:
            ExplorationRecord: 刚创建的记录对象
        """
        # 校验 category
        if category not in ALPHA_SOURCES:
            valid = list(ALPHA_SOURCES.keys())
            raise ValueError(f"category 必须是以下之一: {valid}, 传入: {category}")

        # 计算 params_explored / params_total
        params_explored = 0
        params_total = 0
        if params_grid:
            for k, vals in params_grid.items():
                if params_total == 0:
                    params_total = len(vals)
                else:
                    params_total *= len(vals)
            params_explored = params_total  # 假设此次全覆盖

        # 创建记录
        record = ExplorationRecord(
            session_id=session_id,
            timestamp=datetime.now().isoformat(timespec='seconds'),
            category=category,
            template=template,
            params_grid=params_grid or {},
            params_explored=params_explored,
            params_total=params_total,
            best_params=best_params or {},
            metrics=metrics or {},
            exhausted=exhausted,
            note=note,
        )

        # 追加到 JSONL
        self._append_log(record)

        # 更新内存
        self._records.append(record)
        key = (category, template)
        if key not in self._tree:
            self._tree[key] = FactorTreeNode(category=category, template=template)
        self._tree[key].add_record(record)

        # 持久化树状态
        self._save_state()

        return record

    def mark_redundant(self, category: str, template: str,
                       reason: str = ""):
        """手动标记某模板为冗余（与已有因子同质）"""
        key = (category, template)
        if key in self._tree:
            self._tree[key].status = ExplorationStatus.REDUNDANT
            self._save_state()

    def mark_exhausted(self, category: str, template: str):
        """手动标记某模板为已挖尽"""
        key = (category, template)
        if key in self._tree:
            self._tree[key].status = ExplorationStatus.EXHAUSTED
            self._save_state()

    # ──────────────────────────────────────────────────────────────
    # 查操作
    # ──────────────────────────────────────────────────────────────

    def query(self,
              category: str = None,
              status: ExplorationStatus = None,
              min_sharpe: float = None,
              freq: str = "sharpe_ratio_d",
              limit: int = 20,
              ) -> List[Dict]:
        """按条件查因子库

        Args:
            category: 过滤 alpha source 类别
            status:   过滤探索状态
            min_sharpe: 最小 Sharpe 阈值
            freq:     使用哪个频率的 Sharpe，默认 "sharpe_ratio_d"（日度）
            limit:    返回条数上限

        Returns:
            [{
                "category": ..., "template": ...,
                "status": ..., "best_sharpe_ratio_d": ...,
                "best_params": ..., "explored_ratio": ...,
                "records": ..., "note": ...,
            }, ...]
        """
        results = []
        for key, node in self._tree.items():
            cat, tmpl = key
            metrics = node.metrics

            # 过滤 category
            if category and cat != category:
                continue

            # 过滤 status
            if status and node.status != status:
                continue

            # 过滤 min_sharpe
            shp = metrics.get(freq, -999)
            if min_sharpe is not None and (shp == -999 or shp < min_sharpe):
                continue

            summary = node.summary()
            # 补充最新 note
            if node.records:
                summary["note"] = node.records[-1].note
            results.append(summary)

        # 按指定频率的 Sharpe 降序
        results.sort(key=lambda x: x.get(f"best_{freq}", -999) or -999, reverse=True)
        return results[:limit]

    def suggest_next(self, top_k: int = 5) -> List[Dict]:
        """推荐下一步探索方向

        优先级:
          1. 覆盖率低的类别（该类别已有节点数 < 3）
          2. 已探索但参数在边界上的模板（PARTIAL）
          3. 高 Sharpe 模板的 orthogonal 方向
          4. 主动发现：未被充分探索的 category

        Returns:
            [{"category": ..., "template": ...,
              "reason": ..., "priority": ...}, ...]
        """
        suggestions = []

        # 1. 类别覆盖率检查
        category_counts = {}
        for key in self._tree:
            cat, _ = key
            category_counts[cat] = category_counts.get(cat, 0) + 1
        for cat in ALPHA_SOURCES:
            count = category_counts.get(cat, 0)
            if count < 3:
                suggestions.append({
                    "category": cat,
                    "template": None,
                    "reason": f"类别 '{cat}' 仅有 {count} 个探索记录，远低于建议的 3 个",
                    "priority": "high",
                    "hint": ALPHA_SOURCES[cat],
                })

        # 2. PARTIAL 模板（参数在边界上，需扩展）
        for key, node in self._tree.items():
            if node.status == ExplorationStatus.PARTIAL:
                metrics = node.metrics
                suggestions.append({
                    "category": key[0],
                    "template": key[1],
                    "reason": f"参数在边界上，当前最优: {node.best_params}, "
                              f"Sharpe_D={metrics.get('sharpe_ratio_d', '?')}, "
                              f"已探索 {node._explored_ratio():.0%}",
                    "priority": "medium",
                })

        # 3. 高 Sharpe 模板的正交方向
        top_nodes = sorted(
            [n for n in self._tree.values() if not n.is_redundant()],
            key=lambda n: n.metrics.get("sharpe_ratio_d", -999),
            reverse=True,
        )[:3]
        for node in top_nodes:
            metrics = node.metrics
            bear_sharpe = metrics.get("bear_sharpe", 0)
            if bear_sharpe is not None and bear_sharpe < 0.1:
                suggestions.append({
                    "category": node.category,
                    "template": node.template,
                    "reason": f"全样本 Sharpe={metrics.get('sharpe_ratio_d', '?'):.3f}, "
                              f"但熊市={bear_sharpe:.3f}, 需搭配防御因子保护",
                    "priority": "medium",
                })

        # 按优先级排序
        priority_map = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: priority_map.get(s["priority"], 99))

        return suggestions[:top_k]

    def list_categories(self) -> List[Dict]:
        """按类别统计探索状态"""
        stats = {}
        for cat in ALPHA_SOURCES:
            stats[cat] = {
                "total_templates": 0,
                "exhausted": 0,
                "partial": 0,
                "redundant": 0,
                "untouched": 0,
                "best_sharpe": -999,
                "best_template": "",
            }
        for key, node in self._tree.items():
            cat, tmpl = key
            if cat not in stats:
                continue
            stats[cat]["total_templates"] += 1
            s = node.status.value
            stats[cat][s] = stats[cat].get(s, 0) + 1
            shp = node.metrics.get("sharpe_ratio_d", -999)
            if shp and shp > stats[cat]["best_sharpe"]:
                stats[cat]["best_sharpe"] = shp
                stats[cat]["best_template"] = tmpl

        return [{"category": cat, **info, "hint": ALPHA_SOURCES[cat]}
                for cat, info in stats.items()]

    def get_tree_state(self) -> Dict:
        """获取完整树状态（供 Agent 一次性读取）"""
        return {
            "total_records": len(self._records),
            "total_templates": len(self._tree),
            "categories": self.list_categories(),
            "best_overall": self.query(limit=5),
            "suggestions": self.suggest_next(top_k=5),
        }

    # ──────────────────────────────────────────────────────────────
    # 持久化
    # ──────────────────────────────────────────────────────────────

    def _append_log(self, record: ExplorationRecord):
        """追加一条记录到 JSONL"""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _load(self):
        """从 JSONL 重建树"""
        if not self._log_path.exists():
            return
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    record = ExplorationRecord(**data)
                    self._records.append(record)
                    key = (record.category, record.template)
                    if key not in self._tree:
                        self._tree[key] = FactorTreeNode(
                            category=record.category,
                            template=record.template,
                        )
                    self._tree[key].add_record(record)
                except (json.JSONDecodeError, TypeError) as e:
                    warnings.warn(f"跳过损坏的记录行: {e}\n  {line[:200]}")

    def _save_state(self):
        """将树状态持久化为紧凑 JSON（方便 Agent 一次性读取）"""
        state = {
            "updated_at": datetime.now().isoformat(timespec='seconds'),
            "total_records": len(self._records),
            "total_templates": len(self._tree),
            "nodes": {},
        }
        for key, node in self._tree.items():
            cat, tmpl = key
            if cat not in state["nodes"]:
                state["nodes"][cat] = {}
            state["nodes"][cat][tmpl] = node.summary()

        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def __repr__(self) -> str:
        return (f"FactorKnowledgeBase("
                f"records={len(self._records)}, "
                f"templates={len(self._tree)}, "
                f"log={self._log_path})")
