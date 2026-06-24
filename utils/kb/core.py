"""
utils/kb/core.py — 通用知识库核心

数据模型:
  ExplorationRecord — JSONL 单行 (一次实验的结果)
  KBNode    — 从同 template 的多条记录聚合的节点
  KnowledgeBase     — 主类: 记录/查询/演化/持久化

设计原则:
  - 写一次: 探索后调 record()
  - 随时查: query() / suggest_next()
  - 不自动决策: 只辅助判断方向
"""

import os
import json
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


# ════════════════════════════════════════════════════════════════
# 状态枚举
# ════════════════════════════════════════════════════════════════

class ExplorationStatus(Enum):
    UNTOUCHED = "untouched"    # 尚未探索
    PARTIAL = "partial"        # 已探索但参数边界未穷尽
    EXHAUSTED = "exhausted"    # 参数网格已全部跑完
    REDUNDANT = "redundant"    # 与已有结果同质


# ════════════════════════════════════════════════════════════════
# 数据模型
# ════════════════════════════════════════════════════════════════

@dataclass
class ExplorationRecord:
    """一次探索实验的标准化记录 — 对应 JSONL 的一行"""

    session_id: str
    """会话标识, 请用 {项目}_{描述}_{YYMMDD} 格式. 如: sr_mom_divergence_0623"""

    timestamp: str
    """ISO 时间, 自动生成. 如: 2026-06-23T21:48:44"""

    category: str
    """分类标签. 构造时传了 categories 则校验有效性; 不传则不限制. 如: 反应不足→趋势"""

    template: str
    """模板表达式 (带参数占位符的版本, 非具体值).
       因子: roc(CLOSE,10) - roc(AMOUNT,10)
       信号(方案A): BUY:rsi(CLOSE,14)<30 | SELL:rsi(CLOSE,14)>70
       具体参数值放 best_params."""

    params_grid: Dict[str, list] = field(default_factory=dict)
    """参数搜索空间, 值列表枚举所有候选.
       如: {"W": [5, 15, 30], "T": [0.1, 0.5, 0.9]}"""

    params_explored: int = 0
    """已探索的组合数, 自动从 params_grid 计算."""

    params_total: int = 0
    """网格总组合数, 自动从 params_grid 计算."""

    best_params: Dict[str, Any] = field(default_factory=dict)
    """最优参数组合, 与 params_grid 的键对应.
       如: {"W": 30, "T": 0.5}"""

    metrics: Dict[str, float] = field(default_factory=dict)
    """性能指标字典, 键无硬限制.
       建议至少包含 sharpe_ratio_d 以便排序.
       常用键: sharpe_ratio_d, sharpe_ratio_w, max_drawdown,
               annualized_return, win_rate, avg_hold, ic_mean"""

    exhausted: bool = False
    """True = 参数网格已全部跑完, 无需再探索本模板."""

    orthogonal_to: List[str] = field(default_factory=list)
    """与此记录正交的已有 session_id 列表 (相关系数 < 0.3).
       当前未自动填充, 留为 LLM 自行判断."""

    redundant_with: List[str] = field(default_factory=list)
    """与此记录同质的已有 session_id 列表 (相关系数 > 0.85).
       当前未自动填充, 留为 LLM 自行判断."""

    note: str = ""
    """探索结论/教训/备注. 一句话说明因子的经济学逻辑."""

    parent_templates: List[str] = field(default_factory=list)
    """DAG 父模板列表.
       []       = 独立发现, 无演化来源
       [父模板] = 在父模板基础上改进 (调参/改函数/加条件)
       [父1, 父2] = 融合两个父模板"""

    evolution_note: str = ""
    """演化说明, 描述从父模板到当前模板的变化.
       如: '在delta20基础上改为roc20, 并加入cos包装'"""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat(timespec='seconds')


@dataclass
class KBNode:
    """知识树中的一个模板节点 — 从 ExplorationRecord 聚合生成"""

    category: str
    template: str
    records: List[ExplorationRecord] = field(default_factory=list)

    status: ExplorationStatus = ExplorationStatus.UNTOUCHED
    best_params: Dict = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    explored_grid: Dict[str, set] = field(default_factory=dict)

    def add_record(self, record: ExplorationRecord):
        self.records.append(record)
        for k, vals in record.params_grid.items():
            if k not in self.explored_grid:
                self.explored_grid[k] = set()
            self.explored_grid[k].update(vals)

        current_best = self.metrics.get("sharpe_ratio_d", -999)
        new_val = record.metrics.get("sharpe_ratio_d", -999)
        if new_val > current_best:
            self.best_params = record.best_params
            self.metrics = record.metrics.copy()

        if record.exhausted:
            self.status = ExplorationStatus.EXHAUSTED
        elif self.records:
            self.status = ExplorationStatus.PARTIAL

    def is_redundant(self) -> bool:
        return self.status == ExplorationStatus.REDUNDANT

    def summary(self) -> Dict:
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
        if not self.records:
            return 0.0
        total_explored = sum(r.params_explored for r in self.records)
        total_all = sum(r.params_total or r.params_explored for r in self.records)
        if total_all == 0:
            return 0.0
        return min(total_explored / total_all, 1.0)


# ════════════════════════════════════════════════════════════════
# 核心类: 通用知识库
# ════════════════════════════════════════════════════════════════

DEFAULT_LOG_NAME = "exploration_log.jsonl"
DEFAULT_STATE_NAME = "kb_state.json"


class KnowledgeBase:
    """通用知识树 — 记录已探索空间，供多智能体共享查询

    用法:
      kb = KnowledgeBase(memory_dir=PROJECT_ROOT)
      kb.record(session_id="...", category="...", template="...")
      kb.get_tree_state()
    """

    def __init__(self, memory_dir: str = None, categories: Dict[str, str] = None):
        """
        Args:
            memory_dir: 记忆文件存放目录。None=默认从 __file__ 推算。
            categories: 可选分类字典 {name: description}。传入后 record() 会校验 category。
        """
        if memory_dir is not None:
            memory_dir = Path(memory_dir) / 'memory'
        else:
            memory_dir = Path(__file__).resolve().parent.parent.parent / 'memory'
        self._log_path = memory_dir / DEFAULT_LOG_NAME
        self._state_path = memory_dir / DEFAULT_STATE_NAME
        self._categories = categories or {}

        self._tree: Dict[Tuple[str, str], KBNode] = {}
        self._records: List[ExplorationRecord] = []
        self._load()

    # ── 写操作 ──

    def record(self,
               session_id: str,
               category: str,
               template: str,
               params_grid: Dict[str, list] = None,
               best_params: Dict[str, Any] = None,
               metrics: Dict[str, float] = None,
               exhausted: bool = False,
               note: str = "",
               parent_templates: List[str] = None,
               evolution_note: str = "",
               ) -> ExplorationRecord:
        """记录一次探索结果（追加到 JSONL，更新树）

        Args:
            session_id: 会话ID
            category:   分类标签（如果构造时传了 categories，会校验有效性）
            template:   模板表达式
            params_grid: 参数网格
            best_params: 最优参数
            metrics:    性能指标字典
            exhausted:  True=已穷尽
            note:       探索结论
            parent_templates: 父模板
            evolution_note:   演化说明

        Returns:
            ExplorationRecord
        """
        # 校验 category (如果构造时传了 categories)
        if self._categories and category not in self._categories:
            valid = list(self._categories.keys())
            raise ValueError(
                f"category 必须属于: {valid}, 传入: {category}."
                f" 如不需要校验, 构造时勿传 categories 参数."
            )

        # 计算 params_explored / params_total
        params_explored = 0
        params_total = 0
        if params_grid:
            for k, vals in params_grid.items():
                if params_total == 0:
                    params_total = len(vals)
                else:
                    params_total *= len(vals)
            params_explored = params_total

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
            parent_templates=parent_templates or [],
            evolution_note=evolution_note,
        )

        self._append_log(record)
        self._records.append(record)
        key = (category, template)
        if key not in self._tree:
            self._tree[key] = KBNode(category=category, template=template)
        self._tree[key].add_record(record)
        self._save_state()
        return record

    def mark_redundant(self, category: str, template: str):
        """手动标记某模板为冗余 (与已有因子同质, 相关系数 > 0.85).
        标记后 query() 过滤 status 时可按 REDUNDANT 排除."""
        key = (category, template)
        if key in self._tree:
            self._tree[key].status = ExplorationStatus.REDUNDANT
            self._save_state()

    def mark_exhausted(self, category: str, template: str):
        """手动标记某模板为已挖尽 (参数网格已全部跑完).
        标记后 suggest_next() 的 PARTIAL 推荐中不再出现."""
        key = (category, template)
        if key in self._tree:
            self._tree[key].status = ExplorationStatus.EXHAUSTED
            self._save_state()

    # ── 查操作 ──

    def query(self,
              category: str = None,
              status: ExplorationStatus = None,
              min_sharpe: float = None,
              sort_by: str = "sharpe_ratio_d",
              limit: int = 20,
              ) -> List[Dict]:
        """按条件查询知识库

        Args:
            category:   过滤分类
            status:     过滤状态
            min_sharpe: 最小 Sharpe 阈值
            sort_by:    按哪个 metrics 键排序, 默认 sharpe_ratio_d
            limit:      返回上限

        Returns:
            [{category, template, status, best_sharpe_ratio_d, ...}, ...]
        """
        results = []
        for key, node in self._tree.items():
            cat, tmpl = key
            if category and cat != category:
                continue
            if status and node.status != status:
                continue
            shp = node.metrics.get(sort_by, -999)
            if min_sharpe is not None and (shp == -999 or shp < min_sharpe):
                continue

            summary = node.summary()
            if node.records:
                latest = node.records[-1]
                summary["note"] = latest.note
                summary["evolution_note"] = latest.evolution_note
                summary["parent_templates"] = latest.parent_templates
            results.append(summary)

        results.sort(key=lambda x: x.get(f"best_{sort_by}", -999) or -999, reverse=True)
        return results[:limit]

    def suggest_next(self, top_k: int = 5) -> List[Dict]:
        """推荐下一步探索方向"""
        suggestions = []

        # 1. 类别覆盖率检查
        category_counts = {}
        for key in self._tree:
            cat, _ = key
            category_counts[cat] = category_counts.get(cat, 0) + 1
        for cat in self._categories:
            count = category_counts.get(cat, 0)
            if count < 3:
                suggestions.append({
                    "category": cat,
                    "template": None,
                    "reason": f"类别 '{cat}' 仅有 {count} 个探索记录",
                    "priority": "high",
                    "hint": self._categories.get(cat, ""),
                })

        # 2. PARTIAL 模板
        for key, node in self._tree.items():
            if node.status == ExplorationStatus.PARTIAL:
                suggestions.append({
                    "category": key[0],
                    "template": key[1],
                    "reason": f"参数在边界上, 最优: {node.best_params}, "
                              f"Sharpe_D={node.metrics.get('sharpe_ratio_d', '?')}",
                    "priority": "medium",
                })

        # 3. 高 Sharpe 模板的熊市弱点
        top_nodes = sorted(
            [n for n in self._tree.values() if not n.is_redundant()],
            key=lambda n: n.metrics.get("sharpe_ratio_d", -999),
            reverse=True,
        )[:3]
        for node in top_nodes:
            bear = node.metrics.get("bear_sharpe", 0)
            if bear is not None and bear < 0.1:
                sr = node.metrics.get("sharpe_ratio_d", -999)
                sr_str = f"{sr:.3f}" if isinstance(sr, (int, float)) else "?"
                suggestions.append({
                    "category": node.category,
                    "template": node.template,
                    "reason": f"全样本 SR={sr_str}, 但熊市={bear:.3f}",
                    "priority": "medium",
                })

        # 4. 独立根节点融合推荐
        root_by_cat = {}
        for r in self._records:
            if not r.parent_templates:
                root_by_cat.setdefault(r.category, []).append(r.template)
        for cat, roots in root_by_cat.items():
            if len(roots) >= 2:
                suggestions.append({
                    "category": cat,
                    "template": None,
                    "reason": f"类别 '{cat}' 有 {len(roots)} 个独立根节点可尝试融合",
                    "priority": "medium",
                })

        priority_map = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: priority_map.get(s["priority"], 99))
        return suggestions[:top_k]

    # ── DAG 演化追踪 ──

    def _find_record(self, template: str) -> Optional[ExplorationRecord]:
        for r in reversed(self._records):
            if r.template == template:
                return r
        return None

    def get_evolution_chain(self, template: str, full: bool = False,
                            _visited: set = None) -> List[Dict]:
        """返回指定模板的完整演化链（从根到当前节点）"""
        if _visited is None:
            _visited = set()
        if template in _visited:
            return []
        _visited.add(template)

        chain = []
        record = self._find_record(template)
        if not record:
            return chain

        node_info = {
            "template": template,
            "sharpe_ratio_d": record.metrics.get("sharpe_ratio_d"),
            "note": record.note,
            "evolution_note": record.evolution_note,
        }
        if record.parent_templates:
            node_info["parent_templates"] = record.parent_templates
            if full:
                node_info["ancestry"] = [
                    self.get_evolution_chain(p, full=True, _visited=_visited)
                    for p in record.parent_templates
                ]
        chain.append(node_info)

        if record.parent_templates:
            chain.extend(
                self.get_evolution_chain(
                    record.parent_templates[0], full=full, _visited=_visited
                )
            )
        return list(reversed(chain))

    def find_fusion_candidates(self, category: str = None) -> List[Dict]:
        """找独立根节点，推荐正交配对融合"""
        roots = {}
        for r in self._records:
            if not r.parent_templates:
                cat = r.category
                if category and cat != category:
                    continue
                roots.setdefault(cat, []).append(
                    (r.template, r.metrics.get("sharpe_ratio_d", 0) or 0)
                )
        results = []
        for cat, items in roots.items():
            if len(items) >= 2:
                items.sort(key=lambda x: x[1], reverse=True)
                results.append({
                    "category": cat,
                    "root_templates": [t for t, _ in items],
                    "count": len(items),
                    "sharpe_range": f"{items[-1][1]:.3f}~{items[0][1]:.3f}",
                    "suggestion": f"尝试融合 {items[0][0]} (SR={items[0][1]:.3f}) "
                                  f"和 {items[1][0]} (SR={items[1][1]:.3f})",
                })
        return sorted(results, key=lambda x: x["count"], reverse=True)

    def list_categories(self) -> List[Dict]:
        """按类别统计探索状态"""
        stats = {}
        for cat in self._categories:
            stats[cat] = {
                "total_templates": 0,
                "exhausted": 0,
                "partial": 0,
                "redundant": 0,
                "untouched": 0,
                "best_sharpe": -999,
                "best_template": "",
            }

        # 记录未在 categories 中的动态分类
        for key, node in self._tree.items():
            cat, _ = key
            if cat not in stats:
                stats[cat] = {
                    "total_templates": 0, "exhausted": 0, "partial": 0,
                    "redundant": 0, "untouched": 0,
                    "best_sharpe": -999, "best_template": "",
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

        return [{"category": cat, **info}
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

    # ── 持久化 ──

    def _append_log(self, record: ExplorationRecord):
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def _load(self):
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
                        self._tree[key] = KBNode(
                            category=record.category,
                            template=record.template,
                        )
                    self._tree[key].add_record(record)
                except (json.JSONDecodeError, TypeError) as e:
                    warnings.warn(f"跳过损坏记录: {e}\n  {line[:200]}")

    def _save_state(self):
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
        return (f"KnowledgeBase("
                f"records={len(self._records)}, "
                f"templates={len(self._tree)}, "
                f"log={self._log_path})")
