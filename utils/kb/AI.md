# utils/kb — 通用知识库模块 AI 助手指南

> **版本：v2 | 更新日期：2026-06-24**

## 模块定位

`utils/kb` 是 ft2 项目的**通用知识库基础设施**，为任何需要"试什么→结果如何→下一步试什么"的迭代探索过程提供统一的存储、查询和演化追踪。

核心设计：**基类零领域语义，领域层做薄封装。**

```
utils/kb/
├── __init__.py   # 导出 KnowledgeBase
├── core.py       # 核心类 (零依赖, 纯stdlib)
└── AI.md         # 本文件
```

## 数据模型

### ExplorationRecord — JSONL 单行

```python
@dataclass
class ExplorationRecord:
    session_id: str                # 会话ID      格式: {项目}_{描述}_{YYMMDD}
    timestamp: str                 # ISO时间      自动生成
    category: str                  # 分类标签     传 categories 则校验, 不传则不限制
    template: str                  # 模板表达式   放占位符版本, 具体参数值放 best_params

    params_grid: Dict              # 参数搜索空间 e.g. {"W": [5,15,30]}
    params_explored: int           # 已探索数     自动计算
    params_total: int              # 总数         自动计算
    best_params: Dict              # 最优参数     e.g. {"W": 30}

    metrics: Dict[str, float]      # 性能指标     自由键, 建议含 sharpe_ratio_d 以便排序
    exhausted: bool                # 是否已挖尽
    orthogonal_to: List[str]       # 正交节点     (留空, 供 LLM 自行判断)
    redundant_with: List[str]      # 同质节点     (留空, 供 LLM 自行判断)

    note: str                      # 探索结论     一句话说明逻辑
    parent_templates: List[str]    # DAG父模板    []独立 / [1]继承 / [1,2]融合
    evolution_note: str            # 演化说明     描述从父模板到当前的变化
```

### KBNode — 内存聚合节点

从同 `(category, template)` 的多条记录聚合而成，自动追踪最优指标和参数覆盖率。

## 快速导入

```python
from utils.kb import KnowledgeBase, ExplorationRecord, ExplorationStatus
```

## 完整用法场景

### 场景 1：自由标签（不校验 category）
任何场景都适用, category 随便写。

```python
kb = KnowledgeBase(memory_dir="./my_project")
kb.record(
    session_id="exp_test_0624",
    category="任意标签",
    template="rsi(CLOSE,14) < 30",
    metrics={"sharpe_ratio_d": 1.2},
)
```

### 场景 2：因子探索（校验 category）
通过 `FactorKnowledgeBase` 继承自动传入 ALPHA_SOURCES 7 类。

```python
from factor.v4 import FactorKnowledgeBase
kb = FactorKnowledgeBase(memory_dir=PROJECT_ROOT)
kb.record(
    session_id="sr_mom_div_0623",
    category="反应不足→趋势",       # 必须属于 ALPHA_SOURCES
    template="roc(CLOSE,10) - roc(AMOUNT,10)",
    metrics={"sharpe_ratio_d": 0.857},
    note="价量背离: 缩量上涨=强势",
)
```

### 场景 3：信号探索（BUY+SELL 编码到 template）
对偶信号用 `BUY:{expr} | SELL:{expr}` 编码。

```python
from utils.kb import KnowledgeBase
kb = KnowledgeBase(memory_dir=SIGNALS_ROOT, categories={
    "趋势": "rsi/ema/趋势跟踪",
    "反转": "boll/rsi/均值回归",
})
kb.record(
    session_id="signal_test_0624",
    category="趋势",
    template="BUY:rsi(CLOSE,14)<30 | SELL:rsi(CLOSE,14)>70",
    params_grid={"max_hold": [5, 10, 20]},
    best_params={"max_hold": 10},
    metrics={"sharpe_ratio_d": 1.2, "win_rate": 0.65, "avg_hold": 8.5},
)
```

### 场景 4：模型调参
记录机器学习模型的超参数搜索过程。

```python
kb = KnowledgeBase(memory_dir="./ml_projects")
kb.record(
    session_id="xgb_tune_0624",
    category="分类模型",
    template="XGBoost: n_est={n_est}, lr={lr}, max_depth={md}",
    params_grid={"n_est": [100, 500], "lr": [0.01, 0.1], "md": [3, 6]},
    best_params={"n_est": 500, "lr": 0.01, "md": 6},
    metrics={"auc": 0.873, "logloss": 0.312},
    note="低学习率+深树组合最优",
)
```

### 场景 5：数据源评估
评估不同数据源的质量。

```python
kb.record(
    session_id="data_eval_0624",
    category="数据源",
    template="1min_bar, 2020-2025, 中证500",
    metrics={"coverage": 0.98, "quality_score": 0.92},
    note="1分钟数据质量好, 但2020年前缺失较多",
)
```

## 核心 API

| 方法 | 用途 | 返回 |
|:-----|:-----|:-----|
| `record(...)` | 记录一次探索结果 | `ExplorationRecord` |
| `query(category, min_sharpe, sort_by, limit)` | 按条件查询 | `List[Dict]` |
| `get_tree_state()` | 全貌快照(LLM推荐) | `Dict` |
| `suggest_next(top_k)` | 推荐下一步方向 | `List[Dict]` |
| `get_evolution_chain(template)` | DAG 演化链回溯 | `List[Dict]` |
| `find_fusion_candidates(category)` | 正交根节点融合推荐 | `List[Dict]` |
| `list_categories()` | 按类别统计探索状态 | `List[Dict]` |
| `mark_redundant(cat, tmpl)` | 标记为同质 (query可排除) | — |
| `mark_exhausted(cat, tmpl)` | 标记为挖尽 (suggest中排除) | — |

## 快速查询（LLM 一次性读取）

```python
kb = KnowledgeBase(memory_dir=PROJECT_ROOT)
state = kb.get_tree_state()

# LLM 直接获得:
# {
#   "total_records": 161,
#   "total_templates": 92,
#   "categories": [{"category":"反应不足→趋势","total_templates":15,...}],
#   "best_overall": [{"template":"...","best_sharpe_ratio_d":0.975,...}],
#   "suggestions": [{"category":"资金流动","reason":"仅2个记录",...}]
# }
```

## 演化追踪（DAG）

```python
chain = kb.get_evolution_chain("cos(abs(Δ²P/(ΔP+1)))")
# 返回: [{template, sharpe, note, evolution_note}, ...]
# 从根 → cos_delta → 曲率比, 完整回溯
```

## 使用规范

### session_id 命名

```
{项目缩写}_{描述}_{YYMMDD}
        例: sr_mom_divergence_0623
            gp_math_cos_0623
            kb_01_path_quality_er
            xgb_tune_0624
```

### template 存放原则

- 放**带参数占位符**的模板版本，具体参数值放 `best_params`
- 对偶信号用 `BUY:{expr} | SELL:{expr}` 编码
- metrics 能直接用 `sort_by` 排序的最少放一个关键指标（如 `sharpe_ratio_d`）

### metrics 建议键

| 键 | 含义 | 适用 |
|:----|:-----|:-----|
| `sharpe_ratio_d` | 日频 Sharpe | 因子/信号 (推荐) |
| `sharpe_ratio_w` | 周频 Sharpe | 因子/信号 |
| `max_drawdown` | 最大回撤(%) | 因子/信号 |
| `annualized_return` | 年化收益(%) | 因子/信号 |
| `win_rate` | 胜率 | 信号 |
| `avg_hold` | 平均持仓天数 | 信号 |
| `ic_mean` | IC 均值 | 因子 |
| `auc` | AUC | 模型 |
| `logloss` | 对数损失 | 模型 |
| `f1` | F1 分数 | 模型 |

### DAG 维护

- 独立发现 → `parent_templates=[]`
- 继承改进 → `parent_templates=[父模板]` + `evolution_note="说明改进"` 
- 融合 → `parent_templates=[父1, 父2]`

## 架构图

```
外部项目 (AI_yinzi / 其他)
    │
    │  from utils.kb import KnowledgeBase
    │  kb = KnowledgeBase(memory_dir=PROJECT_ROOT)
    ▼
┌─────────────────────────────────────────────┐
│  utils/kb/KnowledgeBase                      │
│                                              │
│  JSONL (追加日志) ──── 写入 ──→ exploration_log.jsonl │
│  JSON  (状态快照) ──── 覆盖 ──→ kb_state.json          │
└─────────────────────────────────────────────┘
    ▲                    ▲                    ▲
    │                    │                    │
  因子薄封装            信号薄封装          直接使用
  (factor/v4/)        (signals/v4/)        (模型/数据/实验)
```

## 注意事项

1. **memory_dir 必须传**：不传则从 `__file__` 推算，外部项目务必传 `memory_dir=PROJECT_ROOT`
2. **category 校验**：只有构造时传了 `categories` 才会校验，不传则不校验
3. **metrics 类型**：值是 `float`，不要传 `None` 或字符串
4. **JSONL 安全**：只追加不修改，不会丢数据；手动编辑时小心不要破坏 JSON 行格式
5. **state 快照**：每次 `record()` 后自动覆盖 `kb_state.json`，不要手动编辑
