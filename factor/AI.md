# factor 模块 AI 助手指南

> **版本：v5 | 更新日期：2026-07-09**

## 项目定位

因子挖掘 + 因子轮动研究模块。基于 signals.v4 Python AST DSL 构建，共享 72 原语，LLM 原生友好。

## 版本说明

| 版本 | 状态 | 说明 |
|------|------|------|
| **v5** | **主力** | signals.v4 AST DSL + ft2.core Engine + 72 原语, 权重聚焦 GP, `from factor.v5 import ...` |
| v4 | 保留 | 与 v5 接口一致，仍兼容，新探索统一用 v5 |
| v3 | 保留 | 仅供历史测试对照，自研 Parser + 23 原语，回测已统一到 ft2.core |

## v5 核心架构

```
v3:  自研Parser → 23原语 → FactorFacEngine (ft2.core) → FactorExpression
     问题: 语法自定义(add/sub/mul), 原语少, Parser维护成本高

v4:  signals.v4 AST DSL → 72原语实时计算 → ft2.core Engine回测 → FactorExpression
     优势: 原生Python语法, 原语3倍, 回测引擎统一, 探索灵活

v5:  signals.v4 AST DSL → 72原语 → ft2.core Engine → GP引擎独立抽取到utils/gp/v5
     优势: GP核心算法独立迭代, factor层轻量, 权重聚焦+岛屿模型+Motif库
```

## 架构总览

```
                    面板数据 Dict[str, ndarray(T,N)]
                                  │
                   ┌──────────────┼──────────────────┐
                   ▼              ▼                    ▼
           factor/v5/      signals/v4/          factor/v5/
           expression.py    Expression           engine.py
           FactorExpression + rank_panel         FacEngine
           (薄包装层)       (AST DSL)            fast/full/vector 三模式
                   │              │                    │
                   └──────┬───────┘                    │
                          ▼                            │
                  factor/v5/                           │
                  gp_engine.py                         │
                  GPEngine (注入因子evaluator)          │
                          │                            │
                          └──────────┬─────────────────┘
                                     ▼
                            ┌──────────────┐
                            │  utils/gp/v5 │
                            │  GP核心算法   │
                            │  (独立迭代)   │
                            └──────────────┘
```

## 目录结构

```
factor/
├── v5/                     # 主力 ←
│   ├── __init__.py         # 统一导出
│   ├── base.py             # FactorLibrary + FactorMetadata
│   ├── engine.py           # FacEngine (fast/full/vector 三模式, ft2.core 驱动)
│   ├── expression.py       # FactorExpression (基于 signals.v4 AST DSL)
│   ├── gp_engine.py       # GP 因子引擎包装层 (注入因子evaluator, 核心在utils/gp/v5)
│   ├── validator.py        # IC/IR/Bootstrap 检验
│   ├── cache.py            # 因子值缓存
│   ├── industry_fitness.py # 行业适应度 + FitnessCalculator 基类
│   ├── knowledge.py        # 因子知识树 (FactorKnowledgeBase)
│   ├── llm/                # LLM 因子生成器
│   │   ├── generator.py
│   │   ├── prompts.py
│   │   └── eval_utils.py
│   ├── formulas/           # 公式库 (V5 语法)
│   │   ├── wq101.py        # WorldQuant 101 Alpha
│   │   ├── gt191.py        # 国泰安 191 Alpha
│   │   ├── industry.py     # 行业因子
│   │   ├── basic.py        # 因子原子基元
│   │   └── alpha158.py     # Qlib Alpha158
│   ├── discovered/         # GP 发现因子存档
│   └── 设计文档.md
├── v4/                     # 保留 (与v5接口一致)
└── v3/                     # 保留 (仅供历史测试对照)
```

---

## 核心 API (v5)

### 0. 导入规范

```python
from factor.v5 import FactorExpression, FacEngine, FactorLibrary, GPEngine

# 表达式 → 因子面板 → 回测 → 报告 (一站式)
expr = FactorExpression("cs_rank(ts_roc(CLOSE, 20))")
panel = expr.evaluate_ranked(data_dict)      # 因子排名面板
result = FacEngine.backtest(panel, assets, mode='fast', top_n=3, rebalance='W')
# result.sharpe_ratio(), result.annualized_return(), result.max_drawdown()

# full 模式: 验证 + 报告
analyzer = FacEngine.backtest(panel, assets, mode='full', top_n=3,
                               rebalance='W', bench_label='399317.SZ')
analyzer.to_notebook("因子轮动回测")
```

### 1. FactorExpression — 因子表达式

```python
from factor.v5 import FactorExpression

# 字符串 → 因子值面板
expr = FactorExpression("cs_rank(ts_roc(CLOSE, 20))")
panel = expr.evaluate(data_dict)         # → ndarray(T, N)
ranked = expr.evaluate_ranked(data_dict) # → 截面排名 0~1

# 表达式自省
expr.variables    # ['CLOSE']
expr.functions    # ['cs_rank', 'ts_roc']
expr.complexity   # AST 节点数
```

**表达式语法与 signals.v4 完全一致** (72 原语，详见 signals/AI.md)：
- 数据源: `CLOSE OPEN HIGH LOW VOLUME AMOUNT`
- 时序: `ts_mean ts_std ts_rank ts_roc ts_zscore ts_delay ts_delta ...`
- 截面: `cs_rank cs_zscore cs_scale cs_winsorize ...`
- 特征: `rsi atr macd ema tsf kama ...`
- 数学: `abs log sqrt sign exp tanh sigmoid ...`
- Python 原生: `+ - * / > < and or not a if cond else b`

### 2. FacEngine — 因子轮动回测

```python
from factor.v5 import FacEngine

# fast 模式: 搜索 (~400ms/次, 1566天5品种)
analyzer = FacEngine.backtest(panel, assets, mode='fast', top_n=3, rebalance='W')
# → AccountAnalyzer

# full 模式: 验证 + 报告 — 完整快照+交易记录
analyzer = FacEngine.backtest(panel, assets, mode='full', top_n=3,
                               rebalance='W', bench_label='399317.SZ',
                               buffer=2)                        # 缓冲区
analyzer.to_notebook("因子轮动")

# vector 模式: 纯矩阵向量化 (~50-100x 快于 full)
analyzer = FacEngine.backtest(panel, assets, mode='vector', top_n=3, rebalance='W')
```

**FacEngine.backtest() 参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `panel` | 因子排名面板, DataFrame(index=日期, columns=品种) | — |
| `assets` | `{品种代码: OHLCV DataFrame}` | — |
| `returns` | 日收益率 DataFrame (自动合成 OHLCV，二选一) | None |
| `top_n` | 持仓品种数 | 3 |
| `rebalance` | 调仓频率 'D'/'W'/'M'/'ME'/'5D' | 'W' |
| `mode` | 'fast'/'full'/'vector' | 'fast' |
| `initial_capital` | 初始资金 | 1_000_000 |
| `start_date` | 回测起始日 | None=数据首位 |
| `bench_label` | full 模式基准标签 | None |
| `buffer` | 排名缓冲数 | 0 |
| `fee_config` | 费率配置 dict | None |

> fast/full/vector 均返回 AccountAnalyzer，接口一致。vector 模式为纯矩阵运算，不生成 TradeRecord。

### 3. GP 因子引擎 (v5)

```python
from factor.v5 import GPEngine
from factor.v5.gp_engine import TreeGenConfig

gp = GPEngine(
    data=panel_dict,            # {col: ndarray(T,N)}
    fitness_calculator=fitness_calc,  # 可插拔适应度
    seed_expressions=[
        'ts_zscore(CLOSE, 20) > 0 and amt_ratio(AMOUNT, 5, 20) > 1',
        'persist(ts_roc(CLOSE, 5) > 0, 2)',
    ],
    tree_gen_config=TreeGenConfig(
        var_weights={'CLOSE': 1, 'VOLUME': 2, 'AMOUNT': 3},
        mode='continuous',  # 纯数值模式，适合单因子发现
    ),
    config={
        'population_size': 300,
        'generations': 30,
        'max_depth': 4,
        'seed_ratio': 0.35,
        'random_inject_ratio': 0.10,
        'lexicase': True,  # ε-Lexicase 选择
        'num_islands': 3,  # 岛屿模型
    },
    random_seed=42,
)
gp.run()
best = gp.best()
# best.expression_str, best.fitness, best.depth
```

> GP 核心算法已抽取到 `utils/gp/v5/`，factor/v5/gp_engine.py 仅为包装层，注入因子端 evaluator。

### 4. 行业适应度

```python
from factor.v5.industry_fitness import IndustryFitness

# 行业轮动自适应适应度 (N≥100严格, N50~100宽松, N<50仅Sharpe)
fitness = IndustryFitness(
    data=panel_dict,
    future_returns=returns_df,
    returns=returns_df,
    top_n=3,
)

# GPEngine 中使用
gp = GPEngine(
    data=panel_dict,
    fitness_calculator=fitness,
    seed_expressions=[...],
)
```

### 5. 因子检验

```python
from factor.v5 import FactorValidator

validator = FactorValidator(factor_values, future_returns)
ic = validator.information_coefficient()
# ic['ic_mean'], ic['ic_std'], ic['ir'], ic['positive_ratio']
decay = validator.decay_rate(max_lookforward=20)
```

### 6. 因子库 + 持久化

```python
from factor.v5 import FactorLibrary
from factor.v5.base import LibraryEntry

lib = FactorLibrary()
lib.register(LibraryEntry('alpha001', formula, fitness=1.5, source='gp'))
lib.register_batch(entries)

# 查询
lib.top(10, sort_by='sharpe')
lib.to_dataframe()

# 持久化
lib.save('./discovered/gp_round_001.json')
lib2 = FactorLibrary.load('./discovered/gp_round_001.json')
```

### 7. 公式库

```python
from factor.v5.formulas import ALPHA101, ALPHA191, BASIC_FACTORS

len(ALPHA101)       # 101   WorldQuant 101 Alpha
len(ALPHA191)       # 171   国泰安 191 Alpha
len(BASIC_FACTORS)  # 20    因子原子基元
```

### 8. LLM 因子生成

```python
from factor.v5.llm import LLMGenerator, quick_ic_batch

gen = LLMGenerator(provider="deepseek")
exprs = gen.generate("量价背离反转", n=10)
results = quick_ic_batch(exprs, panel_data, returns)
```

---

## v5 vs v4 vs v3 关键差异

| 项目 | v3 | v4 | **v5** |
|------|----|----|-------|
| 表达式语法 | 自研 `add/sub/mul` | Python 原生 `+ - * /` | 同 v4 |
| 原语数 | 23 | 72 (与 signals 共享) | 同 v4 |
| 回测引擎 | FactorFacEngine | FacEngine (fast/full) | **FacEngine (fast/full/vector)** |
| GP 引擎位置 | 内嵌 | 内嵌 | **独立抽取到 utils/gp/v5** |
| GP 特性 | 基础 | 基础 | **权重聚焦 + 岛屿模型 + Motif库 + ε-Lexicase** |
| 参数搜索 | 内置 search.py | 内置 search.py | **移除，GP 内部 _mutate_param 覆盖** |
| 向量化回测 | 无 | 无 | **vector 模式 (~50-100x)** |
| 翻译器 | 无 | translate_v3.py | **移除（ast.v2 更严谨）** |

---

## 数据格式约定

- **因子值面板：** `DataFrame`，index=日期，columns=股票代码，values=因子值
- **收益率面板：** 同上，values=日收益率
- **面板数据字典：** `Dict[str, ndarray(T,N)]`
- **assets 字典：** `Dict[str, DataFrame]`，品种代码 → OHLCV DataFrame

---

## 配套工具：Notebook 可视化报告

因子验证结果可通过 `notebook` 模块生成交互式 HTML 报告。

详见 [`notebook/AI.md`](../notebook/AI.md)

---

## 注意事项

- **v5 主力**：`from factor.v5 import FactorExpression, FacEngine, FactorLibrary, GPEngine`
- v4 保留，接口与 v5 一致，仍兼容
- FactorExpression 是 signals.v4 Expression 的薄包装层，语法完全一致
- FacEngine 与 signals.v4 FacEngine 架构对齐，新增 vector 纯矩阵模式
- fast 模式内部走 `core.Engine.run_fast()` + `FastAccount`，不生成快照/交易记录
- full 模式走 `core.Engine.run()` + `AccountManager`，完整快照+交易记录
- 截面排名推荐使用 `evaluate_ranked()` 或 `Expression.rank_panel()`
- 72 原语覆盖 WorldQuant 时序算子的 96%，截面 100%
- 扩展新原语：`signals.v4.register_function()` 注册 → 因子模块即用
- 知识库指标键频率后缀：`sharpe_ratio_d`（日度）、`sharpe_ratio_w`（周度）、`sharpe_ratio_m`（月度）
- GP 引擎核心算法在 `utils/gp/v5/`，factor/v5/gp_engine.py 仅为包装层
- GP 内部参数调优：`_mutate_param` 权重 40%，自动探索窗口/常数最优值，无需外部网格
- 向量化模式用于 GP 初筛/批量评估，速度比 full 快 50-100 倍

### 变量命名规范（因子探索）

| 场景 | 规范写法 | 避免写法 |
|------|---------|---------|
| 因子排名面板 | `panel` | `ranked` / `factor_df` / `fv` |
| OHLCV 字典 | `assets` | `data` / `ohlcv_dict` |
| 面板数据字典 | `data` 或 `panel_data` |  |
| 回测结果 | `analyzer` | `result` / `r` / `bt` |
| 品种代码列表 | `symbols` | `codes` / `tickers` |
| 表达式字符串 | `expr` | `formula` / `expr_str` |
| 品种名称映射 | 嵌入 DataFrame `name` 列 | 显式传参 |
