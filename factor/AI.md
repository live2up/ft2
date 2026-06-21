# factor 模块 AI 助手指南

> **版本：v4 | 更新日期：2026-06-21**

## 项目定位

因子挖掘 + 因子轮动研究模块。基于 signals.v4 Python AST DSL 构建，共享 67 原语，LLM 原生友好。

## 版本说明

| 版本 | 状态 | 说明 |
|------|------|------|
| **v4** | **主力** | signals.v4 AST DSL + ft2.core Engine + 67 原语, 探索灵活, `from factor.v4 import ...` |
| v3 | 保留 | 仅供历史测试对照，自研 Parser + 23 原语，回测已统一到 ft2.core (FactorEngineCore) |

## v4 核心变革

```
v3:  自研Parser → 23原语 → FactorEngineCore (ft2.core) → FactorExpression
     问题: 语法自定义(add/sub/mul), 原语少, Parser维护成本高

v4:  signals.v4 AST DSL → 67原语实时计算 → ft2.core Engine回测 → FactorExpression
     优势: 原生Python语法, 原语3倍, 回测引擎统一, 探索灵活
```

## 架构总览

```
                    面板数据 Dict[str, ndarray(T,N)]
                                  │
                   ┌──────────────┼──────────────────┐
                   ▼              ▼                    ▼
           factor/v4/      signals/v4/          factor/v4/
           expression.py    Expression           engine.py
           FactorExpression + rank_panel         EngineCore
           (薄包装层)       (AST DSL)            fast/full 双模式
                   │              │                    │
                   └──────┬───────┘                    │
                          ▼                            │
                  factor/v4/                           │
                  gp_engine.py                         │
                  GPEngine (种子驱动组合优化)           │
                          │                            │
                          └──────────┬─────────────────┘
                                     ▼
                            ┌──────────────┐
                            │   ft2.core    │
                            │ Engine/Account │
                            │ Analyzer/      │
                            │ Notebook       │
                            └──────────────┘
```

## 目录结构

```
factor/
├── v4/                     # 主力 ←
│   ├── __init__.py         # 统一导出
│   ├── expression.py       # FactorExpression (基于 signals.v4 AST DSL)
│   ├── engine.py           # EngineCore (fast/full 双模式, ft2.core 驱动)
│   ├── gp_engine.py       # GP 因子组合优化引擎 (Python AST 原生, 种子驱动)
│   ├── validator.py        # IC/IR/Bootstrap 检验
│   ├── search.py           # 网格搜索 + 贝叶斯优化
│   ├── cache.py            # 因子值缓存
│   ├── base.py             # FactorLibrary + FactorMetadata (复用 v3)
│   ├── industry_fitness.py # 行业适应度 + FitnessCalculator 基类
│   ├── llm/                # LLM 因子生成器
│   │   ├── generator.py
│   │   ├── prompts.py
│   │   └── eval_utils.py
│   ├── formulas/           # 公式库 (V4 语法)
│   │   ├── wq101.py        # WorldQuant 101 Alpha
│   │   ├── gt191.py        # 国泰安 191 Alpha
│   │   ├── industry.py     # 行业因子
│   │   └── basic.py        # 因子原子基元
│   └── discovered/         # GP 发现因子存档
├── v3/                     # 保留 (仅供历史测试对照)
```

---

## 核心 API (v4)

### 0. 导入规范

```python
from factor.v4 import FactorExpression, EngineCore, FactorLibrary

# 表达式 → 因子面板 → 回测 → 报告 (一站式)
from signals.v4 import Expression
expr = Expression("cs_rank(ts_roc(CLOSE, 20))")
panel = expr.rank_panel(assets)                              # 因子排名面板
result = EngineCore.backtest(panel, assets, mode='fast', top_n=3, rebalance='W')
# result.sharpe, result.cagr, result.max_drawdown

# full 模式: 验证 + 报告
analyzer = EngineCore.backtest(panel, assets, mode='full', top_n=3,
                               rebalance='W', bench_label='000300.SH')
analyzer.to_notebook("因子轮动回测")
```

### 1. FactorExpression — 因子表达式

```python
from factor.v4 import FactorExpression

# 字符串 → 因子值面板
expr = FactorExpression("cs_rank(ts_roc(CLOSE, 20))")
panel = expr.evaluate(data_dict)         # → ndarray(T, N)
ranked = expr.evaluate_ranked(data_dict) # → 截面排名 0~1

# 表达式自省
expr.variables    # ['CLOSE']
expr.functions    # ['cs_rank', 'ts_roc']
expr.complexity   # AST 节点数
```

**表达式语法与 signals.v4 完全一致** (67 原语，详见 signals/AI.md)：
- 数据源: `CLOSE OPEN HIGH LOW VOLUME AMOUNT`
- 时序: `ts_mean ts_std ts_rank ts_roc ts_zscore ts_delay ts_delta ...`
- 截面: `cs_rank cs_zscore cs_scale cs_winsorize ...`
- 特征: `rsi atr macd ema tsf kama ...`
- 数学: `abs log sqrt sign exp tanh sigmoid ...`
- Python 原生: `+ - * / > < and or not a if cond else b`

### 2. EngineCore — 因子轮动回测

```python
from factor.v4 import EngineCore

# fast 模式: 搜索 (~400ms/次, 1566天5品种) — 引擎内部走 Engine.run_fast() + FastAccount
analyzer = EngineCore.backtest(panel, assets, mode='fast', top_n=3, rebalance='W')
# → AccountAnalyzer (sharpe_ratio()=1.16, annualized_return()=0.148, max_drawdown()=(-0.10,...))

# full 模式: 验证 + 报告 — 引擎内部走 Engine.run() + AccountManager, 完整快照+交易记录
analyzer = EngineCore.backtest(panel, assets, mode='full', top_n=3,
                               rebalance='W', bench_label='000300.SH',
                               buffer=2)                        # 缓冲区: 持仓滑出 top_n+2 名才卖
analyzer.to_notebook("因子轮动")
```

**EngineCore.backtest() 参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `panel` | 因子排名面板, DataFrame(index=日期, columns=品种) | — |
| `assets` | `{品种代码: OHLCV DataFrame}` | — |
| `returns` | 日收益率 DataFrame (自动合成 OHLCV，二选一) | None |
| `top_n` | 持仓品种数 | 3 |
| `rebalance` | 调仓频率 'D'/'W'/'M'/'ME'/'5D' 或 Scheduler 对象 | 'W' |
| `mode` | 'fast' → AccountAnalyzer(仅资产指标), 'full' → AccountAnalyzer(全部指标) | 'fast' |
| `initial_capital` | 初始资金 | 1_000_000 |
| `start_date` | 回测起始日 | None=数据首位 |
| `bench_label` | full 模式基准标签 (自动跑 BenchHolder) | None |
| `buffer` | 排名缓冲数: 持仓滑出 top_n+buffer 名才剔除, 0=严格Top-N | 0 |
| `fee_config` | 费率配置 dict, None=零费率。例: `{'commission_rate': 0.0003, ...}` | None |

> fast/full 均返回 AccountAnalyzer，接口一致。fast 不生成 TradeRecord，交易指标返回 None。两者共用 core.Engine._drive_timeline() 时间线循环，fast 模式下 ctx.account 指向 FastAccount。

### 3. GP 因子组合优化引擎

```python
from factor.v4 import GPEngine

gp = GPEngine(
    data=panel_dict,            # {col: ndarray(T,N)}
    fitness_calculator=fitness_calc,  # 可插拔适应度
    seed_expressions=[
        'ts_zscore(CLOSE, 20) > 0 and amt_ratio(AMOUNT, 5, 20) > 1',
        'persist(ts_roc(CLOSE, 5) > 0, 2)',
    ],
    config={'population_size': 200, 'generations': 20},
)
gp.run()
best = gp.best()
# best.expression_str, best.fitness, best.depth
```

### 4. 行业适应度

```python
from factor.v4.industry_fitness import IndustryFitness, FitnessCalculator

# 行业轮动自适应适应度
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
from factor.v4 import FactorValidator, ValidationResult

validator = FactorValidator(factor_values, future_returns)
ic = validator.information_coefficient()
# ic['ic_mean'], ic['ic_std'], ic['ir']
decay = validator.decay_rate(max_lookforward=20)
groups = validator.group_return(n_groups=10)
```

### 6. 因子库 + 持久化

```python
from factor.v4 import FactorLibrary
from factor.v4.base import LibraryEntry

lib = FactorLibrary()
lib.register(LibraryEntry('alpha001', formula, fitness=1.5, source='gp'))
lib.register_batch(entries)

# 查询
lib.by_source('gp')
lib.seed_expressions(50)
lib.top(10, sort_by='sharpe')
lib.to_dataframe()

# 持久化
lib.save('./discovered/gp_round_001.json')
lib2 = FactorLibrary.load('./discovered/gp_round_001.json')
```

### 7. 公式库

```python
from factor.v4.formulas import ALPHA101, ALPHA191, BASIC_FACTORS

len(ALPHA101)       # 101   WorldQuant 101 Alpha
len(ALPHA191)       # 171   国泰安 191 Alpha
len(BASIC_FACTORS)  # 20    因子原子基元
```

---

## v4 vs v3 关键差异

| 项目 | v3 | v4 |
|------|----|----|
| 表达式语法 | 自研 `add/sub/mul` | **Python 原生 `+ - * /`** |
| 原语数 | 23 | **67 (与 signals 共享)** |
| 回测引擎 | FactorEngineCore (ft2.core) | **EngineCore (ft2.core, 同 signals)** |
| 因子面板回测 | FactorEngineCore (fast/full) | **EngineCore (fast/full + buffer)** |
| 截面排名 | cs_zscore(window) | **cs_rank / evaluate_ranked()** |
| LLM 友好度 | 需学自定义语法 | **原生 Python，即学即用** |

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

- **v4 主力**：`from factor.v4 import FactorExpression, EngineCore, FactorLibrary`
- v3 保留，回测已统一到 ft2.core，语法仍为自研 Parser
- FactorExpression 是 signals.v4 Expression 的薄包装层，语法完全一致
- EngineCore 与 signals.v4 EngineCore 架构对齐，fast/full 双模式，均返回 AccountAnalyzer
- fast 模式内部走 `core.Engine.run_fast()` + `FastAccount`，不生成快照/交易记录；ctx.account 接口与 full 统一
- 截面排名推荐使用 `evaluate_ranked()` 或 `Expression.rank_panel()`
- 67 原语覆盖 WorldQuant 时序算子的 96%，截面 100%
- 扩展新原语：`signals.v4.register_function()` 注册 → 因子模块即用
