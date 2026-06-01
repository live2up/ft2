# factor 模块 - AI 快速上手

> 因子挖掘体系

> **版本：v3.0.0 | 更新日期：2026-06-01**
>
> **AI 助手注意：** 如果发现实际 API 与本文档不一致，说明源码已更新但 AI.md 未同步，请提醒用户更新。

---

## 版本说明
- **v3/** — 推荐使用版本（GP 发现引擎 + 可插拔适应度 + 迭代探索 + 自增长因子库）
- **v2/** — 稳定版本，不再新增功能
- **v1/** — 已废弃

---

## 核心 API (v3)

> v3 的中心不是"合并文件"，而是构建一个**因子发现流水线**：
> ```
> 种子注入(formulas) → 多策略GP并行(IC/Sharpe/多频率) → Pipeline验证 → 入库 → 扩大的种子池 → 下一轮
> ```

### 1. 因子表达式引擎

```python
from factor.v3 import FactorExpression

# 字符串 → 因子值面板
expr = FactorExpression("ts_rank(sub(close, delay(close, 20)), 10)")
panel = expr.evaluate(data_dict)  # → ndarray(T, N)
# expr.terminals  → {'close'}
# expr.depth      → 4
# expr.node_count → 9
```

**表达式语法速查：**
```
终端变量:   open, high, low, close, volume, amount
一元函数:   abs(x), sqrt(x), log(x), neg(x)
二元函数:   add(x,y), sub(x,y), mul(x,y), div(x,y), max(x,y), min(x,y)
时序原语:   ts_rank(x, period)      ts_zscore(x, period)
            delay(x, period)         decay_linear(x, period)
            ts_sum(x, period)        ts_mean(x, period)
            ts_std(x, period)        ts_max(x, period)
            ts_min(x, period)        sma(x, period, lag)
            correlation(x, y, p)     covariance(x, y, p)
            regbeta(x, y, p)         ts_argmin(x, p)
            ts_argmax(x, p)
截面原语:   cs_rank(x)              cs_zscore(x, period)
            signed_power(x, exp)
条件分支:   ifelse(cond, a, b)

示例: ts_rank(mul(cs_zscore(sub(close, delay(close, 20))), abs(close)), 10)
```

### 2. GP 符号回归引擎（核心升级）

```python
from factor.v3 import GPEngine, FitnessMode, Individual

gp = GPEngine(
    data=panel_dict,            # {col: ndarray(T,N)}
    future_returns=returns_df,  # DataFrame(T,N)
    returns=returns_df,         # 原始收益率（Sharpe/MultiFreq 模式需要）
    fitness_mode=FitnessMode.ICIR,  # 适应度：ICIR / SHARPE / MULTI_FREQ
    population_size=300,
    generations=30,
    max_depth=8,
    # 可选：自定义原语集和终端集
    custom_terminals=['close', 'volume', 'open'],
    custom_primitives=[('ts_rank', 'period', [5,10,20,30])],
    seed_expressions=[           # 已知好因子注入
        "ts_rank(sub(close, delay(close, 20)), 10)",
    ],
)
gp.run()
best = gp.best()
# best.expression_str  →  "ts_rank(mul(...), 10)"
# best.fitness         →  2.34
# best.depth, best.node_count

print(gp.report())
```

**v2 vs v3 关键差异：**
| 项目 | v2 `FactorGPMiner` | v3 `GPEngine` |
|------|-------------------|---------------|
| 适应度 | 固定 ICIR | 可插拔 `ICIR/SHARPE/MULTI_FREQ` |
| 调度器 | 全局硬编码 | 构造注入 |
| 原语集 | 硬编码 | 构造参数传入 |

### 3. 因子发现引擎（v3 核心）

```python
from factor.v3 import FactorDiscoveryEngine

engine = FactorDiscoveryEngine(
    data=panel_dict,
    returns=returns_df,
    seed_formulas=ALPHA101,      # 注入公式库作初始种子
    cost_rate=0.0,
    seed_n=50,
)

# 多轮迭代 — 每轮可配置不同的适应度策略、调仓频率、持仓数
report = engine.run_pipeline(rounds=[
    # 第一轮：ICIR 快筛，月末+Top5 验证
    {'mode': 'icir', 'generations': 20, 'top_n': 50,
     'freq': 'ME', 'val_top_n': 5},
    # 第二轮：Sharpe 验证，周度+Top3
    {'mode': 'sharpe', 'generations': 40, 'top_n': 30,
     'freq': 'W', 'val_top_n': 3},
    # 第三轮：多频率稳健性
    {'mode': 'multi_freq', 'generations': 30, 'top_n': 20,
     'freq': '10D', 'val_top_n': 10},
])

# 因子库自增长：每轮验证通过的因子自动入库
print(f"发现 {engine.library.size()} 个因子")
# engine.library.seed_expressions(50)  → 下一轮种子
# engine.library.by_source('gp')       → GP 发现的因子
# engine.library.to_dataframe()        → 导出表格
```

### 4. 适应度策略

```python
from factor.v3 import (
    ICIRFitness,       # |IC_mean| × ICIR × 100（快筛）
    SharpeFitness,     # Pipeline 回测 Sharpe（真实绩效）
    MultiFreqFitness,  # ME/W/N天 多频率取最优
    make_fitness_calculator,
    FitnessMode,
)
from factor.v3 import FixedScheduler, IntervalScheduler, TopNEqualWeight

# 构建自定义适应度计算器（配合自定义调度器）
calc = SharpeFitness(data, future_returns, returns,
    scheduler=IntervalScheduler(5),      # 每5天调仓
    allocator=TopNEqualWeight(10),       # Top10 等权
)
```

### 5. 因子基类

```python
from factor.v3 import Factor, FactorMetadata, FactorCategory, FactorFrequency

class MomentumFactor(Factor):
    def __init__(self, lookback=20):
        meta = FactorMetadata(
            name='Momentum', category=FactorCategory.MOMENTUM,
            frequency=FactorFrequency.DAILY,
        )
        super().__init__(meta)
        self.lookback = lookback

    def calculate(self, data, symbols, dates):
        close = data['close']
        return close / close.shift(self.lookback) - 1
```

### 6. 因子检验

```python
from factor.v3 import FactorValidator

validator = FactorValidator(factor_values, future_returns)
ic = validator.information_coefficient()
# ic['ic_mean'], ic['ic_std'], ic['ir']
decay = validator.decay_rate(max_lookforward=20)
groups = validator.group_return(n_groups=10)
```

### 7. 回测管线

```python
from factor.v3 import FactorPipeline, FixedScheduler, IntervalScheduler, TopNEqualWeight

pipeline = FactorPipeline(
    returns=returns_df,
    scheduler=FixedScheduler('ME'),     # 月末调仓
    allocator=TopNEqualWeight(top_n=3),  # v3默认Top3等权
    cost_rate=0.0,                       # v3默认0手续费
)
result = pipeline.evaluate(factor_values)
# result.sharpe_ratio, result.max_drawdown, result.annual_return

# 调仓周期完全自定义：
# FixedScheduler('W')         周度
# IntervalScheduler(10)       每10个交易日
# IntervalScheduler(5)        每5个交易日

# 多频率对比
results = pipeline.compare_frequencies(fv, ['ME', 'W', '5D', '10D'])
```

### 8. 网格搜索 + BO 搜索

```python
from factor.v3 import FactorGridSearch, FactorBOSearch

# 网格搜索（lookback × freq × topN 全排列）
gs = FactorGridSearch(returns)
results = gs.search(
    factor_name="momentum",
    factor_fn=lambda lb: ...,       # (lookback) → DataFrame
    lookbacks=[20, 40, 60, 80, 120],
    freqs=['ME', 'W', '5D'],
    top_ns=[3, 5, 10],
)
best = gs.best(5)

# 贝叶斯优化
bo = FactorBOSearch(returns)
opt = bo.search("momentum", factor_fn,
    param_space=[(5, 200, 'lookback')],
    n_calls=30, freq='ME', top_n=3)
```

### 9. 因子组合

```python
from factor.v3 import EqualWeightCombiner, ExpandingICCombiner, cross_section_zscore

norm = cross_section_zscore(factor_values)

# 等权组合
ew = EqualWeightCombiner()
combined = ew.combine([('f1', fv1), ('f2', fv2)])

# 动态 IC 加权（无前瞻偏差）
icw = ExpandingICCombiner(min_periods=60)
combined_dyn = icw.combine([('f1', fv1), ('f2', fv2)], returns=returns_df)
```

### 10. 自增长因子库

```python
from factor.v3 import FactorLibrary, LibraryEntry

lib = FactorLibrary()
# 入库
lib.register(LibraryEntry('alpha001', formula, fitness=1.5, source='gp'))
# 批量
lib.register_batch(entries)
# 查询
lib.by_source('gp')           # GP 发现的因子
lib.by_category(FactorCategory.MOMENTUM)
lib.seed_expressions(50)      # 取 Top 50 作下一轮种子
lib.top(10, sort_by='sharpe') # 按指标排名
lib.to_dataframe()            # 导出
```

### 11. 公式库

```python
from factor.v3 import ALPHA101, ALPHA191

len(ALPHA101)  # 101
len(ALPHA191)  # 171（含20个预留基本面缺口）
```

---

## v3 架构速查

```
factor/v3/  (10 文件)
├── __init__.py         统一导出
├── base.py             FactorCategory / FactorMetadata / FactorLibrary
├── primitives.py       19 个时序/截面原语（纯 numpy，无状态）
├── engine.py           表达式引擎：Tokenizer → Parser → AST → FactorExpression
├── backtest.py         回测：Scheduler + Allocator + Combiner + Pipeline
├── formulas.py         WQ101 + GT191 公式字典
├── validator.py        IC/IR/Bootstrap/换手率 检验
├── search.py           网格搜索 + 贝叶斯优化
├── discover.py【核心】 GPEngine + 可插拔适应度 + FactorDiscoveryEngine
└── cache.py            因子值 Parquet 缓存
```

## GP 发现链路

```
formulas → engine(编译) → primitives(原语) → fitness(适应度) → gp(进化) → pipeline(验证) → library(入库)
```

---

## 数据格式约定

- **因子值面板：** `DataFrame`，index=日期，columns=股票代码，values=因子值
- **收益率面板：** 同上，values=日收益率
- **面板数据字典（表达式引擎）：** `Dict[str, ndarray(T,N)]`
- **适应度计算：** ICIR 模式仅需 `future_returns`；Sharpe/MultiFreq 需额外 `returns`

---

## v3 vs v2 关键默认值差异

| 参数 | v2 | v3 |
|------|----|----|
| 默认手续费 | 0.001 (0.1%) | **0.0** |
| 默认TopN | 5 | **3** |
| GP 适应度 | 固定 ICIR | **可插拔** |
| 调度器/分配器 | 硬编码 | **构造注入** |
| 因子库 | 无 | **自增长 FactorLibrary** |

---

## 注意事项
- v1 已废弃，v2 不再新增功能，**新开发全部走 v3**
- `from factor.v3 import ...` 是推荐实践
- `GPEngine` 替代了 v2 的 `FactorGPMiner`，API 不兼容
- `FactorDiscoveryEngine.run_pipeline()` 是迭代探索的推荐入口
- 调度器和分配器全部支持构造注入，无硬编码
- 种子表达式注入仍然是 GP 搜索的关键质量保障
- cost_rate=0.0 是 v3 默认值，如需计入交易成本请显式传入
