# factor 模块 - AI 快速上手

> 因子挖掘体系

> **版本：v3.2.0 | 更新日期：2026-06-05**
>
> **AI 助手注意：** 如果发现实际 API 与本文档不一致，说明源码已更新但 AI.md 未同步，请提醒用户更新。

---

## 版本说明

- **v3/** — 维护中，**推荐使用**。GP 发现引擎 + 可插拔适应度 + 迭代探索 + 自增长因子库 + 持久化存档
- **v2/** — 不再更新。保留 GP 挖掘、表达式引擎、回测管线，但无持久化
- **v1/** — 已归档，仅作历史参考

---

## 核心 API (v3)

> v3 构建一个**可持续的因子发现流水线**：
> ```
> 种子注入(formulas) → 多策略GP并行(IC/Sharpe/多频率) → Pipeline验证 → 入库 → 持久化(discovered/)
>                                                                   ↓
>                                                          扩大的种子池 → 下一轮
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
            ★ 自定义变量: 任何在 data 字典中的字段都可作为终端变量
               (如 rel_close, share, downside_vol, rel_volume, rel_amount)
一元函数:   abs(x), sqrt(x), log(x), exp(x), neg(x), sign(x), tanh(x)
二元函数:   add(x,y), sub(x,y), mul(x,y), div(x,y), max(x,y), min(x,y)
时序原语:   ts_rank(x, period)      ts_zscore(x, period)
            delay(x, period)         delta(x, period)  差分(省1层)
            decay_linear(x, period)  ts_sum/mean/std/max/min(x, p)
            sma(x, period, lag)      correlation/covariance/regbeta(x,y,p)
            ts_argmin/max(x, p)      ts_skew(x, period)  偏度
截面原语:   cs_rank(x)              cs_zscore(x, period)
            cs_mean(x)              signed_power(x, exp)
健壮处理:   winsorize(x, n)         截尾到 ±nσ
语法糖:     ret(period)             收益差（省2层）
            adv(period)             均量（省1层）
            intra_ret               日内收益（(c-o)/o）
条件分支:   ifelse(cond, a, b)

示例: ts_rank(winsorize(mul(delta(close, 20), adv(10)), 3), 10)
自定义: ts_rank(rel_close, 20)      # rel_close 预计算后注入 panel_dict
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
    save_dir='./discovered',     # [新增] GP 结果持久化目录
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

# 每轮结束后自动保存到 save_dir/gp_round_001.json
print(f"发现 {engine.library.size()} 个因子")
# engine.library.seed_expressions(50)  → 下一轮种子
# engine.library.by_source('gp')       → GP 发现的因子
# engine.library.by_time('2026-06-01') → 按时间筛选
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

### 10. 自增长因子库 + 持久化

```python
from factor.v3 import FactorLibrary, LibraryEntry

lib = FactorLibrary()

# 入库 — created_at 自动打时间戳
lib.register(LibraryEntry('alpha001', formula, fitness=1.5, source='gp'))
lib.register_batch(entries)

# 查询
lib.by_source('gp')              # GP 发现的因子
lib.by_category(FactorCategory.MOMENTUM)
lib.by_round(3)                  # 第3轮发现的
lib.by_time('2026-06-01', '2026-06-04')  # [新增] 按时间范围筛选
lib.seed_expressions(50)         # 取 Top 50 作下一轮种子
lib.top(10, sort_by='sharpe')    # 按指标排名
lib.to_dataframe()               # 导出（含 created_at 列）

# [新增] 持久化：保存到磁盘
lib.save('./discovered/gp_round_001.json')

# [新增] 持久化：从磁盘加载
lib2 = FactorLibrary.load('./discovered/gp_round_001.json')
```

### 11. 发现结果存档

```python
from factor.v3 import load_discovered, merge_discovered

# [新增] 加载 discovered/ 目录下所有 .json 文件
lib = load_discovered('./discovered')
# 自动扫描 gp_round_001.json, gp_round_002.json, ...

# [新增] 合并到已有因子库
existing_lib = FactorLibrary()
n = merge_discovered(existing_lib, './discovered')
print(f"合并了 {n} 个新因子")

# 继续探索：用合并后的因子库做种子
seeds = existing_lib.seed_expressions(100, sort_by='sharpe')
```

### 12. 公式库

```python
from factor.v3 import ALPHA101, ALPHA191, BASIC_FACTORS

len(ALPHA101)       # 101   WorldQuant 101 Alpha → formulas/wq101.py
len(ALPHA191)       # 171   国泰安 191 Alpha（含20个预留基本面缺口）→ formulas/gt191.py
len(BASIC_FACTORS)  # 20    因子原子基元 → formulas/basic.py

# 总计: 292 条公式，全部可解析求值
# 旧 import from factor.v3.formulas 仍兼容
```

---

## v3 架构速查

```
factor/v3/
├── __init__.py         统一导出
├── base.py             FactorCategory / FactorMetadata / FactorLibrary (save/load)
├── primitives.py       23 个原语（时序+截面+健壮+语法糖）
├── engine.py           表达式引擎：Tokenizer → Parser → AST → FactorExpression
├── backtest.py         回测一体化：Scheduler + Allocator + Combiner + Pipeline
├── formulas/           公式数据库（拆分为子文件）
│   ├── __init__.py     兼容 from factor.v3.formulas import ...
│   ├── wq101.py        WorldQuant 101 Alpha (101条)
│   ├── gt191.py        国泰安 191 Alpha (171条)
│   └── basic.py        因子原子基元 (20条)
├── discovered/         GP/ML 发现因子存档
│   ├── __init__.py     load_discovered / merge_discovered
│   └── gp_round_*.json 运行时自动生成（持久化）
├── discover.py【核心】 GPEngine + 可插拔适应度 + FactorDiscoveryEngine + 自动落盘
├── validator.py        IC/IR/Bootstrap/换手率 检验
├── search.py           网格搜索 + 贝叶斯优化
└── cache.py            因子值 Parquet 缓存
```

## GP 发现链路（含持久化）

```
formulas/ → engine(编译) → primitives(原语) → fitness(适应度) → gp(进化)
                                                                     ↓
                                             save_dir/gp_round_N.json ←── 入库
                                                                     ↓
                                             下次会话 load_discovered() → 扩大的种子池 → 持续探索
```

---

## 数据格式约定

- **因子值面板：** `DataFrame`，index=日期，columns=股票代码，values=因子值
- **收益率面板：** 同上，values=日收益率
- **面板数据字典（表达式引擎）：** `Dict[str, ndarray(T,N)]`
- **适应度计算：** ICIR 模式仅需 `future_returns`；Sharpe/MultiFreq 需额外 `returns`

---

## v3 vs v2 关键差异

| 参数 | v2 | v3 |
|------|----|----|
| 默认手续费 | 0.001 (0.1%) | **0.0** |
| 默认TopN | 5 | **3** |
| GP 适应度 | 固定 ICIR | **可插拔** |
| 调度器/分配器 | 硬编码 | **构造注入** |
| 因子库 | 无 | **自增长 FactorLibrary** |
| 持久化 | 无 | **save_dir 自动落盘 + load_discovered 恢复** |
| 时间戳 | 无 | **created_at + by_time() 按时间查询** |
| 公式组织 | 单文件 formulas.py | **formulas/ 子包 (wq101/gt191/basic)** |

---

## 配套工具：Notebook 可视化报告

因子验证结果可通过 `notebook` 模块生成交互式 HTML 报告（IC/IR 指标卡片、分组收益图、因子权重饼图等）。

详见 [`notebook/AI.md`](../notebook/AI.md)

---

## 注意事项

- v1 已归档，v2 不再更新，**新开发全部走 v3**
- `from factor.v3 import ...` 是推荐实践
- `GPEngine` 替代了 v2 的 `FactorGPMiner`，API 不兼容
- `FactorDiscoveryEngine.run_pipeline()` 是迭代探索的推荐入口，传入 `save_dir` 启用自动持久化
- 调度器和分配器全部支持构造注入，无硬编码
- 种子表达式注入仍然是 GP 搜索的关键质量保障
- cost_rate=0.0 是 v3 默认值，如需计入交易成本请显式传入
- `discovered/` 目录的 .json 文件可跨会话复用，重启不丢失

## [新增 v3.2] 自定义终端变量

表达式引擎现在支持任意预计算变量作为终端。只需将变量以 `(T,N)` ndarray 形式放入 `data` 字典，即可在表达式中按名引用。

```python
# 在 data 字典中注入自定义变量
panel_dict['rel_close'] = close_arr / bench_close_arr   # 相对基准价格
panel_dict['share'] = amount_arr / amount_arr.sum(axis=1, keepdims=True)  # 资金占比
panel_dict['downside_vol'] = rolling_std(neg_ret, 20)    # 下行波动率

# 表达式直接引用（无需 register_terminal）
expr = FactorExpression("ts_rank(rel_close, 20)")
values = expr.evaluate(panel_dict)  # ✅ 自动从 data 中查找 rel_close
```

**GP 自动发现层面：** `GPEngine` 会自动检测 `data` 中的非标准字段，将其作为终端变量纳入随机树生成。`FactorDiscoveryEngine` 通过 `custom_terminals` 参数可显式控制。

**无需手动注册：** 变量查找优先级为 `VARIABLE_MAP` → `data` 字典直接查找，即放即用。
