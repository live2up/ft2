# factor 模块 - AI 快速上手

> 因子挖掘体系
>
> **版本：v2.0.0 | 更新日期：2026-05-28**
>
> **AI 助手注意：** 如果发现实际 API 与本文档不一致，说明源码已更新但 AI.md 未同步，请提醒用户更新。

---

## 版本说明
- **v2/** — 当前主要使用版本（Pipeline + 表达式引擎 + GP 符号回归 + 网格/BO 搜索）
- **v1/** — 前期版本，已处于非维护状态

---

## 核心 API (v2)

### 1. 因子表达式引擎

```python
from factor.v2 import FactorExpression

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
截面原语:   cs_rank(x)              cs_zscore(x, period)
            signed_power(x, exp)

示例: ts_rank(mul(cs_zscore(sub(close, delay(close, 20))), abs(close)), 10)
```

### 2. 因子 GP 矿机

```python
from factor.v2 import FactorGPMiner

gp = FactorGPMiner(
    data=panel_dict,            # {col: ndarray(T,N)}
    future_returns=returns_df,  # DataFrame(T,N)
    population_size=300,
    generations=30,
    max_depth=8,
    seed_expressions=[           # 已知好因子注入
        "ts_rank(sub(close, delay(close, 20)), 10)",
    ],
)
gp.run()
best = gp.best()
# best.expression_str  →  "ts_rank(mul(...), 10)"
# best.ic_mean         →  0.045
# best.icir            →  0.72

print(gp.report())
```

### 3. 因子基类

```python
from factor.v2 import Factor, FactorMetadata, FactorCategory, FactorFrequency

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

### 4. 因子检验

```python
from factor.v2 import FactorValidator

validator = FactorValidator(factor_values, future_returns)
ic = validator.information_coefficient()
# ic['ic_mean'], ic['ic_std'], ic['ir']
decay = validator.decay_rate(max_lookforward=20)
groups = validator.group_return(n_groups=10)
```

### 5. 回测管线

```python
from factor.v2 import FactorPipeline, FixedScheduler, TopNEqualWeight

pipeline = FactorPipeline(
    returns=returns_df,
    scheduler=FixedScheduler('ME'),     # 月末调仓
    allocator=TopNEqualWeight(top_n=5),  # Top5等权
)
result = pipeline.evaluate(factor_values)
# result.sharpe_ratio, result.max_drawdown, result.annual_return
```

### 6. 网格搜索

```python
from factor.v2 import FactorGridSearch, FactorGridConfig

config = FactorGridConfig(
    factor_class=MomentumFactor,
    param_grid={'lookback': [20, 40, 60, 80, 120]},
    freq_grid=['ME', 'W', '5D'],
    topn_grid=[5, 10, 15],
)
gs = FactorGridSearch(config, data, returns)
results = gs.run()
best = gs.best()  # 最优参数组合
```

### 7. GP 原语

```python
from factor.v2 import ts_rank, ts_zscore, delay, cs_rank

# 所有原语操作 (T,N) ndarray
ranked = ts_rank(factor_values, period=20)
shifted = delay(factor_values, period=5)
cross = cs_rank(factor_values)
```

### 8. 因子组合

```python
from factor.v2 import EqualWeightCombiner, ExpandingICCombiner, cross_section_zscore

# 截面标准化
norm = cross_section_zscore(factor_values)

# 等权组合
ew = EqualWeightCombiner()
combined = ew.combine([('f1', fv1), ('f2', fv2)])

# 动态 IC 加权
icw = ExpandingICCombiner(lookback=60)
combined_dyn = icw.combine([('f1', fv1), ('f2', fv2)])
```

---

## 数据格式约定

**因子值面板：**
- `DataFrame`，index=日期，columns=行业/股票代码，values=因子值

**收益率面板：**
- 格式同上，values=未来收益率

**面板数据字典（表达式引擎）：**
- `Dict[str, ndarray(T,N)]`，如 `{'close': arr(T,N), 'volume': arr(T,N), ...}`

---

## 注意事项
- v1 非维护状态，新开发全部走 v2
- `from factor.v2 import ...` 是最佳实践
- FactorExpression 的 `evaluate()` 接受 `Dict[str, ndarray]` 格式
- FactorGPMiner 的适应度基于每日截面 IC，非回测 Sharpe
- 种子表达式注入是 GP 搜索的关键质量保障
