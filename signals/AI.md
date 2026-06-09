# signals 模块 AI 助手指南

> **版本：v3 | 更新日期：2026-06-10**

## 项目定位

择时信号研究模块，探索、测试、回测全部走 `ft2.core.Engine`。

## 版本说明

| 版本 | 状态 | 说明 |
|------|------|------|
| **v3** | **主力** | 统一引擎 (`EngineV3`), full/fast 双模式, `from signals.v3 import ...` |
| v2 | 冻结 | 保留表达式引擎/特征工厂, 简化回测已弃用 (保留兼容) |
| v1 | 淘汰 | 不再维护, 不再引用 |

## v3 核心变革

v3 与 v2 的最大区别：**搜索和验证用同一把尺子**。

```
v2 问题:  GP/网格搜索 → 简化引擎 (虚高20%) → 发现信号 → core引擎验证 (真实)
          → 选出来的最优信号可能过不了验证关

v3 方案:  GPSearch/GridSearch → EngineV3(mode='fast') → 直接就是真实Sharpe
          → 验证 → EngineV3(mode='full') → AccountAnalyzer → to_notebook
          fast 和 full 同一 Engine.run() 时间线, 同一费率公式, Sharpe 一致
```

## 架构总览

```
                         数据源 (d2_api/TDX)
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
          signals/v2/      signals/v2/    signals/v3/
          features.py      expression     engine.py    ← 统一回测入口
          FeatureSpace      Expression    EngineV3
          (55+特征)         parse/生成     backtest(mode='fast|full')
                  │            │               │
                  └────────────┼───────────────┤
                               │               │
            ┌──────────────────┼───────────────┤
            ▼                  ▼               ▼
     ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
     │ v3/search/   │  │ v3/validate/  │  │ v3/scoring   │
     │              │  │              │  │              │
     │ GPSearch     │  │ single       │  │ Composite    │
     │ GridSearch   │  │ compare      │  │ Scorer       │
     │              │  │ walkforward  │  │ 3-zone       │
     └──────┬───────┘  └──────┬───────┘  └──────────────┘
            │                 │
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │   EngineV3       │
            │                  │
            │ fast → Engine    │
            │   .run() + 自管  │
            │   净值 (无记录)   │
            │                  │
            │ full → Engine    │
            │   .run() +       │
            │   AccountManager │
            │   → Analyzer     │
            └────────┬─────────┘
                     ▼
            ┌─────────────────┐
            │    ft2.core       │
            │ Engine/Account/   │
            │ Analyzer/Notebook │
            └──────────────────┘
```

## 目录结构

```
signals/
├── v1/                     # 淘汰 (不再维护)
├── v2/                     # 冻结 (特征工厂/表达式引擎/管线/打分 仍可用)
│   ├── features.py         # FeatureSpace (v3 继承)
│   ├── expression.py       # Expression (v3 继承)
│   ├── pipeline.py         # SignalPipeline (v3 继承)
│   ├── validator.py        # core_backtest / run_backtest_with_core (v2 桥接)
│   ├── scoring.py          # 多条件打分 (v3 继承)
│   ├── presets.py          # 表达式模板库
│   ├── registry.py         # 表达式注册
│   ├── explainer.py        # 白盒解释
│   ├── decay_monitor.py    # 衰减监控
│   ├── ic_analyzer.py      # IC 分析
│   └── ...
├── v3/                     # 主力 ←
│   ├── engine.py           # EngineV3: full/fast 双模式
│   ├── search/
│   │   ├── gp.py           # GPSearch (v2算法 + v3引擎)
│   │   └── grid.py         # GridSearch (v2算法 + v3引擎)
│   ├── validate/
│   │   ├── single.py       # 单信号验证
│   │   ├── compare.py      # 多信号对比
│   │   └── walkforward.py  # WF 验证
│   ├── scoring.py          # 连续值打分 (继承 v2)
│   ├── monitor/            # 解释/衰减/IC (待桥接)
│   └── __init__.py         # 统一导出
└── AI.md
```

---

## 核心 API (v3)

### 0. 导入规范

```python
# v3 统一入口 (推荐)
from signals.v3 import (
    EngineV3, FastResult,                    # 引擎
    GPSearch, GridSearch,                     # 搜索
    validate_single, compare_signals,         # 验证
    FeatureSpace, Expression,                 # 继承v2
    ScoredSignal, CompositeScorer,            # 打分
)

# 回测
result = EngineV3.backtest(signal, data, mode='fast')    # 搜索
analyzer = EngineV3.backtest(signal, data, mode='full')  # 验证
```

### 1. 特征工厂 (继承 v2, 不变)

```python
from signals.v3 import FeatureSpace, register_feature

fs = FeatureSpace()
features = fs.fit_transform(df)
# features.columns: ['ATR{7}', 'RSI{14}', 'EMA{20}', ...]
```

### 2. 表达式引擎 (继承 v2, 不变)

```python
from signals.v3 import Expression

expr = Expression("thr_mean(ATR{7})", fs)
signals = expr.generate(df)            # pd.Series (0.0/1.0)
# thr_0 / thr_mean / thr_med / thr_roll_mean
# & = AND  | = OR  persist(expr, n) = 连续确认
```

### 3. 回测 — EngineV3 (核心)

```python
from signals.v3 import EngineV3

# ── fast 模式: 搜索 (Engine.run() 时间线, 自管净值, 无记录) ──
r = EngineV3.backtest(signal, data, mode='fast', start_date='2020-01-01')
# → FastResult(sharpe=1.16, cagr=0.148, max_drawdown=-0.10, trades=112)

# ── full 模式: 验证 (Engine.run() + AccountManager) ──
analyzer = EngineV3.backtest(signal, data, mode='full',
                              start_date='2020-01-01',
                              bench_label='399317.SZ')
analyzer.to_notebook("策略回测")

# ── 费率控制 (指数择时默认不扣费) ──
EngineV3.backtest(signal, data, mode='fast', with_fees=False)  # 默认
EngineV3.backtest(signal, data, mode='fast', with_fees=True)   # ETF费率
```

**参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `signal` | — | pd.Series / np.ndarray / list, value>0 做多 |
| `data` | — | OHLCV DataFrame, index=DatetimeIndex |
| `symbol` | `'399317.SZ'` | 交易标的 |
| `initial_capital` | `1_000_000` | 初始资金 |
| `start_date` | `None` | 回测起始日 `'2020-01-01'` |
| `mode` | `'full'` | `'fast'` → FastResult, `'full'` → AccountAnalyzer |
| `with_fees` | `False` | 指数择时默认不扣费率 |
| `bench_label` | `None` | full 模式下基准标签 (自动跑 BenchHolder) |

**模式对比：**

| | full (验证) | fast (搜索) |
|---|---|---|
| 引擎 | `Engine.run()` | `Engine.run()` 同一 |
| 费率 | `fee_config` | `fee_config` 同一 |
| 时间安全 | eob 递进 + context.now | 同一 |
| 持仓管理 | `order_percent()` | 自管 cash/shares/price |
| 输出 | `AccountAnalyzer` | `FastResult` |
| 记录 | TradeRecord + snapshots | 无 |
| 速度/次 | ~3s | ~0.5s |
| GP 4000次 | 3.3h | 33min |

### 4. 搜索 — GPSearch

```python
from signals.v3 import GPSearch, FeatureSpace

fs = FeatureSpace().fit(data)
gs = GPSearch(fs, train_data, test_data,
              population_size=80, generations=50,
              start_date='2020-01-01')
gs.run()

# Top-N elite → full 验证
for ind in gs.elite_set(5):
    signal = ind.tree.evaluate(feature_data)
    analyzer = EngineV3.backtest(signal, data, mode='full')
    analyzer.to_notebook(f"GP: {ind.expression_str}")
```

### 5. 搜索 — GridSearch

```python
from signals.v3 import GridSearch

gs = GridSearch("thr_mean(ATR(?))",
                param_grid={'?': [7, 10, 14, 20]},
                data=data, feature_space=fs,
                start_date='2020-01-01')
df = gs.run(mode='fast')
# df.sort_values('Sharpe', ascending=False).head(5)
```

### 6. 验证 — 单信号

```python
from signals.v3 import validate_single

analyzer = validate_single(signal, data, start_date='2020-01-01',
                            bench_label='399317.SZ')
analyzer.to_notebook("策略回测")
```

### 7. 验证 — 多信号对比

```python
from signals.v3 import compare_signals

df = compare_signals([
    {'name': 'ATR-TSF', 'expr': 'thr_mean(ATR{7}) - thr_mean(TSF{7})'},
    {'name': '量波比', 'expr': 'thr_mean(VOL_RATIO{5,20}) div thr_med((ATR{3} sub TSF{7}))'},
], data=data, start_date='2020-01-01')
# df columns: 排名, 名称, Sharpe, 年化, 最大回撤, 交易, 胜率
```

### 8. 验证 — Walk-Forward

```python
from signals.v3 import walkforward_validate

wf = walkforward_validate(signal, data, symbol='399317.SZ',
                           train_size='2Y', test_size='1Y', step='6M')
# wf.summary['mean_test_sharpe']
# wf.summary['stability_score']
# wf.summary['negative_count']
```

### 9. 多条件打分 (继承 v2)

```python
from signals.v3 import ScoredSignal, CompositeScorer

signals = [
    ScoredSignal('CLOSE_MA_RATIO{10}', weight=0.4),
    ScoredSignal('MOM_CHAIN{10}', weight=0.3, transform='zscore'),
]
scorer = CompositeScorer(signals, features)
score = scorer.compute()
```

---

## 表达式语法速查

```
特征引用:     ATR{7}, RSI{14}, MACD{12,26,9}    ← {} 引用列
阈值函数:     thr_0(x)          x > 0 → 1
             thr_mean(x)       x > mean(x) → 1
             thr_med(x)        x > median(x) → 1
             thr_roll_mean(x,w)  x > rolling_mean(w) → 1
             thr_zscore(x,w,k)   布林带式突破
             thr_pct(x,p)       x > 历史 p 分位数 → 1
二元运算:     expr1 & expr2     AND
             expr1 | expr2     OR
             expr1 + expr2      加法组合
             -expr              反转
信号确认:     persist(expr, n)   连续 n 日同向才触发
完整示例:     persist(thr_mean(ATR{7}) & thr_mean(TRIMA{60}), 3)
```

---

## v2 → v3 迁移速查

| v2 用法 | v3 用法 |
|---------|---------|
| `from signals.v2 import run_backtest_with_core` | `from signals.v3 import EngineV3` |
| `run_backtest_with_core(expr, data)` | `EngineV3.backtest(signal, data, mode='full')` |
| `from signals.v2 import run_backtest` | **弃用, 无替代 — 用 EngineV3 fast 模式** |
| `from signals.v2 import GPOptimizer` | `from signals.v3 import GPSearch` |
| `from signals.v2 import GridSearch` | `from signals.v3 import GridSearch` |
| `from signals.v2 import FeatureSpace, Expression` | `from signals.v3 import FeatureSpace, Expression` (同) |

---

## 注意事项

- **v1 淘汰，不再引用**
- **v2 冻结**：特征工厂/表达式引擎/管线/打分/解释器等纯计算模块仍可用，回测部分建议迁移 v3
- **v3 主力**：`from signals.v3 import ...`
- `EngineV3.backtest(mode='fast')` 的 Sharpe 与 `mode='full'` 一致，搜索出来直接就是真的
- 指数择时默认 `with_fees=False`
- FastResult 不含 TradeRecord，需要交易明细请用 `mode='full'`
- FeatureSpace 一次 fit，多次使用（传给 Expression、GP、GridSearch）
