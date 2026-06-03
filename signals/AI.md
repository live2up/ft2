# signals 模块 AI 助手指南

> **版本：v2.1 | 更新日期：2026-06-03**

## 项目定位
择时信号研究模块，专注计算，不负责输出。

## 版本说明
- **v2.1** — 当前主力（表达式引擎 + core 引擎回测 + Walk-Forward 稳定性分析）
- **v2.0** — 内置简化回测（`run_backtest` / `BacktestResult`），仍可用但建议迁移到 v2.1
- **v1/** — 前期版本，已处于非维护状态

## v2.1 vs v2.0 关键区别

| 特性 | v2.0（兼容保留） | v2.1（推荐） |
|------|-----------------|-------------|
| 回测引擎 | 内置简化回测 (`run_backtest`) | core 引擎 (`run_backtest_with_core`) |
| 回测返回值 | `BacktestResult` (9个标量) | `AccountAnalyzer` (全套指标+链式输出) |
| Walk-Forward | `walk_forward` (训练/测试 Sharpe) | `walk_forward_with_core` (12项指标+稳定性) |
| 报告输出 | 手动拼接 | `analyzer.set_benchmark(...).to_notebook(...)` |

## 架构原则
- **signals/ 专注计算**：信号生成、回测、IC分析
- **输出委托 core.analyzer**：`run_backtest_with_core` 返回 `AccountAnalyzer`，链式调用 `.to_notebook()`
- **禁止跨版本依赖**：v1 与 v2 各自独立

## 目录结构
```
signals/
├── v1/                     # 已归档（TA-Lib 类信号 + 融合器 + 回测）
├── v2/                     # 当前主力 (v2.1)
│   ├── features.py         # 特征工厂（55+ 纯函数 + 声明式配置 → 特征矩阵）
│   ├── expression.py       # 表达式引擎（Tokenizer → Parser → AST → 信号序列）
│   ├── pipeline.py         # 管线编排（多阶段信号处理链）
│   ├── validator.py        # 回测验证（run_backtest / run_backtest_with_core / walk_forward）
│   ├── walk_forward_v2.py  # [v2.1] Walk-Forward (core 引擎版, 全套指标+稳定性分析)
│   ├── scoring.py          # 多条件打分（加权合成 + 三区状态机）
│   ├── grid_search.py      # 参数网格搜索
│   ├── gp_optimizer.py     # 遗传算法优化
│   ├── pysr_adapter.py     # PySR 符号回归适配器
│   ├── ic_analyzer.py      # IC 分析器
│   ├── presets.py          # 表达式模板库
│   ├── registry.py         # 表达式注册与发现
│   ├── explainer.py        # 白盒子解释引擎
│   ├── decay_monitor.py    # 因子衰减监控
│   ├── market_breadth.py   # 市场广度特征
│   ├── timeframe.py        # 多周期特征计算
│   └── __init__.py         # 模块入口
└── AI.md
```

## 核心 API (v2)

### 1. 特征工厂
```python
from signals.v2 import FeatureSpace, register_feature

# 默认配置：55+ 技术指标
fs = FeatureSpace()
features = fs.fit_transform(df)
# features.columns: ['ATR{7}', 'RSI{14}', 'EMA{20}', ...]

# 自定义配置
config = {
    'features': {'my_cat': ['RSI(period=[5,14])']},
    'regime': True,           # 追加市场状态特征
    'market_breadth': False,   # 追加市场广度特征（需 advance/decline 列）
    'normalize': True,
}
fs2 = FeatureSpace(config=config)
```

### 2. 表达式引擎
```python
from signals.v2 import Expression

# 字符串 → 自动编译 → 信号
expr = Expression("thr_mean(ATR{7}) & thr_mean(TRIMA{60})", fs)
signals = expr.generate(df)            # pd.Series (0.0/1.0)

# {} 引用特征列，() 函数调用
# thr_0 / thr_mean / thr_med / thr_roll_mean / thr_zscore / thr_pct / thr_range
# & = AND  | = OR  persist(expr, 3) = 连续3日确认
signals2 = Expression("persist(thr_mean(ATR{7}), 3)", fs).generate(df)
```

### 3. 回测

#### 3a. v2.1 core 引擎回测（推荐）

```python
from signals.v2 import run_backtest_with_core, Expression, FeatureSpace

fs = FeatureSpace()
expr = Expression("thr_mean(ATR{7})", fs)

# 用 core 引擎回测 → 返回 AccountAnalyzer（全套指标 + 链式输出）
analyzer = run_backtest_with_core(expr, df, symbol='399317.SZ')

# 链式调用：注入基准 → 输出报告 → 获取指标
analyzer.set_benchmark(bench_nav, '买入持有') \
        .to_notebook("ATR突破策略")

# 单独获取指标
analyzer.sharpe_ratio()        # → 0.57
analyzer.max_drawdown()        # → (0.185, peak_date, trough_date)
analyzer.win_rate()            # → 0.52
analyzer.volatility()          # → 0.125

# 信号输入支持多种格式
run_backtest_with_core(expr, df)                 # Expression 实例
run_backtest_with_core(signal_series, df)        # pd.Series
run_backtest_with_core(signal_array, df)         # np.ndarray
run_backtest_with_core(lambda d: ..., df)        # callable
```

**参数：**
| 参数 | 默认 | 说明 |
|------|------|------|
| `expr_or_signals` | — | Expression / Series / ndarray / callable |
| `data` | — | OHLCV DataFrame (index=DatetimeIndex) |
| `symbol` | `'399317.SZ'` | 交易标的代码 |
| `freq` | `'1d'` | 数据频率 |
| `initial_capital` | `1_000_000` | 初始资金 |
| `long_only` | `True` | 仅做多（当前不支持做空） |
| `note_fields` | `None` | 附加 bar 字段写入 TradeRecord.note |

#### 3b. v2.0 内置简化回测（兼容保留）

```python
from signals.v2 import run_backtest, walk_forward

# 单次回测 → BacktestResult（9个标量指标）
result = run_backtest(expr, df, initial_capital=1e6)
# result.total_return, result.sharpe, result.max_drawdown, result.win_rate, ...

# Walk-Forward → 每窗口仅 train/test Sharpe
wf = walk_forward(expr, df, train_size='2Y', test_size='1Y', step='6M')
# wf.folds: [{train_sharpe, test_sharpe, ...}, ...]
```

#### 3c. v2.1 Walk-Forward（core 引擎版）

```python
from signals.v2 import walk_forward_with_core, WalkForwardCoreResult

wf = walk_forward_with_core(expr, df, symbol='399317.SZ',
                            train_size='2Y', test_size='1Y', step='6M')

# 汇总属性
wf.stability_score      # 夏普稳定性 = mean/sigma
wf.overfit_ratio        # 过拟合比 = train_sharpe_mean / test_sharpe_mean
wf.negative_count       # 负夏普窗口数
wf.train_sharpes        # List[float] 训练集各窗夏普
wf.test_sharpes         # List[float] 测试集各窗夏普

# 报告方法
wf.to_dataframe()       # 窗口级指标表 (train/test 各12项)
wf.stability_report()   # 指标稳定性: {sharpe: {mean, std, cv}, ...}

# 特殊属性
wf.overfit_ratio        # >2.0 强烈过拟合
wf.stability_score      # 越高越稳定，<0.5 不稳定

# 判断标准
if wf.overfit_ratio > 1.5 and wf.stability_score < 0.5:
    print("⚠ 过拟合高风险")
elif wf.negative_count > 0:
    print("⚠ 部分窗口亏损")
else:
    print("✓ 稳健")
```

### 4. IC 分析
```python
from signals.v2 import ICAnalyzer

signals = expr.generate(df)
ic = ICAnalyzer.analyze(signals, df['close'])
# ic['basic']['summary']['ic_mean']     IC 均值
# ic['significance']['p_value']         显著性
# ic['decay']['decay_rate']             衰减率
# ic['annual']                          年度分解
# ic['turnover']['mean_turnover']       换手率
# ic['distribution']['is_normal']       IC 正态性
```

### 5. 管线编排
```python
from signals.v2 import SignalPipeline, pipe_and, pipe_or

pipeline = SignalPipeline([
    ("signal", expr),
    ("persist", 3),           # 连续3日确认
    ("threshold", lambda x: x > 0.5),
])
signals = pipeline.generate(df)

# 快捷组合
signals = pipe_and(expr1, expr2).generate(df)     # 两个信号取 AND
signals = pipe_or(expr1, expr2).generate(df)      # 两个信号取 OR
```

### 6. 多条件打分
```python
from signals.v2 import ScoredSignal, CompositeScorer, three_zone_backtest

signals = [
    ScoredSignal('CLOSE_MA_RATIO{10}', weight=0.4),
    ScoredSignal('MOM_CHAIN{10}', weight=0.3, transform='zscore'),
]
scorer = CompositeScorer(signals, features)
score = scorer.compute()
bt = three_zone_backtest(score, df['close'], entry_thr=0.3, exit_thr=-0.3)
```

### 7. 参数网格搜索
```python
from signals.v2 import GridSearch

gs = GridSearch("thr_mean(ATR(?))", fs, param_grid={'?': [7, 10, 14, 20]})
results = gs.run(df)
# results.top(n=5, by='sharpe')  → 最佳参数排序
```

### 8. 遗传算法优化
```python
from signals.v2 import GPOptimizer

gp = GPOptimizer(fs, train_data=df_train, population_size=500, generations=50)
best = gp.run()
# best.expression_str  → 最优表达式
# best.train_sharpe    → 训练集夏普
# gp.report()          → 代际收敛报告
```

### 9. 因子衰减监控
```python
from signals.v2 import DecayMonitor, check_decay

decay = check_decay(expr, df)
# decay.alert_level  → NORMAL / DECAYING / DEAD / REVERSED
# decay.rolling_ic   → 滚动 IC 序列
```

### 10. 白盒子解释
```python
from signals.v2 import Explainer

report = Explainer.explain(expr, fs, df)
# 在 12 种市场状态（vol×trend×volume）下切片分析信号表现
```

### 11. 多周期特征
```python
from signals.v2 import compute_multitimeframe_features

features = compute_multitimeframe_features(fs, df, timeframes={'W': 'W-FRI'})
# 第1组: ATR{7}, RSI{14}, ...        (日线)
# 第2组: W_ATR{7}, W_RSI{14}, ...    (周线)
```

### 12. 市场广度特征（需外部数据）
```python
# DataFrame 需预先包含：advance, decline, new_highs, new_lows 等列
config = {'market_breadth': True}
fs = FeatureSpace(config=config)
features = fs.fit_transform(df)
# 自动追加: ADV_DEC_RATIO, NEW_HIGH_LOW, MCCLELLAN_OSC, ARMS_INDEX
```

---

## BacktestResult 字段（v2.0 简化回测）
| 类型 | 字段 |
|------|------|
| 收益 | total_return, annual_return, excess_return |
| 风险 | max_drawdown, annual_vol, downside_vol |
| 比率 | sharpe, sortino, calmar, information_ratio |
| 交易 | trade_count, win_rate, profit_loss_ratio |

> **v2.1**: `run_backtest_with_core` 返回 `AccountAnalyzer`（见 `core/AI.md` 第3章），内置 20+ 指标，支持链式 `.to_notebook()/.to_excel()`。

## WalkForwardCoreResult 字段（v2.1）

| 属性 | 类型 | 说明 |
|------|------|------|
| `windows` | `List[Dict]` | 每窗口 train/test 12项指标 |
| `summary` | `Dict` | 汇总统计 |
| `train_sharpes` | `List[float]` | 训练集各窗夏普 |
| `test_sharpes` | `List[float]` | 测试集各窗夏普 |
| `stability_score` | `float` | 夏普稳定性 = mean/sigma |
| `overfit_ratio` | `float` | 过拟合比 = train_mean / test_mean |
| `negative_count` | `int` | 负夏普窗口数 |
| `to_dataframe()` | `DataFrame` | 窗口级指标表 |
| `stability_report()` | `Dict` | 12项指标均值/标准差/CV |

## IC分析结果结构
```python
ic_result = {
    'basic': {
        'summary': {'ic_mean': ..., 'ic_std': ..., 'ic_ir': ..., 'ic_positive_ratio': ...},
        'IC_1d': {'pearson_ic': ..., 'rank_ic': ...},
        'IC_5d': {...}, 'IC_20d': {...}, ...
    },
    'rolling': {'window_30d': {...}, 'window_60d': {...}, 'window_120d': {...}},
    'decay': {'data': [{'holding_period': ..., 'pearson_ic': ...}, ...], 'decay_rate': ...},
    'significance': {'pearson_ic': ..., 't_statistic': ..., 'p_value': ..., 'significant': ...},
    'annual': {2024: {'pearson_ic': ..., 'sample_size': ...}, ...},
    'cumulative': {'series': ..., 'final_value': ...},
    'turnover': {'mean_turnover': ..., 'max_turnover': ...},
    'distribution': {'mean': ..., 'skewness': ..., 'is_normal': ..., 'percentile_5': ...},
}
```

## 表达式语法速查
```
特征引用:     ATR{7}, RSI{14}, MACD{12,26,9}     ← {} 引用列
阈值函数:     thr_0(x)        x > 0 → 1
             thr_mean(x)     x > mean(x) → 1
             thr_med(x)      x > median(x) → 1
             thr_roll_mean(x, w)    x > rolling_mean(w) → 1
             thr_zscore(x, w, k)    布林带式突破
             thr_pct(x, p)     x > 历史 p 分位数 → 1
             thr_range(x, lo, hi)    区间过滤
二元运算:     expr1 & expr2   AND
             expr1 | expr2   OR
             -expr           反转
信号确认:     persist(expr, n)   连续 n 日同向才触发
条件分支:     if_then(cond, a, b) / regime_switch()

完整示例:     persist(thr_mean(ATR{7}) & thr_mean(TRIMA{60}), 3)
```

## 注意事项
- v1 非维护状态，新开发全部走 v2
- **推荐使用 v2.1 API**：`run_backtest_with_core` + `walk_forward_with_core`
- v2.0 的 `run_backtest` / `walk_forward` 保留兼容，但不建议新项目使用
- `from signals.v2 import ...` 是最佳实践
- 不要直接 import `signals`（会同时触发 v1 和 v2）
- FeatureSpace 一次 fit，多次使用（传给 Expression、GP、GridSearch、回测）
- `run_backtest_with_core` 返回 `AccountAnalyzer`，可直接 `.to_notebook()` 输出 HTML 报告
- 外部数据（行业/广度/日内）通过 DataFrame 附加列注入，不影响框架
- v2.1 Walk-Forward 每窗口重置 FeatureSpace 防前瞻偏差
