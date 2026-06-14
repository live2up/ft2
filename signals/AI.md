# signals 模块 AI 助手指南

> **版本：v4 | 更新日期：2026-06-12**

## 项目定位

择时信号 + 因子轮动研究模块。基于 Python `ast` 模块构建 DSL，LLM 原生友好。

## 版本说明

| 版本 | 状态 | 说明 |
|------|------|------|
| **v4** | **主力** | Python AST DSL, 67 原语, EngineCore, 探索灵活, `from signals.v4 import ...` |
| v3 | 保留 | 仅供历史测试对照，自研 Parser + FeatureSpace |
| v1/v2 | 已归档 | `signals/_archived/v1/`, `signals/_archived/v2/` |

## v4 核心变革

```
v3:  FeatureSpace.precompute → 37列特征表 → 自研Parser → 0/1信号 → EngineV3
     问题: Parser 400行, 特征表1200行, LLM需学自定义语法

v4:  Python ast.parse() → 67原语实时计算 → 连续/0/1信号 → EngineCore
     优势: 零自研Parser, 原生Python语法, LLM即学即用, 特征按需算, 探索灵活
```

## 架构总览

```
                         OHLCV DataFrame (CLOSE/OPEN/HIGH/LOW/VOLUME/AMOUNT)
                                          │
                   ┌──────────────────────┼──────────────────────┐
                   ▼                      ▼                      ▼
           signals/v4/              signals/v4/            signals/v4/
           ast_dsl.py               registry.py            engine.py
           ast.parse()              67 原语注册表           EngineCore
           安全白名单               ts_mean/rsi/atr/...    backtest(fast|full)
                   │                      │                      │
                   └──────────┬───────────┘                      │
                              ▼                                  │
                      signals/v4/                                │
                      expression.py                              │
                      Expression / evaluate_panel / rank_panel    │
                              │                                  │
                              └──────────────┬───────────────────┘
                                             ▼
                                    ┌──────────────┐
                                    │   ft2.core    │
                                    │ Engine/Account │
                                    │ Analyzer/      │
                                    │ Notebook       │
                                    └───────────────┘
```

## 目录结构

```
signals/
├── v4/                     # 主力 ←
│   ├── ast_dsl.py          # Python ast Parser + 安全校验
│   ├── registry.py         # 67 原语注册表
│   ├── expression.py       # Expression 类 (generate/evaluate_panel/rank_panel)
│   ├── engine.py           # EngineCore (fast/full 双模式)
│   ├── scoring.py           # 连续值打分 + 三区状态机
│   ├── market_breadth.py   # 市场宽度
│   ├── wf_result.py        # Walk-Forward 结果
│   └── __init__.py         # 统一导出
├── v3/                     # 保留 (仅供历史测试对照)
└── _archived/              # v1 + v2 已归档
    ├── v1/
    └── v2/
```

---

## 核心 API (v4)

### 0. 导入规范

```python
from signals.v4 import Expression, EngineCore

# 表达式 → 信号 → 回测 → 报告 (一站式)
expr = Expression("rsi(CLOSE, 14) > 0.3")
signal = expr.generate(ohlcv_df)
result = EngineCore.backtest(signal, ohlcv_df, mode='fast')
analyzer = EngineCore.backtest(signal, ohlcv_df, mode='full', bench_label='399317.SZ')
analyzer.to_notebook("V4 策略回测")
```

### 1. Expression — 信号表达式 (核心)

```python
from signals.v4 import Expression

# 单品种择时
expr = Expression("rsi(CLOSE, 14) > 0.3")
signal = expr.generate(ohlcv_df)  # → pd.Series

# 多品种因子轮动
panel = expr.evaluate_panel({'399967': df1, '399970': df2, ...})  # → DataFrame
ranked = expr.rank_panel({'399967': df1, '399970': df2, ...})     # → 每日截面排名

# 表达式自省
expr.variables    # ['CLOSE']
expr.functions    # ['rsi']
expr.complexity   # 5
```

### 2. 回测 — EngineCore

```python
from signals.v4 import EngineCore

# fast 模式: 搜索 (~0.5s/次)
r = EngineCore.backtest(signal, data, mode='fast', start_date='2020-01-01')
# → FastResult(sharpe=1.16, cagr=0.148, max_drawdown=-0.10, trades=112)

# full 模式: 验证 + 报告
analyzer = EngineCore.backtest(signal, data, mode='full',
                               start_date='2020-01-01', bench_label='399317.SZ')
analyzer.to_notebook("策略回测")
```

### 3. 表达式语法

```
# 数据源 (6个)
CLOSE  OPEN  HIGH  LOW  VOLUME  AMOUNT

# 时序统计 (25个)
ts_mean(x, w)      ts_std(x, w)       ts_sum(x, w)
ts_max(x, w)       ts_min(x, w)       ts_median(x, w)
ts_delta(x, w)     ts_delay(x, w)     ts_rank(x, w)
ts_corr(x, y, w)   ts_cov(x, y, w)    ts_skew(x, w)
ts_kurt(x, w)      ts_argmax(x, w)    ts_argmin(x, w)
ts_roc(x, w)       ts_zscore(x, w)    ts_av_diff(x, w)
ts_logret(x)       ts_var(x, w)
ts_decay_linear(x, w)    # 线性衰减加权
ts_product(x, w)         # 滚动乘积 (复利)
ts_regression(y, x, w, t)       # 回归: 0=β 1=α 2=残差 3=预测 4=R²
ts_regression_residual(x, w)    # 对时间做线性回归的残差

# 截面算子 (6个)
cs_rank(x)         cs_zscore(x)        cs_scale(x)
cs_winsorize(x)    cs_normalize(x)     cs_quantile(x)

# 扩张统计 (4个, 无前向偏差)
expanding_mean(x)        expanding_median(x)
expanding_std(x)         expanding_percentile(x, p)

# 特征计算 (21个, 从OHLCV实时算)
rsi(CLOSE, period)       atr(HIGH, LOW, CLOSE, period)
atr_sma(HIGH, LOW, CLOSE, period)   # SMA版ATR
macd(CLOSE, fast, slow, signal)     adx(HIGH, LOW, CLOSE, period)
cci(HIGH, LOW, CLOSE, period)       bb_width(CLOSE, period)
ema(CLOSE, period)       tsf(CLOSE, period)
kama(CLOSE, period)      trima(CLOSE, period)
wma(CLOSE, period)       dema(CLOSE, period)
hv(CLOSE, period)        natr(HIGH, LOW, CLOSE, period)
var(CLOSE, period)       linearreg(CLOSE, period)
stddev(CLOSE, period)    vol_ratio(CLOSE, VOLUME, short, long)
amt_ratio(AMOUNT, short, long)
wilder_smooth(x, period) # Wilder递推平滑

# 信号函数
persist(expr, n)         # 连续 n 日同向才触发
ts_scale(x, w)           # 滚动缩放到 ±1
ts_quantile(x, w)        # 滚动分位数排名

# 数学 (11个)
abs(x)  log(x)  sqrt(x)  sign(x)  exp(x)  tanh(x)
sigmoid(x)  relu(x)  signed_power(x, e)  safe_max(x,y)  safe_min(x,y)

# Python 原生运算符
+ - * / // % **            # 算术
> < >= <= == !=            # 比较
and or not                 # 逻辑
a if cond else b           # 三元
```

### 4. 典型表达式示例

```python
# ═══ 行业结构信号 (v35) ═══
# 需要预计算行业宽度指标注入 extra_features

# 宽度上升趋势 (ts_delay 方向检测)
"BW_MA5 > ts_delay(BW_MA5, 5) and ts_zscore(amt_ratio(AMOUNT,5,20), 60) > 0"

# 宽度 + 资金 OR 互补触发
"(BW_MA5 > 0.65 or AMT5 > 0.55) and ts_zscore(amt_ratio(AMOUNT,5,20), 60) > 0"

# 宽度上升 + NHL确认 (MDD极低)
"BW_MA5 > ts_delay(BW_MA5, 5) and NHL20 > 0 and ts_zscore(amt_ratio(AMOUNT,5,20), 60) > 0"

# 拥挤度负向过滤 (not 原语)
"(BW_MA5 > 0.65 or AMT5 > 0.55) and ts_zscore(amt_ratio(AMOUNT,5,20), 60) > 0 and not (CORR > 0.7)"

# 宽度上升 + 价格OR宽松确认
"(BW_MA5 > ts_delay(BW_MA5, 5)) and (CLOSE > OPEN or ts_roc(CLOSE, 3) > -0.02) and ts_zscore(amt_ratio(AMOUNT,5,20), 60) > 0"

# ═══ 趋势跟踪 ═══
"ema(CLOSE, 20) > ts_mean(CLOSE, 50)"

# 均值回归 (RSI超卖反弹)
"rsi(CLOSE, 14) < -0.3 and ts_roc(CLOSE, 5) > 0"

# 波动突破 (ATR放量 + 方向确认)
"atr(HIGH, LOW, CLOSE, 7) > ts_mean(atr(HIGH, LOW, CLOSE, 7), 30) and macd(CLOSE) > 0"

# 量价共振
"vol_ratio(CLOSE, VOLUME, 5, 20) > 1.5 and ts_roc(CLOSE, 10) > 0"

# 价格 Z-score (布林带位置)
"(CLOSE - ts_mean(CLOSE, 50)) / ts_std(CLOSE, 50)"

# 统计偏离
"ts_regression(CLOSE, ts_mean(CLOSE, 20), 60, 2) / ts_std(CLOSE, 60)"

# 多因子融合 (比率 + RSI + 量比)
"0.5 * ((CLOSE / ts_mean(CLOSE, 50) - 1) * 100) + 0.3 * rsi(CLOSE, 14) + 0.2 * amt_ratio(AMOUNT, 5, 20)"

# 信号确认 (ATR突破连续2日确认)
"persist(atr(HIGH, LOW, CLOSE, 7) > ts_mean(atr(HIGH, LOW, CLOSE, 7), 30), 2)"

# 状态切换 (趋势市追涨, 震荡市抄底)
"ema(CLOSE, 20) if adx(HIGH, LOW, CLOSE, 14) > 25 else rsi(CLOSE, 14)"

# 行业轮动因子
"(CLOSE / ts_mean(CLOSE, 50) - 1) / ts_std(CLOSE, 50)"
# → evaluate_panel() → cs_rank() → Top3
```

### 5. 临时自定义注册 (外部模板热添加)

```python
from signals.v4 import register_function, register_variable, Expression

# 注册自定义函数 (签名为 fn(*np.ndarray) -> np.ndarray)
def my_smooth(x, fast=5, slow=20):
    return np.convolve(x, np.ones(fast)/fast, 'same') - np.convolve(x, np.ones(slow)/slow, 'same')
register_function('my_smooth', my_smooth)

# 注册自定义变量前缀
register_variable('MY_VAR')

# 直接使用
expr = Expression("MY_VAR > 0 and my_smooth(BW_MA5, 5, 20) > 0")
signal = expr.generate(data, extra_features={'MY_VAR': ..., 'BW_MA5': ...})

# 注销 (清理)
from signals.v4 import unregister_function, unregister_variable
unregister_function('my_smooth')
unregister_variable('MY_VAR')
```

> **注意**: 注册是进程级全局操作，重复注册会覆盖并警告。推荐脚本顶部注册、末尾注销。

### 6. V4 原语覆盖 (vs 竞品)

| 类别 | WorldQuant | Hubble | AKQuant | ft2 V4 |
|------|:--:|:--:|:--:|:--:|
| 时序 | 24 | 4 | 12 | **25** |
| 截面 | 6 | 2 | 2 | **6** |
| 特征计算 | 0 | 0 | 0 | **21** |
| 扩张统计 | 0 | 0 | 0 | **4** |
| 数学/逻辑 | 26 | 4 | 7 | **11** |
| **总计** | **56** | **10** | **21** | **67** |

---

## 安全模型

```
ast.parse(expr) → 三层白名单校验:
  1. 节点类型白名单 (Import/Attribute/Lambda 等35种禁止)
  2. 函数名白名单 (67个注册原语 + 临时注册)
  3. 变量名白名单 (6个OHLCV + 行业变量前缀)
```

**LLM 生成 → ast.parse → 白名单校验 → 回测 → 报告，全链路安全。**

## 行业结构变量 (extra_features 注入)

| 变量 | 说明 | 典型用法 |
|------|------|---------|
| `BW_MA5/10/20` | 均线上方行业占比 | `BW_MA5 > 0.65` |
| `AMT5/10` | 成交额均线上方行业占比 | `AMT5 > 0.55` |
| `DISP` | 行业收益截面标准差 | `DISP < 0.01` (一致) |
| `SKEW` | 行业收益截面偏度 | `SKEW < -1` (负偏反转) |
| `NHL20` | 20日新高-新低净差 | `NHL20 > 0` |
| `CORR` | 行业平均两两相关性 | `CORR > 0.3` (系统性), `not (CORR > 0.7)` (拥挤过滤) |
| `ROTSPD` | 行业轮动速度 | 领涨切换频率 |
| `TAILUP/DOWN/NET` | 尾部极端行情密度 | 尾部风险信号 |
| `VMED/VDISP/VSKEW` | 成交量结构 | 资金流向 |
| `SECTOR_UP/AD` | 上涨比例/涨跌比 | 广度基础 |

---

## V3 → V4 语法迁移

| V3 | V4 |
|----|-----|
| `thr_0(ROC{5})` | `ROC_5 > 0` |
| `thr_mean(ATR{7})` | `ATR_7 > expanding_mean(ATR_7)` |
| `thr_roll_mean(EMA{20}, 30)` | `EMA_20 > ts_mean(EMA_20, 30)` |
| `expr1 & expr2` | `(expr1) and (expr2)` |
| `thr_zscore(EMA{20}, 60, 0.5)` | `EMA_20 > ts_mean(EMA_20, 60) + 0.5 * ts_std(EMA_20, 60)` |
| `persist(expr, 3)` | `persist(expr, 3)` (不变) |
| `if_then(cond, a, b)` | `a if cond else b` |
| V3 DSL 表达式 | Python 原生表达式 |

---

## 注意事项

- **v1/v2 已归档** (`signals/_archived/`)，不再使用
- **v3 保留**，仅供历史测试对照
- **v4 主力**：`from signals.v4 import Expression, EngineCore`，探索灵活
- FeatureSpace 仍存在于 v3，可作为兼容层（`extra_features=dict`传入），但不需要
- `evaluate_panel()` 确保索引对齐到 data 尾部（FeatureSpace 冷启动截断）
- `EngineCore.backtest(mode='fast')` 的 Sharpe 与 `mode='full'` 一致
- **滚动窗口预热**: `ts_zscore(60)` 等滚动函数要求满窗口才输出有效值（前59天为 NaN→0），避免早期样本不足时的假信号
- **自定义注册**: `register_function()` / `register_variable()` 进程级全局，推荐脚本顶部注册、末尾注销
- 67 原语覆盖 WorldQuant 时序算子的 96%，截面 100%
- 扩展新原语：写函数 → `register_function()` 注册 → LLM 即用（无需改 registry.py）
