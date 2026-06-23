# signals 模块 AI 助手指南

> **版本：v4 | 更新日期：2026-06-22**

## 项目定位

择时信号 + 因子轮动研究模块。基于 Python `ast` 模块构建 DSL，LLM 原生友好。

## 版本说明

| 版本 | 状态 | 说明 |
|------|------|------|
| **v4** | **主力** | Python AST DSL + `utils/ast` 公共基础设施, 72 原语, SigEngine |
| v3 | 保留 | 仅供历史测试对照，自研 Parser + FeatureSpace |

## v4 核心架构

```
v4:  Python ast.parse() → 72 原语实时计算 → 连续/0/1信号 → SigEngine
                            │
                    utils/ast (公共基础设施)
                     ├── functions.py  (72原语注册表)
                     ├── variables.py  (70+变量前缀白名单)
                     ├── dsl.py        (AST解析/求值/安全校验)
                     ├── resolver.py   (CsResolver 截面函数解算)
                     ├── expr_base.py  (AstExpression 基类)
                     └── spec.py       (构建器/GLM规格)
```

## 依赖关系

```
                    signals/v4/               factor/v4/
                  Expression               FactorExpression
                       │                          │
                       └──────────┬───────────────┘
                                  ▼
                           utils/ast/
                    (公共 DSL 基础设施: 解析/求值/安全)
                                  │
                                  ▼
                             ft2.core
                       (Engine/Account/Analyzer)
```

重要的是：**DSL 的权威实现已全部迁移到 `utils/ast`**。`signals/v4/ast_dsl.py` 和 `signals/v4/registry.py` 现在仅保留向后兼容的 re-export 壳。

新代码推荐：`from utils.ast import register_function, register_variable`

## 目录结构

```
signals/
├── v4/                     # 主力 ←
│   ├── ast_dsl.py          # [兼容重导出] 权威源 → utils/ast/dsl.py
│   ├── registry.py         # [兼容重导出] 权威源 → utils/ast/registry.py + variables.py
│   ├── expression.py       # Expression 类 (继承 AstExpression)
│   ├── engine.py           # SigEngine (fast/full 双模式)
│   ├── market_breadth.py   # 市场宽度特征 (calc_* 核心函数)
│   ├── wf_result.py        # WalkForwardResult 数据类 (v4 独立)
│   ├── search/             # GP遗传算法 + 网格搜索
│   │   ├── gp.py           # GPSearch (基于 Python ast)
│   │   └── grid.py         # GridSearch (参数网格)
│   ├── validate/           # 信号验证
│   │   ├── single.py       # validate_single
│   │   ├── compare.py      # compare_signals / signal_correlation
│   │   └── walkforward.py  # validate_walkforward
│   └── __init__.py         # 统一导出
├── v3/                     # 保留 (仅供历史测试对照)
```

---

## 核心 API (v4)

### 0. 导入规范

```python
from signals.v4 import Expression, SigEngine

# 表达式 → 信号 → 回测 → 报告 (一站式)
expr = Expression("rsi(CLOSE, 14) > 0.3")
signal = expr.generate(data)
analyzer = SigEngine.backtest(signal, data, mode='fast')
analyzer = SigEngine.backtest(signal, data, mode='full', bench_label='399317.SZ')
analyzer.to_notebook("V4 策略回测")
```

### 1. Expression — 信号表达式 (核心)

```python
from signals.v4 import Expression

# 单品种择时
expr = Expression("rsi(CLOSE, 14) > 0.3")
signal = expr.generate(data)  # → pd.Series

# 多品种因子轮动
panel = expr.evaluate_panel({'399967': df1, '399970': df2, ...})  # → DataFrame
ranked = expr.rank_panel({'399967': df1, '399970': df2, ...})     # → 每日截面排名

# 表达式自省
expr.variables    # ['CLOSE']
expr.functions    # ['rsi']
expr.complexity   # 5
```

### 2. 回测 — SigEngine

```python
from signals.v4 import SigEngine

# fast 模式: 搜索 (~25ms/次, 1566天单品种) — 内部走 core.Engine.run_fast() + FastAccount
analyzer = SigEngine.backtest(signal, data, mode='fast', start_date='2020-01-01')
# → AccountAnalyzer (sharpe_ratio()=1.16, annualized_return()=0.148, max_drawdown()=(-0.10,...))

# full 模式: 验证 + 报告 — 内部走 core.Engine.run() + AccountManager, 完整快照+交易记录
analyzer = SigEngine.backtest(signal, data, mode='full',
                               start_date='2020-01-01', bench_label='399317.SZ',
                               fee_config=None)    # 默认零费率
analyzer.to_notebook("策略回测")
```

> fast/full 均返回 AccountAnalyzer，接口一致。fast 不生成 TradeRecord/snapshots，交易指标返回 None。`fee_config` 控制费率，None=零费率。传入 dict 自定义，如 `{'commission_rate': 0.0003, 'stamp_tax_rate': 0.001, 'min_commission': 5.0}`。

### 3. 表达式语法

```
# 数据源 (8个)
CLOSE  OPEN  HIGH  LOW  VOLUME  AMOUNT  VWAP  RETURNS

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

# 特征计算 (22个, 从OHLCV实时算)
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

# 数学 (13个)
abs(x)  log(x)  sqrt(x)  sign(x)  exp(x)  tanh(x)  sigmoid(x)
relu(x)  signed_power(x, e)  safe_max(x,y)  safe_min(x,y)
ts_var(x, w)  ts_logret(x)

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
"BREADTH_S > ts_delay(BREADTH_S, 5) and ts_zscore(amt_ratio(AMOUNT,5,20), 60) > 0"

# 宽度 + 资金 OR 互补触发
"(BREADTH_S > 0.65 or BREADTH_AMT > 0.55) and ts_zscore(amt_ratio(AMOUNT,5,20), 60) > 0"

# 宽度上升 + NHL确认 (MDD极低)
"BREADTH_S > ts_delay(BREADTH_S, 5) and NHL > 0 and ts_zscore(amt_ratio(AMOUNT,5,20), 60) > 0"

# 拥挤度负向过滤 (not 原语)
"(BREADTH_S > 0.65 or BREADTH_AMT > 0.55) and ts_zscore(amt_ratio(AMOUNT,5,20), 60) > 0 and not (CORR > 0.7)"

# 宽度上升 + 价格OR宽松确认
"(BREADTH_S > ts_delay(BREADTH_S, 5)) and (CLOSE > OPEN or ts_roc(CLOSE, 3) > -0.02) and ts_zscore(amt_ratio(AMOUNT,5,20), 60) > 0"

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

# 窗口内条件计数 (比较运算输出0/1, ts_sum=计数)
"ts_sum(CLOSE > OPEN, 20) > 12"  # 过去20天超过12天收阳
"ts_sum(vol_ratio(CLOSE,VOLUME,5,20) > 1.5, 10) > 3"  # 过去10天有3天以上放量

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
expr = Expression("MY_VAR > 0 and my_smooth(BREADTH_S, 5, 20) > 0")
signal = expr.generate(data, extra_features={'MY_VAR': ..., 'BREADTH_S': ...})

# 注销 (清理)
from signals.v4 import unregister_function, unregister_variable
unregister_function('my_smooth')
unregister_variable('MY_VAR')
```

> **注意**: 注册是进程级全局操作，重复注册会覆盖并警告。推荐脚本顶部注册、末尾注销。

### 6. V4 原语覆盖 (vs 竞品)

| 类别 | WorldQuant | Hubble | AKQuant | ft2 V4 |
|------|:--:|:--:|:--:|:--:|
| 时序 | 24 | 4 | 12 | **27** |
| 截面 | 6 | 2 | 2 | **6** |
| 特征计算 | 0 | 0 | 0 | **22** |
| 扩张统计 | 0 | 0 | 0 | **4** |
| 数学/逻辑 | 26 | 4 | 7 | **13** |
| **总计** | **56** | **10** | **21** | **72** |

> **注意**: 权威原语定义在 `utils/ast/functions.py` (FUNC_REGISTRY)，非 signals/v4/registry.py。
> `signals/v4/registry.py` 仅为向后兼容的 re-export。新增原语请用 `register_function()` 注册到
> `utils/ast`，而非 signals/v4。

---

## 安全模型

```
ast.parse(expr) → 三层白名单校验 (实现在 utils/ast/dsl.py):
  1. 节点类型白名单 (Import/Attribute/Lambda 等35种禁止)
  2. 函数名白名单 (72个注册原语 + 临时注册)
  3. 变量名白名单 (70+合法变量前缀)

LLM 生成 → ast.parse → 白名单校验 → 回测 → 报告，全链路安全。
```

## 行业结构变量 (extra_features 注入)

| 变量 | 说明 | 典型用法 |
|------|------|---------|
| `BREADTH_S/M/L` | 均线上方行业占比 | `BREADTH_S > 0.65` |
| `BREADTH_AMT` | 成交额均线上方行业占比 | `BREADTH_AMT > 0.55` |
| `DISP` | 行业收益截面标准差 | `DISP < 0.01` (一致) |
| `SKEW` | 行业收益截面偏度 | `SKEW < -1` (负偏反转) |
| `NHL` | 新高-新低净差 | `NHL > 0` |
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

- **v4 主力**：`from signals.v4 import Expression, SigEngine`，探索灵活
- v3 保留，仅供历史测试对照，回测已统一到 ft2.core
- FeatureSpace 仍存在于 v3，可作为兼容层（`extra_features=dict`传入），但 v4 不需要
- `evaluate_panel()` 确保索引对齐到 data 尾部（FeatureSpace 冷启动截断）
- fast 模式内部走 `core.Engine.run_fast()` + `FastAccount`，不生成快照/交易记录；Sharpe 与 full 一致，`ctx.account` 接口统一
- **滚动窗口预热**: `ts_zscore(60)` 等滚动函数要求满窗口才输出有效值（前59天为 NaN→0），避免早期样本不足时的假信号
- **自定义注册**: `register_function()` / `register_variable()` 进程级全局，推荐脚本顶部注册、末尾注销。
  新代码推荐直接使用 `from utils.ast import register_function, register_variable`
- **72 原语覆盖 WorldQuant 时序算子的 96%**，截面 100%。权威清单在 `utils/ast/functions.py` (FUNC_REGISTRY)
- 扩展新原语：写函数 → `register_function()` 注册 → LLM 即用（无需改 utils/ast/functions.py）

### 变量命名规范（对齐 ft2 外部调用准则）

| 场景 | 规范写法 | 避免写法 |
|------|---------|---------|
| OHLCV 数据 | `data` | `ohlcv_df` / `klines` / `df` |
| 信号序列 | `signal` | `sig` / `pred` / `signals` |
| 表达式字符串 | `expr` / `buy_expr` / `sell_expr` | `formula` / `expr_str` |
| 额外特征 | `extra_features` | `features` / `feats` / `ctx` |
| 回测结果 | `analyzer` | `result` / `r` / `bt` |
| 品种代码列表 | `symbols` | `codes` / `tickers` |
