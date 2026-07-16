# utils/ast v2 — AI 手册

> **v2 权威规范**。AI 重点读 **registry.py**（注册表），`functions.py` 是纯计算实现细节，不需关注。

## 架构

| 文件 | AI 关注度 | 职责 |
|------|-----------|------|
| **`registry.py`** | **★ 核心** | **函数表/变量表/宏表** + `register()` / 热注册 / 宏编译 |
| `dsl.py` | ★ 常用 | parse / evaluate / validate / 安全校验 / 截面排名 |
| `spec.py` | ★ 常用 | AstExpression 基类 / AST 构建器 / 语法规格 |
| `resolver.py` | ☆ 按需 | CsResolver 截面嵌套解算 |
| `functions.py` | ☆ 不需读 | 纯 numba 实现细节 |

## 注册表 (registry.py)

**AI 只需读这里**：所有可用函数、变量、宏的注册定义。

```python
from utils.ast.v2 import (
    FUNC_REGISTRY,    # ~92 函数: {'ts_mean': FunctionSpec, 'cs_rank': FunctionSpec, ...}
    VAR_CATEGORIES,   # 变量分类
    VALID_VAR_PREFIXES,  # 70+ 合法变量前缀
    FunctionSpec, VarSpec, ParamRange,
    register, register_function, register_macro, unregister_function,
    register_variable, unregister_variable,
    is_macro, list_macros, macro_to_str, unregister_macro,
)
```

### 注册 API

```python
# 原语（Callable）
register('my_fn', lambda x, d: ..., category='ts_function', data_args=1, param_pool=[5,10])

# 宏（DSL 字符串，自动编译）
register('sharpe', 'ts_mean(x,d)/ts_std(x,d)', 'ts_function', data_args=1, param_pool=[20,60])

# 变量
register_variable('MY_VAR', category='自定义', is_prefix=False)

# 注销
unregister_function('my_fn')
unregister_macro('sharpe')
```

`category` 必须是: `ts_function`, `cs_function`, `math_function`, `ta_function`, `feature_function`。

## 注册函数表

```
时序 ts_(d): mean/std/sum/max/min/median/delta/delay/rank/roc/zscore
  scale/quantile/av_diff/decay_linear/product/skew/kurt
  argmax/argmin/corr(x,y,d)/cov(x,y,d)/autocorr(x,lag,d)
  slope/resid/rsq/intercept/predict/ar_resid(x,p)/step/hump/var

回归 reg_(y,x,d): slope/intercept/resid/predict/rsq
扩张 expanding_: mean/median/std/percentile
截面 cs_: rank/zscore/scale/winsorize/quantile/normalize

数学: abs/log/sqrt/sign/exp/tanh/sigmoid/relu/softsign/sin/cos
  gauss/p4/neg/square_sigmoid/signed_power/max(x,y)/min(x,y)/logret(x)

特征: rsi/atr/atr_sma/macd/ema/adx/cci/bb_width/vol_ratio/amt_ratio
  hv/stddev/var/linearreg/tsf/kama/wma/dema/trima/natr/wilder_smooth

信号: persist(expr,n)
宏: beta/sharpe/info_ratio/ts_deviate/vol_price_corr
```

## 注册变量表

```
CLOSE OPEN HIGH LOW VOLUME AMOUNT VWAP RETURNS
REL_CLOSE REL_AMOUNT REL_VOLUME BENCH_RETURNS SHARE
PE_TTM_INDEX PB_MRQ TURNOVERRATIO TOTALCAPITAL
SECTOR_UP SECTOR_MOM SECTOR_AD BREADTH_S/M/L/AMT
DISP ROTSPD NHL SKEW IND_CORR
VMED VDISP VSKEW TAILUP TAILDOWN TAILNET
ATR STDDEV HV NATR DOWNSIDE_VOL
RSI CCI MACD MFI ULTOSC ROC
更多见 VALID_VAR_PREFIXES
```

## 宏系统

```python
is_macro('beta')      # → True
list_macros()         # → {'beta': FunctionSpec, ...}
macro_to_str('beta')  # → "beta(x,y,d) = ts_cov(x,y,d) / (ts_std(y,d)**2)"
```

**内置宏**: beta / sharpe / info_ratio / ts_deviate / vol_price_corr

## 常用 API (dsl.py / spec.py)

| 功能 | 用法 |
|------|------|
| 解析 | `tree = parse_expression("ts_roc(CLOSE,20) > 0")` |
| 求值 | `result = evaluate(tree, {'CLOSE': arr})` |
| 校验 | `validate_expression("...", available_vars=[...])` → `{valid, errors, missing_vars}` |
| 面板 | `eval_colwise(tree, data, T, N)` |
| 截面排名 | `cross_sectional_rank(vals_2d)` |
| 基类 | `AstExpression("expr")` → `.variables`, `.functions`, `.complexity` |
| 遍历 | `walk_nodes(tree)` |

## 变量匹配

**精确匹配** (`is_prefix=False`): `CLOSE` ✅，`CLOSE_5` ❌
**前缀通配** (`is_prefix=True`): `REL` → `REL_CLOSE` ✅

## 关键语义（必读）

> NaN = 缺失，不参与排名，不填充极值。

- `cs_rank` / `cross_sectional_rank`: 值域 (0,1], 全 NaN 行 → NaN
- `ts_rank(x,d)`: x[i] 为 NaN → NaN
- `ts_roc(x,d)`: x[t-d]≈0 → NaN
- `ts_corr`/`ts_skew`/`ts_kurt`: std≈0 → NaN
- `ts_resid`/`ts_slope`/`ts_rsq`: window<3 或当前 NaN → NaN
- `eval_colwise`: inf→NaN, 保留 NaN