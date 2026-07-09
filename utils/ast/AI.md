# utils/ast 模块 AI 助手指南

> **🚀 重点提示：本文档即 v2 权威规范**
> `utils/ast` 当前为 **v2**（numba @njit 加速 + 缺失值 NaN 语义 + 函数补全），代码持续演进中。
> 本文件描述 v2 **当前实现**，是与该模块交互的唯一事实来源。涉及表达式求值、截面排名、缺失值处理时，务必遵守文末「关键语义（必读）」一节。
> 修改任何行为（尤其 NaN/排名语义）前，先读 `functions.py` / `dsl.py` 真实实现与同目录 `审核报告.md`，勿依赖记忆或历史文档。

> **版本：v2 | 更新日期：2026-07-09**
>
> [更新] 2026-07-09 同步 v2 代码现状：numba @njit 加速；cs_rank / cross_sectional_rank / 特征函数对缺失值返回 NaN；新增函数 reg_*/ts_var/ts_logret/signed_power/safe_max/safe_min/cs_quantile/cs_normalize/atr_sma/stddev/var/linearreg；3 个旧名别名（ts_resi/ts_regression_residual/ts_rsquare）标记待删除。

## 模块定位

`utils/ast` 是 ft2 项目的**公共 DSL 基础设施**，为 signals（择时信号）和 factor（因子轮动）两模块提供统一的表达式解析、求值、校验能力。

## 五层架构

| 层 | 文件 | 职责 |
|---|---|---|
| 语法层 | `dsl.py` | parse/evaluate/validate/安全校验 |
| 原语层+变量层 | `functions.py` | 87 函数注册表 + 63 变量注册表 + 元数据 dataclass |
| 编排层 | `resolver.py` | CsResolver 截面函数嵌套解算 |
| 规格层 | `spec.py` | AST构建器/规范化/LLM语法规格/表达式基类 |

> 变量层已合并入 functions.py（`_VAR_REGISTRY`），不再有独立 variables.py。
> [重构] 2026-07-08 `expr_base.py` 并入 `spec.py`（AstExpression 基类移入），`registry.py` 已删除；`variables.py` 仅保留 re-export 兼容。

## 核心元数据

```python
# 函数规格 — 87 个 FUNC_REGISTRY 条目 (84 函数定义 + 3 旧名别名 ts_resi/ts_regression_residual/ts_rsquare, 后续删除)
FunctionSpec(func, category, data_args, param_pool, param_ranges, data_vars)

# 变量规格 — 63 个注册变量
VarSpec(name, category, is_prefix, description)

# 参数值域约束 — 描述连续/离散范围参数
ParamRange(name, dtype, min_val, max_val, pool)
```

## 快速导入

```python
from utils.ast import (
    # 语法层
    parse_expression, evaluate, validate_expression,
    get_variables, get_functions,
    eval_colwise, cross_sectional_rank,
    normalize_data_keys, ast_depth, ast_node_count, SAFE_CONSTANTS,

    # 基类
    AstExpression,

    # 原语层
    FUNC_REGISTRY, register_function, unregister_function,
    FunctionSpec, ParamRange, VarSpec,
    FUNC_CATEGORIES, VALID_FUNC_CATEGORIES, get_func_category,

    # 变量层
    VALID_VAR_PREFIXES, is_valid_variable, register_variable, unregister_variable,
    VAR_CATEGORIES, get_var_category,

    # 编排层
    CsResolver,

    # 规格层
    make_var, make_call, make_compare, make_binop, make_const, make_unaryop,
    make_boolop, make_ifexp,
    normalize_expression, normalize_ast, describe_expression,
    grammar_spec_for_llm, grammar_spec_compact,
)
```

## 核心 API

### 1. 解析 + 求值

```python
tree = parse_expression("ts_roc(CLOSE, 20) > 0")
# → ast.Expression (已通过安全校验)

result = evaluate(tree, {'CLOSE': np.array([100, 101, 99, ...])})
# → np.array([0, 0, 0, 1, ...])
```

### 2. LLM 前置校验

```python
# LLM 生成表达式后立即校验，不等 evaluate() 才发现变量缺失
v = validate_expression(
    "ts_roc(CLOSE, 20) > 0 and BREADTH_S > 0.6",
    available_vars=['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME']
)
# → {'valid': False, 'missing_vars': ['BREADTH_S'],
#     'errors': ["变量 'BREADTH_S' 不在可用数据中"]}
```

### 3. 面板逐列求值（调试模式）

```python
# 批量回测: 某列异常 → NaN，不中断
result = eval_colwise(tree, panel_data, T, N)

# 调试模式: 异常立即抛出，带列号定位
result = eval_colwise(tree, panel_data, T, N, strict=True)
# → RuntimeError: "eval_colwise 第 3 列求值失败: ..."
```

### 4. 表达式基类

```python
from utils.ast import AstExpression

expr = AstExpression("ts_roc(CLOSE, 20) > 0")
expr.variables   # ['CLOSE']
expr.functions   # ['ts_roc']
expr.complexity  # 6
```

signals/Expression 和 factor/FactorExpression 均继承自此基类。

### 5. 热注册（LLM / 探索用）

```python
# 注册自定义函数
register_function('my_signal', lambda x, d: ..., category='ts_function',
                  data_args=1, param_pool=[5, 10, 20])

# 注册自定义变量（精确匹配）
register_variable('MY_DATA', category='自定义', is_prefix=False)

# 注册自定义变量（前缀通配）
register_variable('MY_PREFIX', category='自定义', is_prefix=True)

# 注销
unregister_function('my_signal')
unregister_variable('MY_DATA')
```

`category` 必须属于 `VALID_FUNC_CATEGORIES`，否则 `_register` 立即抛出 `ValueError`。

### 6. CsResolver（截面函数嵌套解算）

```python
resolver = CsResolver()
ranked = resolver.resolve(tree, panel_data)  # → ndarray(T,N) 0~1
# 自动识别 cs_* 前缀函数，支持任意嵌套:
#   cs_rank(ts_delta(cs_rank(A), 5))
#   cs_rank(A) + cs_zscore(B)*0.3
```

## 表达式语法速查

```python
# 数据源 (63 个注册变量，含精确匹配 + 前缀通配)
CLOSE  OPEN  HIGH  LOW  VOLUME  AMOUNT  VWAP  RETURNS
REL_CLOSE  REL_AMOUNT  REL_VOLUME  BENCH_RETURNS  SHARE
PE_TTM_INDEX  PB_MRQ  TURNOVERRATIO  TOTALCAPITAL
RSI  ATR  MACD  EMA  ...  SECTOR_UP  BREADTH_S  TAILUP  ...

# 时序函数 (窗口参数 d)
ts_mean(x, d)     ts_std(x, d)      ts_sum(x, d)      # ts_sum(CLOSE>OPEN,20)=过去20天涨的天数
ts_max(x, d)      ts_min(x, d)      ts_median(x, d)
ts_delta(x, d)    ts_delay(x, d)    ts_rank(x, d)
ts_corr(x, y, d)  ts_cov(x, y, d)   ts_skew(x, d)
ts_kurt(x, d)     ts_argmax(x, d)   ts_argmin(x, d)
ts_roc(x, d)      ts_zscore(x, d)   ts_scale(x, d)
ts_quantile(x, d, p)                ts_av_diff(x, d)
ts_decay_linear(x, d)  ts_product(x, d)
ts_slope(x, d)    ts_resid(x, d)    ts_rsq(x, d)
ts_intercept(x, d)  ts_predict(x, d)

# 双变量回归 (y~x, 3 参数: y, x, d)
reg_slope(y, x, d)     reg_intercept(y, x, d)
reg_resid(y, x, d)     reg_predict(y, x, d)
reg_rsq(y, x, d)

# 扩张统计 (expanding_)
expanding_mean(x, d)  expanding_median(x, d)
expanding_std(x, d)   expanding_percentile(x, d)

# 截面函数 (需完整2D面板)
cs_rank(x)  cs_zscore(x)  cs_scale(x, scale)
cs_winsorize(x, std)  cs_quantile(x)  cs_normalize(x)

# 数学运算
abs(x)  log(x)  sqrt(x)  sign(x)  exp(x)  tanh(x)
sigmoid(x)  relu(x)  softsign(x)  sin(x)  cos(x)
gauss(x)  p4(x)  neg(x)  square_sigmoid(x)
signed_power(x, exponent)  safe_max(x,y)  safe_min(x,y)

# 特征计算
rsi(CLOSE, 14)    atr(H,L,C,14)     atr_sma(H,L,C,14)  macd(CLOSE)
ema(CLOSE, 20)    adx(H,L,C,14)     vol_ratio(C,V,5,20)
amt_ratio(A,5,20)  hv(CLOSE, 20)    bb_width(CLOSE, 20)
cci(H,L,C,14)     natr(H,L,C,14)    tsf(CLOSE, 10)
kama(CLOSE, 30)   wma(CLOSE, 10)    dema(CLOSE, 10)
trima(CLOSE, 40)  wilder_smooth(x, 10)  stddev(C,20)  var(C,20)  linearreg(C,20)

# 信号确认
persist(expr, n)  — 连续 n 日同向才触发

# 运算符
+ - * / > < and or not    a if cond else b
```

## 变量系统说明

变量注册表 `_VAR_REGISTRY` 统一管理 63 个变量，分两类匹配模式：

- **精确匹配**（`is_prefix=False`）：只认变量名本身，拒绝后缀（如 `CLOSE` ✅, `CLOSE_5` ❌）
- **前缀通配**（`is_prefix=True`）：允许 `prefix_xxx` 任意后缀（如 `REL` → `REL_CLOSE` ✅, `REL_AMOUNT` ✅）

`VALID_VAR_PREFIXES` 和 `VAR_CATEGORIES` 由 `_VAR_REGISTRY` 自动推导，保持向后兼容。

## 注意事项

- `ts_sum(cond, d)` 天然支持计数——比较运算输出 0/1
- `ts_regression(y,x,d,rettype=2)` 残差是冠军信号核心
- `cat_scale(x, scale)` 和 `cs_winsorize(x, std)` 的配置参数由 `ParamRange` 描述范围
- CsResolver 不需要任何配置，新增 cs_* 函数自动获得嵌套支持
- `validate_expression()` 不依赖数据，只做语法+变量校验
- `eval_colwise(strict=True)` 仅在调试时使用
- `register_function` 的 `category` 参数必须属于 `VALID_FUNC_CATEGORIES`（5 个有效分类）
- `register_variable` 新增 `category`/`is_prefix`/`description` 参数，旧式单参数调用仍兼容

## 关键语义（必读）

> 以下为 v2 numba 重构后的最终语义，是与该模块交互的唯一事实来源。修改任何行为（尤其 NaN/排名语义）前，务必先读 `functions.py` / `dsl.py` 真实实现与同目录 `审核报告.md`。

### 性能：numba @njit 加速
- ts_*/cs_* 的计算核心（`_ts_*_core` / `_cs_rank_core` 等）用 `@njit(cache=True)` 重写，cs_rank 约 28x 加速。
- `_rolling` 保留为 Python fallback（用于 `_feature_hv` 等少数路径），仅作兼容；新代码应优先走 numba core。
- 截面函数 cs_rank/cs_zscore/cs_scale/cs_winsorize 的 2D 路径走 numba core；1D 输入为兼容回退（cs_rank→0.5, cs_zscore→0 等），截面语义需 2D 面板。

### NaN / 缺失值语义（对齐 WQ 规范：NaN=缺失，不参与排名，不填充极值）
- `cs_rank(x)` 与 `cross_sectional_rank(vals)`：值域 (0,1]，整行全 NaN 时返回 NaN。
- `ts_rank(x, d)`：当前值 x[i] 为 NaN 时返回 NaN。
- `ts_roc(x, d)`：x[t-d]≈0 时返回 NaN（不伪造变化率）。
- 特征函数 `_feature_hv` / `_feature_bbwidth` / `_feature_vol_ratio` / `_feature_amt_ratio`：停牌/缺失导致的 NaN 透传返回 NaN。
- `ts_corr` / `ts_skew` / `ts_kurt`：std≈0 时返回 NaN（与 ts_zscore 一致）。
- `ts_resid`/`ts_slope`/`ts_rsq`/`ts_intercept`/`ts_predict`：window<3 或当前值 NaN 时返回 NaN，不返回 0.0。

### 已废弃的别名（不要再使用）
- `ts_resi` → 用 `ts_resid`；`ts_regression_residual` → 用 `reg_resid`；`ts_rsquare` → 用 `ts_rsq`。

### 审核报告状态
- 同目录 `审核报告.md`（2026-07-06）第一轮 P0/P1 已全部修复；第二轮发现的 P1-A/B/C/D（特征函数缺失值处理）已在 2026-07-09 修复。剩余待处理项：P2-1/2/3（窗口含 NaN 时语义偏差，对干净数据无影响）、P2-7/8（性能，不影响正确性）、P3（文档/边界）。