# utils/ast 模块 AI 助手指南

> **版本：v1 | 更新日期：2026-06-22**

## 模块定位

`utils/ast` 是 ft2 项目的**公共 DSL 基础设施**，为 signals（择时信号）和 factor（因子轮动）两模块提供统一的表达式解析、求值、校验能力。

## 五层架构

| 层 | 文件 | 职责 |
|---|---|---|
| 语法层 | `dsl.py` | parse/evaluate/validate/安全校验 |
| 原语层 | `functions.py` | 72 函数注册表 |
| 变量层 | `variables.py` | 70+ 变量前缀白名单 |
| 编排层 | `resolver.py` | CsResolver 截面函数嵌套解算 |
| 规格层 | `spec.py` | AST构建器/规范化/LLM语法规格 |

## 快速导入

```python
from utils.ast import (
    # 语法层
    parse_expression, evaluate, validate_expression,
    get_variables, get_functions,
    eval_colwise, cross_sectional_rank,
    normalize_data_keys,

    # 基类
    AstExpression,

    # 原语层
    FUNC_REGISTRY, register_function,

    # 变量层
    VALID_VAR_PREFIXES, is_valid_variable, register_variable,

    # 编排层
    CsResolver,

    # 规格层
    make_var, make_call, make_compare,
    normalize_expression, describe_expression,
    grammar_spec_for_llm,
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

### 5. 热注册（LLM 探索用）

```python
# 注册自定义函数
register_function('my_signal', lambda x, d: ...)
# 注册自定义变量
register_variable('MY_DATA')
```

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
# 数据源
CLOSE  OPEN  HIGH  LOW  VOLUME  AMOUNT  VWAP  RETURNS

# 时序函数 (窗口参数 d)
ts_mean(x, d)     ts_std(x, d)      ts_sum(x, d)      # ts_sum(CLOSE>OPEN,20)=过去20天涨的天数
ts_max(x, d)      ts_min(x, d)      ts_median(x, d)
ts_delta(x, d)    ts_delay(x, d)    ts_rank(x, d)
ts_corr(x, y, d)  ts_cov(x, y, d)   ts_skew(x, d)
ts_kurt(x, d)     ts_argmax(x, d)   ts_argmin(x, d)
ts_roc(x, d)      ts_zscore(x, d)   ts_scale(x, d)
ts_av_diff(x, d)  ts_decay_linear(x, d)  ts_product(x, d)
ts_regression(y, x, d, rettype=2)   # 0=β 1=α 2=残差 3=预测 4=R²

# 截面函数 (需完整2D面板)
cs_rank(x)  cs_zscore(x)  cs_scale(x)  cs_winsorize(x)

# 特征计算
rsi(CLOSE, 14)    atr(H,L,C,14)     macd(CLOSE)
ema(CLOSE, 20)    adx(H,L,C,14)     vol_ratio(C,V,5,20)
amt_ratio(A,5,20)

# 运算符
+ - * / > < and or not    a if cond else b
```

## 注意事项

- `ts_sum(cond, d)` 天然支持计数——比较运算输出 0/1
- `ts_regression(y,x,d,rettype=2)` 残差是冠军信号核心
- CsResolver 不需要任何配置，新增 cs_* 函数自动获得嵌套支持
- `validate_expression()` 不依赖数据，只做语法+变量校验
- `eval_colwise(strict=True)` 仅在调试时使用
