# 因子探索 Prompt 模板 — ft2 因子模块

> 丢给 Qwen/Claude/GPT，每次生成一批候选因子表达式，跑回测，反馈结果，迭代。

---

## 模板 A：首次探索（给定数据空间）

```
你是一个量化因子研究员。我有一个因子表达式引擎，支持以下语法：

【终端变量（6 个 OHLCV 基础 + 可扩展）】
open, high, low, close, volume, amount
（也支持自定义预计算变量，如 rel_close, downside_vol, share 等）

【一元函数】
abs, sqrt, log, exp, neg, sign, tanh, not

【二元函数】
add, sub, mul, div, max, min, gt, lt, ge, le, eq, ne, and, or

【时序原语（window 为回看窗口）】
ts_rank(x, window)          — 时序排名
ts_zscore(x, window)        — 时序标准化
delay(x, window)            — 滞后
delta(x, window)            — 差分
decay_linear(x, window)     — 线性衰减加权
ts_sum(x, window)           — 滚动和
ts_mean(x, window)          — 滚动均值
ts_std(x, window)           — 滚动标准差
ts_max(x, window)           — 滚动最大值
ts_min(x, window)           — 滚动最小值
sma(x, window, offset)      — 简单移动平均
correlation(x, y, window)   — 滚动相关系数
covariance(x, y, window)    — 滚动协方差
regbeta(x, y, window)       — 滚动回归 beta
ts_argmax(x, window)        — 最大值位置
ts_argmin(x, window)        — 最小值位置
ts_skew(x, window)          — 偏度

【截面原语】
cs_rank(x)                  — 截面排名
cs_zscore(x, window)        — 截面标准化
cs_mean(x)                  — 截面均值
signed_power(x, exp)        — 带符号幂

【健壮处理】
winsorize(x, n)             — 截尾到 ±nσ

【语法糖（减少嵌套层数）】
ret(window)                 — 收益率: (close - delay(close, window)) / delay(close, window)
adv(window)                 — 均量: ts_mean(volume, window)
intra_ret                   — 日内收益: (close - open) / open

【条件分支】
ifelse(cond, a, b)          — 三元条件

---

【任务】

在以下数据空间内，生成 {N} 个候选因子表达式：

数据空间描述：{DATA_SPACE_DESCRIPTION}
（例如："OHLCV 纯变形空间，目标是预测未来 5 日收益"）

要求：
1. 表达式深度 3-8 层，不要过于简单（如纯 ret(5)）
2. 尽量组合不同类别的原语（时序 + 截面 + 健壮处理）
3. 每个表达式附带一行中文注释，说明因子逻辑
4. 避免完全等价或高度相似的重复表达式
5. 输出格式：每行一个表达式，格式为 `表达式  # 注释`

示例输出：
ts_rank(winsorize(mul(delta(close, 20), adv(10)), 3), 10)  # 20日价格动量 × 10日均量，时序排名
ts_zscore(div(ret(5), ts_std(ret(1), 20)), 60)  # 5日收益 / 20日波动率，60日标准化
cs_rank(sub(ts_mean(close, 5), ts_mean(close, 20)))  # 5日均线与20日均线差距的截面排名
```

---

## 模板 B：迭代反馈（基于回测结果）

```
上一轮你生成了 {M} 个因子表达式，回测结果如下：

【Top 5】
1. {expression_1}  → IC={ic1}, Sharpe={sharpe1}
2. {expression_2}  → IC={ic2}, Sharpe={sharpe2}
3. {expression_3}  → IC={ic3}, Sharpe={sharpe3}
4. {expression_4}  → IC={ic4}, Sharpe={sharpe4}
5. {expression_5}  → IC={ic5}, Sharpe={sharpe5}

【淘汰的典型】
- {bad_expression_1}  → IC={ic}, 原因：{reason}
- {bad_expression_2}  → IC={ic}, 原因：{reason}

---

【任务】

基于以上反馈，再生成 {N} 个新因子表达式：

分析方向：
1. 哪些类型的因子表现好？（动量型？波动率型？量价结合？截面型？）
2. 表现好的因子有什么共同特征？（窗口参数范围？操作符组合模式？）
3. 表现差的因子有什么共性？如何避免？

生成策略：
1. 变异优秀因子：对 Top 因子做小幅改动（换窗口、换操作符、加减一层）
2. 探索新方向：基于优秀因子的特征，设计全新组合
3. 跨类别杂交：把不同 Top 因子的核心思路融合

要求同上轮，输出 {N} 个表达式 + 注释。
```

---

## 模板 C：专题探索（给定研究方向）

```
我正在研究以下因子方向：{DIRECTION}

方向描述：{DESCRIPTION}
（例如："价量弹性——价格变动对成交量的敏感度"）

【已尝试的公式】
{EXISTING_FORMULAS}

【当前最佳】
{best_expression}  → IC={ic}, Sharpe={sharpe}

---

【任务】

在这个方向上深入挖掘，生成 {N} 个新变体：

思考维度：
1. 窗口参数变化：短窗口(5-10) vs 中窗口(20-40) vs 长窗口(60-120)
2. 平滑/标准化：raw → winsorize → ts_zscore → cs_zscore
3. 组合方式：乘法 vs 除法 vs 加法 vs 条件分支
4. 引入辅助变量：加入 volume/amount 维度、加入截面排名
5. 非线性变换：sqrt、log、signed_power、tanh

输出 {N} 个表达式 + 注释。
```

---

## 模板 D：批量生成（大规模探索）

```
请生成 {N} 个因子表达式，覆盖以下类别，每类约 {per_category} 个：

类别清单：
1. 动量类（momentum）：ret、delta、ts_rank 组合
2. 波动率类（volatility）：ts_std、ts_zscore、偏度相关
3. 量价类（volume-price）：volume/amount 与 price 的交叉组合
4. 反转类（mean-reversion）：delay 与当前值的偏离度
5. 相关性类（correlation）：correlation/covariance/regbeta 应用
6. 截面类（cross-section）：cs_rank、cs_zscore 组合
7. 路径类（path）：intra_ret、ts_argmax/min、条件分支
8. 复合类（hybrid）：跨类别的多层嵌套

要求：
- 每个表达式带类别标签和一行注释
- 深度 4-8 层
- 避免类内重复

输出格式：
[动量] ts_rank(delta(close, 20), 10)  # 20日价格动量时序排名
[波动率] div(ret(5), winsorize(ts_std(ret(1), 20), 3))  # 5日收益 / 截尾20日波动率
...
```

---

## 使用流程

```
第 1 轮: 用模板 A（首次探索），N=20
  → 拿到 20 个表达式
  → 批量编译 → 批量 evaluate → 批量 IC/Sharpe 排名
  → 保留 Top 5-10

第 2 轮: 用模板 B（迭代反馈），N=15
  → 反馈 Top 5 + 淘汰典型
  → 拿到 15 个变异/新方向表达式
  → 再跑回测 → 更新排名

第 3-N 轮: 模板 B 继续迭代，或模板 C 专题深入
  → 直到新表达式不再超越已有最优

最终: 用模板 C 对最优方向做最后一轮精调
```

---

## 批量跑回测的代码片段（给 AI 参考）

```python
import sys; sys.path.insert(0, r'd:\01-Doc\Quant\ft2')
from factor.v3 import FactorExpression, FactorValidator

# 假设已有 panel_dict 和 future_returns
expressions = [
    "ts_rank(delta(close, 20), 10)",
    "div(ret(5), ts_std(ret(1), 20))",
    # ... AI 生成的表达式列表
]

results = []
for expr_str in expressions:
    try:
        expr = FactorExpression(expr_str)
        vals = expr.evaluate(panel_dict)
        fv = pd.DataFrame(vals, index=dates, columns=symbols)
        
        validator = FactorValidator(fv, future_returns)
        ic = validator.information_coefficient()
        
        results.append({
            'expression': expr_str,
            'ic_mean': ic['ic_mean'],
            'ic_std': ic['ic_std'],
            'ir': ic['ir'],
        })
    except Exception as e:
        results.append({'expression': expr_str, 'error': str(e)})

# 排序输出
df = pd.DataFrame(results).sort_values('ic_mean', ascending=False)
print(df.to_string())
```
