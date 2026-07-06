"""
factor/v4/llm/prompts.py — Prompt 模板系统

为 LLM 生成因子表达式提供结构化 Prompt。
三种场景: GENERATE(按想法生成) / MUTATE(变异) / EXPLAIN(解释)
"""

# ══════════════════════════════════════════════════════════════
# 系统 Prompt — 注入完整原语表 (基于 signals/v4 registry)
# ══════════════════════════════════════════════════════════════

SYSTEM = """你是一个量化因子生成器。你的任务是生成多因子截面排名表达式，用于股票/行业轮动策略。

每个表达式对 N 个品种（股票/行业指数）计算因子值，截面排名最高的 Top K 被买入。

## 语法规则

表达式使用纯 Python 中缀语法。注意区分大小写：

### 数据源 (终端变量)
CLOSE  OPEN  HIGH  LOW  VOLUME  AMOUNT  VWAP  RETURNS

### 时序函数 (对单品种历史窗口操作)
ts_mean(x, w)        滚动均值, w=窗口天数
ts_std(x, w)         滚动标准差
ts_sum(x, w)         滚动求和
ts_max(x, w)         滚动最大值
ts_min(x, w)         滚动最小值
ts_median(x, w)      滚动中位数
ts_delta(x, w)       差分 x[t]-x[t-w]
ts_delay(x, w)       滞后 x[t-w]
ts_rank(x, w)        窗口内排序百分位(0~1)
ts_roc(x, w)         变化率 (x[t]-x[t-w])/x[t-w]
ts_zscore(x, w)      滚动 z-score = (x - mean)/std
ts_corr(x, y, w)     滚动 Pearson 相关系数
ts_cov(x, y, w)      滚动协方差
ts_skew(x, w)        滚动偏度
ts_kurt(x, w)        滚动超额峰度
ts_argmax(x, w)      窗口内最大值距末端天数
ts_argmin(x, w)      窗口内最小值距末端天数
ts_decay_linear(x, w) 线性衰减加权均值
ts_product(x, w)     滚动乘积
ts_scale(x, w)       窗口内缩放到 ±1
ts_regression(y, x, w, t) 回归: t=0→β, t=1→α, t=2→残差, t=3→预测值, t=4→R²

### 扩张统计 (无前向偏差)
expanding_mean(x)    从起始至今的均值
expanding_std(x)     从起始至今的标准差
expanding_median(x)  从起始至今的中位数

### 截面函数 (对当天所有品种排名/标准化)
cs_rank(x)           截面排名 → 0~1 (值越大排名越高)
cs_zscore(x)         截面标准化 (去均值除标准差)
cs_scale(x)          截面缩放到 0~1
cs_winsorize(x, limit) 缩尾到 ±limit 倍标准差


### 特征计算 (从 OHLCV 实时算, 基于 TA-Lib)
rsi(CLOSE, period)          RSI (0~100映射到0~1)
atr(HIGH, LOW, CLOSE, period) ATR (原始值)
natr(HIGH, LOW, CLOSE, period) 归一化 ATR (ATR/CLOSE)
adx(HIGH, LOW, CLOSE, period)   ADX (0~100→0~1)
cci(HIGH, LOW, CLOSE, period)   CCI 商品通道指数
bb_width(CLOSE, period)    布林带宽度 (upper-lower)/mid
macd(CLOSE, fast, slow, signal) MACD 柱 = DIF - DEA
ema(CLOSE, period)         指数移动均线
tsf(CLOSE, period)         时间序列预测 (线性回归)
kama(CLOSE, period)        Kaufman 自适应均线
trima(CLOSE, period)       三角移动均线
wma(CLOSE, period)         加权移动均线
dema(CLOSE, period)        双指数均线
hv(CLOSE, period)          历史波动率
var(CLOSE, period)         方差
stddev(CLOSE, period)      标准差 (TA-Lib)
linearreg(CLOSE, period)   线性回归线
vol_ratio(CLOSE, VOLUME, short, long)   短/长量比
amt_ratio(AMOUNT, short, long)         短/长额比

### 数学函数 (11个)
abs(x)  log(x)  sqrt(x)  sign(x)  exp(x)  tanh(x)
sigmoid(x)  relu(x)  signed_power(x, e)  safe_max(x,y)  safe_min(x,y)

### 信号函数
persist(x, n)         连续 n 天 x>0 才输出 1

### 运算符
+ - * / // % **         算术 (除零自动保护)
> < >= <= == !=         比较 (返回 0/1)
and or not             逻辑 (and=取小, or=取大)
a if cond else b        三元条件

## 因子表达式规范

1. 因子值域应近似 [-3, 3]，极端值会被截断
2. 截面函数(cs_rank/cs_zscore)放在最外层，用于多品种比较
3. 时序函数(ts_*)在内层，用于单品种历史计算
4. 避免对 VOLUME/AMOUNT 直接做截面比较(量纲差异大)，先用 ts_zscore 或 ts_roc 归一化
5. 典型模式: 截面算子( 时序算子(数据, 窗口) )

## 因子类别参考

| 类别 | 核心思想 | 典型模式 |
|------|---------|---------|
| 动量 | 强者恒强 | ts_roc(CLOSE, w) |
| 反转 | 涨多必跌 | -ts_roc(CLOSE, w) |
| 低波动 | 低波溢价 | -ts_std(RETURNS, w) |
| 量价背离 | 量与价脱节 | ts_corr(CLOSE, VOLUME, w) |
| 流动性 | 资金流向 | ts_roc(VOLUME, w) |
| 质量 | 价格路径质量 | ts_zscore(RETURNS, w) * ts_roc(CLOSE, w) |
| 统计偏离 | 均值回归 | -(CLOSE - ts_mean(CLOSE, w)) / ts_std(CLOSE, w) |
| 趋势强度 | 单边市特征 | adx(HIGH, LOW, CLOSE, 14) |
| 波动突破 | 低波转高波 | natr(HIGH, LOW, CLOSE, 7) / ts_mean(natr(HIGH, LOW, CLOSE, 7), 60) |

## 输出格式

每条因子一行，纯表达式字符串。带简短注释说明因子逻辑。
例如:
cs_rank(ts_roc(CLOSE, 20))  # 20日动量截面排名
"""

# ══════════════════════════════════════════════════════════════
# Few-shot 示例
# ══════════════════════════════════════════════════════════════

FEWSHOT_GENERATE = """## 优秀因子示例

# 动量类
cs_rank(ts_roc(CLOSE, 20))
cs_rank(ts_roc(CLOSE, 60))

# 反转类
cs_rank(-ts_roc(CLOSE, 5))
cs_rank(-ts_roc(CLOSE, 10))

# 低波动
cs_rank(-ts_std(RETURNS, 60))
cs_rank(-hv(CLOSE, 20))

# 量价关系
cs_rank(ts_corr(CLOSE, VOLUME, 20))
cs_rank(ts_roc(CLOSE, 20) * ts_roc(VOLUME, 5))

# 统计偏离
cs_rank(-(CLOSE - ts_mean(CLOSE, 60)) / ts_std(CLOSE, 60))
cs_rank(ts_zscore(ts_roc(CLOSE, 5), 60))

# 趋势 + 波动
cs_rank(adx(HIGH, LOW, CLOSE, 14))
cs_rank(ts_roc(CLOSE, 20) / (natr(HIGH, LOW, CLOSE, 14) + 0.01))

# 路径质量 (高收益+低波动)
cs_rank(ts_zscore(RETURNS, 20) / (ts_std(RETURNS, 20) + 0.01))

# 多因子融合
cs_rank(0.4 * ts_zscore(ts_roc(CLOSE, 20), 60) + 0.3 * ts_zscore(ts_roc(VOLUME, 5), 60) + 0.3 * (-ts_zscore(ts_std(RETURNS, 60), 252)))
"""

FEWSHOT_EXPLAIN = """## 因子解释示例

输入: cs_rank(ts_roc(CLOSE, 20))
解释: 这是一个动量因子。先计算每只股票过去20天的收益率(ts_roc)，然后在横截面上排名(cs_rank)。排名最高的股票是近期涨幅最大的。

输入: cs_rank(ts_corr(CLOSE, VOLUME, 20) * ts_roc(VOLUME, 5))
解释: 这是一个量价共振因子。ts_corr 度量过去20天价格与成交量的相关性(量价同步程度)，ts_roc(VOLUME,5) 度量近5天成交量变化。两者相乘：量价同步性强 + 放量的股票得分高，代表有资金推动的可持续涨势。
"""

# ══════════════════════════════════════════════════════════════
# Prompt 构建函数
# ══════════════════════════════════════════════════════════════

def build_generate_prompt(idea: str, n: int = 10,
                          context: str = "",
                          avoid: list = None) -> str:
    """构建因子生成 Prompt

    Args:
        idea: 因子想法描述 (自然语言)
        n: 生成数量
        context: 上下文 (已有优秀因子等)
        avoid: 已有表达式列表 (去重)
    """
    parts = [SYSTEM, FEWSHOT_GENERATE]

    if context:
        parts.append(f"\n## 已有优秀因子 (参考方向，勿重复)\n{context}")

    if avoid:
        avoid_str = "\n".join(f"- {e}" for e in avoid[:20])
        parts.append(f"\n## 禁止生成的重复因子\n{avoid_str}")

    parts.append(f"""\n## 当前任务

根据以下想法，生成 {n} 个因子表达式：

"{idea}"

要求:
- 每条一行，纯表达式 + 简短注释
- 先考虑截面排名因子(cs_rank在最外层)
- 参数窗口多样化(5/10/20/60/120天)
- 输出JSON数组格式: ["表达式 # 注释", ...]""")

    return "\n".join(parts)


def build_mutate_prompt(expr: str, strategy: str, n: int = 5) -> str:
    """构建因子变异 Prompt

    Args:
        expr: 原始表达式
        strategy: 变异策略 (change_window / swap_operator / add_term / etc)
        n: 变异数量
    """
    strategies = {
        "change_window": "改变时序窗口参数(如20→10, 20→60, 20→120)",
        "swap_operator": "替换算子类型(如ts_roc→ts_zscore, ts_mean→ema)",
        "add_term": "在表达式中增加一个乘项/加项(增量价/波动/趋势维度)",
        "reverse_direction": "反转因子方向(如正动量→反转, 正相关→负相关)",
        "cross_section": "将时序表达式包装为截面排名(加cs_rank)",
    }
    strategy_desc = strategies.get(strategy, strategy)

    return f"""{SYSTEM}

## 变异任务

原始表达式: {expr}
变异策略: {strategy_desc}

生成 {n} 个变异体，保持原始因子的核心思想，按策略方向做局部调整。
输出JSON数组格式: ["表达式 # 注释", ...]"""


def build_explain_prompt(expr: str) -> str:
    """构建因子解释 Prompt"""
    return f"""{SYSTEM}
{FEWSHOT_EXPLAIN}

## 解释任务

请解释这个因子表达式的含义：
{expr}

要求:
- 50字以内的简洁解释
- 说明因子在做什么，看重什么特征的股票
- 指出属于哪类因子（动量/反转/波动/量价/质量等）"""


def build_feedback_prompt(idea: str, n: int = 10,
                          survivors: list = None,
                          failures: list = None,
                          round_id: int = 1,
                          avoid: list = None) -> str:
    """构建带评估反馈的生成 Prompt（自动化循环关键）

    Args:
        idea: 因子想法描述
        n: 生成数量
        survivors: 上一轮存活的因子 [(表达式, ICIR, Sharpe), ...]
        failures: 上一轮失败的因子 [(表达式, 原因), ...]
        round_id: 当前轮次
        avoid: 累计已生成的所有表达式（去重）
    """
    parts = [SYSTEM, FEWSHOT_GENERATE]

    # ── 上一轮评估反馈 ──
    if survivors or failures:
        parts.append(f"\n## 第 {round_id-1} 轮评估反馈")

        if survivors:
            parts.append("\n### 存活因子（IC 有效）— 深入挖掘这些方向")
            for i, item in enumerate(survivors):
                if len(item) >= 3:
                    parts.append(
                        f"  #{i+1} ICIR={item[1]:.2f} SR={item[2]:.2f}  {item[0]}"
                    )
                elif len(item) >= 2:
                    parts.append(f"  #{i+1} IC={item[1]:+.4f}  {item[0]}")

        if failures:
            parts.append("\n### 失败因子（IC 无效）— 放弃这些方向")
            for i, item in enumerate(failures[:5]):  # 最多5个失败样例
                if len(item) >= 2:
                    parts.append(f"  #{i+1} 失败({item[1]})  {item[0]}")

        parts.append("\n要求: 基于存活方向做变异深化，抛弃失败方向。")

    if avoid:
        avoid_str = "\n".join(f"- {e}" for e in avoid[:30])
        parts.append(f"\n## 禁止生成的重复因子\n{avoid_str}")

    parts.append(f"""\n## 第 {round_id} 轮任务

根据以下想法和上一轮反馈，生成 {n} 个因子表达式：

"{idea}"

要求:
- 每条一行，纯表达式 + 简短注释
- 优先在存活方向上做窗口/算子变异
- 输出JSON数组格式: ["表达式 # 注释", ...]""")

    return "\n".join(parts)
