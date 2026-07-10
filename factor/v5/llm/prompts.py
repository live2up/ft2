"""
factor/v5/llm/prompts.py — Prompt 模板系统

为 LLM 生成因子表达式提供结构化 Prompt。
三种场景: GENERATE(按想法生成) / MUTATE(变异) / EXPLAIN(解释)

[重构] 2026-07-10 SYSTEM 原语表改为动态同步自 utils.ast.grammar_spec_compact()
         彻底消除与 ast.v2 FUNC_REGISTRY 的手工不同步 (原 ts_regression 等未注册函数 /
         漏列 sin/cos/gauss 等 6 个数学函数 / 变量仅 8 个), 任何 v2 更新自动生效。
"""

from utils.ast import grammar_spec_compact

# ════════════════════════════════════════════════════════════
# 系统 Prompt — 角色/规范 (静态) + 原语表 (动态同步自 ast.v2)
# ════════════════════════════════════════════════════════════

_SYSTEM_HEAD = """你是一个量化因子生成器。你的任务是生成多因子截面排名表达式，用于股票/行业轮动策略。

每个表达式对 N 个品种（股票/行业指数）计算因子值，截面排名最高的 Top K 被买入。

## 语法规则

表达式使用纯 Python 中缀语法。注意区分大小写。

## 可用原语（自动同步自 ast.v2 — FUNC_REGISTRY / 变量注册表）

> 以下原语表由代码自动生成, 与表达式引擎完全一致。不要使用表中未列出的函数或变量。

"""

_SYSTEM_TAIL = """## 因子表达式规范

1. 因子值域应近似 [-3, 3]，极端值会被截断
2. 截面函数(cs_rank/cs_zscore)放在最外层，用于多品种比较
3. 时序函数(ts_*)在内层，用于单品种历史计算
4. 避免对 VOLUME/AMOUNT 直接做截面比较(量纲差异大)，先用 ts_zscore 或 ts_roc 归一化
5. 典型模式: 截面算子( 时序算子(数据, 窗口) )

## 双变量回归（v2 实际可用函数名）

- reg_slope(y, x, d)    斜率 β
- reg_intercept(y, x, d) 截距 α
- reg_resid(y, x, d)    残差
- reg_predict(y, x, d)   预测值
- reg_rsq(y, x, d)      R²
- 单变量趋势: ts_slope(x,d) / ts_resid(x,d) / ts_rsq(x,d) / ts_intercept(x,d) / ts_predict(x,d)

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

# 动态拼接: 原语表取自 ast.v2, 保证与引擎完全一致 (含 63 变量 / 88 函数)
SYSTEM = _SYSTEM_HEAD + grammar_spec_compact() + _SYSTEM_TAIL

# ════════════════════════════════════════════════════════════
# Few-shot 示例
# ════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════
# Prompt 构建函数
# ════════════════════════════════════════════════════════════

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
