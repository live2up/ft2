# Notebook — HTML 报告格式化输出指南

> 基于 Jinja2 + Vue3 + ECharts 的轻量级报告生成器。**本文档重点说明输出规范：何时用文字、何时用表格、何时用图表。**

---

## 输出格式选择决策树

```
你需要输出什么？
  │
  ├─ 几个关键数字（3~8个）────────────→ nb.metrics()
  │   如：夏普=1.5、年化=18%、回撤=-12%
  │
  ├─ 结构化行列表数据──────────────→ nb.table()
  │   如：Top10排名、交易明细、参数网格结果
  │   行数多 → 自动分页（默认每页10条）
  │   列有数值 → 可选 heatmap 热力图着色
  │
  ├─ 时间序列 / 趋势 / 对比 / 分布 ──→ nb.chart()
  │   如：净值曲线、回撤序列、IC序列、月度收益热力图
  │
  ├─ 一句话结论 / 注释 / 警告 ───────→ nb.text()
  │   如："该因子在2024年出现显著衰减"
  │
  ├─ 多段分析 / 分段标题 ───────────→ nb.markdown() 或 nb.title()
  │
  └─ 需要分组折叠的复杂内容 ────────→ with nb.section("标题"):
```

---

## 一、格式规范详解

### 1. metrics — 核心指标卡片

**适用场景**：3~8 个关键 KPI，需要一目了然。

**规范**：
- 数量控制在 **4~8 个**，过多会失去"一目了然"的效果
- 正值用绿色 `'green'`，负值/风险用红色 `'red'`
- 百分比保留 1 位小数，夏普/比率保留 2 位小数

**示例**：

```python
nb.metrics([
    {'name': '总收益率', 'value': '45.2%', 'color': 'green'},
    {'name': '年化收益', 'value': '18.5%', 'color': 'green'},
    {'name': '夏普比率', 'value': '1.85'},
    {'name': '最大回撤', 'value': '-12.5%', 'color': 'red'},
    {'name': '胜率', 'value': '58.3%'},
    {'name': '交易次数', 'value': '127'},
], title="核心指标", columns=3)
```

**反例**（不推荐）：
- ❌ 10+ 个指标堆在 metrics 里 → 改用 table
- ❌ 所有指标都用同样颜色 → 缺乏重点

---

### 2. table — 结构化表格

**适用场景**：
- 排名表（Top10/Top50）
- 交易明细
- 参数网格搜索结果
- 因子对比表
- 任何需要**逐行查看**的结构化数据

**规范**：
- 列名用中文，简洁明了
- 数值列统一小数位数
- 超过 20 行自动分页（默认每页 10 条）
- 数值密集的表格用 `heatmap` 着色增强可读性

```python
# 基础表格
nb.table(rank_data, columns=['排名', '表达式', '夏普', '年化(%)', '回撤(%)'],
         title='Top10 信号排名')

# 数值热力图表格（列级着色，蓝→白→红）
nb.table(factor_data, columns=['因子', 'IC均值', 'ICIR', 'IR', '胜率'],
         heatmap={'start': 2, 'axis': 'column'}, title='因子IC对比')

# 冻结左侧列
nb.table(wide_data, freeze={'left': 2}, title='宽表（冻结前2列）')

# 不分页（数据少时）
nb.table(short_data, page=False, title='简短列表')
```

**反例**：
- ❌ 只有 2-3 个数值用 table → 改用 metrics
- ❌ 1000+ 行且不需要逐行查看 → 考虑聚合后用 chart

---

### 3. chart — 可视化图表

**适用场景**：

| 你想展示什么 | 用哪种 chart | 数据类型 |
|-------------|-------------|---------|
| 趋势变化 | `'line'` | 时间序列 |
| 趋势变化（强调面积） | `'area'` | 时间序列 |
| 数量对比 | `'bar'` | 类别 vs 数值 |
| 占比分布 | `'pie'` | 名称 vs 占比 |
| 相关性/分布 | `'scatter'` | X vs Y |
| 价格走势 | `'kline'` | OHLC |
| 矩阵热力 | `'heatmap'` | 嵌套字典 |

**规范**：
- 所有图表**必须填 `title`**，说明图表内容
- 折线图有多条 series 时，每条必须有 `'name'` 区分
- 净值曲线用 `series_opts={'is_smooth': True}` 平滑

```python
# 标准时序数据格式
{
    'xAxis': ['2024-01', '2024-02', '2024-03'],
    'series': [
        {'name': '策略', 'data': [1.0, 1.05, 1.12]},
        {'name': '基准', 'data': [1.0, 1.02, 1.05]},
    ]
}

# 折线图（净值曲线）
nb.chart('line', nav_data, title='净值曲线', series_opts={'is_smooth': True})

# 柱状图（IC序列）
nb.chart('bar', ic_data, title='滚动IC序列')

# 饼图（资产配置）
nb.chart('pie', [{'name': '股票', 'value': 60}, {'name': '债券', 'value': 40}],
         title='资产配置')

# 热力图（月度收益）
nb.chart('heatmap', {'2024': {'01': 0.05, '02': -0.02, '03': 0.08}},
         title='月度收益热力图')

# 两个图垂直合并（如：净值 + 成交量）
nb.chartg('line', nav_data, height=300)
nb.chartg('bar', volume_data, height=150)
# 自动合并为一个 Grid 输出
```

**DataFrame 自动转换**（line/area/bar/heatmap 都支持直接传 DataFrame）：

```python
# df 列: ['date', 'strategy', 'benchmark']
# 自动：第一列→xAxis，其余列→series
nb.chart('line', df, title='净值对比')
```

---

### 4. text / markdown — 文字、结论与注释

**适用场景**：

| 方法 | 适用 | 示例 |
|------|------|------|
| `text()` | 一句话结论、警告、简短注释 | `"该因子在 2024 年出现显著衰减，建议暂停使用"` |
| `markdown()` | 多段分析、带格式文本 | 列表、粗体、二级标题 |
| `title()` | 报告内部分段大标题 | `"三、风险分析"` |

```python
# 关键结论（带颜色强调）
nb.text("该因子在 2024H2 出现显著衰减，建议观察后再启用", color='red')
nb.text("ATR(7) & TRIMA(60) 在所有组合中表现最优", color='green')

# 多段 Markdown（推荐用于"结论与建议"部分）
nb.markdown("""
## 分析结论

1. **波动率类指标持续领先**，ATR/BBWIDTH 包揽前三
2. 简单组合（深度 ≤ 3）优于复杂组合
3. 建议以 `ATR(7) & TRIMA(60)` 作为基准信号
""")
```

**反例**：
- ❌ 用 `markdown()` 写长篇纯文本替代表格 → 数据应该用 table
- ❌ 每个 text 都写一大段 → 长文本用 markdown 分段

---

### 5. section — 报告章节组织

**规范**：
- 每个 section 代表一个**逻辑分析单元**（如"收益分析"、"风险分析"、"因子对比"）
- section 内先放 metrics（总览），再放 chart（细节），最后放 table（明细）
- 辅助性 / 过长的内容用 `collapsed=True` 默认折叠

```python
nb = Notebook("择时因子分析报告")

# 顶层：总览
nb.metrics({'夏普': '1.85', '年化': '18.5%'}, title="核心指标")

# 章节：按分析主题组织
with nb.section("收益分析"):
    nb.chart('line', nav_data, title='净值曲线')

with nb.section("风险分析"):
    nb.metrics({'最大回撤': '-12.5%', '波动率': '18.3%'}, title="风险指标")
    nb.chart('line', dd_data, title='回撤序列')

# 折叠：详细数据（默认隐藏，减少视觉干扰）
with nb.section("交易明细", collapsed=True):
    nb.table(trade_log, columns=['日期', '操作', '价格'], page={'size': 20})
```

---

## 二、快速参考：常用报告模板

### 回测报告

```python
nb = Notebook("策略回测报告")
nb.metrics({'总收益率': '45.2%', '夏普': '1.85', '最大回撤': '-12.5%'}, title="核心指标")
nb.chart('line', nav_data, title='净值曲线', series_opts={'is_smooth': True})

with nb.section("风险分析"):
    nb.chart('line', dd_data, title='回撤序列')

with nb.section("交易明细", collapsed=True):
    nb.table(trades, page={'size': 20})

nb.export_html()
```

### 因子分析报告

```python
nb = Notebook("因子分析报告")
nb.metrics({'IC均值': '0.052', 'ICIR': '0.285', 'IC>0占比': '62.3%'}, title="IC统计")
nb.chart('bar', ic_data, title='IC序列')
nb.chart('heatmap', monthly_returns, title='月度收益热力图')
nb.table(factor_rank, columns=['因子', 'IC', 'ICIR', '夏普'], heatmap={'start': 2, 'axis': 'column'})

nb.export_html()
```

### 网格搜索结果报告

```python
nb = Notebook("参数网格搜索结果")
nb.text(f"共测试 {len(results)} 个参数组合", color='blue')
nb.metrics(top1_metrics, title="最优参数表现")

with nb.section("Top10 结果"):
    nb.table(top10, columns=['表达式', '夏普', '年化(%)', '回撤(%)'],
             heatmap={'start': 2, 'axis': 'column'})

with nb.section("全部结果", collapsed=True):
    nb.table(all_results, page={'size': 20})

nb.export_html()
```

---

## 三、通用输出原则

1. **先总览，后细节** — 任何报告第一屏都应是 metrics + 净值图
2. **能用图表不用表格，能用表格不用纯文字** — 数据天生适合可视化
3. **标题必填** — `title=` 参数是给读者和 AI 的索引，不填等于浪费
4. **颜色有语义** — green=好, red=风险/警告, blue=中性信息
5. **折叠不是藏** — 辅助数据折叠，不要藏核心结论
6. **数字精度统一** — 同一列的小数位数一致，百分比 1 位，比率 2 位

---

## 四、单元格 API 速查

| 方法 | 输出类型 | 核心参数 |
|------|---------|---------|
| `nb.metrics(data, title, columns=4)` | 指标卡片 | data: List[Dict] 或 Dict |
| `nb.table(data, columns, title, **opts)` | 表格 | data, columns, freeze, page, heatmap |
| `nb.chart(type, data, title, height)` | 图表 | type: line/bar/pie/heatmap/kline/scatter/area |
| `nb.chartg(type, data, height)` | Grid 叠加 | 连续调用自动合并 |
| `nb.text(text, color)` | 纯文本 | color: red/green/blue/yellow 等 |
| `nb.markdown(text)` | Markdown | 支持标题/列表/粗体/链接 |
| `nb.title(text, level=1)` | 标题 | level: 1-6 |
| `nb.divider()` | 分隔线 | — |
| `nb.code(code, language, output)` | 代码块 | — |
| `nb.pyecharts(chart, title)` | pyecharts 原生 | chart: pyecharts 对象 |

### 图表数据格式速记

| 图表类型 | 数据格式 |
|---------|---------|
| line / bar / area | `{'xAxis': [...], 'series': [{'name': '', 'data': []}]}` 或 DataFrame |
| pie | `[{'name': '', 'value': 0}]` 或 2 列 DataFrame |
| heatmap | `{'Y': {'X': value}}` 或 DataFrame（首列→X轴） |
| kline | `{'xAxis': [日期], 'series': [{'data': [[开,收,低,高]]}]}` |
| scatter | 仅标准格式 dict |

### 表格可选参数

```python
freeze={'left': 2}           # 冻结前2列
page=False                   # 禁用分页
page={'size': 20}            # 每页20条
heatmap={'start': 2, 'end': 5, 'axis': 'column'}  # 第2到5列着色
```

---

## 依赖

- Python >= 3.8
- pandas
- jinja2
- pyecharts >= 2.1.0

---

> 最后更新：2026-05-12
