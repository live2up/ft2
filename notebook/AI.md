# Notebook 模块 AI 助手指南

> **AI 助手请先阅读此文件**，掌握 Notebook 输出模块的调用规范和数据格式
>
> **版本：v2.1.0 | 更新日期：2026-05-30**

---

## 项目定位

Notebook 是 ft2 框架的 **HTML 报告生成器**，基于 Jinja2 + Vue3 + ECharts + ft-table 构建。核心职责是将 Python 数据结构转换为美观的交互式 HTML 报告。

**设计原则：**
- Python 端只负责数据组装，不负责渲染逻辑
- 图表通过 min_pyecharts 构建，输出 ECharts option JSON
- 表格通过 ft-table Vue 组件渲染
- 最终由 Jinja2 模板 + Vue3 组装为完整 HTML
- 数据以 JSON-LD 格式注入 `<head>`，AI 可优先抓取
- **输入即输出**：xAxis 数据原样通过，前端不做日期格式转换
- **后端管布局，前端管位置**：高度/间距/legend 空间由后端管理，前端仅做 legend 偏移

---

## 文件结构

```
notebook/
├── __init__.py      # 导出 Notebook 类
├── cell.py          # Cell/Section 数据类 + CellBuilder + 图表构建器
├── min_pyecharts.py # 精简版 pyecharts（自研，668行，含 baseline diff 优化）
├── notebook.py      # Notebook 主类（用户 API 入口）
├── README.md        # 用户文档
└── AI.md            # 本文件

template/
├── notebook.html    # Jinja2 HTML 模板（数据以 JSON-LD 注入 <head>）
└── js/
    ├── notebook3D.js   # Vue3 渲染逻辑（CHART_AXIS_RULES 共享规则 + applyGridAxisRules）
    ├── notebook3D.css  # 样式
    ├── ft-table.js     # 表格组件
    └── echarts.min.js  # ECharts 库
```

---

## 核心 API

### 导入

```python
from notebook import Notebook
```

### 构造

```python
nb = Notebook(title: str = "Notebook Report")
# title: 报告标题，同时作为默认输出文件名
```

### 输出

```python
nb.export_html(name=None, template_path=None, local_assets=False)
# name: 输出文件名（不含 .html 扩展名），默认使用标题
# template_path: 自定义模板路径，默认 ../template/notebook.html
# local_assets: True=本地 file:// 资源（离线），False=CDN 资源（默认）
nb.to_json()                                    # 导出 JSON 字符串
nb.to_dict()                                    # 导出字典
```

**输出路径规则：** HTML 文件输出到 **调用者脚本所在目录**（`base_dir`），非 notebook 模块目录。
**本地测试：** 使用 `local_assets=True`，浏览器通过 `file://` 协议加载本地 JS/CSS。

---

## 单元格类型一览

| 方法 | 类型 | 说明 | 必填参数 |
|------|------|------|----------|
| `nb.title(text, level=1)` | 文本 | 标题 | `text: str` |
| `nb.text(text, color=None)` | 文本 | 文本 | `text: str` |
| `nb.markdown(text)` | 文本 | Markdown | `text: str` |
| `nb.code(code, language='python', output=None)` | 代码 | 代码块 | `code: str` |
| `nb.table(data, columns=None, title=None, **options)` | 数据 | 表格 | `data: List[Dict] / DataFrame` |
| `nb.metrics(data, title=None, columns=4)` | 数据 | 指标卡片 | `data: List[Dict] / Dict` |
| `nb.chart(chart_type, data, title=None, height='400px', **kwargs)` | 图表 | 图表 | 见下方图表规范 |
| `nb.pyecharts(chart, title=None, height='400px', width='100%')` | 图表 | pyecharts 原生 | `chart: pyecharts 对象` |
| `nb.chartg(chart_type, data, height=200, **kwargs)` | 图表 | Grid 累加 | 同 chart |
| `nb.divider()` | 布局 | 分隔线 | 无 |
| `nb.html(html_content)` | 布局 | 原始 HTML | `html_content: str` |
| `nb.section(title, collapsed=None)` | 布局 | 章节容器 | `title: str` |

**所有方法返回 `self`，支持链式调用。**

---

## 数据格式规范（关键）

### 1. 表格 `nb.table()`

**数据输入：**

```python
# 方式1：List[Dict]（推荐）
data = [
    {'code': '000001', 'name': '平安银行', 'return': 0.15},
    {'code': '000002', 'name': '万科A', 'return': -0.05},
]

# 方式2：DataFrame（自动转换）
df = pd.DataFrame(data)
```

**columns 参数：**

```python
columns=['code', 'name', 'return']  # 字符串列表，控制显示列及顺序
# None 时显示所有列
```

**options 参数（**kwargs）：**

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `freeze` | `dict` | 冻结列 | `{'left': 2, 'right': 1}` |
| `page` | `dict / False` | 分页配置 | `{'size': 20}` 或 `False` 禁用 |
| `heatmap` | `dict` | 热力图效果 | `{'start': 2, 'end': -1, 'axis': 'column'}` |

**freeze（列冻结）使用建议：**

```python
# 场景1：多列表格，冻结标识列（最常见的推荐用法）
# 股票行情、持仓明细：冻结左侧 1-2 列标识列
nb.table(data, freeze={'left': 2})          # 前2列（代码+名称）固定

# 场景2：右侧汇总列固定
nb.table(data, freeze={'left': 2, 'right': 1})  # 左侧2列+右侧1列固定

# 何时用 freeze：
# ✅ 列数 ≥ 6 且前两列为标识列（代码+名称/日期+品种）
# ✅ 最后列为合计/汇总时，冻结 right:1
# ❌ 列数 < 5 → 不需要冻结，屏幕够宽
```

**heatmap（热力图效果）使用建议：**

```python
# 核心参数
heatmap={
    'start': 2,      # 从第2列开始着色（跳过代码、名称列）
    'end': -1,       # 到倒数第1列（-1 排除最后的汇总列）
    'axis': 'column' # 'column'=每列独立归一化 / 'table'=全表统一
}

# 场景1：收益矩阵 → column 模式（每列独立看涨跌）
heatmap={'start': 2, 'axis': 'column'}
# 效果：收益%列 红涨绿跌，换手率列 深浅看活跃度，各自独立

# 场景2：因子暴露 → table 模式（全表统一比较强度）
heatmap={'start': 1, 'axis': 'table', 'colors': ['#fff', '#ff9800']}
# 效果：全局按强度着色，发现最强/最弱因子

# 场景3：自定义配色
heatmap={'start': 2, 'colors': ['#2196f3', '#fff', '#f44336']}  # 蓝→白→红（A股风格）
heatmap={'start': 2, 'colors': ['#4caf50', '#fff', '#f44336']}  # 绿→白→红

# 何时用 heatmap：
# ✅ 数据列 ≥ 3 且有数值列（收益率、因子值、权重等）
# ✅ 需要快速识别数值大小/正负的矩阵数据
# ❌ 纯文本列（策略名、分类标签）→ 不需要热力图
```

**freeze + heatmap 组合用法（推荐）：**

```python
# 场景：行业轮动矩阵
nb.table(industry_data,
    columns=['行业', '近1月', '近3月', '近6月', '近1年', '年均'],
    freeze={'left': 1},                        # 冻结行业列
    heatmap={'start': 2, 'end': -1, 'axis': 'column'},  # 数值列着色，汇总列除外
    title='行业轮动收益矩阵')

# 场景：选股结果
nb.table(stock_data,
    columns=['代码', '名称', '收益率', '因子得分', '市值', '行业'],
    freeze={'left': 2},                        # 代码+名称固定
    heatmap={'start': 3, 'end': 5, 'axis': 'column'},  # 数值列着色
    page={'size': 20},                         # 分页
    title='优选股票池')
```

**page 配置：**

```python
page=False                              # 禁用分页（≤20行时推荐）
page={'size': 20}                       # 每页 20 条
page={'size': 20, 'options': [10, 20, 50, 100]}  # 自定义选项
# 不传 page → 默认分页，每页 10 条
```

---

### 2. 指标卡片 `nb.metrics()`

**数据输入：**

```python
# 方式1：List[Dict]（推荐，支持 color）
data = [
    {'name': '总收益率', 'value': '45.2%', 'color': 'green'},
    {'name': '夏普比率', 'value': '1.85'},
    {'name': '最大回撤', 'value': '-12.5%', 'color': 'red'},
]

# 方式2：Dict（便捷，自动转换）
data = {'收益率': '15%', '夏普': '1.5'}

# columns: 每行显示卡片数，默认 4
nb.metrics(data, title='核心指标', columns=4)
```

**颜色支持：** `red`, `green`, `blue`, `yellow`, `orange`, `purple`, `gray` 等

---

### 3. 图表 `nb.chart()`

**chart_type 支持的类型：**

| 类型 | 说明 | 数据格式 |
|------|------|----------|
| `'line'` | 折线图 | 标准格式 / DataFrame |
| `'area'` | 面积图 | 标准格式 / DataFrame |
| `'bar'` | 柱状图 | 标准格式 / DataFrame |
| `'scatter'` | 散点图 | 标准格式 / DataFrame |
| `'kline'` | K线图 | 标准格式 / DataFrame |
| `'pie'` | 饼图 | 列表格式 / DataFrame |
| `'heatmap'` | 热力图 | 嵌套字典 / DataFrame |

#### 3.1 标准格式（line / area / bar / scatter / kline）

```python
{
    'xAxis': ['2024-01', '2024-02', '2024-03'],  # X 轴类目
    'series': [
        {'name': '策略', 'data': [1.0, 1.05, 1.12]},
        {'name': '基准', 'data': [1.0, 1.02, 1.05]},
    ]
}
```

#### 3.2 DataFrame 自动转换规则（按图表类型不同）

**line / area / bar：** 第一列 → xAxis，其余列 → series

```python
# df 列: ['date', 'nav', 'benchmark']
# 转换: xAxis=df['date'], series=[{'name':'nav','data':[...]}, {'name':'benchmark','data':[...]}]
nb.chart('line', df, title='净值曲线')
```

**scatter：** 第一列→名称，第二列→X，第三列→Y，第四列(可选)→气泡大小

```python
# 推荐: DataFrame
# 3列 → 散点图 (name, x, y)
df = pd.DataFrame({'股票': ['茅台','平安'], '波动率': [15.2,8.1], '收益率': [0.82,0.45]})
nb.chart('scatter', df, title='波动率 vs 收益率')
# 自动: X轴名=波动率, Y轴名=收益率, series名=波动率 vs 收益率, tooltip=茅台 (15.2, 0.82)

# 4列 → 气泡图 (name, x, y, size)
df = pd.DataFrame({'股票': ['茅台','平安'], '波动率': [15.2,8.1], '收益率': [0.82,0.45], '权重': [60,30]})
nb.chart('scatter', df, title='气泡图')
# 第4列自动映射气泡大小

# 进阶: pyecharts 原生 dict（类目散点，xAxis 为标签）
{'xAxis': ['A', 'B', 'C'], 'series': [{'name': '', 'data': [10, 20, 30]}]}
# 进阶: pyecharts 原生 dict（数值散点，xAxis 可选）
{'series': [{'name': '', 'data': [[1, 10], [2, 20]]}]}
# 进阶: pyecharts 原生 dict（气泡图）
{'series': [{'name': '', 'data': [[1, 10, 60], [2, 20, 80]]}]}
```

**kline：** 第一列 → xAxis（日期），需包含 open/close/low/high 字段（自动识别中英文列名）

```python
# 标准格式
{
    'xAxis': ['2024-01-01', '2024-01-02'],
    'series': [{'name': 'K线', 'data': [[开,收,低,高], [开,收,低,高]]}]
}
# DataFrame 格式: df 列 ['date', 'open', 'close', 'low', 'high']
# 字段映射: open→['open','开盘','Open'], close→['close','收盘','Close'],
#           low→['low','最低','Low'], high→['high','最高','High']
nb.chart('kline', df_kline, title='K线')
```

**pie：** 第一列 → name，第二列 → value

```python
# 列表格式（标准）
[{'name': '股票', 'value': 60}, {'name': '债券', 'value': 30}]
# DataFrame 格式: df 列 ['类别', '权重'] → name=类别, value=权重
nb.chart('pie', df_pie, title='资产分布')
```

**heatmap：** 第一列 → X 轴，其余列名 → Y 轴

```python
# 嵌套字典格式: {Y: {X: value}}
{'2023': {'1月': 0.02, '2月': -0.01}, '2024': {'1月': 0.05}}
# DataFrame 格式: df 列 ['月份', '策略A', '策略B']
# → X轴=['1月','2月',...], Y轴=['策略A','策略B']
nb.chart('heatmap', df_heatmap, title='月度收益')
```

#### 3.7 图表可选参数

```python
nb.chart('line', data,
    title='净值曲线',           # Cell 标题
    height='400px',             # 容器高度
    width='100%',               # 容器宽度

    # 全局配置（dict 格式，自动转为 pyecharts opts 对象）
    title_opts={'title': '图表标题'},
    legend_opts={'is_show': True},
    tooltip_opts={'trigger': 'axis'},
    xaxis_opts={'name': '日期'},
    yaxis_opts={'min_': 0.9, 'max_': 1.3},
    datazoom_opts=[{'type_': 'slider', 'range_start': 0, 'range_end': 100}],
    visualmap_opts={'min_': 0, 'max_': 100},
    grid_opts={'contain_label': True},

    # 系列配置（统一应用到所有系列）
    series_opts={'is_smooth': True, 'symbol_size': 6},
)
```

---

### 4. Grid 图表 `nb.chartg()`

将多个图表垂直合并为一个 Grid 布局，**累加模式**。
主要用于**同一时间轴的多个数据纵向堆叠**（如净值+仓位+信号）。

```python
nb.chartg('line', nav_data, height=300)
nb.chartg('bar',  pos_data, height=150)
nb.chartg('line', sig_data, height=100)
# 在下一个 cell 或 section 退出时自动合并输出
```

**高度语义（px 绝对定位）：**
- `height=300` → chart 绘图区 300px（不包含 legend 和间距）
- 总高度 = `sum(heights) + gap×N + legend_h×N`
- `gap=30px`（子图间距，后端常量）
- `legend_h=28px`（每子图 legend 预留，后端常量，前端 `GRID_LEGEND_HEIGHT` 同步）

**Grid 特性：**
- 所有子图**共享 xAxis**（联动的 datazoom），各自独立 yAxis
- datazoom 滑块**联动全部子图**
- 各子图的 legend 分别定位在各自 grid 上方 28px
- Y 轴左对齐（统一 `left: 80px`，`containLabel: false`）
- 折线图/面积图从 X 轴起点开始（`boundaryGap: false`），柱状图/K线留白（`true`）

**触发合并的时机：**
- 调用任何非 chartg 的 cell 方法时
- 退出 `with nb.section(...)` 时
- 调用 `export_html()` 时

---

### 5. pyecharts 原生 `nb.pyecharts()`

当 `nb.chart()` 的简化封装不够用时，直接使用 pyecharts 对象：

```python
from pyecharts.charts import Line
from pyecharts import options as opts

line = Line()
line.add_xaxis(['1月', '2月', '3月'])
line.add_yaxis('策略', [1.0, 1.05, 1.12], is_smooth=True)
line.add_yaxis('基准', [1.0, 1.02, 1.04], is_smooth=False)
line.set_global_opts(yaxis_opts=opts.AxisOpts(min_=0.9))

nb.pyecharts(line, title='净值曲线', height='500px')
```

---

### 6. 章节容器 `nb.section()`

```python
# 基础章节
with nb.section("收益分析"):
    nb.metrics([...], title="核心指标")
    nb.chart('line', {...}, title="净值曲线")

# 可折叠章节
with nb.section("详细数据", collapsed=True):   # 默认折叠
    nb.table(data)

# 可折叠但默认展开
with nb.section("风险分析", collapsed=False):
    nb.chart('line', {...})

# 嵌套章节（level 自动递增）
with nb.section("风险分析"):
    with nb.section("回撤分析"):
        nb.chart('line', {...})
```

**collapsed 参数：**
- `None`（默认）：不可折叠
- `True`：可折叠，默认折叠
- `False`：可折叠，默认展开

---

## 输出 JSON-LD 结构

Notebook 最终输出到 HTML `<head>` 中的 JSON-LD 数据：

```python
{
    "@context": "https://schema.org",
    "@type": "Dataset",
    "title": "报告标题",
    "createdAt": "2026-05-11 10:30:00",
    "children": [
        {
            "type": "title",
            "content": "标题文本",
            "options": {"level": 1}
        },
        {
            "type": "metrics",
            "content": [{"name": "收益率", "value": "15%"}],
            "options": {"columns": 4}
        },
        {
            "type": "table",
            "content": [{"code": "000001", ...}],
            "options": {"columns": [{"field": "code", "title": "代码"}], "freeze": {"left": 2}}
        },
        {
            "type": "chart",
            "content": {"charts": {/* echarts option */}, "width": "100%", "height": "400px"}
        },
        {
            "type": "section",
            "title": "章节标题",
            "options": {"level": 1},
            "children": [...]
        }
    ]
}
```

**模板渲染：** 数据以 `<script id="notebook-data" type="application/ld+json">` 注入 `<head>`，AI 优先读取，Vue3 从同一数据源初始化。

---

## 常见模式

### 回测报告

```python
from notebook import Notebook

nb = Notebook("策略回测报告")

nb.metrics({
    '总收益率': '45.2%',
    '年化收益': '18.5%',
    '夏普比率': '1.85',
    '最大回撤': '-12.5%',
}, title="核心指标")

nb.chart('line', {
    'xAxis': dates,
    'series': [
        {'name': '策略', 'data': nav_list},
        {'name': '基准', 'data': benchmark_list},
    ]
}, title='净值曲线', series_opts={'is_smooth': True})

with nb.section("风险分析"):
    nb.chart('line', {
        'xAxis': dates,
        'series': [{'name': '回撤', 'data': drawdown_list}]
    }, title='回撤曲线')

with nb.section("交易明细", collapsed=True):
    nb.table(trade_data, columns=['date', 'code', 'action', 'price', 'amount'])

nb.export_html()
```

### 因子分析报告

```python
nb = Notebook("因子分析报告")

nb.metrics({
    'IC均值': '0.0523',
    'IC标准差': '0.1834',
    'IR': '0.2852',
    'IC>0占比': '62.3%',
}, title="IC统计")

nb.chart('bar', {
    'xAxis': dates,
    'series': [{'name': 'IC', 'data': ic_series}]
}, title='IC序列')

nb.chart('heatmap', monthly_returns_dict, title='月度收益热力图')

with nb.section("分组收益"):
    nb.table(group_data, columns=['group', 'return', 'count'],
             heatmap={'start': 2, 'axis': 'column'})

nb.export_html()
```

### Grid 多图合并（同时间轴）

```python
nb = Notebook("策略综合分析")

# 同一时间轴的多个指标纵向堆叠
nb.chartg('line', nav_data, height=300)   # 净值：chart 绘图区 300px
nb.chartg('bar',  pos_data, height=150)   # 仓位：chart 绘图区 150px
nb.chartg('line', sig_data, height=100)   # 信号：chart 绘图区 100px
nb.divider()  # 触发合并，总高 = 300+150+100 + 30×2 + 28×3 = 694px

nb.export_html()
```

---

## 注意事项

1. **xAxis 键名**：图表标准格式使用 `'xAxis'`（不是 `'x'`）
2. **DataFrame 转换规则（按类型不同）**：line/area/bar → 第一列→xAxis其余→series；kline → 第一列→xAxis+自动识别OHLC字段；pie → 第一列→name第二列→value；heatmap → 第一列→X轴其余列名→Y轴
3. **scatter DataFrame 格式**：3列→散点(name,X,Y)，4列→气泡(第4列=大小)，自动推断轴名+series名
4. **chartg 自动合并**：chartg 是累加模式，在下一个非 chartg 操作时自动 flush
5. **输入即输出原则**：xAxis 日期字符串原样传入，前端用 `category` 类型直接显示，不做时区/格式转换
6. **Grid 高度语义**：`height` = chart 绘图区高度（px），不包含 legend 和间距，后端自动累加
7. **输出路径**：`export_html()` 输出到调用者脚本所在目录，非 notebook 模块目录
8. **链式调用**：所有 cell 方法返回 self，可 `nb.title("A").text("B").divider()`
9. **section 内 title**：在 `with nb.section(...)` 内，cell 的 `title` 参数作为小标题；在 section 外，`title` 参数会自动创建一个单元素 section
10. **pyecharts 延迟导入**：pyecharts 仅在首次使用图表时导入，不影响纯文本/表格场景的性能

---

## AI 建议性用法

### 表格 ft-table 最佳实践

| 场景 | freeze | heatmap | page | 说明 |
|------|--------|---------|------|------|
| 持仓明细 | `left:2` | `start:3` column | `{size:20}` | 代码+名称固定，盈亏列着色 |
| 交易记录 | — | — | `{size:20}` | 纯文本为主，不需要冻结 |
| 行业轮动矩阵 | `left:1` | `start:2,end:-1` column | False | 行业名固定，收益列红涨绿跌 |
| 因子暴露分析 | `left:1` | `start:2` table | False | 全表统一比较，找最强因子 |
| 选股结果 | `left:2` | `start:3,end:5` column | `{size:20}` | 多列数值指标着色 |
| 绩效汇总（≤10行） | — | — | False | 行少直接展示，不用分页 |

**推荐原则：**
- 标识列（代码/名称/日期）→ freeze 左冻结
- 数值列 ≥ 3 → heatmap 着色，`axis: 'column'` 最常见
- 汇总/合计列 → `end: -1` 排除，不参与着色
- 行数 ≤ 20 → `page: False`，一眼看完
- freeze + heatmap 组合是最出彩的表格用法

### Section 层次结构（推荐）

`with nb.section()` 是组织报告层次的核心机制，遵循以下推荐结构：

```python
# ═══════ 推荐报告结构 ═══════
with nb.section("核心指标"):    # 一级 section
    nb.metrics({...})          # 顶层概览，无需 title

with nb.section("收益分析"):    # 一级 section
    nb.chart('line', {...}, title='净值曲线')     # title 作为小标题
    nb.chart('bar', {...}, title='月度收益')

with nb.section("风险分析"):    # 一级 section
    nb.metrics({...}, title='风险指标')
    nb.chart('area', {...}, title='回撤序列')

    with nb.section("VaR 分析"):   # 二级 section（嵌套）
        nb.metrics({...})
        nb.chart('line', {...})

with nb.section("交易明细", collapsed=True):  # 默认折叠
    nb.table(trades)
```

**推荐原则：**

| 原则 | 说明 |
|------|------|
| **顶层放概览** | 报告标题后直接 `nb.metrics()` 放核心 KPI，让读者一眼掌握全局 |
| **一级 section 按主题划分** | 核心指标 / 收益分析 / 风险分析 / 交易分析，每个 `<h2>` 大标题 |
| **二级 section 用于细分** | 风险分析下可嵌套 VaR / 最大回撤 / 波动率 等子主题 |
| **深度 ≤ 2** | 最多嵌套到二级 section，更深会影响可读性 |
| **section 内 cell 加 title** | 在 section 内部，图表/表格的 `title` 作为 `<h3>` 小标题展示 |
| **section 外 cell 也加 title** | 不在 section 内时，cell 的 `title` 会自动创建单元素 section 包裹 |
| **长列表 section 折叠** | 交易明细、原始数据等用 `collapsed=True`，保持报告简洁 |
| **4-6 个一级 section** | 太少内容拥挤，太多层级琐碎，4-6 个主题是最佳平衡 |

**典型报告层次参考：**

```
策略回测报告
├── 核心指标 (metrics, 无 section)         ← 顶层 KPI 卡片
├── 收益分析 (一级 section)
│   ├── 净值曲线 (chart, title='净值曲线')
│   └── 月度收益 (chart, title='月度收益')
├── 风险分析 (一级 section)
│   ├── 风险指标 (metrics, title='风险指标')
│   ├── 回撤序列 (chart, title='回撤序列')
│   └── VaR 分析 (二级 section)
│       └── VaR/CVaR (metrics)
├── 交易分析 (一级 section)
│   ├── 交易统计 (metrics, title='交易统计')
│   └── 盈亏分布 (chart, title='盈亏分布')
└── 交易明细 (一级 section, collapsed=True)  ← 默认折叠
    └── 全部成交记录 (table)
```

**不使用 section 的反模式：**

```python
# ❌ 不推荐：完全不用 section，所有 cell 平铺
nb.metrics({...})
nb.chart('line', {...}, title='净值')
nb.chart('bar', {...}, title='收益')
nb.table(data)
# 缺点：无层次感，阅读体验差，前端无法提供折叠/导航
```

### 图表高度选择

| 场景 | 推荐 height | 说明 |
|------|------------|------|
| 主净值图 | 300-400 | 核心图表，应占较多视觉空间 |
| K线图 | 300 | 蜡烛图需要足够高度才能看清 |
| 柱状图 | 150-200 | 适合月度收益、仓位等 |
| 信号/辅助图 | 100-150 | 次要信息，紧凑即可 |
| Grid 总计 | 500-700 | 2-3 个子图组合的推荐总高 |

### 何时用 chart vs chartg

```
chart()  → 独立图表，需要单独标题/legend/坐标轴
chartg() → 多个图表共享时间轴，纵向堆叠（如净值+仓位+信号）
```

### Grid 高度计算规则

```python
# 后端自动计算，无需手动
total = sum(chart_heights) + 30×(n-1) + 28×n
# 例: 300+150+150 → total = 600 + 30×2 + 28×3 = 744px
```

### 前端渲染规则（AI 无需关心，仅供参考）

- xAxis: Grid 统一 `type='category'`，单图 `time`(日期)/`category`(非日期) 自动检测
- yAxis: line/scatter/kline → `scale:true`（自适应），bar → `scale:false`（从0起步）
- legend: 始终 `type='scroll'`，ECharts 自动根据容器宽度决定是否翻页
- Grid Y轴对齐: `left: 80px` + `containLabel: false`

---

## 依赖

- Python >= 3.8
- pandas
- jinja2
- pyecharts >= 2.1.0

---

> 最后更新：2026-05-30
