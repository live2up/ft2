# Notebook 模块 AI 助手指南

> **AI 助手请先阅读此文件**，掌握 Notebook 输出模块的调用规范和数据格式
>
> **版本：v2.0.0 | 更新日期：2026-05-11**

---

## 项目定位

Notebook 是 ft2 框架的 **HTML 报告生成器**，基于 Jinja2 + Vue3 + ECharts + ft-table 构建。核心职责是将 Python 数据结构转换为美观的交互式 HTML 报告。

**设计原则：**
- Python 端只负责数据组装，不负责渲染逻辑
- 图表通过 min_pyecharts 构建，输出 ECharts option JSON
- 表格通过 ft-table Vue 组件渲染
- 最终由 Jinja2 模板 + Vue3 组装为完整 HTML
- 数据以 JSON-LD 格式注入 `<head>`，AI 可优先抓取

---

## 文件结构

```
notebook/
├── __init__.py      # 导出 Notebook 类
├── cell.py          # Cell/Section 数据类 + CellBuilder + 图表构建器
├── min_pyecharts.py # 精简版 pyecharts（自研，402行）
├── notebook.py      # Notebook 主类（用户 API 入口）
├── README.md        # 用户文档
└── ai_index.md      # 本文件

template/
├── notebook.html    # Jinja2 HTML 模板（数据以 JSON-LD 注入 <head>）
└── js/
    ├── notebook3C.js   # Vue3 渲染逻辑
    ├── notebook3.css   # 样式
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
nb.export_html(name=None, template_path=None)  # 导出 HTML（name 不含扩展名）
nb.to_json()                                    # 导出 JSON 字符串
nb.to_dict()                                    # 导出字典
```

**输出路径规则：** HTML 文件输出到 **调用者脚本所在目录**（`base_dir`），非 notebook 模块目录。

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
| `heatmap` | `dict` | 热力图配置 | `{'start': 2, 'end': 5, 'axis': 'column'}` |

**page 配置：**

```python
page=False                    # 禁用分页
page={'size': 20}             # 每页 20 条
page={'size': 20, 'options': [10, 20, 50, 100]}  # 自定义选项
# 不传 page → 默认分页，每页 10 条
```

**heatmap 配置：**

```python
heatmap={'start': 2, 'end': 5, 'axis': 'column'}
# start/end: 列索引（1-based，支持负数）
# axis: 'column'（每列独立归一化）或 'table'（全表统一）
# colors: 自定义颜色，默认 A 股配色 ['#2196f3', '#fff', '#f44336']（蓝→白→红）
# columns: 直接指定列名数组（优先级最高）
# exclude: 排除的列索引数组
# excludeRows: 排除的行索引（如 [-1] 排除汇总行）
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
| `'scatter'` | 散点图 | **仅标准格式** |
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

**scatter：** ❌ 不支持 DataFrame，必须用标准格式 dict

```python
# 类目散点图
{'xAxis': ['A', 'B', 'C'], 'series': [{'name': '', 'data': [10, 20, 30]}]}
# 数值散点图
{'xAxis': [], 'series': [{'name': '', 'data': [[1, 10], [2, 20]]}]}
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
nb.chartg('line', nav_data, height=300, yaxis_opts={'min_': 0.9})
nb.chartg('bar', pos_data, height=150)
# 在下一个 cell 或 section 退出时自动合并输出
# 总高度 = sum(heights)
```

**Grid 特性：**
- 所有子图**共享 xAxis**（联动的 datazoom），各自独立 yAxis
- datazoom 滑块**联动全部子图**
- 各子图的 legend 分别定位在各自 grid 顶部
- 最后一个子图显示 x 轴标签

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

### Grid 多图合并（同时间轴)

```python
nb = Notebook("策略综合分析")

# 同一时间轴的多个指标纵向堆叠
nb.chartg('line', nav_data, height=300, yaxis_opts={'min_': 0.9})  # 净值
nb.chartg('bar', pos_data, height=150)                               # 仓位
nb.chartg('line', sig_data, height=100)                              # 信号
# 自动合并为一个 Grid 布局，datazoom 联动全部

nb.export_html()
```

---

## 注意事项

1. **xAxis 键名**：图表标准格式使用 `'xAxis'`（不是 `'x'`）
2. **DataFrame 转换规则（按类型不同）**：line/area/bar → 第一列→xAxis其余→series；kline → 第一列→xAxis+自动识别OHLC字段；pie → 第一列→name第二列→value；heatmap → 第一列→X轴其余列名→Y轴
3. **scatter 不支持 DataFrame**：散点图必须使用标准格式 dict
4. **chartg 自动合并**：chartg 是累加模式，在下一个非 chartg 操作时自动 flush
5. **输出路径**：`export_html()` 输出到调用者脚本所在目录，非 notebook 模块目录
6. **链式调用**：所有 cell 方法返回 self，可 `nb.title("A").text("B").divider()`
7. **section 内 title**：在 `with nb.section(...)` 内，cell 的 `title` 参数作为小标题；在 section 外，`title` 参数会自动创建一个单元素 section
8. **pyecharts 延迟导入**：pyecharts 仅在首次使用图表时导入，不影响纯文本/表格场景的性能

---

## 依赖

- Python >= 3.8
- pandas
- jinja2
- pyecharts >= 2.1.0

---

> 最后更新：2026-05-11
