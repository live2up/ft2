# Notebook - HTML 报告生成器

> 基于 Jinja2 + Vue3 + ECharts 的轻量级报告生成组件

---

## 特性

- 美观的 HTML 输出 — 基于 Jinja2 模板，数据以 JSON-LD 注入 `<head>`
- 丰富的图表类型 — 折线图、柱状图、饼图、散点图、K线图、热力图
- 智能表格组件 — 支持冻结列、分页、热力图
- 指标卡片 — 一键生成核心指标展示
- 章节容器 — 支持嵌套、折叠的章节结构
- 链式调用 — 流畅的 API 设计

---

## 快速开始

```python
from notebook import Notebook

# 创建报告
nb = Notebook("策略回测报告")

# 添加指标卡片
nb.metrics([
    {'name': '总收益率', 'value': '45.2%', 'color': 'green'},
    {'name': '夏普比率', 'value': '1.85'},
    {'name': '最大回撤', 'value': '-12.5%', 'color': 'red'},
])

# 添加表格
nb.table(
    data=[{'code': '000001', 'name': '平安银行', 'return': 0.15}],
    columns=['code', 'name', 'return'],
    title='持仓明细'
)

# 添加图表
nb.chart('line', {
    'xAxis': ['2024-01', '2024-02', '2024-03'],
    'series': [{'name': '净值', 'data': [1.0, 1.05, 1.12]}]
}, title='净值曲线')

# 导出 HTML
nb.export_html()
```

---

## API 文档

### Notebook 类

```python
Notebook(title: str = "Notebook Report")
```

**参数：** `title` — 报告标题，同时作为默认输出文件名

---

### 指标卡片

```python
nb.metrics(data, title=None, columns=4)
```

**参数：**
- `data` — 指标数据
  - `List[Dict]`：`[{'name': '指标名', 'value': '指标值', 'color': '颜色'}, ...]`
  - `Dict`：`{'指标名': '指标值', ...}`（自动转换）
- `title` — 标题（可选）
- `columns` — 每行显示的卡片数量，默认 4

```python
# List[Dict] 格式（推荐）
nb.metrics([
    {'name': '收益率', 'value': '15%', 'color': 'green'},
    {'name': '夏普', 'value': '1.5'},
])

# Dict 格式（便捷）
nb.metrics({'收益率': '15%', '夏普': '1.5'})
```

颜色支持：`red`, `green`, `blue`, `yellow`, `orange`, `purple`, `gray`

---

### 表格

```python
nb.table(data, columns=None, title=None, **options)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `data` | List[Dict] / DataFrame | 表格数据 |
| `columns` | List[str] | 列名列表，指定显示的列及顺序 |
| `title` | str | 标题（可选） |
| `freeze` | Dict | 冻结列配置，如 `{'left': 2, 'right': 1}` |
| `page` | Dict / False | 分页配置，如 `{'size': 20}` 或 `False` 禁用 |
| `heatmap` | Dict | 热力图配置 |

```python
nb.table(data, columns=['code', 'name', 'return'])
nb.table(data, freeze={'left': 2})
nb.table(data, page={'size': 20, 'options': [10, 20, 50, 100]})
nb.table(data, page=False)
nb.table(data, heatmap={'start': 2, 'end': 5, 'axis': 'column'})
```

---

### 图表

```python
nb.chart(chart_type, data, title=None, height='400px', **kwargs)
```

**支持的图表类型：**

| 类型 | 说明 | 数据格式 |
|------|------|----------|
| `line` | 折线图 | 标准格式 / DataFrame |
| `area` | 面积图 | 标准格式 / DataFrame |
| `bar` | 柱状图 | 标准格式 / DataFrame |
| `scatter` | 散点图 | 标准格式（仅 dict） |
| `kline` | K线图 | 标准格式 / DataFrame |
| `pie` | 饼图 | 列表格式 / DataFrame |
| `heatmap` | 热力图 | 嵌套字典 / DataFrame |

**标准格式（line / area / bar / scatter / kline）：**

```python
{
    'xAxis': ['2024-01', '2024-02', '2024-03'],
    'series': [
        {'name': '策略', 'data': [1.0, 1.05, 1.12]},
        {'name': '基准', 'data': [1.0, 1.02, 1.05]},
    ]
}
```

**DataFrame 自动转换：**

```python
# 第一列 → xAxis，其余列 → series
nb.chart('line', df, title='净值曲线')
```

**饼图：**

```python
[{'name': '股票', 'value': 60}, {'name': '债券', 'value': 30}]
```

**热力图：**

```python
# 嵌套字典：{Y: {X: value}}
{'2023': {'1月': 0.02, '2月': -0.01}, '2024': {'1月': 0.05}}
```

**可选参数：**

```python
nb.chart('line', data,
    title='净值曲线',
    height='400px',
    width='100%',
    title_opts={'title': '图表标题'},
    legend_opts={'is_show': True},
    tooltip_opts={'trigger': 'axis'},
    xaxis_opts={'name': '日期'},
    yaxis_opts={'min_': 0.9, 'max_': 1.3},
    datazoom_opts=[{'type_': 'slider', 'range_start': 0, 'range_end': 100}],
    visualmap_opts={'min_': 0, 'max_': 100},
    grid_opts={'contain_label': True},
    series_opts={'is_smooth': True, 'symbol_size': 6},
)
```

---

### 章节容器

```python
with nb.section(title, collapsed=None):
    nb.metrics([...])
    nb.table(data)
```

- `collapsed=None`（默认）：不可折叠
- `collapsed=True`：可折叠，默认折叠
- `collapsed=False`：可折叠，默认展开

```python
with nb.section("收益分析"):
    nb.metrics([...], title="核心指标")
    nb.chart('line', {...}, title="净值曲线")

with nb.section("详细数据", collapsed=True):
    nb.table(data)

with nb.section("风险分析"):
    with nb.section("回撤分析"):
        nb.chart('line', {...})
```

---

### Grid 图表（多图合并）

```python
nb.chartg(chart_type, data, height=200, **kwargs)
```

将多个图表累加合并为一个 Grid 布局，在下一个非 chartg 操作时自动输出。

```python
nb.chartg('line', data1, height=200)
nb.chartg('bar', data2, height=150)
# 自动合并，总高度 = 200 + 150
```

---

### 文本与布局

```python
nb.title("标题", level=1)          # level: 1-6
nb.text("普通文本")                  # 纯文本
nb.text("红色文本", color='red')     # 带颜色
nb.markdown("## 标题\n- 列表项")     # Markdown 渲染
nb.code("print('hello')", language='python', output='hello')  # 代码块
nb.divider()                        # 分隔线
nb.html("<div>原始HTML</div>")       # 原始 HTML
```

---

### 导出

```python
nb.export_html(name=None, template_path=None)
```

- `name` — 输出文件名（不含扩展名），默认使用标题
- `template_path` — 自定义模板路径

输出到调用者脚本所在目录。

---

## 数据格式

### 表格数据

```python
# List[Dict]（推荐）
data = [
    {'code': '000001', 'name': '平安银行', 'return': 0.15},
    {'code': '000002', 'name': '万科A', 'return': -0.05},
]

# DataFrame（自动转换）
df = pd.DataFrame(data)
```

### 图表数据

```python
# 折线图/柱状图（标准格式）
{
    'xAxis': ['2024-01', '2024-02', '2024-03'],
    'series': [
        {'name': '净值', 'data': [1.0, 1.05, 1.12]},
        {'name': '基准', 'data': [1.0, 1.02, 1.05]},
    ]
}

# 饼图
[{'name': '股票', 'value': 60}, {'name': '债券', 'value': 30}]

# 热力图
{'2023': {'1月': 0.02, '2月': -0.01}, '2024': {'1月': 0.05}}
```

---

## 完整示例

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
    nb.table(trade_data, columns=['date', 'code', 'action', 'price'])

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

---

## 依赖

- Python >= 3.8
- pandas
- jinja2
- pyecharts >= 2.1.0

---

> 最后更新：2026-05-11
