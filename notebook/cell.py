"""
Cell 模块 - Notebook 的单元格类型定义和构建器

本模块包含：
1. 单元格类型枚举 (CellType)
2. 数据类定义 (Cell, Section)
3. 图表构建器注册表和构建函数
4. 单元格构建器 (CellBuilder)
5. 网格布局构建器 (_build_grid)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import json
import pandas as pd


# =============================================================================
# 第一部分：类型定义
# =============================================================================

class CellType(Enum):
    """单元格类型枚举"""
    # 文本类
    TITLE = "title"
    TEXT = "text"
    MARKDOWN = "markdown"
    # 代码类
    CODE = "code"
    # 数据类
    TABLE = "table"
    METRICS = "metrics"
    # 图表类
    CHART = "chart"
    PYECHARTS = "pyecharts"
    # 布局类
    DIVIDER = "divider"
    HTML = "html"
    SECTION = "section"


CONTAINER_TYPES = {CellType.SECTION}


@dataclass
class Cell:
    """原子类型 Cell - 表示一个不可再分的单元格"""
    type: CellType
    content: Any
    title: Optional[str] = None
    options: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        result = {"type": self.type.value, "content": self.content}
        if self.title is not None:
            result["title"] = self.title
        if self.options:
            result["options"] = self.options
        return result


@dataclass
class Section:
    """容器类型 Section - 可以包含多个 Cell 的章节"""
    children: List[Union[Cell, 'Section']] = field(default_factory=list)
    title: Optional[str] = None
    options: Dict = field(default_factory=dict)
    
    @property
    def type(self) -> CellType:
        return CellType.SECTION
    
    def to_dict(self) -> Dict:
        result = {
            "type": "section",
            "children": [c.to_dict() for c in self.children]
        }
        if self.title is not None:
            result["title"] = self.title
        if self.options:
            result["options"] = self.options
        return result


CellLike = Union[Cell, 'Section']


# =============================================================================
# 第二部分：图表配置转换辅助函数
# =============================================================================

def _create_opts(opts_name: str, opts_dict: dict):
    """
    将 dict 转换为 pyecharts opts 对象
    
    Args:
        opts_name: 配置项名称，如 'title_opts', 'xaxis_opts' 等
        opts_dict: 配置字典
    
    Returns:
        pyecharts options 对象
    """
    from pyecharts import options as opts
    
    opts_map = {
        'title_opts': opts.TitleOpts,
        'legend_opts': opts.LegendOpts,
        'tooltip_opts': opts.TooltipOpts,
        'xaxis_opts': opts.AxisOpts,
        'yaxis_opts': opts.AxisOpts,
        'visualmap_opts': opts.VisualMapOpts,
        'grid_opts': opts.GridOpts,
    }
    
    if opts_name == 'datazoom_opts':
        return [opts.DataZoomOpts(**item) for item in opts_dict]
    
    OptClass = opts_map.get(opts_name)
    return OptClass(**opts_dict) if OptClass else opts_dict


# =============================================================================
# 第三部分：图表注册表
# =============================================================================

CHART_REGISTRY = None


def _init_chart_registry():
    """
    延迟初始化图表注册表，避免 pyecharts 导入开销
    
    注册表结构：
    {
        'chart_type': {
            'class': ChartClass,
            'builder': builder_function
        }
    }
    """
    global CHART_REGISTRY
    if CHART_REGISTRY is not None:
        return
    
    from .min_pyecharts import MinLine, MinBar, MinPie, MinHeatMap, MinKline, MinScatter
    
    CHART_REGISTRY = {
        # ---------- XY 轴系列（通用构建器）----------
        'line': {
            'class': MinLine,
            'builder': lambda data, opts: _build_xy_chart(MinLine, data, opts)
        },
        'perf': {
            'class': MinLine,
            'builder': lambda data, opts: _build_xy_chart(MinLine, data, opts)
        },
        'bar': {
            'class': MinBar,
            'builder': lambda data, opts: _build_xy_chart(MinBar, data, opts)
        },
        'area': {
            'class': MinLine,
            'builder': lambda data, opts: _build_xy_chart(MinLine, data, opts, is_area=True)
        },
        'scatter': {
            'class': MinScatter,
            'builder': _build_scatter
        },
        'kline': {
            'class': MinKline,
            'builder': _build_kline
        },
        
        # ---------- 特殊类型（独立构建器）----------
        'pie': {
            'class': MinPie,
            'builder': _build_pie
        },
        'heatmap': {
            'class': MinHeatMap,
            'builder': _build_heatmap
        },
    }


def get_chart_registry():
    """获取图表注册表（自动初始化）"""
    _init_chart_registry()
    return CHART_REGISTRY


# =============================================================================
# 第四部分：图表构建器实现
# =============================================================================

# ---------- 4.1 通用 XY 轴图表构建器 ----------

def _build_xy_chart(ChartClass, data, series_opts, is_area=False):
    """
    通用 XY 轴图表构建器
    
    支持图表类型：line, bar, area, scatter, kline
    
    Args:
        ChartClass: pyecharts 图表类
        data: 图表数据，支持两种格式：
              1. 标准格式: {'xAxis': [...], 'series': [{'name': '', 'data': []}]}
              2. DataFrame 或嵌套字典（自动转换）
        series_opts: 系列配置选项
        is_area: 是否为面积图（仅对 Line 有效）
    
    Returns:
        pyecharts Chart 对象
    """
    from pyecharts import options as opts
    chart = ChartClass()
    
    # 判断是否为标准格式
    is_standard_format = isinstance(data, dict) and 'xAxis' in data and 'series' in data
    
    if not is_standard_format:
        # 处理非标准格式：DataFrame 或嵌套字典
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.DataFrame(data)
        
        # 【DataFrame转换逻辑】
        # 简化设计：第一列 → xAxis，其余列 → series
        # 无论索引类型如何，始终使用第一列作为X轴
        
        if len(df.columns) == 0:
            raise ValueError("DataFrame 没有列")
        
        # 第一列作为 xAxis
        x_axis = df.iloc[:, 0].astype(str).tolist()
        
        # 其余列作为 series
        series_list = []
        for col in df.columns[1:]:  # 从第二列开始
            series_list.append({
                'name': str(col),
                'data': df[col].tolist()
            })
        
        data = {
            'xAxis': x_axis,
            'series': series_list
        }
    
    # 构建图表
    chart.add_xaxis(data['xAxis'])
    for s in data['series']:
        params = {'series_name': s.get('name', ''), 'y_axis': s['data'], **series_opts}
        if is_area:
            params['areastyle_opts'] = opts.AreaStyleOpts(opacity=0.3)
        chart.add_yaxis(**params)
    
    return chart


# ---------- 4.2 散点图构建器 ----------

def _build_scatter(data, series_opts):
    """
    散点图构建器
    
    注意：散点图不支持 DataFrame 自动转换，请使用标准格式
    
    支持的数据格式：
    1. 类目散点图: {'xAxis': ['A','B','C'], 'series': [{'name': '', 'data': [10,20,30]}]}
    2. 数值散点图: {'xAxis': [], 'series': [{'name': '', 'data': [[1,10], [2,20]]}]}
    
    Args:
        data: 图表数据（仅支持标准格式 dict）
        series_opts: 系列配置选项
    
    Returns:
        pyecharts Scatter 对象
    """
    from .min_pyecharts import MinScatter
    from pyecharts import options as opts
    
    chart = MinScatter()
    
    # 散点图仅支持标准格式
    if not (isinstance(data, dict) and 'xAxis' in data and 'series' in data):
        raise ValueError(
            "散点图不支持 DataFrame 自动转换，请使用标准格式:\n"
            "1. 类目散点图: {'xAxis': ['A','B'], 'series': [{'name': '', 'data': [10,20]}]}\n"
            "2. 数值散点图: {'xAxis': [], 'series': [{'name': '', 'data': [[1,10], [2,20]]}]}"
        )
    
    # 构建图表
    chart.add_xaxis(data['xAxis'])
    for s in data['series']:
        params = {
            'series_name': s.get('name', ''),
            'y_axis': s['data'],
            **series_opts
        }
        chart.add_yaxis(**params)
    
    return chart


# ---------- 4.3 K线图构建器 ----------

def _build_kline(data, series_opts):
    """
    K线图构建器
    
    支持两种数据格式：
    1. 标准格式: {'xAxis': [...], 'series': [{'name': '', 'data': [[开,收,低,高], ...]}]}
    2. DataFrame: 第一列 → X轴（日期），自动识别 open/close/low/high 字段
    
    Args:
        data: 图表数据
        series_opts: 系列配置选项
    
    Returns:
        pyecharts Kline 对象
    """
    from .min_pyecharts import MinKline
    chart = MinKline()
    
    # 判断是否为标准格式
    is_standard_format = isinstance(data, dict) and 'xAxis' in data and 'series' in data
    
    if not is_standard_format:
        # DataFrame 格式转换
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.DataFrame(data)
        
        # 【DataFrame转换逻辑】第一列 → X轴（日期），指定字段 → K线数据
        if len(df.columns) < 5:
            raise ValueError("K线DataFrame需要至少5列：日期 + open/close/low/high")
        
        # 字段映射：支持中英文
        field_map = {
            'open': ['open', '开盘', 'Open', 'OPEN', 'o', 'O'],
            'close': ['close', '收盘', 'Close', 'CLOSE', 'c', 'C'],
            'low': ['low', '最低', 'Low', 'LOW', 'l', 'L'],
            'high': ['high', '最高', 'High', 'HIGH', 'h', 'H']
        }
        
        # 自动查找字段
        def find_field(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            available = list(df.columns)
            raise ValueError(f"找不到字段，候选: {candidates}，可用: {available}")
        
        open_col = find_field(field_map['open'])
        close_col = find_field(field_map['close'])
        low_col = find_field(field_map['low'])
        high_col = find_field(field_map['high'])
        
        # 构建 xAxis 和 K线数据
        # 【统一规范】第一列 → X轴（日期）
        x_axis = df.iloc[:, 0].astype(str).tolist()
        kline_data = []
        for _, row in df.iterrows():
            kline_data.append([
                row[open_col],
                row[close_col],
                row[low_col],
                row[high_col]
            ])
        
        data = {
            'xAxis': x_axis,
            'series': [{
                'name': series_opts.get('name', 'K线'),
                'data': kline_data
            }]
        }
    
    # 构建图表
    chart.add_xaxis(data['xAxis'])
    for s in data['series']:
        chart.add_yaxis(
            series_name=s.get('name', ''),
            y_axis=s['data'],
            **series_opts
        )
    
    return chart


# ---------- 4.4 饼图构建器 ----------

def _build_pie(data, series_opts):
    """
    饼图构建器
    
    支持两种数据格式：
    1. 标准格式: [{'name': '类别', 'value': 数值}, ...]
    2. DataFrame: 第一列 → name，第二列 → value
    
    Args:
        data: 图表数据
        series_opts: 系列配置选项
    
    Returns:
        pyecharts Pie 对象
    """
    from .min_pyecharts import MinPie
    chart = MinPie()
    
    # DataFrame 转换：第一列 → name，第二列 → value
    if isinstance(data, pd.DataFrame):
        names = data.iloc[:, 0].astype(str).tolist()
        values = data.iloc[:, 1].tolist()
        data = [{'name': name, 'value': value} for name, value in zip(names, values)]
    
    # 构建图表
    data_pair = [(item['name'], item['value']) for item in data]
    chart.add('', data_pair, **series_opts)
    return chart


# ---------- 4.5 热力图构建器 ----------

def _build_heatmap(data, series_opts):
    """
    热力图构建器
    
    Args:
        data: DataFrame 或嵌套字典 {y: {x: value}}
        series_opts: 系列配置选项
    
    Returns:
        pyecharts HeatMap 对象
    """
    from .min_pyecharts import MinHeatMap
    chart = MinHeatMap()
    
    # 【核心逻辑】
    # 1. 字典格式：{Y: {X: value}} → 提取 X/Y 轴，转换为 [[x_idx, y_idx, value], ...]
    # 2. DataFrame 格式：第一列作为 X 轴，其余列作为 Y 轴

    if isinstance(data, pd.DataFrame):
        # DataFrame 格式：第一列 → X 轴，其余列 → Y 轴
        if len(data.columns) == 0:
            raise ValueError("DataFrame 没有列")

        # 提取 X 轴数据（第一列）
        xaxis_data = data.iloc[:, 0].astype(str).tolist()

        # 提取 Y 轴数据（列名）
        yaxis_data = data.columns[1:].astype(str).tolist()

        # 重新构建 DataFrame（第一列为索引）
        df_heatmap = data.set_index(data.columns[0])

        # 构建坐标映射
        x_map = {str(x): i for i, x in enumerate(df_heatmap.index)}
        y_map = {str(y): i for i, y in enumerate(df_heatmap.columns)}

        # 转换为 [[x_idx, y_idx, value], ...]
        values = []
        for (x, y), v in df_heatmap.stack().items():
            values.append([x_map[str(x)], y_map[str(y)], v])

        chart.add_xaxis(xaxis_data)
        chart.add_yaxis('', yaxis_data, values, **series_opts)

    else:
        # 字典格式：{Y: {X: value}}
        # 提取 Y 轴数据（外层 keys）
        yaxis_data = list(data.keys())

        # 提取 X 轴数据（内层 keys，去重并保持顺序）
        x_set = set()
        for inner_dict in data.values():
            x_set.update(inner_dict.keys())
        xaxis_data = list(x_set)

        # 构建坐标映射
        x_map = {str(x): i for i, x in enumerate(xaxis_data)}
        y_map = {str(y): i for i, y in enumerate(yaxis_data)}

        # 转换为 [[x_idx, y_idx, value], ...]
        values = []
        for y, inner_dict in data.items():
            for x, v in inner_dict.items():
                values.append([x_map[str(x)], y_map[str(y)], v])

        chart.add_xaxis(xaxis_data)
        chart.add_yaxis('', yaxis_data, values, **series_opts)
    
    return chart


# =============================================================================
# 第五部分：单元格构建器
# =============================================================================

class CellBuilder:
    """
    单元格构建器 - 提供静态方法创建各种类型的 Cell
    
    分类：
    - 文本类：title, text, markdown
    - 代码类：code
    - 数据类：table, metrics
    - 图表类：chart, pyecharts
    - 布局类：divider, html, section
    """
    
    # ---------- 5.1 文本类 ----------
    
    @staticmethod
    def title(text: str, level: int = 1) -> Cell:
        """创建标题单元格"""
        return Cell(CellType.TITLE, text, options={"level": level})
    
    @staticmethod
    def text(text: str, color: str = None) -> Cell:
        """创建文本单元格，color支持: red, green, blue, yellow, orange, purple, gray等"""
        opts = {}
        if color:
            opts["color"] = color
        return Cell(CellType.TEXT, text, options=opts)
    
    @staticmethod
    def markdown(text: str) -> Cell:
        """创建 Markdown 单元格"""
        return Cell(CellType.MARKDOWN, text)
    
    # ---------- 5.2 代码类 ----------
    
    @staticmethod
    def code(code: str, language: str = 'python', output: str = None) -> Cell:
        """创建代码块单元格"""
        return Cell(
            CellType.CODE,
            {"code": code, "language": language, "output": output}
        )
    
    # ---------- 5.3 数据类 ----------
    
    @staticmethod
    def table(data: List[Dict], columns: List[str] = None, options: dict = None) -> Cell:
        """
        创建表格单元格
        
        Args:
            data: 表格数据（List[dict]）
            columns: 列名列表，如 ["代码", "名称"]
            options: 额外选项，如 freeze, page, heatmap 等
        """
        # 将字符串数组转换为对象数组格式
        # 输入: ["代码", "名称"]
        # 输出: [{"field": "代码", "title": "代码"}, {"field": "名称", "title": "名称"}]
        if columns:
            columns = [{"field": col, "title": col} for col in columns]
        
        opts = {"columns": columns} if columns else {}
        
        if options:
            opts.update(options)
        
        return Cell(CellType.TABLE, data, options=opts)
    
    @staticmethod
    def metrics(data: List[Dict], columns: int = 4) -> Cell:
        """创建指标卡片单元格"""
        return Cell(CellType.METRICS, data, options={"columns": columns})
    
    # ---------- 5.4 图表类 ----------
    
    @staticmethod
    def chart(chart_type: str, data, height: str = '400px', **kwargs) -> Cell:
        """
        创建图表单元格（简化封装）
        
        Args:
            chart_type: 图表类型，如 'line', 'bar', 'pie' 等
            data: 图表数据
            height: 容器高度，默认 '400px'
            **kwargs: 额外配置选项
        
        Returns:
            Cell 对象
        """
        width = kwargs.pop('width', '100%')
        
        # 初始化注册表
        _init_chart_registry()
        
        # 查找图表构建器
        spec = CHART_REGISTRY.get(chart_type)
        if not spec:
            supported = list(CHART_REGISTRY.keys())
            raise ValueError(f"不支持的图表类型: {chart_type}，可用: {supported}")
        
        # 提取全局配置和系列配置
        global_opts_keys = [
            'title_opts', 'legend_opts', 'tooltip_opts',
            'xaxis_opts', 'yaxis_opts', 'datazoom_opts',
            'visualmap_opts', 'grid_opts'
        ]
        global_opts = {k: kwargs.pop(k) for k in global_opts_keys if k in kwargs}
        series_opts = kwargs.pop('series_opts', {})
        
        # 构建图表
        chart = spec['builder'](data, series_opts)
        
        # 应用全局配置
        if global_opts:
            chart.set_global_opts(**{k: _create_opts(k, v) for k, v in global_opts.items()})
        
        # 输出为字典
        option_dict = json.loads(chart.dump_options())
        return Cell(
            CellType.CHART,
            {"charts": option_dict, "width": width, "height": height}
        )
    
    @staticmethod
    def pyecharts(chart, height: str = '400px', width: str = '100%') -> Cell:
        """
        创建 pyecharts/min_pyecharts 图表单元格（高级需求，输出自动精简）

        Args:
            chart: pyecharts 或 min_pyecharts 图表对象
            height: 容器高度
            width: 容器宽度
        """
        from .min_pyecharts import minimize_option
        option_dict = json.loads(chart.dump_options())
        option_dict = minimize_option(option_dict)
        
        return Cell(
            CellType.PYECHARTS,
            {
                "charts": option_dict,
                "width": width,
                "height": height
            }
        )
    
    # ---------- 5.5 布局类 ----------
    
    @staticmethod
    def divider() -> Cell:
        """创建分隔线单元格"""
        return Cell(CellType.DIVIDER, None)
    
    @staticmethod
    def html(html_content: str) -> Cell:
        """创建 HTML 单元格"""
        return Cell(CellType.HTML, html_content)
    
    @staticmethod
    def section(title: str, children: List[CellLike] = None,
                level: int = 1, collapsed: bool = None) -> Section:
        """
        创建章节容器
        
        Args:
            title: 章节标题
            children: 子单元格列表
            level: 层级（1-3）
            collapsed: 折叠状态（None=不可折叠, True=默认折叠, False=默认展开）
        """
        opts = {"level": level}
        if collapsed is not None:
            opts["collapsed"] = collapsed
        return Section(children or [], title, opts)


# =============================================================================
# 第六部分：网格布局构建器
# =============================================================================

def _build_grid(charts_config, total_height=600, gap=12, legend_h=20):
    """
    构建网格布局（多个图表垂直排列）

    使用 pyecharts Grid 组织布局, 再经 min_pyecharts 精简输出。
    [重构] 2026-05-30 绝对像素定位：后端负责全部布局
      cum_top += legend_h  →  给 legend 留空间（与 pyecharts 一致）
      cum_top += h         →  chart 绘图区
      cum_top += gap       →  子图间距

    Args:
        charts_config: [{type, data, height, kwargs}, ...]
        total_height: 总高度（px）
        gap: 子图之间的间距（px）
        legend_h: 每个子图 legend 预留高度（px），与前端 GRID_LEGEND_HEIGHT 同步

    Returns:
        ECharts option 字典（已精简）
    """
    import json
    from pyecharts.charts import Grid
    from pyecharts import options as opts
    from .min_pyecharts import minimize_grid_option

    _init_chart_registry()

    grid = Grid()
    cum_top = 0
    for i, cfg in enumerate(charts_config):
        chart_type = cfg['type']
        data = cfg['data']
        kwargs = cfg.get('kwargs', {})
        series_opts = kwargs.pop('series_opts', {})

        spec = CHART_REGISTRY.get(chart_type)
        if not spec:
            raise ValueError(f"不支持的图表类型: {chart_type}")

        chart = spec['builder'](data, series_opts)
        chart.set_global_opts(tooltip_opts=opts.TooltipOpts(trigger='axis'))

        global_opts_keys = [
            'title_opts', 'legend_opts', 'tooltip_opts',
            'xaxis_opts', 'yaxis_opts', 'datazoom_opts',
            'visualmap_opts', 'grid_opts'
        ]
        global_opts = {k: kwargs.pop(k) for k in global_opts_keys if k in kwargs}
        if global_opts:
            chart.set_global_opts(**{k: _create_opts(k, v) for k, v in global_opts.items()})

        cum_top += legend_h                         # legend 空间
        h = cfg['height']
        grid.add(chart, grid_opts=opts.GridOpts(
            pos_top=f"{cum_top}px", height=f"{h}px",
            is_contain_label=True,
        ))
        cum_top += h + gap                          # chart + 间距

    option_dict = json.loads(grid.dump_options())
    option_dict = minimize_grid_option(option_dict, charts_config, gap=gap, legend_h=legend_h)
    return option_dict
