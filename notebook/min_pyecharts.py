"""
最小化 ECharts 图表类 — pyecharts 的替代品

继承 pyecharts 图表类,用法完全一致,仅 dump_options() 输出精简版。
通过 baseline diff 剥离 v5 默认值,只保留用户显式传入的参数,让 ECharts v6 缺省值自动生效。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
实现逻辑
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

super().dump_options()                    ← pyecharts v5 完整输出
        │
        ▼
json.loads()                              ← 解析为 current dict
        │
        ├─ 含: 用户数据 + 用户配置 + pyecharts 默认配置
        │
        ▼
_create_reactive_baseline(ChartClass)     ← 生成 baseline dict
        │
        ├─ ChartClass()                   ← 空白图表实例
        ├─ _add_minimal_data()            ← 添加最小数据触发完整输出
        ├─ _build_core_opts()             ← 设置基础 opts
        ─ 含: 最小数据 + pyecharts 默认配置（无用户配置）
        │
        ▼
deep_diff(current, baseline, path=())     ← 核心对比逻辑
        │
        ├─ path in _DATA_PATHS            ← 核心数据，直接保留，跳过 diff
        │   保护: series.data, xAxis.data, series.name, title.text
        │
        ├─ key not in baseline            ← 用户新增配置，保留
        │
        ├─ value 是 dict                  ← 递归 deep_diff
        │
        ├─ value 是 list                  ← _diff_config_list 逐项 diff
        │
        └─ value != baseline[key]         ← 用户修改配置，保留
        │
        ▼
_strip_reactive_opts()                    ← 补空标记（toolbox/dataZoom 等）
        │
        ▼
_preserve_structural()                    ← 强制保留 type 等结构字段
        │
        ▼
deep_merge(self._extra)                   ← 合并 v6 专属参数
        │
        ▼
输出 JSON                                 ← 仅用户数据 + 用户配置

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

直接替代 pyecharts 的用法:
    # 旧: from pyecharts.charts import Line
    # 新: from min_pyecharts import MinLine
    chart = MinLine()
    chart.add_xaxis(['1月', '2月'])
    chart.add_yaxis('策略', [1.0, 1.05])
    chart.set_global_opts(title_opts=opts.TitleOpts(title="策略净值"))
    chart.dump_options()  # 自动精简,无 v5 默认值

工厂函数用法:
    chart = MinECharts('line')
    chart.add_xaxis(['1月', '2月'])
    chart.add_yaxis('策略', [1.0, 1.05])
    chart.dump_options()

任意 pyecharts 图表类型:
    from pyecharts.charts import Radar
    MinRadar = make_min_class(Radar)
    chart = MinRadar()
    ...
    chart.dump_options()
"""
import json
from pyecharts.charts import Line, Bar, Pie, HeatMap, Kline, Scatter
from pyecharts.charts.chart import RectChart
from pyecharts import options as opts
from pyecharts.commons.utils import remove_key_with_none_value


_STRUCTURAL_PRESERVE_KEYS = {'type'}

# [重构] 2026-05-21 基于路径的数据保护白名单
# 精确到层级，避免误伤配置 key
_DATA_PATHS = frozenset({
    ('series', 'data'),        # series[i].data → 核心数据
    ('xAxis', 'data'),         # xAxis[i].data → X 轴标签
    ('yAxis', 'data'),         # yAxis[i].data → Y 轴数据
    ('legend', 'data'),        # legend[i].data → 图例名称
    ('title', 'text'),         # title[i].text → 标题文本
    ('series', 'name'),        # series[i].name → 系列名称
})


def deep_diff(current: dict, baseline: dict, path=()) -> dict:
    """递归比较 current 与 baseline,只保留差异部分(剥离 v5 默认值)
    
    [重构] 2026-05-21 基于路径保护核心数据
    """
    result = {}
    for key, value in current.items():
        current_path = path + (key,)
        
        # 核心数据路径 → 直接保留，不做 diff
        if current_path in _DATA_PATHS:
            result[key] = value
            continue
        
        if key not in baseline:
            result[key] = value
        elif isinstance(value, dict) and isinstance(baseline[key], dict):
            sub = deep_diff(value, baseline[key], current_path)
            if sub:
                result[key] = sub
        elif isinstance(value, list) and isinstance(baseline[key], list):
            diff_list = _diff_config_list(value, baseline[key], current_path)
            if diff_list:
                result[key] = diff_list
        elif value != baseline[key]:
            result[key] = value
    return result


def _preserve_structural(result: dict, current: dict, path=()) -> dict:
    """保留结构性必需字段:即使 diff 结果中缺失,也从 current 强制拷回
    
    [重构] 2026-05-21 同步加入 path 参数，跳过数据路径
    """
    for key, value in current.items():
        current_path = path + (key,)
        
        # 数据路径已在 deep_diff 中保留，无需处理
        if current_path in _DATA_PATHS:
            continue
        
        if key in _STRUCTURAL_PRESERVE_KEYS and key not in result:
            result[key] = value
        elif isinstance(value, list) and key in result and isinstance(result[key], list):
            current_items = value
            result_items = result[key]
            for ri, ci in zip(result_items, current_items):
                if isinstance(ci, dict) and isinstance(ri, dict):
                    _preserve_structural(ri, ci, current_path)
        elif isinstance(value, dict) and key in result and isinstance(result[key], dict):
            _preserve_structural(result[key], value, current_path)
    return result


def _diff_config_list(current_list: list, baseline_list: list, parent_path: tuple) -> list:
    """逐项比较配置列表(元素为 dict),按 series type 匹配对应类型的 baseline
    
    [重构] 2026-05-21 接收 parent_path，传给 deep_diff
    """
    result = []
    for i, item in enumerate(current_list):
        if isinstance(item, dict):
            base_item = _pick_series_baseline(item, i, baseline_list)
            if isinstance(base_item, dict):
                sub = deep_diff(item, base_item, parent_path)
                if sub:
                    result.append(sub)
            else:
                result.append(item)
        elif i < len(baseline_list) and item == baseline_list[i]:
            pass
        elif i >= len(baseline_list) or item != baseline_list[i]:
            result.append(item)
    return result


def _pick_series_baseline(item: dict, index: int, baseline_list: list) -> dict:
    """为 series dict 选择最合适的 baseline 项"""
    if index < len(baseline_list) and isinstance(baseline_list[index], dict):
        if item.get("type") == baseline_list[index].get("type"):
            return baseline_list[index]
    series_type = item.get("type", "")
    type_baseline = _get_series_type_baseline(series_type)
    if type_baseline:
        return type_baseline
    if baseline_list and isinstance(baseline_list[-1], dict):
        return baseline_list[-1]
    return {}


def deep_merge(base: dict, extra: dict) -> dict:
    """递归合并 extra 到 base(extra 覆盖 base)"""
    result = dict(base)
    for key, value in extra.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _opt_to_dict(opt_obj) -> dict:
    """将 pyecharts Opts 对象转为清理后的 dict"""
    if isinstance(opt_obj, list):
        return [remove_key_with_none_value(item.opts) if hasattr(item, 'opts') else item for item in opt_obj]
    return remove_key_with_none_value(opt_obj.opts) if hasattr(opt_obj, 'opts') else opt_obj


_CORE_BASELINE_CACHE: dict = {}
_OPT_BASELINE_CACHE: dict = {}
_SERIES_TYPE_BASELINE_CACHE: dict = {}


def _build_core_opts(c) -> dict:
    """根据图表实例构建核心 baseline 的 global_opts 参数"""
    core_opts = dict(
        title_opts=opts.TitleOpts(),
        legend_opts=opts.LegendOpts(),
        tooltip_opts=opts.TooltipOpts(),
    )
    if isinstance(c, RectChart):
        is_kline = isinstance(c, Kline)
        core_opts['xaxis_opts'] = opts.AxisOpts(is_scale=is_kline)
        core_opts['yaxis_opts'] = opts.AxisOpts(is_scale=is_kline)
    return core_opts


def _get_series_type_baseline(series_type: str) -> dict:
    """获取指定 series type 的默认 series dict"""
    if series_type in _SERIES_TYPE_BASELINE_CACHE:
        return _SERIES_TYPE_BASELINE_CACHE[series_type]
    chart_key = _SERIES_TYPE_TO_CHART_KEY.get(series_type)
    if not chart_key:
        _SERIES_TYPE_BASELINE_CACHE[series_type] = {}
        return {}
    ChartClass = CHART_MAP.get(chart_key)
    if not ChartClass:
        _SERIES_TYPE_BASELINE_CACHE[series_type] = {}
        return {}
    c = ChartClass()
    _add_minimal_data(c)
    core_opts = _build_core_opts(c)
    c.set_global_opts(**core_opts)
    full_dict = json.loads(c.dump_options())
    series_list = full_dict.get("series", [])
    if series_list:
        _SERIES_TYPE_BASELINE_CACHE[series_type] = series_list[0]
        return series_list[0]
    _SERIES_TYPE_BASELINE_CACHE[series_type] = {}
    return {}


def _opt_baseline_dict(ChartClass, **opt_kwargs) -> dict:
    """通过创建临时图表获取某个 opt 的完整默认值 dict"""
    cache_key = (ChartClass, tuple(sorted(opt_kwargs.keys())))
    if cache_key in _OPT_BASELINE_CACHE:
        return _OPT_BASELINE_CACHE[cache_key]

    c = ChartClass()
    _add_minimal_data(c)
    core_opts = _build_core_opts(c)
    core_opts.update(opt_kwargs)
    c.set_global_opts(**core_opts)
    full_dict = json.loads(c.dump_options())

    _OPT_NAME_TO_JSON_KEY = {
        'toolbox_opts': 'toolbox',
        'brush_opts': 'brush',
        'visualmap_opts': 'visualMap',
        'datazoom_opts': 'dataZoom',
        'grid_opts': 'grid',
        'axispointer_opts': 'axisPointer',
        'matrix_opts': 'matrix',
        'thumbnail_opts': 'thumbnail',
    }
    result = {}
    for key in opt_kwargs:
        json_key = _OPT_NAME_TO_JSON_KEY.get(key)
        if json_key and json_key in full_dict:
            result[json_key] = full_dict[json_key]

    _OPT_BASELINE_CACHE[cache_key] = result
    return result


_OPT_KEY_TO_FACTORY = {
    'toolbox':     lambda cc: _opt_baseline_dict(cc, toolbox_opts=opts.ToolboxOpts()),
    'brush':       lambda cc: _opt_baseline_dict(cc, brush_opts=opts.BrushOpts()),
    'visualMap':   lambda cc: _opt_baseline_dict(cc, visualmap_opts=opts.VisualMapOpts()),
    'dataZoom':    lambda cc: _opt_baseline_dict(cc, datazoom_opts=[opts.DataZoomOpts()]),
    'grid':        lambda cc: _opt_baseline_dict(cc, grid_opts=opts.GridOpts()),
    'axisPointer': lambda cc: _opt_baseline_dict(cc, axispointer_opts=opts.AxisPointerOpts()),
    'matrix':      lambda cc: _opt_baseline_dict(cc, matrix_opts=opts.MatrixOpts()),
    'thumbnail':   lambda cc: _opt_baseline_dict(cc, thumbnail_opts=opts.ThumbnailOpts()),
}

_REACTIVE_OPT_MINIMAL = {
    'toolbox': {},
    'brush': {},
    'visualMap': {},
    'dataZoom': [{}],
    'grid': {},
    'axisPointer': {},
    'matrix': {},
    'thumbnail': {},
}


def _add_minimal_data(c):
    """自动检测图表类型并添加最小数据"""
    fn = _MINIMAL_DATA_REGISTRY.get(type(c))
    if fn:
        fn(c)
        return
    if isinstance(c, RectChart):
        c.add_xaxis(['_'])
        try:
            c.add_yaxis('_', [0])
        except TypeError:
            try:
                c.add_yaxis('_', ['_'], [['_', '_', 0]])
            except TypeError:
                try:
                    c.add_yaxis('_', [[0, 0, 0, 0]])
                except TypeError:
                    pass
    elif hasattr(c, 'add'):
        try:
            c.add('_', [('_', 0)])
        except TypeError:
            pass


def _rect_minimal_data(c):
    c.add_xaxis(['_'])
    c.add_yaxis('_', [0])


def _kline_minimal_data(c):
    c.add_xaxis(['_'])
    c.add_yaxis('_', [[0, 1, 0, 1]])


def _heatmap_minimal_data(c):
    c.add_xaxis(['_'])
    c.add_yaxis('_', ['_'], [[0, 0, 0]])


def _pie_minimal_data(c):
    c.add('_', [('_', 0)])


_MINIMAL_DATA_REGISTRY = {
    Line: _rect_minimal_data,
    Bar: _rect_minimal_data,
    Scatter: _rect_minimal_data,
    Kline: _kline_minimal_data,
    HeatMap: _heatmap_minimal_data,
    Pie: _pie_minimal_data,
}


def _create_core_baseline(ChartClass) -> dict:
    """创建核心 baseline(仅包含 pyecharts 始终输出的 opts 默认值)"""
    if ChartClass in _CORE_BASELINE_CACHE:
        return _CORE_BASELINE_CACHE[ChartClass]

    c = ChartClass()
    _add_minimal_data(c)
    core_opts = _build_core_opts(c)
    c.set_global_opts(**core_opts)
    _CORE_BASELINE_CACHE[ChartClass] = json.loads(c.dump_options())
    return _CORE_BASELINE_CACHE[ChartClass]


def _create_reactive_baseline(ChartClass, current_options: dict) -> dict:
    """创建响应式 baseline(核心 baseline + 用户实际设置的 opts 默认值)"""
    baseline = dict(_create_core_baseline(ChartClass))

    for key in current_options:
        if key not in baseline:
            factory = _OPT_KEY_TO_FACTORY.get(key)
            if factory:
                baseline.update(factory(ChartClass))

    return baseline


def _strip_reactive_opts(result: dict, current: dict) -> dict:
    """对响应式 opts 做二次精简:diff 后若为空则保留最小标记"""
    for key, minimal in _REACTIVE_OPT_MINIMAL.items():
        if key in current and key not in result:
            result[key] = minimal
    return result


def make_min_class(ChartClass):
    """为指定图表类动态生成继承版本,重写 dump_options 剥离默认值"""
    _cc = ChartClass

    class MinChart(ChartClass):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._extra = {}

        def update_options(self, extra: dict):
            """注入 v6 专属参数(pyecharts 不支持的选项)"""
            self._extra = deep_merge(self._extra, extra)

        def dump_options_dict(self) -> dict:
            """输出精简的 option dict(仅用户参数)"""
            current = json.loads(super().dump_options())
            baseline = _create_reactive_baseline(_cc, current)
            result = deep_diff(current, baseline)
            result = _strip_reactive_opts(result, current)
            result = _preserve_structural(result, current)
            if self._extra:
                result = deep_merge(result, self._extra)
            return result

        def dump_options(self) -> str:
            """输出精简的 option JSON(仅用户参数)"""
            return json.dumps(self.dump_options_dict(), ensure_ascii=False)

    MinChart.__name__ = f'Min{ChartClass.__name__}'
    MinChart.__qualname__ = f'Min{ChartClass.__name__}'
    return MinChart


_SERIES_TYPE_TO_CHART_KEY = {
    'line': 'line',
    'bar': 'bar',
    'scatter': 'scatter',
    'candlestick': 'kline',
    'heatmap': 'heatmap',
    'pie': 'pie',
}

CHART_MAP = {
    'line':    Line,
    'bar':     Bar,
    'scatter': Scatter,
    'kline':   Kline,
    'heatmap': HeatMap,
    'pie':     Pie,
}


def MinECharts(chart_type: str = 'line', **kwargs):
    """工厂函数:创建继承 pyecharts 的精简图表实例"""
    ChartClass = CHART_MAP.get(chart_type)
    if not ChartClass:
        raise ValueError(f"不支持的图表类型: {chart_type}, 可用: {list(CHART_MAP.keys())}")
    MinClass = make_min_class(ChartClass)
    return MinClass(**kwargs)


MinLine    = make_min_class(Line)
MinBar     = make_min_class(Bar)
MinScatter = make_min_class(Scatter)
MinKline   = make_min_class(Kline)
MinHeatMap = make_min_class(HeatMap)
MinPie     = make_min_class(Pie)


def minimize_option(option_dict: dict) -> dict:
    """对已有的 option dict 做精简(不需要创建 MinChart 实例)"""
    # [优化] 2026-05-19 Grid 布局自动走 minimize_grid_option
    if option_dict.get('grid'):
        return minimize_grid_option(option_dict)

    series = option_dict.get('series', [])
    chart_key = None
    if series:
        series_type = series[0].get('type', '')
        chart_key = _SERIES_TYPE_TO_CHART_KEY.get(series_type)
    if chart_key is None:
        return option_dict

    ChartClass = CHART_MAP.get(chart_key)
    if not ChartClass:
        return option_dict

    baseline = _create_reactive_baseline(ChartClass, option_dict)
    result = deep_diff(option_dict, baseline)
    result = _strip_reactive_opts(result, option_dict)
    result = _preserve_structural(result, option_dict)
    return result


_GRID_BASELINE_CACHE: dict = {}


def _build_grid_baseline(charts_config: list, gap: int = 12, legend_space: int = 24) -> dict:
    """
    为 Grid 布局构建同结构 baseline

    [重构] 2026-05-30 改为绝对像素定位，与 _build_grid 保持一致
    
    Args:
        charts_config: [{'type': str, 'height': int, 'kwargs': dict}, ...]
        gap: 子图间距（px）
        legend_space: legend 区域高度（px）
    """
    from pyecharts.charts import Grid

    cache_key = (gap, legend_space) + tuple((c['type'], c['height']) for c in charts_config)
    if cache_key in _GRID_BASELINE_CACHE:
        return _GRID_BASELINE_CACHE[cache_key]

    baseline_grid = Grid()
    cum_top = legend_space
    for i, cfg in enumerate(charts_config):
        ChartClass = CHART_MAP.get(cfg['type'])
        if not ChartClass:
            ChartClass = Line
        c = ChartClass()
        _add_minimal_data(c)
        core_opts = _build_core_opts(c)
        c.set_global_opts(**core_opts)

        h = cfg['height']
        baseline_grid.add(
            c,
            grid_opts=opts.GridOpts(
                pos_top=f"{cum_top}px",
                height=f"{h}px",
            )
        )
        cum_top += h + gap

    baseline_dict = json.loads(baseline_grid.dump_options())
    _GRID_BASELINE_CACHE[cache_key] = baseline_dict
    return baseline_dict


def minimize_grid_option(option_dict: dict, charts_config: list = None, gap: int = 12, legend_space: int = 24) -> dict:
    """
    对 Grid 布局的 option dict 做精简
    
    和 minimize_option 的区别: Grid 包含多种 series type,
    需要构建"同结构 baseline"(同数量子图、同类型、同位置)才能正确 diff。
    
    Args:
        option_dict: Grid 布局的 option dict(含 grid/xAxis/yAxis/series 等)
        charts_config: 子图配置列表,用于构建同结构 baseline
            [{'type': str, 'height': int, 'kwargs': dict}, ...]
            如果为 None, 从 option_dict 中自动推断
        gap: 子图间距（px），需与 _build_grid 保持一致
        legend_space: legend 区域高度（px）

    [新增] 2026-05-19 替代 _build_grid 的手工合并方式,
    让 pyecharts Grid 输出后走 deep_diff 精简, 维护边界更清晰
    """
    if charts_config is None:
        charts_config = _infer_charts_config(option_dict)

    baseline = _build_grid_baseline(charts_config, gap=gap, legend_space=legend_space)
    result = deep_diff(option_dict, baseline)
    result = _strip_reactive_opts(result, option_dict)
    result = _preserve_structural(result, option_dict)
    result = _preserve_grid_structural(result, option_dict)
    return result


_GRID_STRUCTURAL_KEYS = {'gridIndex', 'xAxisIndex', 'yAxisIndex'}


def _preserve_grid_structural(result: dict, current: dict) -> dict:
    """
    Grid 布局结构性字段保护

    gridIndex/xAxisIndex/yAxisIndex 即使和 baseline 相同也不能移除,
    因为 ECharts 默认值为 0, Grid 多子图场景下必须显式指定索引。
    grid 数组也必须保留(含 top/height 定位信息)。
    yAxis/xAxis 条目即使被完全精简也需要保留 gridIndex 占位,
    否则 ECharts 无法将轴与 grid 正确关联。

    [修复] 2026-05-19 新增 grid top/height 和 xAxis/yAxis type 的保护
    旧逻辑只在 result_grids 为空时才补充 top/height,但 deep_diff 后 grid
    数组非空(如 containLabel 存活),导致 top/height 永远补不回来。
    同样 xAxis/yAxis 的 type 字段被 deep_diff 剥离后也需补回。
    """
    _GRID_LAYOUT_KEYS = {'top', 'height', 'containLabel'}
    _GRID_AXIS_KEYS = {'type', 'name', 'gridIndex'}

    for key in ('series', 'xAxis', 'yAxis'):
        current_list = current.get(key, [])
        result_list = result.get(key, [])
        for i, ci in enumerate(current_list):
            if not isinstance(ci, dict):
                continue
            if i >= len(result_list):
                if any(k in ci for k in _GRID_STRUCTURAL_KEYS):
                    result_list.append({k: ci[k] for k in _GRID_STRUCTURAL_KEYS if k in ci})
            elif isinstance(result_list[i], dict):
                for sk in _GRID_STRUCTURAL_KEYS:
                    if sk in ci and sk not in result_list[i]:
                        result_list[i][sk] = ci[sk]
        if current_list and key in ('xAxis', 'yAxis'):
            # 确保 xAxis/yAxis 条目数 >= current, 每个至少保留 gridIndex
            if not isinstance(result.get(key), list):
                result[key] = []
            result_list = result[key]
            for i, ci in enumerate(current_list):
                if i >= len(result_list):
                    result_list.append({'gridIndex': ci.get('gridIndex', i)})
                else:
                    # [修复] 补回 type 等关键字段
                    for ak in _GRID_AXIS_KEYS:
                        if ak in ci and ak not in result_list[i]:
                            result_list[i][ak] = ci[ak]

    # [修复] 确保 grid 数组的 top/height/containLabel 不被剥离
    current_grids = current.get('grid', [])
    if isinstance(current_grids, list) and current_grids:
        if not isinstance(result.get('grid'), list):
            result['grid'] = []
        # 保证 grid 条目数一致, 逐项合并 layout 字段
        for i, g in enumerate(current_grids):
            if i >= len(result['grid']):
                result['grid'].append(
                    {k: v for k, v in g.items() if k in _GRID_LAYOUT_KEYS}
                )
            elif isinstance(result['grid'][i], dict):
                for lk in _GRID_LAYOUT_KEYS:
                    if lk in g and lk not in result['grid'][i]:
                        result['grid'][i][lk] = g[lk]

    return result


def _infer_charts_config(option_dict: dict) -> list:
    """
    从 Grid option dict 中推断 charts_config(自动推断模式)
    
    从 series 类型 + grid 数量反推子图配置,
    用于没有显式 charts_config 的场景(如 CellBuilder.pyecharts 传入 Grid)
    """
    series_list = option_dict.get('series', [])
    grid_list = option_dict.get('grid', [])

    if not series_list:
        return []

    n_grids = len(grid_list) or 1
    default_height = 200

    configs = []
    for i in range(n_grids):
        if i < len(series_list):
            series_type = series_list[i].get('type', 'line')
        else:
            series_type = 'line'
        chart_key = _SERIES_TYPE_TO_CHART_KEY.get(series_type, 'line')

        grid_item = grid_list[i] if i < len(grid_list) else {}
        height_str = grid_item.get('height', '45%')
        try:
            height_val = int(float(height_str.rstrip('%')) * default_height / 45)
        except (ValueError, TypeError):
            height_val = default_height

        configs.append({
            'type': chart_key,
            'height': height_val,
            'kwargs': {}
        })

    return configs
