"""
最小化 ECharts 图表类 — pyecharts 的替代品

继承 pyecharts 图表类,用法完全一致,仅 dump_options() 输出精简版。
通过 baseline diff 剥离 v5 默认值,只保留用户显式传入的参数,让 ECharts v6 缺省值自动生效。

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


def deep_diff(current: dict, baseline: dict) -> dict:
    """递归比较 current 与 baseline,只保留差异部分(剥离 v5 默认值)"""
    result = {}
    for key, value in current.items():
        if key not in baseline:
            result[key] = value
        elif isinstance(value, dict) and isinstance(baseline[key], dict):
            sub = deep_diff(value, baseline[key])
            if sub:
                result[key] = sub
        elif isinstance(value, list) and isinstance(baseline[key], list):
            diff_list = _diff_list(value, baseline[key])
            if diff_list:
                result[key] = diff_list
        elif value != baseline[key]:
            result[key] = value
    return result


def _preserve_structural(result: dict, current: dict) -> dict:
    """保留结构性必需字段:即使 diff 结果中缺失,也从 current 强制拷回"""
    for key, value in current.items():
        if key in _STRUCTURAL_PRESERVE_KEYS and key not in result:
            result[key] = value
        elif isinstance(value, list) and key in result and isinstance(result[key], list):
            current_items = value
            result_items = result[key]
            for ri, ci in zip(result_items, current_items):
                if isinstance(ci, dict) and isinstance(ri, dict):
                    _preserve_structural(ri, ci)
        elif isinstance(value, dict) and key in result and isinstance(result[key], dict):
            _preserve_structural(result[key], value)
    return result


def _diff_list(current_list: list, baseline_list: list) -> list:
    """逐项比较列表,按 series type 匹配对应类型的 baseline"""
    result = []
    for i, item in enumerate(current_list):
        if isinstance(item, dict):
            base_item = _pick_series_baseline(item, i, baseline_list)
            if isinstance(base_item, dict):
                sub = deep_diff(item, base_item)
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
