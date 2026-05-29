/**
 * Notebook Vue3 - Vue 3 组合式 API 版本的 Notebook 应用逻辑
 * 组件模式 + Composable 复用逻辑
 * 
 * v2.0 — 2026-05-17
 *   新增: 滚动轴(line/area/bar/kline)、全屏(五个组件)
 *   重构: float → flex 统一布局
 *   优化: 全局ESC、updateChart不合并、滚动轴切换不重建
 */

const { createApp, ref, computed, watch, onMounted, onUnmounted, nextTick } = Vue;

// [优化] 2026-05-18 全局ESC注册表(仅一个keydown监听器)
window.__fsEsc = new Set();

// [优化] 2026-05-27 全局 resize 防抖管理器（解决多图表卡顿问题）
// 原理：所有图表共用一个 resize 监听，通过 requestAnimationFrame 批量执行
// 延迟约 16ms（1帧），肉眼几乎无感知，不影响图表交互
window.__resizeManager = (() => {
    let instances = new Set();
    let rafId = null;
    let inited = false;

    const scheduleResize = () => {
        if (rafId) return;
        rafId = requestAnimationFrame(() => {
            instances.forEach(fn => fn());
            rafId = null;
        });
    };

    const init = () => {
        if (inited) return;
        inited = true;
        window.addEventListener('resize', scheduleResize);
        window.addEventListener('colorSchemeChanged', scheduleResize);
    };

    return {
        register(fn) {
            init();
            instances.add(fn);
            return () => instances.delete(fn);
        }
    };
})();

// [提取] 2026-05-18 默认色板常量，消除 useChart/GridChart 中的硬编码重复
const DEFAULT_COLORS = ['#e74c3c', '#f39c12', '#af7ac5', '#5499c7', '#f4d03f', '#82e0aa'];

// [优化] 2026-05-27 提取公共 ECharts 配置，提升性能且便于统一维护
const COMMON_CHART_OPTIONS = {
    animation: false
};

// =============================================================================
// 第一部分：Chart Composable - 图表组件共用逻辑
// =============================================================================

function useChart(props, chartOptions = {}) {
    const chartRef = ref(null);
    let chartInstance = null;

    const getColors = (chartType) => {
        const colorPalettes = window.colorPalettes;
        if (!colorPalettes) return DEFAULT_COLORS;
        const group = colorPalettes.typeToGroup?.[chartType] || 'chart';
        const paletteKey = colorPalettes.groups?.[group] || colorPalettes.global;
        const palette = colorPalettes.palettes?.[paletteKey];
        return palette ? palette.colors : DEFAULT_COLORS;
    };

    const extractData = (charts) => {
        if (!charts?.series?.[0]) return null;

        // [修复] 2026-05-19 pyecharts Line/Bar add_yaxis 合并 xAxis+y 为 2D 格式
        // 检测: data[0] 是 [x, y] 两元素数组 → 转回 1D (x 已在 xAxis.data 中)
        // [修复] 2026-05-27 scatter 散点图数据保持 [[x,y]] 格式，不做转换
        const series = (charts.series || []).map(s => {
            const d = s.data;
            if (!Array.isArray(d) || d.length === 0) return s;
            const first = d[0];
            if (Array.isArray(first) && first.length === 2 && s.type !== 'candlestick' && s.type !== 'scatter') {
                return { ...s, data: d.map(item => item[1]) };
            }
            return s;
        });

        return {
            chart_type: series[0]?.type || charts.series[0].type,
            series: series,
            xAxis: charts.xAxis?.[0]?.data || [],
            yAxis: charts.yAxis?.[0]?.data || [],
            raw: { ...charts, series }
        };
    };

    const buildOption = chartOptions.buildOption || ((extracted, colors) => ({ color: colors, series: extracted.series }));

    const getChartOption = () => {
        const charts = props.cell.content?.charts;
        if (!charts) return {};
        const extracted = extractData(charts);
        if (!extracted) return charts;
        const colors = getColors(extracted.chart_type);
        return buildOption(extracted, colors);
    };

    const initChart = () => {
        if (!chartRef.value || !props.cell.content?.charts) return;
        if (chartInstance) chartInstance.dispose();
        chartInstance = echarts.init(chartRef.value);
        chartInstance.setOption(getChartOption());
    };

    // [优化] 2026-05-17 不合并,避免dataZoom等组件残留
    const updateChart = () => {
        if (!chartInstance) return;
        chartInstance.setOption(getChartOption(), true);
    };

    watch(() => props.cell.content?.charts, () => {
        if (chartInstance) updateChart();
    });

    const handleResize = () => chartInstance?.resize();

    // [优化] 2026-05-27 使用全局 resize 管理器防抖，避免多图表同时 resize 卡顿
    let unregisterResize = null;

    onMounted(() => {
        nextTick(() => initChart());
        unregisterResize = window.__resizeManager.register(handleResize);
        window.addEventListener('colorSchemeChanged', initChart);
    });

    onUnmounted(() => {
        if (unregisterResize) unregisterResize();
        window.removeEventListener('colorSchemeChanged', initChart);
        chartInstance?.dispose();
    });

    return { chartRef, initChart, updateChart, getChartOption, getColors, extractData, refreshChart: initChart };
}

// [提取] 2026-05-18 全屏 composable，消除 GenericChart/HeatmapChart/StackedChart/GridChart 中的重复
function useFullscreen() {
    const isFullscreen = ref(false);
    const toggleFullscreen = () => {
        isFullscreen.value = !isFullscreen.value;
        document.body.style.overflow = isFullscreen.value ? 'hidden' : '';
        setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
    };
    const handleKeydown = () => { if (isFullscreen.value) toggleFullscreen(); };
    onMounted(() => { window.__fsEsc.add(handleKeydown); });
    onUnmounted(() => {
        window.__fsEsc.delete(handleKeydown);
        document.body.style.overflow = '';
    });
    return { isFullscreen, toggleFullscreen };
}

// =============================================================================
// 第二部分：图表组件
// =============================================================================

// ---------- GenericChart - 通用图表 ----------
const GenericChart = {
    name: 'GenericChart',
    props: { cell: { type: Object, required: true } },
    setup(props) {
        const showDataZoom = ref(false);
        const showBarLabel = ref(true);         // Bar/柱状图：显示数值标签
        const intervalCompare = ref(false);  // 区间收益模式
        const intervalStart = ref(0);         // dataZoom start%（缓存当前滑块位置）
        const intervalEnd = ref(100);         // dataZoom end%
        let intervalListener = null;          // dataZoom 事件监听器
        const { isFullscreen, toggleFullscreen } = useFullscreen();

        const { chartRef, updateChart } = useChart(props, {
            buildOption: (extracted, colors) => {
                const chartType = extracted.chart_type;
                const rawSeries = extracted.series || [];  // 原始数据（不被修改）
                const option = { color: colors, series: rawSeries };
                if (['line', 'bar', 'area'].includes(chartType)) {
                    const isBarChart = chartType === 'bar';
                    const first = extracted.xAxis[0];
                    const isDateStr = typeof first === 'string' && /^\d{4}-\d{2}-\d{2}/.test(first);
                    const xAxisType = isDateStr ? 'time' : 'category';
                    option.xAxis = { type: xAxisType, boundaryGap: isBarChart, data: extracted.xAxis };
                    option.yAxis = { type: 'value', scale: !isBarChart };  // [修复] 2026-05-21 柱状图必须 scale:false 才能让 min:0 生效，否则 auto-scale 会忽略 min 导致柱子不从 0 起步
                    // [优化] 2026-05-22 legend自适应：策略<=10用plain换行，>10用scroll翻页
                    const legendCount = rawSeries.length;
                    const legendType = legendCount > 10 ? 'scroll' : 'plain';
                    option.grid = {
                        left: 8, right: 40,
                        bottom: showDataZoom.value ? 50 : 5,
                        top: legendCount > 10 ? 32 : 50,
                        containLabel: true
                    };
                    option.legend = {
                        data: rawSeries.map(s => ({ name: s.name, icon: 'rect' })),
                        top: 5, type: legendType, width: '90%',
                        pageIconSize: 10, pageTextStyle: { fontSize: 11 }
                    };
                    option.tooltip = { trigger: 'axis', axisPointer: { type: 'shadow' } };

                    // === 区间收益：基于滚动区间将原始数据转为累计收益百分比（仅 line/area） ===
                    let displaySeries = rawSeries;  // 默认不转换
                    if (intervalCompare.value && (chartType === 'line' || chartType === 'area')) {
                        const dataLen = rawSeries[0]?.data?.length || 0;
                        // [修复] 如果没有启用滚动轴，默认用 0-100%（从第一个数据点开始）
                        const startPercent = showDataZoom.value ? intervalStart.value : 0;
                        const startIdx = Math.floor(dataLen * startPercent / 100);
                        const baseIdx = startIdx > 0 ? startIdx - 1 : 0;  // 基准点：可见起点前一个数据点
                        
                        // [修复] 区间收益模式：保留原始xAxis日期，仅在pyecharts 2D格式且原始xAxis为空时才从data提取
                        const firstDataPoint = rawSeries[0]?.data?.[0];
                        const is2DFormat = Array.isArray(firstDataPoint) && firstDataPoint.length >= 2;
                        const xData = is2DFormat ? (rawSeries[0]?.data || []).map(v => v[0]) : extracted.xAxis;
                        option.xAxis = {
                            type: 'category', boundaryGap: false, data: xData
                        };
                        
                        // [核心修复] 使用 Array.from() 创建新数组，避免 Proxy 对象引用问题
                        displaySeries = Array.from(rawSeries).map((s, seriesIdx) => {
                            // [核心修复] 处理 pyecharts 二维数组格式 [[日期, 净值], ...]
                            const numericData = (s.data || []).map(v => {
                                if (Array.isArray(v) && v.length >= 2) {
                                    // pyecharts 格式：[日期, 净值]，取第二个元素
                                    return parseFloat(v[1]);
                                } else if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
                                    // 对象格式：{value: 1.05} 或 {date: '2020-01-02', value: 1.05}
                                    return parseFloat(v.value || v[1] || 0);
                                }
                                // 一维数组：直接是数值
                                return parseFloat(v);
                            }).filter(v => !isNaN(v));  // 过滤掉 NaN
                            
                            const baseValue = numericData[baseIdx];
                            
                            // [核心修复] 计算累计收益率：所有数据点相对于基准点的收益率
                            // baseIdx 之前的数据点填充 null（不显示）
                            const returnsData = numericData.map((v, idx) => {
                                if (idx < baseIdx) return null;  // 基准点之前的数据不显示
                                if (!baseValue || baseValue === 0) return 0;  // 基准值为 0 时返回 0
                                return parseFloat(((v - baseValue) / baseValue * 100).toFixed(4));
                            });
                            
                            // [核心修复] 创建全新的对象，避免 Proxy 引用问题
                            return {
                                name: s.name,
                                type: s.type,
                                data: returnsData,
                                stack: s.stack
                            };
                        });
                        
                        // Y轴百分比标注，区间收益模式固定从0%起步
                        option.yAxis = {
                            type: 'value', scale: false,
                            axisLabel: { formatter: '{value}%' }
                        };
                    }

                    // 统一构建 series（displaySeries 可能是原始数据或转换后的数据）
                    option.series = displaySeries.map((s, i) => {
                        
                        const baseOption = { name: s.name, type: chartType === 'area' ? 'line' : chartType, data: s.data };
                        if (s.stack) baseOption.stack = s.stack;
                        if (chartType === 'line' || chartType === 'area') {
                            baseOption.smooth = true;
                            baseOption.showSymbol = false;
                            if (chartType === 'area') baseOption.areaStyle = { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: colors[i % colors.length] + '60' }, { offset: 1, color: colors[i % colors.length] + '10' }] } };
                        }
                        if (isBarChart) baseOption.itemStyle = { color: rawSeries.length === 1 && !s.stack ? function(params) { return params.value >= 0 ? colors[0] : colors[1]; } : colors[i % colors.length], borderRadius: [4, 4, 0, 0] };
                        if (chartType === 'bar') baseOption.label = { show: showBarLabel.value, position: 'top' };
                        return baseOption;
                    });
                    
                } else if (chartType === 'scatter') {
                    // [修复] 2026-05-27 散点图轴类型：有 xAxis.data 用 category（类目散点），否则用 value（数值散点）
                    const hasXData = extracted.xAxis && extracted.xAxis.length > 0;
                    option.xAxis = { type: hasXData ? 'category' : 'value', data: hasXData ? extracted.xAxis : [] };
                    option.yAxis = { type: 'value' };
                } else if (chartType === 'candlestick') {
                    // [新增] 2026-05-27 K线图支持
                    // K线图X轴用 category，因为 pyecharts 输出的 data 是 [[开,收,低,高]] 不含日期
                    // 日期在 xAxis.data 中，用 category 类型显示
                    option.xAxis = { type: 'category', data: extracted.xAxis, boundaryGap: true };
                    option.yAxis = { type: 'value', scale: true };
                    // K线图需要指定 series 类型为 candlestick
                    option.series = extracted.series.map(s => ({
                        name: s.name,
                        type: 'candlestick',
                        data: s.data
                    }));
                }

                // [修复] 2026-05-27 dataZoom 对所有支持图表类型生效（line/bar/area/scatter/candlestick）
                if (showDataZoom.value && (chartType === 'line' || chartType === 'area' || chartType === 'bar' || chartType === 'scatter' || chartType === 'candlestick')) {
                    // [区间收益] 开启时保留当前滑块位置，不重置为 0-100
                    const dzStart = intervalCompare.value ? intervalStart.value : 0;
                    const dzEnd = intervalCompare.value ? intervalEnd.value : 100;
                    option.dataZoom = [
                        { type: 'inside', xAxisIndex: [0], start: dzStart, end: dzEnd },
                        { type: 'slider', show: true, xAxisIndex: [0], start: dzStart, end: dzEnd, bottom: 10, height: 20 }
                    ];
                } else {
                    option.dataZoom = [];  // 显式清空,避免setOption合并残留
                }
                return { ...COMMON_CHART_OPTIONS, ...option };
            }
        });

        watch([showDataZoom, showBarLabel], updateChart);

        // === 区间收益逻辑 ===
        const handleDataZoom = () => {
            const instance = chartRef.value ? echarts.getInstanceByDom(chartRef.value) : null;
            if (instance) {
                const opt = instance.getOption();
                const zoom = (opt.dataZoom || []).find(z => z.type === 'slider') || (opt.dataZoom || [])[0];
                intervalStart.value = zoom?.start ?? 0;
                intervalEnd.value = zoom?.end ?? 100;
            }
            updateChart();
        };

        watch(intervalCompare, (val) => {
            if (val) {
                // 开启：先读取当前 dataZoom 位置，再注册监听
                handleDataZoom();
                if (!intervalListener) {
                    intervalListener = handleDataZoom;
                    const instance = chartRef.value ? echarts.getInstanceByDom(chartRef.value) : null;
                    if (instance) instance.on('dataZoom', intervalListener);
                }
            } else {
                // 关闭：移除监听，恢复原始数据
                if (intervalListener) {
                    const instance = chartRef.value ? echarts.getInstanceByDom(chartRef.value) : null;
                    if (instance) instance.off('dataZoom', intervalListener);
                    intervalListener = null;
                }
                updateChart();
            }
        });

        onUnmounted(() => {
            if (intervalListener) {
                const instance = chartRef.value ? echarts.getInstanceByDom(chartRef.value) : null;
                if (instance) instance.off('dataZoom', intervalListener);
                intervalListener = null;
            }
        });

        return { chartRef, showDataZoom, showBarLabel, isFullscreen, toggleFullscreen, intervalCompare, chartType: computed(() => { const charts = props.cell.content?.charts; return charts?.series?.[0]?.type || 'line'; }) };
    },
    template: `
        <div class="cell-chart" :class="{ 'chart-zoomed': isFullscreen }">
            <h3 v-if="cell.title">{{ cell.title }}</h3>
            <div class="cell-chart-body">
                <div class="chart-wrapper">
                    <div ref="chartRef" class="chart-container chart-container-main"
                         :style="{ width: cell.content?.width || '100%', height: cell.content?.height || '400px' }"></div>
                    <div class="chart-toolbar">
                        <!-- 上方：视图控制 -->
                        <button class="tool-btn" :class="{ active: isFullscreen }" @click="toggleFullscreen" :title="isFullscreen ? '退出全屏' : '全屏'">⛶</button>
                        <!-- 中间：数据操作 -->
                        <button v-if="chartType === 'bar'" class="tool-btn" :class="{ active: showBarLabel }" @click="showBarLabel = !showBarLabel" title="显示数值">V</button>
                        <button v-if="chartType === 'line' || chartType === 'area'" class="tool-btn" :class="{ active: intervalCompare }" @click="intervalCompare = !intervalCompare" title="区间收益">%</button>
                        <!-- 下方：滚动轴（靠近底部滑块） -->
                        <button class="tool-btn" :class="{ active: showDataZoom }" @click="showDataZoom = !showDataZoom" title="滚动轴">⇄</button>
                    </div>
                </div>
            </div>
        </div>
    `
};

// ---------- PieChart ----------
const PieChart = {
    name: 'PieChart',
    props: { cell: { type: Object, required: true } },
    setup(props) {
        const pieShowValue = ref(true);
        const pieShowPercent = ref(true);

        const { chartRef, refreshChart } = useChart(props, {
            buildOption: (extracted, colors) => {
                const data = extracted.series[0]?.data || [];
                let labelFormatter = '{b}';
                if (pieShowValue.value && pieShowPercent.value) labelFormatter = '{b}\n{c} ({d}%)';
                else if (pieShowValue.value) labelFormatter = '{b}\n{c}';
                else if (pieShowPercent.value) labelFormatter = '{b}\n({d}%)';
                return {
                    ...COMMON_CHART_OPTIONS,
                    color: colors,
                    legend: { data: data.map((item, i) => ({ name: item.name, itemStyle: { color: colors[i % colors.length] } })), top: 10, left: 'center', orient: 'horizontal' },
                    series: [{ type: 'pie', data: data, radius: ['40%', '70%'], center: ['45%', '55%'], label: { show: true, formatter: labelFormatter }, labelLine: { show: true, length: 15, length2: 10 }, emphasis: { label: { show: true, fontSize: 14, fontWeight: 'bold' } } }]
                };
            }
        });

        watch([pieShowValue, pieShowPercent], refreshChart);

        return { chartRef, pieShowValue, pieShowPercent };
    },
    template: `
        <div class="cell-chart">
            <h3 v-if="cell.title">{{ cell.title }}</h3>
            <div class="cell-chart-body">
                <div class="chart-wrapper">
                    <div ref="chartRef" class="chart-container chart-container-main"
                         :style="{ width: cell.content?.width || '100%', height: cell.content?.height || '400px' }"></div>
                    <div class="chart-toolbar">
                        <!-- 中间：数据操作 -->
                        <button class="tool-btn" :class="{ active: pieShowValue }" @click="pieShowValue = !pieShowValue" title="显示数值">V</button>
                        <button class="tool-btn" :class="{ active: pieShowPercent }" @click="pieShowPercent = !pieShowPercent" title="显示百分比">%</button>
                    </div>
                </div>
            </div>
        </div>
    `
};

// ---------- HeatmapChart ----------
const HeatmapChart = {
    name: 'HeatmapChart',
    props: { cell: { type: Object, required: true } },
    setup(props) {
        const heatmapShowData = ref(true);
        const heatmapMultiplier = ref(1);
        const { isFullscreen, toggleFullscreen } = useFullscreen();

        const { chartRef, updateChart } = useChart(props, {
            buildOption: (extracted, colors) => {
                const rawData = extracted.series[0]?.data || [];
                const multiplier = heatmapMultiplier.value;
                const HEATMAP_COLORS = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'];
                let minValue = Infinity, maxValue = -Infinity;
                const displayData = rawData.map(d => { const scaled = d[2] * multiplier; if (scaled < minValue) minValue = scaled; if (scaled > maxValue) maxValue = scaled; return [d[0], d[1], scaled]; });
                const valueRange = maxValue - minValue;
                let step = 0.01, decimalPlaces = 2;
                if (valueRange >= 10) { step = 5; decimalPlaces = 0; }
                else if (valueRange >= 1) { step = 0.5; decimalPlaces = 1; }
                const visualMin = Math.floor(minValue / step) * step;
                const visualMax = Math.ceil(maxValue / step) * step;
                return {
                    ...COMMON_CHART_OPTIONS,
                    grid: { left: '10%', right: '18%', top: '10%', bottom: '12%' },
                    xAxis: { type: 'category', data: extracted.xAxis, splitArea: { show: true } },
                    yAxis: { type: 'category', data: extracted.yAxis, splitArea: { show: true } },
                    visualMap: { min: visualMin, max: visualMax, range: [visualMin, visualMax], calculable: true, orient: 'vertical', right: '2%', top: 'center', text: [visualMax.toFixed(decimalPlaces) + ' (×' + multiplier + ')', visualMin.toFixed(decimalPlaces) + ' (×' + multiplier + ')'], inRange: { color: HEATMAP_COLORS } },
                    series: [{ type: 'heatmap', data: displayData, label: { show: heatmapShowData.value, formatter: params => params.value[2].toFixed(2) }, emphasis: { itemStyle: { shadowBlur: 10 } } }]
                };
            }
        });

        watch([heatmapShowData, heatmapMultiplier], () => { updateChart(); });

        return { chartRef, heatmapShowData, heatmapMultiplier, isFullscreen, toggleFullscreen };
    },
    template: `
        <div class="cell-chart heatmap" :class="{ 'chart-zoomed': isFullscreen }">
            <h3 v-if="cell.title">{{ cell.title }}</h3>
            <div class="cell-chart-body">
                <div class="chart-wrapper">
                    <div ref="chartRef" class="chart-container chart-container-main"
                         :style="{ width: cell.content?.width || '100%', height: cell.content?.height || '400px' }"></div>
                    <div class="chart-toolbar">
                        <!-- 上方：视图控制 -->
                        <button class="tool-btn" :class="{ active: isFullscreen }" @click="toggleFullscreen" :title="isFullscreen ? '退出全屏' : '全屏'">⛶</button>
                        <!-- 中间：数据操作 -->
                        <button class="tool-btn" :class="{ active: heatmapShowData }" @click="heatmapShowData = !heatmapShowData" title="显示数据">T</button>
                        <!-- 下方：热力图专用 -->
                        <select class="tool-select" v-model.number="heatmapMultiplier" title="数据缩放">
                            <option value="1000">x1000</option>
                            <option value="100">x100</option>
                            <option value="10">x10</option>
                            <option value="1">原始</option>
                            <option value="0.1">1/10</option>
                            <option value="0.01">1/100</option>
                        </select>
                    </div>
                </div>
            </div>
        </div>
    `
};

// ---------- StackedChart ----------
// 【识别逻辑】父组件 ChartContainer 检测到 series 中有 stack 属性时，使用此组件
// 【关键问题】堆叠柱状图必须使用 scale: false 强制从0开始，否则小数值系列会被压缩看不见
const StackedChart = {
    name: 'StackedChart',
    props: { cell: { type: Object, required: true } },
    setup(props) {
        const stackNormalize = ref(false);
        const stackShowRaw = ref(true);
        const stackShowPercent = ref(false);
        const { isFullscreen, toggleFullscreen } = useFullscreen();

        const { chartRef, updateChart } = useChart(props, {
            buildOption: (extracted, colors) => {
                const chartType = extracted.chart_type;
                const series = extracted.series || [];
                const { normalize, showRaw, showPercentStack } = { normalize: stackNormalize.value, showRaw: stackShowRaw.value, showPercentStack: stackShowPercent.value };
                
                // 【数据处理】提取原始数据并计算每列总和（用于百分比计算）
                const rawData = series.map(s => [...(s.data || [])]);
                const dataLength = rawData[0]?.length || 0;
                const totals = new Array(dataLength).fill(0);
                rawData.forEach(sData => { sData.forEach((v, i) => { totals[i] = (totals[i] || 0) + (v || 0); }); });
                
                // 【标签格式化】支持显示原始值、百分比或两者
                const buildLabelFormatter = (rawData, totals, showRaw, showPercent) => (seriesIndex) => (params) => {
                    const rawValue = rawData[seriesIndex][params.dataIndex];
                    const total = totals[params.dataIndex];
                    const percent = total > 0 ? (rawValue / total * 100).toFixed(1) : 0;
                    if (showRaw && showPercent) return `${rawValue}\n(${percent}%)`;
                    else if (showRaw) return String(rawValue);
                    else if (showPercent) return `${percent}%`;
                    return '';
                };
                
                // 【提示框格式化】显示总计和各系列详情
                const buildTooltipFormatter = (rawData, totals, showRaw, showPercent) => (params) => {
                    const xValue = params[0].axisValue;
                    const total = totals[params[0].dataIndex];
                    let result = `<strong>${xValue}</strong><br/><div style="color:#666;margin-bottom:4px;">总计: ${total}</div>`;
                    params.forEach(p => {
                        const rawValue = rawData[p.seriesIndex][p.dataIndex];
                        const percent = total > 0 ? (rawValue / total * 100).toFixed(1) : 0;
                        let label = `${p.seriesName}: `;
                        if (showRaw && showPercent) label += `${rawValue} (${percent}%)`;
                        else if (showRaw) label += rawValue;
                        else if (showPercent) label += `${percent}%`;
                        else label += rawValue;
                        result += `${p.marker} ${label}<br/>`;
                    });
                    return result;
                };
                
                const labelFormatter = buildLabelFormatter(rawData, totals, showRaw, showPercentStack);
                const tooltipFormatter = buildTooltipFormatter(rawData, totals, showRaw, showPercentStack);
                
                let displaySeries, yAxisConfig;
                
                if (normalize) {
                    // 【归一化模式】数据转为百分比，Y轴固定0-100
                    displaySeries = series.map((s, i) => ({ ...s, data: rawData[i].map((v, j) => totals[j] > 0 ? (v / totals[j] * 100) : 0), type: chartType === 'area' ? 'line' : chartType }));
                    yAxisConfig = { type: 'value', min: 0, max: 100, axisLabel: { formatter: '{value}%' } };
                } else {
                    // 【原始数据模式】关键：柱状图必须从0开始
                    displaySeries = series.map((s, i) => ({ ...s, data: [...rawData[i]], type: chartType === 'area' ? 'line' : chartType }));
                    
                    // 【关键修复】堆叠柱状图必须强制从0开始，否则小数值系列（如72 vs 172）会被压缩看不见
                    // scale: true 会让Y轴自适应最小值，导致小数值几乎不可见
                    const isBarChart = chartType === 'bar';
                    yAxisConfig = { 
                        type: 'value', 
                        scale: !isBarChart,              // 柱状图禁用自适应，强制从0开始
                        boundaryGap: isBarChart ? [0, '10%'] : ['10%', '10%']  // 柱状图底部无间隙
                    };
                }
                const isBarChart = chartType === 'bar';
                const legendCount = series.length;
                const legendType = legendCount > 10 ? 'scroll' : 'plain';
                const option = {
                    color: colors,
                    xAxis: { type: 'category', boundaryGap: isBarChart, data: extracted.xAxis },
                    yAxis: yAxisConfig,
                    grid: { left: 8, right: 40, bottom: 5, top: legendCount > 10 ? 32 : 40, containLabel: true },
                    legend: {
                        data: series.map(s => ({ name: s.name, icon: 'rect' })),
                        top: 5, type: legendType, width: '90%',
                        pageIconSize: 10, pageTextStyle: { fontSize: 11 }
                    },
                    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: tooltipFormatter },
                    series: displaySeries.map((s, i) => {
                        const baseOption = {
                            name: s.name,
                            type: chartType === 'area' ? 'line' : chartType,
                            data: s.data, stack: s.stack,
                            label: { show: showRaw || showPercentStack, position: 'inside', formatter: labelFormatter(i) }
                        };
                        if (chartType === 'line' || chartType === 'area') {
                            baseOption.smooth = true;
                            baseOption.showSymbol = false;
                            if (chartType === 'area') {
                                baseOption.areaStyle = {
                                    color: {
                                        type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
                                        colorStops: [
                                            { offset: 0, color: colors[i % colors.length] + '60' },
                                            { offset: 1, color: colors[i % colors.length] + '10' }
                                        ]
                                    }
                                };
                            }
                        }
                        if (isBarChart) baseOption.itemStyle = { color: colors[i % colors.length], borderRadius: [4, 4, 0, 0] };
                        return baseOption;
                    })
                };
                return { ...COMMON_CHART_OPTIONS, ...option };
            }
        });

        watch([stackNormalize, stackShowRaw, stackShowPercent], updateChart);

        return { chartRef, stackNormalize, stackShowRaw, stackShowPercent, isFullscreen, toggleFullscreen };
    },
    template: `
        <div class="cell-chart" :class="{ 'chart-zoomed': isFullscreen }">
            <h3 v-if="cell.title">{{ cell.title }}</h3>
            <div class="cell-chart-body">
                <div class="chart-wrapper">
                    <div ref="chartRef" class="chart-container chart-container-main"
                         :style="{ width: cell.content?.width || '100%', height: cell.content?.height || '400px' }"></div>
                    <div class="chart-toolbar">
                        <!-- 上方：视图控制 -->
                        <button class="tool-btn" :class="{ active: isFullscreen }" @click="toggleFullscreen" :title="isFullscreen ? '退出全屏' : '全屏'">⛶</button>
                        <!-- 中间：数据操作 -->
                        <button class="tool-btn" :class="{ active: stackNormalize }" @click="stackNormalize = !stackNormalize" title="归一化">N</button>
                        <button class="tool-btn" :class="{ active: stackShowRaw }" @click="stackShowRaw = !stackShowRaw" title="显示数值">V</button>
                        <button class="tool-btn" :class="{ active: stackShowPercent }" @click="stackShowPercent = !stackShowPercent" title="显示百分比">%</button>
                    </div>
                </div>
            </div>
        </div>
    `
};

// ---------- GridChart ----------
const GridChart = {
    name: 'GridChart',
    props: { cell: { type: Object, required: true } },
    setup(props) {
        const chartRef = ref(null);
        let chartInstance = null;
        const showDataZoom = ref(false);
        const intervalCompare = ref(false);     // 区间收益：仅转换第一个子图（xAxisIndex=0）
        const intervalStart = ref(0);
        const intervalEnd = ref(100);
        let intervalListener = null;
        // 判断第一个子图类型，仅 line/area 支持区间收益
        const firstGridType = computed(() => {
            const series = props.cell.content?.charts?.series || [];
            const first = series.find(s => (s.xAxisIndex ?? s.gridIndex ?? 0) === 0);
            return first?.type || 'line';
        });
        const hasIntervalCompare = computed(() =>
            ['line', 'area'].includes(firstGridType.value)
        );
        const { isFullscreen, toggleFullscreen } = useFullscreen();

        const getColors = () => {
            const colorPalettes = window.colorPalettes;
            if (!colorPalettes) return DEFAULT_COLORS;
            const paletteKey = colorPalettes.groups?.chart || colorPalettes.global;
            const palette = colorPalettes.palettes?.[paletteKey];
            return palette ? palette.colors : DEFAULT_COLORS;
        };

        const buildGridOption = () => {
            const charts = props.cell.content?.charts;
            if (!charts) return {};
            const colors = getColors();
            const option = JSON.parse(JSON.stringify(charts));
            option.color = colors;
            option.tooltip = { trigger: 'axis', axisPointer: { type: 'shadow' } };

            // [适配] pyecharts Grid 输出适配: legend 数组 → 每个对应其 grid 位置
            // legend[i] 对应 grid[i], 放在各自 grid 顶部
            if (option.legend && Array.isArray(option.legend)) {
                const grids = option.grid || [];
                option.legend = option.legend.map((leg, i) => {
                    const g = grids[i] || {};
                    const topPct = g.top ? parseFloat(g.top) - 3 : 5;
                    return {
                        data: (leg.data || []).map(d => ({
                            name: typeof d === 'string' ? d : (d.name || d),
                            icon: 'rect'
                        })),
                        top: Math.max(0, topPct) + '%',
                        left: 'center',
                        textStyle: { fontSize: 11 }
                    };
                });
            } else if (option.legend) {
                const legendData = option.legend.data || [];
                const legendCount = legendData.length;
                option.legend.icon = 'rect';
                option.legend.width = '90%';
                // [优化] 2026-05-22 策略>10用scroll翻页，否则plain换行
                if (legendCount > 10) {
                    option.legend.type = 'scroll';
                    option.legend.pageIconSize = 10;
                    option.legend.pageTextStyle = { fontSize: 11 };
                }
            }

            // [适配] xAxis 补 type + axisLabel 防溢出
            if (option.xAxis && Array.isArray(option.xAxis)) {
                option.xAxis = option.xAxis.map(x => {
                    if (!x.type && x.data) {
                        const first = String(x.data[0] || '');
                        x.type = /^\d{4}-\d{2}-\d{2}/.test(first) ? 'time' : 'category';
                    }
                    x.axisLabel = { margin: 8, ...(x.axisLabel || {}) };
                    return x;
                });
            }

            // [适配] yAxis 补 type + scale
            if (option.yAxis && Array.isArray(option.yAxis)) {
                option.yAxis = option.yAxis.map(y => ({ type: 'value', ...y }));  // [修复] 2026-05-21 移除 scale:true，默认 scale:false 让柱状图从 0 起步；折线图如需 auto-scale 可显式传 scale_: true
            }

            if (option.grid && Array.isArray(option.grid)) {
                option.grid = option.grid.map(g => ({ ...g, left: 8, right: 40, top: g.top, height: g.height }));
            }
            if (option.series && Array.isArray(option.series)) {
                option.series.forEach(s => { if (s.type === 'line') { s.smooth = true; s.showSymbol = false; } });
            }

            // [新增] Grid 区间收益：仅转换第一个子图的 line/area series
            if (intervalCompare.value && hasIntervalCompare.value) {
                const firstGridSeries = (option.series || []).filter(s =>
                    (s.xAxisIndex ?? s.gridIndex ?? 0) === 0
                );
                if (firstGridSeries.length > 0) {
                    const dataLen = firstGridSeries[0]?.data?.length || 0;
                    const startPercent = showDataZoom.value ? intervalStart.value : 0;
                    const startIdx = Math.floor(dataLen * startPercent / 100);
                    const baseIdx = startIdx > 0 ? startIdx - 1 : 0;

                    if (option.yAxis && Array.isArray(option.yAxis) && option.yAxis[0]) {
                        option.yAxis[0] = { ...option.yAxis[0], axisLabel: { formatter: '{value}%' } };
                    }

                    option.series = (option.series || []).map(s => {
                        const idx = s.xAxisIndex ?? s.gridIndex ?? 0;
                        if (idx !== 0) return s;
                        const numericData = (s.data || []).map(v => {
                            if (Array.isArray(v) && v.length >= 2) return parseFloat(v[1]);
                            if (typeof v === 'object' && v !== null) return parseFloat(v.value || 0);
                            return parseFloat(v);
                        }).filter(v => !isNaN(v));
                        const baseValue = numericData[baseIdx];
                        const returnsData = numericData.map((v, i) => {
                            if (i < baseIdx) return null;
                            if (!baseValue || baseValue === 0) return 0;
                            return parseFloat(((v - baseValue) / baseValue * 100).toFixed(4));
                        });
                        return { ...s, data: returnsData };
                    });
                }
            }

            if (showDataZoom.value) {
                const gridLen = (option.grid || []).length;
                const allIdx = Array.from({length: gridLen}, (_, i) => i);
                const dc = [];
                // 一个 inside 联动所有 grid
                if (allIdx.length) dc.push({ type: 'inside', xAxisIndex: allIdx, start: 0, end: 100 });
                // 只在最后 grid 底部显示滑块，联动所有 grid
                if (allIdx.length) {
                    dc.push({ type: 'slider', show: true, xAxisIndex: allIdx,
                              start: 0, end: 100, bottom: 6, height: 16 });
                    const lastGrid = option.grid[gridLen - 1];
                    lastGrid.bottom = 22;
                    delete lastGrid.height;
                }
                option.dataZoom = dc;
            } else {
                option.dataZoom = [];
            }
            return { ...COMMON_CHART_OPTIONS, ...option };
        };

        const initChart = () => {
            if (!chartRef.value || !props.cell.content?.charts) return;
            if (chartInstance) chartInstance.dispose();
            chartInstance = echarts.init(chartRef.value);
            chartInstance.setOption(buildGridOption());
        };

        const handleResize = () => chartInstance?.resize();

        // [优化] 2026-05-27 使用全局 resize 管理器防抖，避免多图表同时 resize 卡顿
        let unregisterResize = null;

        watch([showDataZoom, intervalCompare], () => { if (chartInstance) chartInstance.setOption(buildGridOption(), true); });

        onMounted(() => {
            nextTick(() => initChart());
            unregisterResize = window.__resizeManager.register(handleResize);
            window.addEventListener('colorSchemeChanged', initChart);
        });

        onUnmounted(() => {
            if (unregisterResize) unregisterResize();
            window.removeEventListener('colorSchemeChanged', initChart);
            const inst = chartRef.value ? echarts.getInstanceByDom(chartRef.value) : null;
            if (inst && intervalListener) inst.off('dataZoom', intervalListener);
            if (chartInstance) {
                chartInstance.dispose();
            }
        });

        // [新增] Grid 区间收益：监听 dataZoom 滑块位置作为收益率基准点
        watch(intervalCompare, (val) => {
            if (val) {
                nextTick(() => {
                    const inst = chartRef.value ? echarts.getInstanceByDom(chartRef.value) : null;
                    if (!inst) return;
                    intervalListener = (params) => {
                        if (params.batch) {
                            const dz = params.batch.find(d => d.dataZoomIndex === 0);
                            if (dz) { intervalStart.value = dz.start; intervalEnd.value = dz.end; }
                        } else if (params.dataZoomIndex === 0) {
                            intervalStart.value = params.start;
                            intervalEnd.value = params.end;
                        }
                    };
                    inst.on('dataZoom', intervalListener);
                });
            } else {
                const inst = chartRef.value ? echarts.getInstanceByDom(chartRef.value) : null;
                if (inst && intervalListener) { inst.off('dataZoom', intervalListener); intervalListener = null; }
            }
        });

        return { chartRef, showDataZoom, intervalCompare, hasIntervalCompare, isFullscreen, toggleFullscreen };
    },
    template: `
        <div class="cell-chart" :class="{ 'chart-zoomed': isFullscreen }">
            <h3 v-if="cell.title">{{ cell.title }}</h3>
            <div class="cell-chart-body">
                <div class="chart-wrapper">
                    <div ref="chartRef" class="chart-container chart-container-main"
                         :style="{ width: cell.content?.width || '100%', height: cell.content?.height || '400px' }"></div>
                    <div class="chart-toolbar">
                        <!-- 上方：视图控制 -->
                        <button class="tool-btn" :class="{ active: isFullscreen }" @click="toggleFullscreen" :title="isFullscreen ? '退出全屏' : '全屏'">&#x26F6;</button>
                        <!-- 中间：数据操作 -->
                        <button v-if="hasIntervalCompare" class="tool-btn" :class="{ active: intervalCompare }" @click="intervalCompare = !intervalCompare" title="区间收益（仅第一图）">%</button>
                        <!-- 下方：滚动轴（靠近底部滑块） -->
                        <button class="tool-btn" :class="{ active: showDataZoom }" @click="showDataZoom = !showDataZoom" title="滚动轴">&#x21C4;</button>
                    </div>
                </div>
            </div>
        </div>
    `
};

// ---------- ChartRenderer - 入口组件 ----------
const ChartRenderer = {
    name: 'ChartRenderer',
    components: {
        GridChart,
        GenericChart,
        PieChart,
        HeatmapChart,
        StackedChart
    },
    props: { cell: { type: Object, required: true } },
    setup(props) {
        const chartType = computed(() => {
            const charts = props.cell.content?.charts;
            if (!charts) return 'generic';
            if (Array.isArray(charts.grid) && charts.grid.length > 1) return 'grid';
            const type = charts.series?.[0]?.type;
            if (charts.series?.some(s => s.stack)) return 'stacked';
            return type || 'generic';
        });
        return { chartType };
    },
    template: `
        <grid-chart v-if="chartType === 'grid'" :cell="cell" />
        <pie-chart v-else-if="chartType === 'pie'" :cell="cell" />
        <heatmap-chart v-else-if="chartType === 'heatmap'" :cell="cell" />
        <stacked-chart v-else-if="chartType === 'stacked'" :cell="cell" />
        <generic-chart v-else :cell="cell" />
    `
};

// =============================================================================
// 第三部分：CellRenderer 组件
// =============================================================================

const CellRenderer = {
    name: 'CellRenderer',
    components: {
        FtTable: typeof window !== 'undefined' && window.FtTable ? window.FtTable : null,
        ChartRenderer
    },
    props: {
        cell: { type: Object, required: true },
        cellId: { type: [String, Number], required: true },
        level: { type: Number, default: 0 }
    },
    setup(props) {
        const renderMarkdown = (content) => {
            if (!content) return '';
            return content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\*(.*?)\*/g, '<em>$1</em>').replace(/`(.*?)`/g, '<code>$1</code>').replace(/\n/g, '<br>');
        };

        const getMetricClass = (value) => {
            if (typeof value !== 'string') return '';
            if (value.includes('%')) { const num = parseFloat(value); if (!isNaN(num)) return num >= 0 ? 'positive' : 'negative'; }
            return '';
        };

        const getTableCols = (cell) => {
            if (cell.options?.columns) return cell.options.columns.map(col => typeof col === 'string' ? { field: col, title: col } : { field: col.field, title: col.title || col.field });
            if (cell.content && cell.content.length > 0) return Object.keys(cell.content[0]).map(key => ({ field: key, title: key }));
            return [];
        };

        const getTableOptions = (cell) => {
            const opts = {};
            if (cell.options?.freeze) opts.freeze = cell.options.freeze;
            if (cell.options?.heatmap) opts.heatmap = cell.options.heatmap;
            if (cell.options?.page !== undefined) opts.page = cell.options.page;
            return opts;
        };

        const handleSectionClick = (cell) => {
            if (cell.options?.collapsed !== undefined) {
                cell.options.collapsed = !cell.options.collapsed;
                nextTick(() => {
                    window.dispatchEvent(new Event('resize'));
                });
            }
        };

        return { renderMarkdown, getMetricClass, getTableCols, getTableOptions, handleSectionClick };
    },

    template: `
        <!-- Section -->
        <div v-if="cell.type === 'section'" class="section" :class="{ 'nested-section': level > 0, 'collapsible-section': cell.options?.collapsed !== undefined }" :id="'section-' + cellId">
            <div v-if="cell.title" class="section-title" :class="{ 'collapsible-header': cell.options?.collapsed !== undefined }" @click="handleSectionClick(cell)">
                <span>{{ cell.title }}</span>
                <span v-if="cell.options?.collapsed !== undefined" class="collapse-icon">{{ cell.options?.collapsed ? '▶' : '▼' }}</span>
            </div>
            <div class="section-content" v-show="cell.options?.collapsed !== true">
                <cell-renderer v-for="(subCell, idx) in cell.children" :key="idx" :cell="subCell" :cell-id="cellId + '-' + idx" :level="level + 1"></cell-renderer>
            </div>
        </div>

        <!-- 其他类型 -->
        <div v-else class="cell">
            <!-- 标题 -->
            <div v-if="cell.type === 'title'" class="cell-title">
                <h1 v-if="cell.options?.level === 1">{{ cell.content }}</h1>
                <h2 v-else-if="cell.options?.level === 2">{{ cell.content }}</h2>
                <h3 v-else>{{ cell.content }}</h3>
            </div>

            <!-- 文本 -->
            <div v-else-if="cell.type === 'text'" class="cell-text" :style="cell.options?.color ? { color: cell.options.color } : {}">{{ cell.content }}</div>

            <!-- Markdown -->
            <div v-else-if="cell.type === 'markdown'" class="markdown-content" v-html="renderMarkdown(cell.content)"></div>

            <!-- 代码 -->
            <div v-else-if="cell.type === 'code'" class="code-block">
                <div class="code-header">{{ cell.content?.lang || 'Python' }}</div>
                <div class="code-input"><pre>{{ cell.content?.code }}</pre></div>
                <div v-if="cell.content?.output" class="code-output">{{ cell.content.output }}</div>
            </div>

            <!-- 表格 -->
            <div v-else-if="cell.type === 'table'" class="cell-table">
                <h3 v-if="cell.title">{{ cell.title }}</h3>
                <div v-if="!cell.content || cell.content.length === 0" class="table-empty">暂无数据</div>
                <ft-table v-else :id="'table-' + cellId" :data="cell.content" :cols="getTableCols(cell)" v-bind="getTableOptions(cell)"></ft-table>
            </div>

            <!-- 指标 -->
            <div v-else-if="cell.type === 'metrics'" class="cell-metrics">
                <h3 v-if="cell.title">{{ cell.title }}</h3>
                <div class="metrics-grid" :style="{'--columns': cell.options?.columns || 4}">
                    <div v-for="metric in cell.content" :key="metric.name" class="metric-card" :class="getMetricClass(metric.value)">
                        <div class="metric-value">{{ metric.value }}</div>
                        <div class="metric-label">{{ metric.name }}</div>
                        <div v-if="metric.desc" class="metric-desc">{{ metric.desc }}</div>
                    </div>
                </div>
            </div>

            <!-- 图表（使用 ChartRenderer 入口） -->
            <chart-renderer v-else-if="(cell.type === 'chart' || cell.type === 'pyecharts') && cell.content?.charts" :cell="cell"></chart-renderer>

            <!-- HTML -->
            <div v-else-if="cell.type === 'html'" class="html-block">
                <div class="html-block-inner" v-html="cell.content"></div>
            </div>

            <!-- 分隔线 -->
            <div v-else-if="cell.type === 'divider'" class="cell-divider"></div>

            <!-- 可折叠 -->
            <div v-else-if="cell.type === 'collapsible'" class="cell-collapsible">
                <button class="collapse-toggle" @click="cell.options.collapsed = !cell.options.collapsed">
                    <span>{{ cell.title }}</span>
                    <span>{{ cell.options?.collapsed ? '▶' : '▼' }}</span>
                </button>
                <div v-show="!cell.options?.collapsed" class="collapse-content">
                    <cell-renderer v-for="(subCell, idx) in cell.children" :key="idx" :cell="subCell" :cell-id="cellId + '-' + idx" :level="level + 1"></cell-renderer>
                </div>
            </div>
        </div>
    `
};

// =============================================================================
// 第四部分：ColorPicker 组件
// =============================================================================

const ColorPicker = {
    name: 'ColorPicker',
    setup() {
        const showColorPicker = ref(false);
        const colorPalettes = window.colorPalettes;
        const STORAGE_KEY = 'nb_palette_global';

        const savePaletteToStorage = (scope, paletteKey) => {
            try {
                const existingStr = localStorage.getItem(STORAGE_KEY);
                const existing = existingStr ? JSON.parse(existingStr) : {};
                existing[scope] = paletteKey;
                localStorage.setItem(STORAGE_KEY, JSON.stringify(existing));
            } catch (e) { console.warn('保存配色失败:', e); }
        };

        const setColorPalette = (scope, paletteKey) => {
            if (scope === 'global') {
                colorPalettes.global = paletteKey;
                Object.keys(colorPalettes.groups).forEach(group => { colorPalettes.groups[group] = paletteKey; });
            } else if (colorPalettes.groups[scope] !== undefined) {
                colorPalettes.groups[scope] = paletteKey;
            }
            savePaletteToStorage(scope, paletteKey);
            window.dispatchEvent(new CustomEvent('colorSchemeChanged'));
        };

        return { showColorPicker, colorPalettes, toggleColorPicker: () => { showColorPicker.value = !showColorPicker.value; }, setColorPalette };
    },
    template: `
        <div>
            <div class="color-float-btn" @click="toggleColorPicker" :class="{ active: showColorPicker }" title="配色"><span>🎨</span></div>
            <div class="drawer-overlay" v-if="showColorPicker" @click="showColorPicker = false"></div>
            <aside class="color-drawer" :class="{ open: showColorPicker }">
                <div class="drawer-header"><span>🎨 配色方案</span><button class="close-btn" @click="showColorPicker = false">✕</button></div>
                <div class="drawer-body">
                    <div class="palette-group">
                        <h5>全局配色</h5>
                        <div class="palette-options">
                            <button v-for="(palette, key) in colorPalettes.palettes" :key="key" class="palette-btn" :class="{ active: colorPalettes.global === key }" @click="setColorPalette('global', key)">
                                <div class="palette-preview"><span v-for="(color, idx) in palette.colors.slice(0, 5)" :key="idx" class="palette-color-dot" :style="{ backgroundColor: color }"></span></div>
                                <span class="palette-name">{{ palette.name }}</span>
                            </button>
                        </div>
                    </div>
                    <div class="palette-group">
                        <h5>按图表类型</h5>
                        <div class="palette-options">
                            <div class="chart-type-picker"><label>通用类:</label><select :value="colorPalettes.groups.chart" @change="setColorPalette('chart', $event.target.value)"><option v-for="(palette, key) in colorPalettes.palettes" :key="key" :value="key">{{ palette.name }}</option></select></div>
                            <div class="chart-type-picker"><label>占比类:</label><select :value="colorPalettes.groups.pie" @change="setColorPalette('pie', $event.target.value)"><option v-for="(palette, key) in colorPalettes.palettes" :key="key" :value="key">{{ palette.name }}</option></select></div>
                        </div>
                    </div>
                </div>
            </aside>
        </div>
    `
};

// =============================================================================
// 第五部分：TocMenu 组件
// =============================================================================

const TocMenu = {
    name: 'TocMenu',
    props: { title: { type: String, default: '' }, createdAt: { type: String, default: '' }, cells: { type: Array, default: () => [] } },
    setup(props) {
        const selectedIndices = ref(new Set());
        const menuExpanded = ref(true);
        const isNarrow = ref(false);

        const tocItems = computed(() => {
            const items = [];
            if (props.title) items.push({ title: props.title, type: 'header', index: -1 });
            props.cells.forEach((cell, index) => { if (cell.type === 'section' && cell.title) items.push({ title: cell.title, type: 'section', index }); });
            return items;
        });

        const selectedCount = computed(() => selectedIndices.value.size);
        const isSelected = (index) => selectedIndices.value.has(index);
        const toggleSelection = (index) => { const newSet = new Set(selectedIndices.value); if (newSet.has(index)) newSet.delete(index); else newSet.add(index); selectedIndices.value = newSet; };
        const selectAll = () => { selectedIndices.value = new Set(tocItems.value.map(item => item.index)); };
        const clearSelection = () => { selectedIndices.value = new Set(); };
        const scrollToSection = (index) => { if (index === -1) window.scrollTo({ top: 0, behavior: 'smooth' }); else { const el = document.getElementById('section-' + index); if (el) el.scrollIntoView({ behavior: 'smooth' }); } };
        const toggleMenu = () => { menuExpanded.value = !menuExpanded.value; setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 350); };
        // [调整] 2026-05-29 窄屏自动收起，宽屏自动展开（覆盖用户手动收起状态）
        const checkScreenWidth = () => { 
            const nowNarrow = window.innerWidth <= 1200; 
            if (nowNarrow && !isNarrow.value) { 
                // 从宽变窄 → 自动收起
                menuExpanded.value = false; 
            } else if (!nowNarrow && isNarrow.value) { 
                // 从窄变宽 → 自动展开
                menuExpanded.value = true; 
            }
            isNarrow.value = nowNarrow; 
        };
        const showToast = (message, type = 'info', duration = 3000) => { if (typeof window !== 'undefined' && window.showToast) window.showToast(message, type, duration); };

        const stitchImages = async (imageBlobs) => {
            const images = await Promise.all(imageBlobs.map(blob => new Promise((resolve, reject) => { const img = new Image(); img.onload = () => resolve(img); img.onerror = reject; img.src = URL.createObjectURL(blob); })));
            const MARGIN_TOP = 12, MARGIN_BOTTOM = 12, CONTAINER_PADDING = 20;
            const maxContentWidth = Math.max(...images.map(img => img.width));
            const maxWidth = maxContentWidth + CONTAINER_PADDING * 2;
            const contentHeight = images.reduce((sum, img) => sum + img.height, 0);
            const spacingHeight = (images.length - 1) * (MARGIN_TOP + MARGIN_BOTTOM);
            const totalHeight = contentHeight + spacingHeight + CONTAINER_PADDING * 2;
            const canvas = document.createElement('canvas'); canvas.width = maxWidth; canvas.height = totalHeight;
            const ctx = canvas.getContext('2d'); ctx.fillStyle = '#f5f5f5'; ctx.fillRect(0, 0, maxWidth, totalHeight);
            let currentY = CONTAINER_PADDING;
            images.forEach((img, index) => {
                const x = Math.floor((maxWidth - img.width) / 2);
                if (index > 0) { ctx.fillStyle = '#f5f5f5'; ctx.fillRect(0, currentY, maxWidth, MARGIN_TOP); currentY += MARGIN_TOP; }
                ctx.drawImage(img, x, currentY); currentY += img.height;
                if (index < images.length - 1) { ctx.fillStyle = '#f5f5f5'; ctx.fillRect(0, currentY, maxWidth, MARGIN_BOTTOM); currentY += MARGIN_BOTTOM; }
                URL.revokeObjectURL(img.src);
            });
            return new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
        };

        const captureScreenshot = async () => {
            if (selectedIndices.value.size === 0) return;
            const mainContainer = document.querySelector('.notebook-container');
            const elementsToCapture = [];
            if (selectedIndices.value.has(-1)) { const header = mainContainer.querySelector('.notebook-header'); if (header) elementsToCapture.push(header); }
            const sortedIndices = [...selectedIndices.value].filter(index => index !== -1).sort((a, b) => a - b);
            sortedIndices.forEach(index => { const section = document.getElementById('section-' + index); if (section) elementsToCapture.push(section); });
            try {
                await new Promise(resolve => setTimeout(resolve, 300));
                const imageBlobs = [];
                for (const el of elementsToCapture) {
                    const hasTable = el.querySelector('ft-table, .ft-table, table');
                    if (hasTable) await new Promise(resolve => setTimeout(resolve, 200));
                    const result = await snapdom(el, { scale: 2, backgroundColor: '#f5f5f5', cache: 'auto' });
                    const blob = await result.toBlob({ type: 'png' });
                    imageBlobs.push(blob);
                }
                const finalBlob = await stitchImages(imageBlobs);
                try { await navigator.clipboard.write([new ClipboardItem({ 'image/png': finalBlob })]); showToast(`已复制 ${selectedIndices.value.size} 个选中区域到剪贴板`, 'success'); }
                catch (clipboardErr) { console.error('复制到剪贴板失败:', clipboardErr); showToast('复制到剪贴板失败，正在下载图片...', 'info'); const reader = new FileReader(); reader.onload = function(e) { const link = document.createElement('a'); link.download = `${props.title || 'notebook'}-选中部分.png`; link.href = e.target.result; link.click(); }; reader.readAsDataURL(finalBlob); }
            } catch (err) { console.error('截图失败:', err); showToast('截图失败: ' + err.message, 'error'); }
        };

        const captureAll = async () => {
            const element = document.querySelector('.notebook-container');
            if (!element) { showToast('未找到截图区域', 'error'); return; }
            try {
                document.body.focus();
                const result = await snapdom(element, { backgroundColor: '#f5f5f5', scale: 1, cache: 'auto' });
                const blob = await result.toBlob({ type: 'png' });
                try { await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]); showToast('全页截图已复制到剪贴板', 'success'); }
                catch (err) { console.error('复制到剪贴板失败:', err); showToast('复制到剪贴板失败，正在下载图片...', 'info'); const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.download = `截图_${new Date().toLocaleString().replace(/[/:]/g, '-')}.png`; link.href = url; link.click(); URL.revokeObjectURL(url); }
            } catch (err) { console.error('截图失败:', err); showToast('截图失败: ' + err.message, 'error'); }
        };

        onMounted(() => { checkScreenWidth(); window.addEventListener('resize', checkScreenWidth); });
        onUnmounted(() => { window.removeEventListener('resize', checkScreenWidth); });

        return { tocItems, selectedCount, isSelected, toggleSelection, selectAll, clearSelection, scrollToSection, captureScreenshot, captureAll, menuExpanded, toggleMenu, isNarrow };
    },
    template: `
        <nav class="toc-float-menu" :class="{ expanded: menuExpanded }" v-if="tocItems.length > 0">
            <div class="menu-wide">
                <!-- [调整] 2026-05-29 宽屏也显示收起按钮，用户可手动收起/展开 -->
                <div class="menu-header"><span>📑 目录</span><button class="collapse-btn" @click="toggleMenu" title="收起">✕</button></div>
                <ul class="menu-list">
                    <li v-for="(item, index) in tocItems" :key="index" class="menu-item" :class="{ selected: isSelected(item.index) }" @click="scrollToSection(item.index)">
                        <input type="checkbox" :checked="isSelected(item.index)" @click.stop="toggleSelection(item.index)">
                        <span class="menu-title">{{ item.title }}</span>
                    </li>
                </ul>
                <div class="menu-footer">
                    <button @click="selectAll">全选</button>
                    <button @click="clearSelection">清空</button>
                    <button @click="captureScreenshot" :disabled="selectedCount === 0">截图选中</button>
                    <button @click="captureAll" class="full-btn">📋 全页</button>
                </div>
            </div>
            <div class="menu-narrow">
                <div class="collapsed-header" @click="toggleMenu" title="展开目录">📑</div>
                <ul class="collapsed-list">
                    <li v-for="(item, index) in tocItems" :key="index" class="collapsed-item" :class="{ selected: isSelected(item.index) }" @click="scrollToSection(item.index)" :title="item.title">{{ item.title.charAt(0) }}</li>
                </ul>
            </div>
        </nav>
    `
};

// =============================================================================
// 第六部分：Toast 组件
// =============================================================================

const Toast = {
    name: 'Toast',
    setup() {
        const toastMessage = ref('');
        const toastType = ref('info');
        let toastTimer = null;
        const showToast = (message, type = 'info', duration = 3000) => { toastMessage.value = message; toastType.value = type; if (toastTimer) clearTimeout(toastTimer); toastTimer = setTimeout(() => { toastMessage.value = ''; }, duration); };
        if (typeof window !== 'undefined') window.showToast = showToast;
        return { toastMessage, toastType };
    },
    template: `<div class="toast-container" v-if="toastMessage"><div class="toast" :class="toastType">{{ toastMessage }}</div></div>`
};

// =============================================================================
// 第七部分：createNotebookApp
// =============================================================================

// [优化] 2026-05-17 全局ESC监听(仅一个,替代每个图表的独立keydown)
if (typeof document !== 'undefined' && !window.__fsEscInited) {
    window.__fsEscInited = true;
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            window.__fsEsc.forEach(fn => fn());
        }
    });
}

function createNotebookApp() {
    const FtTableComponent = typeof window !== 'undefined' ? window.FtTable : null;

    return createApp({
        components: { CellRenderer, ColorPicker, TocMenu, Toast, FtTable: FtTableComponent },
        setup() {
            const config = window.notebookConfig || { title: '未命名 Notebook', createdAt: new Date().toLocaleString(), children: [] };
            const title = ref(config.title);
            const createdAt = ref(config.createdAt);
            const cells = ref(config.children || config.cells || []);
            const isScreenshotMode = ref(false);

            const defaultPalettes = {
                global: 'warmToCool',
                typeToGroup: { line: 'chart', bar: 'chart', area: 'chart', scatter: 'chart', radar: 'chart', pie: 'pie', doughnut: 'pie', funnel: 'pie', gauge: 'pie' },
                groups: { chart: 'warmToCool', pie: 'warmToCool' },
                palettes: {
                    warmToCool: { name: '暖冷渐变系', desc: '珊瑚橙粉紫青金绿', colors: ['#e74c3c', '#f39c12', '#af7ac5', '#5499c7', '#f4d03f', '#82e0aa', '#d35400', '#9b59b6', '#76d7c4'] },
                    contrast: { name: '高对比度系', desc: '清晰易辨识', colors: ['#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#3498db', '#e74c3c', '#2ecc71', '#e67e22', '#95a5a6'] },
                    dahongdazi: { name: '大红大紫系', desc: '红紫粉金，柔和现代', colors: ['#e74c3c', '#9b59b6', '#f39c12', '#e91e63', '#f1c40f', '#8e44ad', '#ff6b6b', '#af7ac5', '#daa520'] },
                    echartsDefault: { name: 'ECharts默认', desc: '官方默认配色', colors: ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc'] }
                }
            };

            if (!window.colorPalettes || !window.colorPalettes.groups) window.colorPalettes = Vue.reactive(defaultPalettes);
            const colorPalettes = window.colorPalettes;

            const STORAGE_KEY = 'nb_palette_global';
            const loadPaletteFromStorage = () => { try { const saved = localStorage.getItem(STORAGE_KEY); if (saved) return JSON.parse(saved); } catch (e) { console.warn('读取配色失败:', e); } return null; };
            const savedPalette = loadPaletteFromStorage();
            if (savedPalette) {
                if (savedPalette.global && colorPalettes.palettes[savedPalette.global]) { colorPalettes.global = savedPalette.global; Object.keys(colorPalettes.groups).forEach(group => { colorPalettes.groups[group] = savedPalette.global; }); }
                Object.keys(savedPalette).forEach(scope => { if (scope !== 'global' && colorPalettes.groups[scope] !== undefined && colorPalettes.palettes[savedPalette[scope]]) colorPalettes.groups[scope] = savedPalette[scope]; });
            }

            return { title, createdAt, cells, isScreenshotMode };
        }
    });
}

// =============================================================================
// 导出
// =============================================================================

if (typeof module !== 'undefined' && module.exports) {
    module.exports = { CellRenderer, ColorPicker, TocMenu, Toast, createNotebookApp, useChart };
}
