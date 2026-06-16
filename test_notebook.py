"""
Notebook 模块测试
"""

import sys, os
# [修复] 2026-05-30 使用相对路径获取项目根目录，替代硬编码的绝对路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
from notebook import Notebook


def test_comprehensive():
    """综合测试：混合自由内容与模块嵌套"""
    print("\n[综合测试]...")
    
    nb = Notebook("综合测试报告")
    
    # ========== 【自由内容】报告头部 ==========
    nb.title("策略回测报告", level=1)
    nb.divider()
    nb.markdown("""
**策略类型**: 趋势跟踪 | **回测区间**: 2024-01 至 2024-05 | **初始资金**: 1,000,000
    """)
    
    # 【自由内容】核心指标（全局可见，不在section中）
    nb.metrics([
        {"name": "总收益", "value": "45.6%", "desc": "累计收益"},
        {"name": "夏普", "value": "1.85", "desc": "风险调整"},
        {"name": "最大回撤", "value": "-12.3%", "desc": "2024-03"},
        {"name": "胜率", "value": "58%", "desc": "盈利占比"},
    ])
    
    nb.divider()
    
    # ========== 【Section嵌套】收益分析 ==========
    dates = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
    
    with nb.section("收益分析"):
        # 数据定义
        x_data = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
        strategy_data = [1.0, 1.05, 1.12, 1.08, 1.15]
        benchmark_data = [1.0, 1.02, 1.04, 1.06, 1.08]

        # 字典格式：xAxis + series
        nb.chart('line', {
            'xAxis': x_data,
            'series': [
                {"name": "策略", "data": strategy_data},
                {"name": "基准", "data": benchmark_data}
            ]
        }, title="净值曲线（字典格式：xAxis + series）")
        
        # 100条数据的折线图测试
        x_100 = [f"2024-{i:02d}" for i in range(1, 101)]
        y_100_1 = [1.0 + np.sin(i * 0.1) * 0.2 + i * 0.005 for i in range(100)]
        y_100_2 = [1.0 + np.cos(i * 0.1) * 0.15 + i * 0.003 for i in range(100)]
        nb.chart('line', {
            'xAxis': x_100,
            'series': [
                {"name": "策略A", "data": y_100_1},
                {"name": "策略B", "data": y_100_2}
            ]
        }, title="100条数据折线图测试")

        # DataFrame 格式：第一列=X轴，其余列=series
        # 与字典格式对应关系：
        #   xAxis = x_data = df["月份"]
        #   series[0]["name"] = "策略" = df["策略"].name
        #   series[0]["data"] = strategy_data = df["策略"].tolist()
        df_line = pd.DataFrame({
            "月份": x_data,
            "策略": strategy_data,
            "基准": benchmark_data
        })
        nb.chart('line', df_line, title="净值曲线（DataFrame格式：第一列=X轴）")
        
        # 堆叠柱状图
        month_data = ["1月", "2月", "3月", "4月", "5月"]
        profit_data = [100, 150, 80, 200, 120]
        loss_data = [30, 20, 50, 10, 40]

        nb.chart('bar', {
            'xAxis': month_data,
            'series': [
                {"name": "盈利", "data": profit_data},
                {"name": "亏损", "data": loss_data}
            ]
        }, title="月度盈亏（字典格式）")

        df_stack = pd.DataFrame({
            "月份": month_data,
            "盈利": profit_data,
            "亏损": loss_data
        })
        nb.chart('bar', df_stack, series_opts={'stack': 'total'}, title="月度盈亏（DataFrame格式）")
        
        # 嵌套：月度分析
        with nb.section("月度分析"):
            nb.chart('bar', {
                'xAxis': ["1月", "2月", "3月", "4月", "5月"],
                'series': [{"name": "月收益%", "data": [5, -2, 8, 3, 6]}]
            }, title="月度收益")
            
            nb.chart('pie', [
                {"name": "股票", "value": 60},
                {"name": "债券", "value": 25},
                {"name": "现金", "value": 15}
            ], title="资产配置", height='250px')
    
    # ========== 【自由内容】简短说明 ==========
    nb.text("【自由内容】以下是风险分析部分，可直接阅读或点击目录跳转。")
    
    # ========== 【Section嵌套】风险分析 ==========
    with nb.section("风险分析"):
        nb.chart('area', {
            'xAxis': dates,
            'series': [{"name": "回撤%", "data": [0, -2, -5, -3, -1]}]
        }, title="回撤曲线", height=250)
        
        # 热力图
        year_month_data = {
            "2023": {"01": 0.02, "02": -0.01, "03": 0.03, "04": 0.01, "05": -0.02, "06": 0.04},
            "2024": {"01": 0.05, "02": -0.02, "03": 0.08, "04": 0.03, "05": 0.06, "06": -0.01}
        }
        nb.chart('heatmap', year_month_data, title="月度收益热力图（字典格式：Y轴=年份，X轴=月份）")

        # DataFrame 格式：第一列=X轴，其余列=Y轴
        month_labels = ["01", "02", "03", "04", "05", "06"]
        y2023 = [0.02, -0.01, 0.03, 0.01, -0.02, 0.04]
        y2024 = [0.05, -0.02, 0.08, 0.03, 0.06, -0.01]

        df_heatmap = pd.DataFrame({
            "月份": month_labels,
            "2023": y2023,
            "2024": y2024
        })
        nb.chart('heatmap', df_heatmap, title="月度收益热力图（DataFrame格式：第一列=X轴）")

        # [新增] 2026-06-16 DataFrame 散点图（推荐）：第一列→名称，第二列→X，第三列→Y
        df_scatter = pd.DataFrame({
            '股票': ['茅台', '平安银行', '万科A', '招商银行', '宁德时代'],
            '波动率': [15.2, 8.1, 22.3, 10.5, 28.7],
            '收益率': [0.82, 0.45, 0.31, 0.56, -0.15],
        })
        nb.chart('scatter', df_scatter, title="DataFrame散点图（推荐: 名称+X+Y，tooltip显示股票名）")

        # [新增] 2026-05-27 散点图进阶格式（pyecharts 原生 dict）
        # 数值散点图：data 为 [x, y] 二维数组
        np.random.seed(42)
        n = 30
        _sx = np.random.uniform(0, 20, n)
        _sy = _sx * 0.8 + np.random.normal(0, 2, n)
        scatter_data = [[round(x, 2), round(y, 2)] for x, y in zip(_sx, _sy)]
        nb.chart('scatter', {
            'series': [
                {"name": "波动率-收益", "data": scatter_data},
            ]
        }, title="数值散点图（进阶: [x,y] 二维数组）")

        # 类目散点图：xAxis 为标签，data 为 y 值
        nb.chart('scatter', {
            'xAxis': ['浦发银行', '平安银行', '万科A', '贵州茅台', '中国平安'],
            'series': [
                {"name": "收益率", "data": [5.2, -3.1, 8.7, 12.3, -1.5]},
            ]
        }, title="类目散点图（进阶: xAxis标签 + y值）")

        # [新增] 2026-05-27 K线图测试（两种格式）
        # 标准格式：xAxis + series（data 为 [[开,收,低,高], ...]）
        kline_dates = ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08']
        kline_data = [
            [100, 105, 98, 106],   # 开, 收, 低, 高
            [105, 103, 101, 107],
            [103, 108, 102, 109],
            [108, 106, 104, 110],
            [106, 112, 105, 113]
        ]
        nb.chart('kline', {
            'xAxis': kline_dates,
            'series': [
                {"name": "日K", "data": kline_data}
            ]
        }, title="K线图（标准格式）")

        # DataFrame 格式：自动识别 open/close/low/high 字段
        df_kline = pd.DataFrame({
            "日期": kline_dates,
            "open": [100, 105, 103, 108, 106],
            "close": [105, 103, 108, 106, 112],
            "low": [98, 101, 102, 104, 105],
            "high": [106, 107, 109, 110, 113]
        })
        nb.chart('kline', df_kline, title="K线图（DataFrame格式）")

        # [新增] 2026-05-27 Grid多图测试
        # 场景1：K线 + 成交金额
        grid_dates = ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08']
        grid_kline = [
            [100, 105, 98, 106],
            [105, 103, 101, 107],
            [103, 108, 102, 109],
            [108, 106, 104, 110],
            [106, 112, 105, 113]
        ]
        grid_volume = [10000, 15000, 8000, 12000, 20000]

        nb.chartg('kline', {
            'xAxis': grid_dates,
            'series': [{'name': '日K', 'data': grid_kline}]
        }, height=300)
        nb.chartg('bar', {
            'xAxis': grid_dates,
            'series': [{'name': '成交金额', 'data': grid_volume}]
        }, height=150)  # 场景1结束
        nb.divider()    # 触发 Grid 合并输出

        # 场景2：净值 + 模拟信号 + 仓位
        nav_dates = ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08']
        nav_data = [1.0, 1.02, 1.01, 1.05, 1.08]
        signal_data = [0, 1, 1, 0, 1]  # 0=空仓, 1=持仓
        position_data = [0, 0.5, 0.8, 0, 0.6]  # 仓位比例

        nb.chartg('line', {
            'xAxis': nav_dates,
            'series': [{'name': '净值', 'data': nav_data}]
        }, height=200)
        nb.chartg('bar', {
            'xAxis': nav_dates,
            'series': [{'name': '信号', 'data': signal_data}]
        }, height=100)
        nb.chartg('line', {
            'xAxis': nav_dates,
            'series': [{'name': '仓位', 'data': position_data}]
        }, height=150)
        nb.divider()  # 触发 Grid 合并输出

        # 场景3：资产净值 + 仓位（仅2层，数据更真实）
        asset_dates = ['2024-01-02', '2024-01-05', '2024-01-10', '2024-01-15', '2024-01-22',
                       '2024-01-29', '2024-02-05', '2024-02-12', '2024-02-19', '2024-02-26']
        asset_values = [1000000, 1020000, 1015000, 1050000, 1040000,
                        1080000, 1100000, 1130000, 1110000, 1150000]  # 策略资产
        position_pct = [0, 0.5, 0.8, 0.8, 0.5, 0, 0.6, 0.6, 0.4, 0.8]  # 仓位比例%

        nb.chartg('line', {
            'xAxis': asset_dates,
            'series': [{'name': '策略净值', 'data': asset_values}],
        }, height=300, series_opts={'is_smooth': True})

        nb.chartg('bar', {
            'xAxis': asset_dates,
            'series': [{'name': '仓位', 'data': position_pct}],
        }, height=150)
        nb.divider()  # 触发 Grid 合并输出

        # 嵌套：风险评估
        with nb.section("风险评估", collapsed=True):
            nb.markdown("""
**VaR(95%)**: 2.5%  
**CVaR(95%)**: 3.8%  
**下行波动率**: 8.2%
            """)
    
    # ========== 【自由内容】持仓概览 ==========
    nb.html("""
    <div style="background: linear-gradient(90deg, #f3f4f6 0%, #fff 100%); padding: 12px 16px; border-radius: 6px; margin: 12px 0;">
        <strong>💡 【自由内容提示】</strong> 持仓明细如下，详细数据可点击 Section 查看。
    </div>
    """)
    
    # ========== 【Section嵌套】持仓明细 ==========
    with nb.section("持仓明细"):
        df = pd.DataFrame({
            "代码": ["600000", "000001", "000002"],
            "名称": ["浦发银行", "平安银行", "万科A"],
            "数量": [10000, 5000, 8000],
            "成本": [10.5, 15.2, 8.3],
            "现价": [11.2, 16.8, 7.5],
            "盈亏": ["+6.7%", "+10.5%", "-9.6%"]
        })
        nb.table(df, title="持仓列表")
    
    # ========== 【Section嵌套】股票行情 ==========
    with nb.section("股票行情"):
        nb.text("多列表格，冻结左侧2列：")
        wide_data = []
        for i in range(20):
            wide_data.append({
                "股票代码": f"60{i:04d}",
                "股票名称": f"股票{i}名称",
                "收盘价": round(10 + i * 0.5, 2),
                "涨跌幅": f"{i-10}%",
                "成交量": 1000000 + i * 100000,
                "成交额": 10000000 + i * 1000000,
                "振幅": f"{i*0.5}%",
                "换手率": f"{i*0.3}%",
                "市盈率": round(20 + i * 0.5, 2),
                "市净率": round(2 + i * 0.1, 2),
                "总市值": 1000000000 + i * 10000000,
                "流通市值": 800000000 + i * 8000000,
            })
        nb.table(
            wide_data,
            columns=["股票代码", "股票名称", "收盘价", "涨跌幅", "成交量", "成交额", "振幅", "换手率", "市盈率", "市净率", "总市值", "流通市值"],
            title="股票行情（冻结左侧2列，共12列）",
            freeze={"left": 2, "right": 0}
        )
    
    # ========== 【自由内容】分隔线 ==========
    nb.divider()
    
    # ========== 【Section嵌套】文本换行测试 ==========
    with nb.section("文本换行测试"):
        nb.text("""【text三引号测试】第一行
第二行
第三行""")
        
        nb.text('【text单引号+\\n测试】第一行\n第二行\n第三行')
        
        nb.text('【text变量拼接测试】\n' + '\n'.join(['项目1', '项目2', '项目3']))
    
    # ========== 【Section嵌套】Markdown换行测试 ==========
    with nb.section("Markdown换行测试"):
        nb.markdown("""【markdown三引号测试】
# 一级标题
## 二级标题
正文第一行
正文第二行
**加粗文本**
*斜体文本*""")
        
        nb.markdown('【markdown单引号+\\n测试】\n# 标题\n第一行\n第二行\n**加粗**')
        
        nb.markdown("""【markdown列表测试】
- 列表项1
- 列表项2
- 列表项3

1. 有序列表1
2. 有序列表2
3. 有序列表3""")
        
        nb.markdown("""【markdown代码块测试】
```python
def hello():
    print("world")
```
行内代码: `print(1)`""")
        
        nb.markdown("""【markdown引用测试】
> 这是引用内容
> 第二行引用
>
> 嵌套引用""")
    
    # ========== 【Section嵌套】历史记录（折叠） ==========
    with nb.section("历史记录", collapsed=True):
        nb.table(
            [
                {"日期": "2024-01-05", "方向": "买入", "代码": "600000", "价格": 10.5, "数量": 1000},
                {"日期": "2024-02-10", "方向": "卖出", "代码": "600000", "价格": 11.2, "数量": 1000},
                {"日期": "2024-02-15", "方向": "买入", "代码": "000001", "价格": 15.0, "数量": 500},
                {"日期": "2024-03-20", "方向": "卖出", "代码": "000001", "价格": 16.8, "数量": 500},
            ],
            title="历史交易"
        )
    
    # ========== 【Section嵌套】策略代码 ==========
    with nb.section("策略代码"):
        nb.code(
            code="""# 策略核心逻辑
def on_bar(bar):
    if bar.close > ma20:
        buy(symbol=bar.symbol, quantity=100)
    elif bar.close < ma20:
        sell(symbol=bar.symbol, quantity=100)
""",
            language="python",
            output="策略初始化完成\n回测结束"
        )
    
    # ========== 【自由内容】风险提示（HTML） ==========
    nb.html("""
<div style="background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 6px; margin-top: 20px;">
    <strong>⚠️ 【自由内容】风险提示</strong>
    <p style="margin: 8px 0 0 0; color: #856404;">
        本报告仅供参考，不构成投资建议。历史业绩不代表未来表现。
    </p>
</div>
    """)
    
    nb.export_html(local_assets=True)
    


if __name__ == '__main__':
    print("=" * 60)
    print("Notebook 模块测试")
    print("=" * 60)
    
    test_comprehensive()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
