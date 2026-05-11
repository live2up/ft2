import os
from typing import List, Optional, Union
from pathlib import Path
from datetime import datetime
import json
from jinja2 import Environment, FileSystemLoader

from .cell import Cell, Section, CellType, CellBuilder, CellLike, _build_grid, _init_chart_registry


class SectionContext:
    """Section 上下文管理器"""
    
    def __init__(self, notebook: 'Notebook', title: str, 
                 level: int = 1, collapsed: bool = None):
        self.notebook = notebook
        self.title = title
        self.level = level
        self.collapsed = collapsed
        self.children: List[CellLike] = []
    
    def __enter__(self) -> 'Notebook':
        self.notebook._push_section(self)
        return self.notebook
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.notebook._flush_chartg()  # 先 flush，此时 section_stack 还有当前 section
        self.notebook._pop_section()
        section = CellBuilder.section(self.title, self.children, self.level, self.collapsed)
        self.notebook._add_cell(section)
        return False


class Notebook:
    """
    Notebook风格输出类
    
    使用方式：
        nb = Notebook("策略分析报告")
        nb.title("回测结果")
        nb.table(data, columns=['code', 'name'], freeze={'left': 2})
        nb.metrics([{'name': '收益率', 'value': '15%'}])
        nb.chart('line', {'x': dates, 'series': series})
        nb.export_html("report.html")
        
        # Section 容器
        with nb.section("收益分析"):
            nb.metrics([...], title="核心指标")
            nb.chart('line', {'x': dates, 'series': series}, title="净值曲线")
        
        # 可折叠 Section
        with nb.section("详细数据", collapsed=True):
            nb.table(data)
    """
    
    def __init__(self, title: str = "Notebook Report"):
        import inspect
        
        self.nb_title = title
        self.children: List[CellLike] = []
        self.created_at = datetime.now()
        self._cell_counter = 0
        self._section_stack: List[SectionContext] = []
        self._chartg_buffer = []
        
        caller_frame = None
        for frame_info in inspect.stack():
            if frame_info.filename != __file__:
                caller_frame = frame_info
                break
        
        if caller_frame:
            self.base_dir = os.path.dirname(os.path.abspath(caller_frame.filename))
        else:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
    
    def _push_section(self, section):
        """进入 Section"""
        self._section_stack.append(section)
    
    def _pop_section(self):
        """退出 Section"""
        return self._section_stack.pop()
    
    def _flush_chartg(self):
        """合并 chartg buffer"""
        if not self._chartg_buffer:
            return
        _init_chart_registry()
        heights = [c['height'] for c in self._chartg_buffer]
        total_height = sum(heights)
        option_dict = _build_grid(self._chartg_buffer, total_height)
        cell = Cell(
            CellType.CHART,
            {"charts": option_dict, "width": "100%", "height": f"{total_height}px"}
        )
        self._chartg_buffer.clear()
        self._cell_counter += 1
        if self._section_stack:
            self._section_stack[-1].children.append(cell)
        else:
            self.children.append(cell)
    
    def chartg(self, chart_type, data, height=200, **kwargs):
        """添加 Grid 图表（累加模式）"""
        self._chartg_buffer.append({
            'type': chart_type,
            'data': data,
            'height': height,
            'kwargs': kwargs
        })
        return self
    
    def _add_cell(self, cell: CellLike, title: str = None):
        """
        添加单元格并返回self以支持链式调用

        逻辑:
        1. 如果在 with 内 -> 添加到当前 Section，设置 Cell.title 作为小标题
        2. 如果不在 with 内但有 title -> 自动创建 Section（Cell 无 title）
        3. 如果不在 with 内且无 title -> 普通 Cell 添加到顶层
        """
        self._flush_chartg()
        self._cell_counter += 1

        if self._section_stack:
            if isinstance(cell, Cell) and title:
                cell.title = title
            self._section_stack[-1].children.append(cell)
        elif title:
            section = CellBuilder.section(title, [cell], level=1)
            self.children.append(section)
        else:
            self.children.append(cell)

        return self
    
    def section(self, title: str, collapsed: bool = None):
        """
        创建 Section 容器（上下文管理器）
        
        Args:
            title: Section 标题
            collapsed: 折叠状态
                - None: 不可折叠（默认）
                - True: 可折叠，默认折叠
                - False: 可折叠，默认展开
        
        Returns:
            SectionContext: 上下文管理器
        
        Usage:
            with nb.section("收益分析"):
                nb.metrics([...], title="核心指标")
            
            with nb.section("详细数据", collapsed=True):
                nb.table(data)
        """
        level = len(self._section_stack) + 1
        return SectionContext(self, title, level, collapsed)
    
    # ========== 标题和文本 ==========
    
    def title(self, text: str, level: int = 1):
        """添加标题"""
        return self._add_cell(CellBuilder.title(text, level))
    
    def text(self, text: str, color: str = None):
        """添加文本，color支持: red, green, blue, yellow, orange, purple, gray等"""
        return self._add_cell(CellBuilder.text(text, color))
    
    def markdown(self, text: str):
        """添加Markdown内容"""
        return self._add_cell(CellBuilder.markdown(text))
    
    def divider(self):
        """添加分隔线"""
        return self._add_cell(CellBuilder.divider())
    
    # ========== 代码 ==========
    
    def code(self, code: str, language: str = 'python', output: str = None):
        """添加代码块"""
        return self._add_cell(CellBuilder.code(code, language, output))
    
    # ========== 表格 ==========
    
    def table(self, data, columns=None, title=None, **options):
        """
        添加表格

        核心参数:
            data: 表格数据（List[dict] 或 DataFrame）
                - List[dict]: [{'code': '000001', 'name': '基金A'}, ...]
                - DataFrame: pd.DataFrame 对象，自动转换
            columns: 列名列表（指定要显示的列及顺序）
                - ['code', 'name', 'type']  # 只显示这3列，按此顺序
                - None 时显示数据中的所有列
            title: 标题

        可选参数 (**options):
            freeze: 冻结列配置
                - dict: {'left': n, 'right': m}
            page: 分页配置（对应 ft-table.js 的 page 参数，默认启用分页）
                - 不传参数: 默认分页，每页 10 条
                - False: 禁用分页
                - {'size': 20}: 每页 20 条
                - {'size': 20, 'options': [10, 20, 50, 100]}: 自定义选项
            heatmap: 热力图配置（列级别或全局）
                - 全局: {'start': 2, 'end': 5, 'axis': 'column', 'colors': [...]}
                - 列级别: 在 columns 数组中每列单独配置
                - 详细说明见 ft-table.js 注释

        Examples:
            nb.table(data)                      # 默认分页，每页 10 条
            nb.table(data, columns=['code', 'name'], title='基金列表')
            nb.table(df, title='数据表')
            nb.table(data, freeze={'left': 2})
            nb.table(data, page=False)  # 不分页
            nb.table(data, page={'size': 20})  # 每页 20 条
            nb.table(data, heatmap={'start': 2, 'axis': 'column'})
        """
        import pandas as pd
        
        if isinstance(data, pd.DataFrame):
            df_data = data.to_dict('records')
            cols = columns or list(data.columns)
        else:
            df_data = data
            # 从数据中提取 columns（取第一个元素的 key）
            if not columns and df_data and len(df_data) > 0:
                cols = list(df_data[0].keys())
            else:
                cols = columns
        
        cell = CellBuilder.table(df_data, cols, options)
        return self._add_cell(cell, title)
    
    # ========== 指标卡片 ==========
    
    def metrics(self, data, title: str = None, columns: int = 4):
        """
        添加指标卡片

        data格式:
            - List[Dict]: [{'name': '指标名', 'value': '指标值'}, ...]  # 本质
            - Dict: {'指标名': '指标值', ...}  # 便捷输入，自动转换
        """
        if isinstance(data, dict):
            data = [{'name': k, 'value': str(v)} for k, v in data.items()]
        return self._add_cell(CellBuilder.metrics(data, columns), title)
    
    # ========== 图表 ==========
    
    def chart(self, chart_type, data, title=None, height='400px', **kwargs):
        """
        添加图表（pyecharts 简化封装）
        
        基础参数:
            chart_type: 图表类型
                - 'line': 折线图
                - 'area': 面积图
                - 'bar': 柱状图
                - 'pie': 饼图
                - 'heatmap': 热力图
                - 'kline': K线图
            data: 图表数据（格式因类型而异）
                - line/area/bar/kline: {'xAxis': [...], 'series': [{'name': '', 'data': []}, ...]}
                - pie: [{'name': '', 'value': 0}, ...]
                - heatmap: {'2024': {'01': 0.05, ...}, ...} 或 DataFrame
            title: Cell 标题（推荐填写）
        
        容器参数（有默认值）:
            height: 容器高度，默认 '400px'
            width: 容器宽度，默认 '100%'
        
        全局参数（可选，遵循 pyecharts 规范）:
            title_opts: 标题配置
            legend_opts: 图例配置
            tooltip_opts: 提示框配置
            xaxis_opts: X轴配置
            yaxis_opts: Y轴配置
            datazoom_opts: 数据缩放
            visualmap_opts: 视觉映射
            grid_opts: 网格配置
        
        系列参数（可选，统一应用到所有系列）:
            series_opts: 系列配置
        
        Examples:
            # 折线图
            nb.chart('line', {'xAxis': dates, 'series': series}, title='净值曲线')
            
            # 柱状图
            nb.chart('bar', {'xAxis': categories, 'series': series}, title='收益分布')
            
            # 饼图
            nb.chart('pie', [{'name': '股票', 'value': 60}, ...], title='资产配置')
            
            # 热力图
            nb.chart('heatmap', monthly_returns, title='月度收益')
            
            # K线图
            nb.chart('kline', {'xAxis': dates, 'series': [kline_data]}, title='K线')
            
            # 带可选参数
            nb.chart('line', data, title='净值曲线',
                yaxis_opts={'min_': 0.9},
                series_opts={'is_smooth': True}
            )
            
            # 高级需求 → 使用 pyecharts() 方法
            # from pyecharts.charts import Line
            # line = Line()
            # line.add_xaxis([...])
            # line.add_yaxis(...)
            # nb.pyecharts(line, title='净值曲线')
        """
        return self._add_cell(CellBuilder.chart(chart_type, data, height, **kwargs), title)
    
    def pyecharts(self, chart, title=None, height='400px', width='100%'):
        """
        添加 pyecharts 对象（高级需求）
        
        Args:
            chart: pyecharts 图表对象（如 Line, Bar, Pie 等）
            title: Cell 标题（推荐填写）
            height: 容器高度，默认 '400px'
            width: 容器宽度，默认 '100%'
        
        Returns:
            Notebook: 支持链式调用
        
        Examples:
            from pyecharts.charts import Line
            from pyecharts import options as opts
            
            line = Line()
            line.add_xaxis(['1月', '2月', '3月'])
            line.add_yaxis('策略', [1.0, 1.05, 1.08], is_smooth=True)
            line.add_yaxis('基准', [1.0, 1.02, 1.04], is_smooth=False)
            line.set_global_opts(yaxis_opts=opts.AxisOpts(min_=0.9))
            
            nb.pyecharts(line, title='净值曲线')
        """
        return self._add_cell(CellBuilder.pyecharts(chart, height, width), title)
    
    # ========== HTML ==========
    
    def html(self, html_content: str) -> 'Notebook':
        """添加原始HTML"""
        return self._add_cell(CellBuilder.html(html_content))
    
    # ========== 输出 ==========
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'title': self.nb_title,
            'createdAt': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'children': [c.to_dict() for c in self.children]
        }
    
    def to_json(self) -> str:
        """导出为JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def export_html(self, name: str = None, template_path: str = None):
        """
        导出为HTML文件
        
        :param name: 输出文件名（不含扩展名），默认使用标题
        :param template_path: 自定义模板路径
        :return: 输出文件路径
        """
        self._flush_chartg()
        if name is None:
            name = self.nb_title.replace('/', '_').replace('\\', '_')
        
        output_path = os.path.join(self.base_dir, f"{name}.html")
        if template_path is None:
            template_dir = Path(__file__).parent.parent / 'template'
            template_path = str(template_dir / 'notebook.html')
        else:
            template_dir = Path(template_path).parent
            template_path = str(template_path)
        
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template(Path(template_path).name)
        
        data = {
            '@context': 'https://schema.org',
            '@type': 'Dataset',
            'title': self.nb_title,
            'createdAt': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'children': [c.to_dict() for c in self.children]
        }
        data_json = json.dumps(data, ensure_ascii=False, default=str, indent=2)
        
        html_content = template.render(title=self.nb_title, data_json=data_json)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding='utf-8')

        print(f"输出: {output_path.name}")
        return str(output_path)
    
    def export_markdown(self, name: str = None) -> str:
        """
        导出为 Markdown 文件
        
        :param name: 输出文件名（不含扩展名），默认使用标题
        :return: 输出文件路径
        """
        self._flush_chartg()
        if name is None:
            name = self.nb_title.replace('/', '_').replace('\\', '_')
        
        output_path = os.path.join(self.base_dir, f"{name}.md")
        
        md_lines = []
        md_lines.append(f"# {self.nb_title}")
        md_lines.append("")
        md_lines.append(f"> 创建时间: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
        
        for child in self.children:
            md_lines.extend(self._cell_to_markdown(child))
        
        md_content = "\n".join(md_lines)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md_content, encoding='utf-8')
        
        print(f"输出: {output_path.name}")
        return str(output_path)
    
    def _cell_to_markdown(self, cell: CellLike, indent: int = 0) -> List[str]:
        """
        将 Cell 或 Section 转换为 Markdown 格式
        
        :param cell: Cell 或 Section 对象
        :param indent: 缩进层级（用于嵌套 Section）
        :return: Markdown 行列表
        """
        if isinstance(cell, Section):
            return self._section_to_markdown(cell, indent)
        
        lines = []
        cell_type = cell.type
        
        if cell.title:
            level = indent + 2
            lines.append(f"{'#' * level} {cell.title}")
            lines.append("")
        
        if cell_type == CellType.TITLE:
            level = indent + cell.options.get("level", 1)
            lines.append(f"{'#' * level} {cell.content}")
            lines.append("")
        
        elif cell_type == CellType.TEXT:
            lines.append(cell.content)
            lines.append("")
        
        elif cell_type == CellType.MARKDOWN:
            lines.append(cell.content)
            lines.append("")
        
        elif cell_type == CellType.CODE:
            language = cell.content.get("language", "python")
            code = cell.content.get("code", "")
            output = cell.content.get("output")
            lines.append(f"```{language}")
            lines.append(code)
            lines.append("```")
            lines.append("")
            if output:
                lines.append("**输出:**")
                lines.append("")
                lines.append("```")
                lines.append(output)
                lines.append("```")
                lines.append("")
        
        elif cell_type == CellType.TABLE:
            lines.extend(self._table_to_markdown(cell))
            lines.append("")
        
        elif cell_type == CellType.METRICS:
            lines.extend(self._metrics_to_markdown(cell))
            lines.append("")
        
        elif cell_type in (CellType.CHART, CellType.PYECHARTS):
            lines.extend(self._chart_to_markdown(cell))
            lines.append("")
        
        elif cell_type == CellType.DIVIDER:
            lines.append("---")
            lines.append("")
        
        elif cell_type == CellType.HTML:
            lines.append(cell.content)
            lines.append("")
        
        return lines
    
    def _section_to_markdown(self, section: Section, indent: int = 0) -> List[str]:
        """
        将 Section 转换为 Markdown 格式
        
        :param section: Section 对象
        :param indent: 缩进层级
        :return: Markdown 行列表
        """
        lines = []
        level = indent + section.options.get("level", 1)
        is_collapsed = section.options.get("collapsed")
        
        if is_collapsed is not None:
            lines.append(f"<details>")
            lines.append(f"<summary>{'#' * level} {section.title}</summary>")
            lines.append("")
        else:
            lines.append(f"{'#' * level} {section.title}")
            lines.append("")
        
        for child in section.children:
            lines.extend(self._cell_to_markdown(child, indent + 1))
        
        if is_collapsed is not None:
            lines.append("</details>")
            lines.append("")
        
        return lines
    
    def _table_to_markdown(self, cell: Cell) -> List[str]:
        """
        将表格 Cell 转换为 Markdown 表格
        
        :param cell: Cell 对象（类型为 TABLE）
        :return: Markdown 表格行列表
        """
        data = cell.content
        columns = cell.options.get("columns")
        
        if not data or len(data) == 0:
            return ["*（无数据）*", ""]
        
        if columns:
            cols = [col.get("field", col) if isinstance(col, dict) else col for col in columns]
        else:
            cols = list(data[0].keys())
        
        lines = []
        
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        separator = "| " + " | ".join("---" for _ in cols) + " |"
        lines.append(header)
        lines.append(separator)
        
        for row in data:
            row_values = [str(row.get(col, "")) for col in cols]
            lines.append("| " + " | ".join(row_values) + " |")
        
        lines.append("")
        return lines
    
    def _metrics_to_markdown(self, cell: Cell) -> List[str]:
        """
        将指标卡片 Cell 转换为 Markdown 表格
        
        :param cell: Cell 对象（类型为 METRICS）
        :return: Markdown 表格行列表
        """
        data = cell.content
        
        if not data or len(data) == 0:
            return ["*（无指标）*", ""]
        
        lines = []
        lines.append("| 指标 | 值 |")
        lines.append("|------|------|")
        
        for item in data:
            name = item.get("name", "")
            value = item.get("value", "")
            lines.append(f"| {name} | {value} |")
        
        lines.append("")
        return lines
    
    def _chart_to_markdown(self, cell: Cell) -> List[str]:
        """
        将图表 Cell 转换为 Markdown（折叠 + JSON）
        
        :param cell: Cell 对象（类型为 CHART 或 PYECHARTS）
        :return: Markdown 行列表
        """
        chart_data = cell.content.get("charts", {})
        
        lines = []
        lines.append("<details>")
        lines.append("<summary>📊 图表数据</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(chart_data, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("</details>")
        
        return lines
    
    def __repr__(self):
        return f"<Notebook '{self.nb_title}' with {len(self.children)} items>"
    
    def __len__(self):
        return len(self.children)
    
    def __getitem__(self, index):
        return self.children[index]
