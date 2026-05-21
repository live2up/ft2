# -*- coding: utf-8 -*-
"""
表格格式化工具类
专为解决中文字符宽度问题而设计

Version: 2026.03.27

========================================
与 pip tabulate 的差异
========================================
| 特性               | 本工具              | pip tabulate           |
|-------------------|--------------------|-----------------------|
| 自动打印           | ❌ 只返回字符串     | ❌ 只返回字符串        |
| 返回字符串         | ✅ 始终返回         | ✅ 始终返回            |
| 中文字符宽度       | ✅ 自动按2计算      | ❌ 按1计算，对齐错乱   |
| 函数调用           | print(tabulate(d, h)) | print(tabulate(d, h)) |
| headers参数        | ✅ 可选（自动提取） | ❌ 必需              |
| HTML格式支持       | ✅ tablefmt='html'   | ✅ tablefmt='html'   |
| 表格样式           | ⚠️ 固定ASCII样式   | ✅ 多种样式可选      |
| pandas DataFrame   | ✅ 原生支持         | ✅ 需安装pandas       |

注意：
- 本工具默认不打印，返回字符串，用户可自由决定何时打印
- 如需自动打印，可直接调用：print(tabulate(data))
- 中文字符宽度按2计算，确保对齐完美
- 支持 pandas DataFrame 直接传入，自动转换为字典列表格式

========================================
支持的数据格式
========================================
1. pandas DataFrame格式:
   import pandas as pd
   df = pd.DataFrame({
       'Time': ['2026-01-06', '2026-01-07'],
       'Close': [2654.33, 2668.61],
       'Ma': [2631.15, 2633.39],
   })
   tabulate(df)

2. 字典列表格式 (推荐):
   data = [
       {'Time': '2026-01-06', 'Close': 2654.33, 'Ma': 2631.15},
       {'Time': '2026-01-07', 'Close': 2668.61, 'Ma': 2633.39},
   ]
   
   # 方式1：自动从字典key获取表头（推荐）
   tabulate(data)
   
   # 方式2：自定义表头
   tabulate(data, ['Time', 'Close', 'Ma'])

3. 二维列表格式:
   data = [
       ['2026-01-06', 2654.33, 2631.15],
       ['2026-01-07', 2668.61, 2633.39],
   ]
   
   # 方式1：自动生成默认列名
   tabulate(data)
   
   # 方式2：自定义表头
   tabulate(data, ['Time', 'Close', 'Ma'])

注意: 
- pandas DataFrame、字典列表和二维列表格式同时支持 tabulate() 和 tabulate(tablefmt='html')
- headers 参数可选，不传则自动生成表头
- tablefmt 参数可选，'text'为纯文本（默认），'html'为HTML表格
- 函数返回表格字符串，不自动打印，用户可自由决定何时打印

========================================
使用示例
========================================
from tabulate import tabulate

# 示例1: 字典列表（纯文本格式）
data = [
    {'Time': '2026-01-06 13:15:00', 'Close': 2654.33, 'Ma': 2631.15, 'Upper Band': 2656.61, 'Lower Band': 2605.69},
    {'Time': '2026-01-06 13:30:00', 'Close': 2668.61, 'Ma': 2633.39, 'Upper Band': 2663.58, 'Lower Band': 2603.21},
]
# 获取字符串后打印
table_str = tabulate(data, title="布林带历史数据")
print(table_str)

# 或者直接打印（两种写法等效）
print(tabulate(data, title="布林带历史数据"))

# 输出:
# +---------------------+-----------+-----------+--------------+--------------+
# |        Time         |   Close   |    Ma     |  Upper Band  |  Lower Band  |
# +---------------------+-----------+-----------+--------------+--------------+
# | 2026-01-06 13:15:00 |  2654.33  |  2631.15  |   2656.61    |   2605.69    |
# | 2026-01-06 13:30:00 |  2668.61  |  2633.39  |   2663.58    |   2603.21    |
# +---------------------+-----------+-----------+--------------+--------------+

# 示例2: 字典列表（HTML格式）
data = [
    {'Time': '2026-01-06 13:15:00', 'Close': 2654.33, 'Ma': 2631.15, 'Upper Band': 2656.61, 'Lower Band': 2605.69},
    {'Time': '2026-01-06 13:30:00', 'Close': 2668.61, 'Ma': 2633.39, 'Upper Band': 2663.58, 'Lower Band': 2603.21},
]
# 获取HTML字符串后打印
html_str = tabulate(data, tablefmt='html', title="布林带数据")
print(html_str)

# 或者直接打印（两种写法等效）
print(tabulate(data, tablefmt='html', title="布林带数据"))

# 输出:
# <h3 class="tabulate-title">布林带数据</h3>
# <table class="tabulate">
# <thead>
# <tr>
# <th>Time</th>
# <th>Close</th>
# <th>Ma</th>
# <th>Upper Band</th>
# <th>Lower Band</th>
# </tr>
# </thead>
# ...

# 示例3: 自动从字典key获取表头
data = [
    {'Time': '2026-01-06', 'Close': 2654.33, 'Ma': 2631.15},
    {'Time': '2026-01-07', 'Close': 2668.61, 'Ma': 2633.39},
]
tabulate(data, title="布林带数据")  # 不传headers，自动从key获取表头

# 示例4: 自定义表头
data = [
    {'Time': '2026-01-06', 'Close': 2654.33, 'Ma': 2631.15},
    {'Time': '2026-01-07', 'Close': 2668.61, 'Ma': 2633.39},
]
tabulate(data, headers=['时间', '收盘价', '均线'], title="布林带数据")

========================================
注意事项
========================================
- 小数位数需要在数据准备阶段处理 (如 df[col].round(2))
- 字典格式按 key 匹配列，二维列表按位置索引匹配列
- 函数返回表格字符串，不自动打印，用户可自由决定何时打印
- 如需自动打印，可直接调用：print(tabulate(data))
- 中文字符宽度按2计算，确保对齐完美

========================================
返回字符串的优势
========================================
1. 灵活性：用户可以自由决定何时、何地打印（控制台、文件、邮件等）
2. 可组合性：可以拼接多个表格字符串
3. 可保存性：可以保存到文件或数据库
4. 可测试性：可以断言返回的字符串内容
5. 兼容性：仍然支持直接打印的方式（print(tabulate(data))）

========================================
HTML 样式说明
========================================
tabulate(tablefmt='html') 输出纯净 HTML，使用以下 class，可自行定义 CSS:

<style>
.tabulate { border-collapse: collapse; width: 100%; border: 1px solid #ddd; font-family: Arial, sans-serif; }
.tabulate th { background-color: #f2f2f2; border: 1px solid #ddd; padding: 8px; text-align: left; }
.tabulate td { border: 1px solid #ddd; padding: 8px; }
.tabulate tr.odd { background-color: #ffffff; }
.tabulate tr.even { background-color: #f9f9f9; }
.tabulate-title { font-family: Arial, sans-serif; margin-bottom: 10px; }
</style>
"""


def _str_width(text):
    """
    计算字符串的显示宽度，中文字符宽度为2，英文字符宽度为1
    
    参数:
        text: 需要计算宽度的字符串
        
    返回:
        字符串的显示宽度
    """
    width = 0
    for char in str(text):
        code = ord(char)
        if code < 128:  # ASCII字符
            width += 1
        elif code in range(0x2713, 0x2714) or code in range(0x274C, 0x274D):  # ✓, ❌ 等符号
            width += 1
        elif code in range(0x2600, 0x27FF):  # 其他符号范围
            width += 1
        elif code in range(0x2190, 0x21FF):  # 箭头符号
            width += 1
        elif code in range(0x2300, 0x23FF):  # 技术符号
            width += 1
        else:  # 中文宽字符
            width += 2
    return width


def _format_string(text, width, align='left'):
    """
    格式化字符串使其按照指定宽度对齐，正确处理中英文混合文本
    
    参数:
        text: 需要格式化的文本
        width: 目标宽度
        align: 对齐方式 ('left', 'right', 'center')
        
    返回:
        格式化后的字符串
    """
    text = str(text)
    text_width = _str_width(text)
    padding = width - text_width
    
    if padding <= 0:
        return text
    
    if align == 'left':
        return text + ' ' * padding
    elif align == 'right':
        return ' ' * padding + text
    elif align == 'center':
        left_padding = padding // 2
        right_padding = padding - left_padding
        return ' ' * left_padding + text + ' ' * right_padding
    else:
        return text + ' ' * padding


def _format_table(headers, data):
    """
    格式化表格数据，正确处理中英文字符宽度
    
    参数:
        headers: 表头列表或None
            - 列表: ['Time', 'Close', 'Ma']
            - None: 自动从字典列表的key获取表头
        data: 数据列表，支持三种格式：
            - pandas DataFrame: 自动转换为字典列表
            - 字典列表: [{'Time': '...', 'Close': 123}, ...] (按 key 匹配)
            - 二维列表: [['...', 123], ...] (按位置索引匹配)
        
    返回:
        格式化后的表格字符串
    """
    if hasattr(data, 'to_dict'):
        data = data.to_dict('records')
    
    if not data:
        return ""
    
    is_dict_list = isinstance(data[0], dict)
    
    # 如果 headers 为 None，自动从数据中提取
    if headers is None:
        if is_dict_list:
            # 字典列表：从第一个字典的key获取表头
            headers = list(data[0].keys())
        else:
            # 二维列表：无法自动生成表头，使用默认列名
            headers = [f"Column {i+1}" for i in range(len(data[0]) if data[0] else 0)]
    
    # 计算每列的最大显示宽度
    col_widths = []
    for i, header in enumerate(headers):
        max_width = _str_width(header)
        for row in data:
            if is_dict_list:
                value = str(row.get(header, ""))
            else:
                value = str(row[i] if i < len(row) else "")
            max_width = max(max_width, _str_width(value))
        col_widths.append(max_width)
    
    # 创建分隔线
    separator = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"
    
    # 创建表头行
    header_row = "|"
    for i, header in enumerate(headers):
        formatted_header = _format_string(header, col_widths[i], 'center')
        header_row += f" {formatted_header} |"
    
    # 创建数据行
    rows = []
    for row in data:
        data_row = "|"
        for i, header in enumerate(headers):
            if is_dict_list:
                value = str(row.get(header, ""))
            else:
                value = str(row[i] if i < len(row) else "")
            formatted_value = _format_string(value, col_widths[i], 'left')
            data_row += f" {formatted_value} |"
        rows.append(data_row)
    
    # 组合表格组件
    table = separator + "\n"
    table += header_row + "\n"
    table += separator + "\n"
    table += "\n".join(rows) + "\n"
    table += separator
    
    return table


def _format_html_table(headers, data):
    """
    生成HTML格式的表格，正确处理中英文字符宽度
    
    参数:
        headers: 表头列表或None
            - 列表: ['Time', 'Close', 'Ma']
            - None: 自动从字典列表的key获取表头
        data: 数据列表，支持三种格式：
            - pandas DataFrame: 自动转换为字典列表
            - 字典列表: [{'Time': '...', 'Close': 123}, ...] (按 key 匹配)
            - 二维列表: [['...', 123], ...] (按位置索引匹配)
        
    返回:
        格式化后的HTML表格字符串
    """
    if hasattr(data, 'to_dict'):
        data = data.to_dict('records')
    
    if not data:
        return ""
    
    is_dict_list = isinstance(data[0], dict)
    
    # 如果 headers 为 None，自动从数据中提取
    if headers is None:
        if is_dict_list:
            # 字典列表：从第一个字典的key获取表头
            headers = list(data[0].keys())
        else:
            # 二维列表：无法自动生成表头，使用默认列名
            headers = [f"Column {i+1}" for i in range(len(data[0]) if data[0] else 0)]
    
    html = "<table class=\"tabulate\">\n"
    
    html += "<thead>\n<tr>\n"
    for header in headers:
        html += f"<th>{header}</th>\n"
    html += "</tr>\n</thead>\n"
    
    html += "<tbody>\n"
    for i, row in enumerate(data):
        row_class = "even" if i % 2 ==1 else "odd"
        html += f"<tr class=\"{row_class}\">\n"
        for j, header in enumerate(headers):
            if is_dict_list:
                value = str(row.get(header, ""))
            else:
                value = str(row[j] if j < len(row) else "")
            html += f"<td>{value}</td>\n"
        html += "</tr>\n"
    html += "</tbody>\n"
    html += "</table>\n"
    return html


def tabulate(data, headers=None, title=None, tablefmt='text'):
    """
    类似tabulate的函数式接口，生成表格
    
    参数:
        data: 数据列表，支持三种格式：
            - pandas DataFrame: 自动转换为字典列表
            - 字典列表: [{'Time': '...', 'Close': 123}, ...] (按 key 匹配)
            - 二维列表: [['...', 123], ...] (按位置索引匹配)
        headers: 表头列表或None
            - 列表: ['Time', 'Close', 'Ma']
            - None: 自动从字典列表的key获取表头
        title: 表格标题（可选）
        tablefmt: 表格格式
            - 'text': 纯文本表格（默认）
            - 'html': HTML表格
        
    返回:
        格式化后的表格字符串
    """
    if tablefmt == 'html':
        table = _format_html_table(headers, data)
    else:
        table = _format_table(headers, data)
    
    if title:
        if tablefmt == 'html':
            title_str = f"<h3 class=\"tabulate-title\">{title}</h3>\n"
        else:
            title_str = f"{title}\n"
        return title_str + table
    else:
        return table