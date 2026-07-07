# -*- coding: utf-8 -*-
"""
utils/ast/expr_base.py — AstExpression 表达式基类 (公共基础设施)
=============================================================================

在五层架构中的位置: 规格层 — DSL 表达式对象模型

AstExpression 是所有 DSL 表达式的通用基类，封装了:
  1. 解析 (parse_expression)           — 字符串 → Python AST 树
  2. 自省 (get_variables/get_functions) — 提取依赖的变量和函数
  3. 复杂度 (ast_node_count)            — 节点数/深度度量
  4. 截面检测 (_has_any_cs)            — 是否有 cs_* 函数

职责边界:
  - AstExpression: 只做解析 + 自省，不关心数据形状
  - signals/v4/Expression: 继承后 + generate() (1D 择时信号)
  - factor/v4/FactorExpression: 继承后 + evaluate() (2D 因子面板)

设计理由:
  - 两个子类的 __init__ 逻辑 100% 相同 (parse→vars→funcs→complexity)
  - 差异仅在求值方法的输入/输出类型 (pd.Series vs ndarray)
  - 提取基类消除 ~30 行重复代码，统一解析入口

用法:
  >>> from utils.ast.expr_base import AstExpression
  >>> expr = AstExpression("ts_roc(CLOSE, 20) > 0")
  >>> expr.variables   # ['CLOSE']
  >>> expr.functions   # ['ts_roc']
  >>> expr.complexity  # 6
  >>> expr._tree       # ast.Expression 对象 (可直接传给 evaluate/eval_colwise)

[新增] 2026-06-22 从 signals/v4 Expression 和 factor/v4 FactorExpression 提取公共基类
=============================================================================
"""
import ast
from typing import List

from .dsl import parse_expression, get_variables, get_functions, ast_node_count
from .resolver import _has_any_cs, _is_outer_cs_rank_call


class AstExpression:
    """DSL 表达式基类

    封装了 DSL 表达式从字符串到 AST 树的完整解析和自省流程。
    子类只需实现各自的求值方法 (generate / evaluate / evaluate_ranked)。

    Attributes:
        expr_str (str):     原始表达式字符串
        name (str):         表达式名称 (默认取前60字符)
        _tree (ast.Expression): 解析后的 Python AST 树
        variables (list):   表达式引用的变量名列表 (ALL_CAPS)
        functions (list):   表达式调用的函数名列表 (snake_case)
        complexity (int):   AST 节点总数
    """

    def __init__(self, expr_str: str, name: str = None):
        """解析并自省表达式

        Args:
            expr_str: 表达式字符串 (如 "ts_roc(CLOSE, 20) > 0")
            name: 可选名称标识 (默认用表达式前60字符)

        Raises:
            DSLSyntaxError:  语法错误
            DSLSecurityError: 安全校验失败 (禁止节点/未注册函数/未注册变量)
        """
        self.expr_str = expr_str.strip()
        self.name = name or self.expr_str[:60]
        self._tree = parse_expression(self.expr_str)
        self.variables = get_variables(self._tree)
        self.functions = get_functions(self._tree)
        self.complexity = ast_node_count(self._tree)

        # 截面函数检测 (子类 evaluate_ranked 需要)
        self._has_cs = _has_any_cs(self._tree)
        self._is_outer_cs_rank = _is_outer_cs_rank_call(self._tree)

    @property
    def features_used(self) -> List[str]:
        """表达式依赖的所有变量 (兼容旧 API)"""
        return self.variables

    def __repr__(self):
        cls_name = type(self).__name__
        return f"{cls_name}({self.expr_str[:60]!r})"

    def __str__(self):
        return self.expr_str
