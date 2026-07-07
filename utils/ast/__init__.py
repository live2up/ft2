"""
utils.ast — 公共 AST 基础设施

[重构] 2026-07-08 v2 为当前版本。此 __init__.py 将 utils.ast.* 重定向到 v2,
保持 from utils.ast import X / from utils.ast.dsl import Y 等旧路径兼容。

v2 内部结构 (5 个实质文件):
  functions.py  — 原语层 + 变量层 (函数/变量注册表, FunctionSpec, ParamConstraint)
  dsl.py        — 语法层 (解析/求值/安全校验)
  spec.py       — 规格层 + AstExpression 基类 (AST 构建/规范化/语法规格)
  resolver.py   — 编排层 (截面函数嵌套解算)
  variables.py  — 兼容层 (内容已并入 functions.py, 仅 re-export)
"""
import os as _os

# [重构] 2026-07-08 将 utils.ast 的模块搜索路径扩展到 v2 子目录,
# 使 from utils.ast.dsl import X 等旧路径自动解析到 v2/dsl.py
_v2_path = _os.path.join(__path__[0], 'v2')  # type: ignore
if _v2_path not in __path__:  # type: ignore
    __path__.insert(0, _v2_path)  # type: ignore

from .v2 import *  # noqa: F401,F403
