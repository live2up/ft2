"""
factor/v4/discovered/ — GP/ML 发现的因子存档
=============================================================================

目录约定：
  gp_round_{N}.json          — GP 第 N 轮全量结果（自动生成，FactorLibrary 读写）
  manual_{date}.py           — 手工精选因子（人类编辑）
  gp_round_{N}_top20.py      — GP 每轮 Top 20 精选（自动生成，方便 code review）

查询接口：
  >>> from factor.v4.discovered import load_discovered
  >>> lib = load_discovered(r'path/to/discovered')
  >>> lib.top(20, sort_by='sharpe')
  >>> lib.by_time('2026-06-01', '2026-06-04')  # 按时间范围筛选
"""

import os
import json

from factor.v4.base import FactorLibrary, LibraryEntry, FactorCategory


def load_discovered(discovered_dir: str = None) -> FactorLibrary:
    """从 discovered/ 目录加载所有已发现因子

    自动扫描目录下所有 .json 文件并合并到 FactorLibrary。
    同名 alpha_id 以首次注册的为准（后发现的不会覆盖）。

    Args:
        discovered_dir: 目录路径，默认当前模块所在目录

    Returns:
        FactorLibrary: 包含所有已发现因子的库
    """
    if discovered_dir is None:
        discovered_dir = os.path.dirname(os.path.abspath(__file__))

    lib = FactorLibrary()
    if not os.path.isdir(discovered_dir):
        return lib

    json_files = sorted(
        [f for f in os.listdir(discovered_dir) if f.endswith('.json')]
    )
    for fn in json_files:
        path = os.path.join(discovered_dir, fn)
        temp_lib = FactorLibrary.load(path)
        for entry in temp_lib._entries.values():
            if entry.alpha_id not in lib._entries:
                lib.register(entry)

    return lib


def merge_discovered(
    existing_lib: FactorLibrary,
    discovered_dir: str = None,
) -> int:
    """将 discovered/ 中的因子合并到已有 FactorLibrary

    Returns:
        int: 新增因子数
    """
    discovered = load_discovered(discovered_dir)
    count = 0
    for entry in discovered._entries.values():
        if existing_lib.register(entry):
            count += 1
    return count
