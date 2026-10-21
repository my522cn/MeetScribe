"""运行时路径解析：源码运行 vs PyInstaller 打包后运行。

打包成 exe 后，模块里的 ``__file__`` 会指向 PyInstaller 的临时解包目录
（``_internal``），如果照旧用 ``Path(__file__).parents[n]`` 推项目根，
就会去 ``_internal/models`` 找模型，一启动就报“模型不存在”。
所以所有“项目根”一律走这里：冻结态取 exe 所在目录，源码态取仓库根。
"""
from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    """模型、config、data 所在的可写目录（exe 旁边 / 仓库根）。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def bundle_root() -> Path:
    """PyInstaller 解包目录（只读资源），源码态与 project_root 相同。"""
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass).resolve() if meipass else project_root()
