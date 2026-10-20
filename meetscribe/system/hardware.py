"""硬件探测与配置建议。

只用注册表 + psutil，不调 wmic/PowerShell（那些在界面上会卡几百毫秒）。
结果缓存一次，配置页点"重新检测"才刷新。
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

import psutil

_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def _reg(path: str, name: str) -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as k:
            return str(winreg.QueryValueEx(k, name)[0]).strip()
    except Exception:  # noqa: BLE001
        return ""


def cpu_name() -> str:
    name = _reg(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString")
    if not name:
        name = os.environ.get("PROCESSOR_IDENTIFIER", "未知 CPU")
    return " ".join(name.split())


def gpu_names() -> list[str]:
    names: list[str] = []
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS) as root:
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(root, i)
                except OSError:
                    break
                i += 1
                if not sub.isdigit():
                    continue
                desc = _reg(_DISPLAY_CLASS + "\\" + sub, "DriverDesc")
                if desc and desc not in names:
                    names.append(desc)
    except Exception:  # noqa: BLE001
        pass
    return names


def has_cuda() -> bool:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


@dataclass
class Hardware:
    cpu: str
    cores_physical: int
    cores_logical: int
    ram_gb: float
    gpus: list[str] = field(default_factory=list)
    cuda: bool = False

    # ---------- 建议 ----------
    @property
    def asr_threads(self) -> int:
        return max(2, min(4, self.cores_logical // 4))

    @property
    def refine_threads(self) -> int:
        return max(2, min(4, self.cores_logical // 4))

    @property
    def can_two_pass(self) -> bool:
        return self.cores_logical >= 8 and self.ram_gb >= 8

    @property
    def can_live_refine(self) -> bool:
        return self.cores_logical >= 12 and self.ram_gb >= 12

    @property
    def llm_tier(self) -> str:
        """给 Ollama 选规模的建议：3b / 4b / 8b。"""
        if self.ram_gb >= 24 and self.cores_logical >= 12:
            return "8b"
        if self.ram_gb >= 12:
            return "4b"
        return "1.7b"

    @property
    def verdict(self) -> str:
        if self.can_live_refine and self.can_two_pass:
            return "配置充足：可同时开启两遍识别与实时滑窗校正"
        if self.can_two_pass:
            return "配置够用：建议开启两遍识别，滑窗校正在长会议中可能吃紧"
        return "配置偏紧：建议只开流式识别，或关闭回环轨降低负载"

    def summary(self) -> str:
        gpu = "、".join(self.gpus) if self.gpus else "未检测到独立显卡"
        if self.cuda:
            gpu += "（支持 CUDA）"
        elif self.gpus:
            gpu += "（CPU 推理）"
        return (
            f"{self.cpu}\n"
            f"{self.cores_physical} 核 {self.cores_logical} 线程 · {self.ram_gb:.0f} GB 内存\n"
            f"{gpu}"
        )


_cache: Hardware | None = None


def profile(force: bool = False) -> Hardware:
    global _cache
    if _cache is not None and not force:
        return _cache
    _cache = Hardware(
        cpu=cpu_name(),
        cores_physical=psutil.cpu_count(logical=False) or 0,
        cores_logical=psutil.cpu_count(logical=True) or 0,
        ram_gb=round(psutil.virtual_memory().total / 1024**3, 1),
        gpus=gpu_names(),
        cuda=has_cuda(),
    )
    return _cache
