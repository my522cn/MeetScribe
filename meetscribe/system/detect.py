"""会议软件检测：进程轮询 + 音频会话确认（可选）。

仅靠进程会出现误判（软件开着但没开会），所以第二层用 WASAPI 音频会话判断该进程
是否真的在占用音频设备。没有安装 pycaw 时自动降级为纯进程判断。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import psutil

WATCHED: dict[str, str] = {
    "wemeetapp.exe": "腾讯会议",
    "wemeet.exe": "腾讯会议",
    "dingtalk.exe": "钉钉",
    "feishu.exe": "飞书",
    "lark.exe": "飞书",
    "ms-teams.exe": "Teams",
    "teams.exe": "Teams",
    "wxwork.exe": "企业微信",
    "wechat.exe": "微信",
    "weixin.exe": "微信",
    "zoom.exe": "Zoom",
}

_HAS_AUDIO_API = False
try:  # pycaw 为可选依赖
    from pycaw.pycaw import AudioUtilities  # type: ignore

    _HAS_AUDIO_API = True
except Exception:  # noqa: BLE001
    AudioUtilities = None  # type: ignore


@dataclass
class MeetingApp:
    process: str
    app_name: str
    pid: int
    audio_active: bool


def _audio_active_pids() -> set[int]:
    if not _HAS_AUDIO_API:
        return set()
    pids: set[int] = set()
    try:
        for session in AudioUtilities.GetAllSessions():
            if session.Process and session.State == 1:  # 1 = Active
                pids.add(session.Process.pid)
    except Exception:  # noqa: BLE001
        return set()
    return pids


def scan() -> list[MeetingApp]:
    """返回当前正在运行的会议软件；audio_active 表示确实在占用音频。"""
    audio_pids = _audio_active_pids()
    found: list[MeetingApp] = []
    for proc in psutil.process_iter(["name", "pid"]):
        try:
            name = (proc.info["name"] or "").lower()
            if name not in WATCHED:
                continue
            found.append(
                MeetingApp(
                    process=name,
                    app_name=WATCHED[name],
                    pid=proc.info["pid"],
                    audio_active=proc.info["pid"] in audio_pids,
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def capability() -> str:
    return "进程 + 音频会话" if _HAS_AUDIO_API else "仅进程（未安装 pycaw）"


class MeetingDetector:
    """只在"从无到有"的瞬间触发一次，避免重复录音。"""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._last_scan = datetime.now()

    def poll(self) -> MeetingApp | None:
        apps = scan()
        current = {a.app_name for a in apps}
        new_names = current - self._seen
        self._seen = current
        for app in apps:
            if app.app_name in new_names:
                return app
        return None

    def prime(self) -> None:
        """启动时先把"已经在跑的软件"记为已知，避免一开机就误触发录音。"""
        self._seen = {a.app_name for a in scan()}

    def reset(self) -> None:
        self._seen = set()
