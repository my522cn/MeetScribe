"""会议软件检测：进程轮询 + 通话状态确认。

关键修正（微信通话检测不到的根因）：
微信/钉钉/企业微信这类软件**常驻后台**，进程一直在，只看"进程从无到有"永远
触发不了——必须判断"它是不是正在通话"。所以判定分两层：
  1. 该应用的进程里有没有处于 Active 的 WASAPI 音频会话（通话必然占用音频设备）
  2. 有没有标题明显写着"正在通话/会议中"的窗口（无 pycaw 时的兜底，也是强证据）
触发时机也从"进程出现"改成"通话状态 false → true"。

去抖：音频活跃要连续几次轮询都为真才算通话，否则微信来一条语音消息的
提示音（1~2 秒）就会误触发录音。
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass, field
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

# 窗口标题里出现这些词 = 明确在通话/开会（注意用"会议中"而不是"会议"，
# 否则群名"项目会议群"会被误判成开会）
CALL_TITLE_KEYS = (
    "通话", "语音通话", "视频通话", "正在通话", "会议中", "正在共享",
    "屏幕共享", "zoom meeting", "meeting |", "teams 会议", "呼叫中",
)

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
    in_call: bool = False          # 是否真的在通话/开会（本次扫描的结论）
    title: str = ""                # 命中的通话窗口标题（没有则空）
    pids: list[int] = field(default_factory=list)


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


def _window_titles() -> dict[int, str]:
    """枚举可见窗口标题：{pid: 标题}。非 Windows 或失败时返回空。"""
    out: dict[int, str] = {}
    try:
        user32 = ctypes.windll.user32
        enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

        def cb(hwnd: wintypes.HWND, _lparam: wintypes.LPARAM) -> bool:
            try:
                if not user32.IsWindowVisible(hwnd):
                    return True
                n = user32.GetWindowTextLengthW(hwnd)
                if not n:
                    return True
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                out.setdefault(int(pid.value), buf.value)
            except Exception:  # noqa: BLE001
                pass
            return True

        user32.EnumWindows(enum_proc(cb), 0)
    except Exception:  # noqa: BLE001
        return {}
    return out


def _is_call_title(title: str) -> bool:
    low = (title or "").lower()
    return any(k in low for k in CALL_TITLE_KEYS)


def scan() -> list[MeetingApp]:
    """返回当前运行的会议软件（每个应用一条），并判断是否正在通话。"""
    audio_pids = _audio_active_pids()
    titles = _window_titles()
    grouped: dict[str, MeetingApp] = {}
    for proc in psutil.process_iter(["name", "pid"]):
        try:
            name = (proc.info["name"] or "").lower()
            if name not in WATCHED:
                continue
            pid = int(proc.info["pid"])
            app_name = WATCHED[name]
            app = grouped.get(app_name)
            if app is None:
                app = MeetingApp(process=name, app_name=app_name, pid=pid,
                                 audio_active=False, pids=[])
                grouped[app_name] = app
            app.pids.append(pid)
            if pid in audio_pids:
                app.audio_active = True
            t = titles.get(pid, "")
            if t and _is_call_title(t) and not app.title:
                app.title = t
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for app in grouped.values():
        # 窗口明写"通话中"是强证据；否则靠音频活跃（由 detector 做连续去抖）
        app.in_call = bool(app.title) or app.audio_active
    return list(grouped.values())


def capability() -> str:
    return "进程 + 音频会话" if _HAS_AUDIO_API else "仅进程（未安装 pycaw）"


class MeetingDetector:
    """在"通话状态 false → true"的瞬间触发一次。

    只认"进程从无到有"是错的：微信/钉钉/企微常驻后台，进程一直在，
    真正的信号是"它开始占用音频设备 / 弹出通话窗口"。
    """

    def __init__(self, confirm_polls: int = 2) -> None:
        # 连续多少次轮询都活跃才算真在通话（默认 2 次 ≈ 6 秒，滤掉提示音）
        self.confirm_polls = max(1, confirm_polls)
        self._in_call: set[str] = set()
        self._streak: dict[str, int] = {}
        self._last_scan = datetime.now()

    def poll(self) -> MeetingApp | None:
        apps = scan()
        now_in_call: set[str] = set()
        for app in apps:
            if app.title:
                self._streak[app.app_name] = self.confirm_polls  # 窗口证据：立即算数
            elif app.audio_active:
                self._streak[app.app_name] = self._streak.get(app.app_name, 0) + 1
            else:
                self._streak[app.app_name] = 0
            if self._streak.get(app.app_name, 0) >= self.confirm_polls:
                now_in_call.add(app.app_name)
        for name in list(self._streak):
            if name not in {a.app_name for a in apps}:
                self._streak.pop(name, None)

        new = now_in_call - self._in_call
        self._in_call = now_in_call
        self._last_scan = datetime.now()
        for app in apps:
            if app.app_name in new:
                app.in_call = True
                return app
        return None

    def prime(self) -> None:
        """启动时把"当前正在通话的软件"记为已知，避免一启动就自动录音。"""
        apps = scan()
        self._in_call = set()
        for app in apps:
            if app.title:
                self._in_call.add(app.app_name)
                self._streak[app.app_name] = self.confirm_polls
            elif app.audio_active:
                # 启动时已在放声音：先观察一轮再说，避免把播放语音消息当通话
                self._streak[app.app_name] = self.confirm_polls - 1

    def reset(self) -> None:
        self._in_call = set()
        self._streak = {}
