"""会话回放：跨分段连续播放，时间轴与会议时间一一对应。

设计（虚拟混音时间轴）：
- 不合并文件：按 session.json 的分段清单，把 mic/loop 两轨各自看成一条
  "会议时间轴上的连续音轨"，每次取 0.1 秒的会议时间窗口；
- 每个窗口里，从覆盖该时间窗的分段中取样本（不足补静音），两轨相加成
  一路混音喂给声卡——段间无缝、两轨严格同步、重叠发言同时可闻；
- player.play() 按真实时长阻塞，天然实现实时节拍；暂停时冻结时间锚点，
  拖动进度条 = 用新的会议时间重建锚点继续播。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import soundcard as sc
import soundfile as sf

FEED_MS = 100               # 每个混音窗口的会议时长
SAMPLE_RATE = 16000
TRACKS = ("mic", "loop")


class _TrackReader:
    """按会议时间从一条轨的分段文件里取样本（跨段自动衔接，惰性解码）。"""

    def __init__(self, chunks: list[tuple[int, Path, int]]) -> None:
        self.chunks = chunks  # [(start_ms, path, duration_ms)] 已按时间排序
        self._idx = -1
        self._audio: np.ndarray | None = None

    def read(self, start_ms: int, n: int) -> np.ndarray:
        """取会议时间 start_ms 起的 n 个样本；没有音频覆盖的区间补零。"""
        out = np.zeros(n, dtype=np.float32)
        if not self.chunks:
            return out
        filled = 0
        while filled < n:
            seg_start, path, seg_dur = self.chunks[self._idx] if self._idx >= 0 else (None, None, None)
            if self._idx < 0 or start_ms + int(filled / SAMPLE_RATE * 1000) >= seg_start + seg_dur:
                # 需要下一段
                nxt = self._idx + 1
                if nxt >= len(self.chunks):
                    break
                seg_start, path, seg_dur = self.chunks[nxt]
                if seg_start >= start_ms + int(n / SAMPLE_RATE * 1000):
                    break  # 这一段在窗口之后，剩下的都是静音
                self._idx = nxt
                data, sr = sf.read(path, dtype="float32", always_2d=True)
                self._audio = data.mean(axis=1).astype(np.float32)
                if sr != SAMPLE_RATE:
                    import soxr
                    self._audio = soxr.resample(self._audio, sr, SAMPLE_RATE)
                continue
            # 当前段与请求窗口有重叠，取交集
            t = start_ms + int(filled / SAMPLE_RATE * 1000)
            if t < seg_start:
                gap = min(int((seg_start - t) * SAMPLE_RATE / 1000), n - filled)
                filled += gap  # 段前空隙补零
                continue
            off = int((t - seg_start) * SAMPLE_RATE / 1000)
            take = min(len(self._audio) - off, n - filled)
            if take <= 0:
                self._idx += 1
                if self._idx >= len(self.chunks):
                    break
                self._audio = None
                continue
            out[filled : filled + take] = self._audio[off : off + take]
            filled += take
        return out


class SessionPlayer:
    """一次会话的播放器。同一时刻只跑一个播放线程。"""

    def __init__(self, session_dir: Path, on_end=None) -> None:
        self.session_dir = Path(session_dir)
        data = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        self.duration_ms = int(data.get("duration_ms") or 0)
        self.on_end = on_end
        self._readers = {
            t: _TrackReader(sorted(
                ((int(s["start_ms"]), session_dir / s["file"], int(s["duration_ms"]))
                 for s in data.get("segments", []) if s["track"] == t
                 and (session_dir / s["file"]).exists()),
                key=lambda x: x[0],
            ))
            for t in TRACKS
        }
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._seek_to: int | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._anchor_wall = 0.0   # 当前锚点对应的墙钟
        self._anchor_ms = 0       # 当前锚点对应的会议时间
        # play() 被调用的时刻：开播前要开 WASAPI 设备（数百毫秒），
        # 位置必须从“点播放那一刻”起算，否则进度条会先跳到片尾再跳回来
        self._start_wall: float | None = None
        # 播放线程完成首次锚定之前，position 一律返回起始值
        # （否则旧 _anchor_wall=0 会算出越界的巨大位置）
        self._anchored = threading.Event()

    # ---------- 对外 ----------
    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def paused(self) -> bool:
        return self.active and self._pause.is_set()

    def play(self, from_ms: int = 0) -> None:
        """从会议时间 from_ms 开始/继续/跳转播放。"""
        with self._lock:
            self._seek_to = max(0, int(from_ms))
            self._pause.clear()
            if self.active:
                return
            self._stop.clear()
            self._anchored.clear()
            self._anchor_ms = max(0, int(from_ms))
            self._anchor_wall = time.perf_counter()
            self._start_wall = self._anchor_wall
            self._thread = threading.Thread(target=self._run, name="replay", daemon=True)
            self._thread.start()

    def pause(self) -> None:
        if self.active:
            self._pause.set()
            # 固定等 0.15s（略大于一个 0.1s 的喂入块）：正在播的那块结束后
            # 播放线程会把锚点写到真实的已播位置，此后冻结才是准确位置
            time.sleep(0.15)

    def resume(self) -> None:
        if self.active and self._pause.is_set():
            self._anchor_wall = time.perf_counter()
            self._pause.clear()

    def stop(self) -> None:
        self._stop.set()
        self._pause.clear()
        self._anchored.clear()

    def position_ms(self) -> int:
        if not self._anchored.is_set() or not self.active or self._pause.is_set():
            return self._anchor_ms
        elapsed = (time.perf_counter() - self._anchor_wall) * 1000
        return min(self.duration_ms, int(self._anchor_ms + elapsed))

    # ---------- 播放线程 ----------
    def _run(self) -> None:
        try:
            self._loop()
        finally:
            self._pause.set()  # 线程退出后 position 冻结在锚点
            if not self._stop.is_set() and self.on_end is not None:
                self.on_end()

    def _loop(self) -> None:
        speaker = sc.default_speaker()
        with speaker.player(samplerate=SAMPLE_RATE, channels=1) as p:
            while not self._stop.is_set():
                with self._lock:
                    start = self._seek_to
                    self._seek_to = None
                if start is None:
                    return
                self._anchor_ms = start
                # 首次开播：锚点从 play() 调用时刻起算，把开设备的耗时算进已播时间
                if self._start_wall is not None:
                    self._anchor_wall = self._start_wall
                    self._start_wall = None
                else:
                    self._anchor_wall = time.perf_counter()
                self._anchored.set()
                t = start
                while t < self.duration_ms and not self._stop.is_set():
                    # seek 请求：跳出内层，重建锚点
                    if self._seek_to is not None:
                        break
                    # 暂停：墙钟随暂停同步后移，恢复后位置连续
                    while self._pause.is_set() and not self._stop.is_set():
                        self._anchor_wall += 0.05
                        time.sleep(0.05)
                    if self._stop.is_set():
                        return
                    mixed = np.zeros(int(FEED_MS * SAMPLE_RATE / 1000), dtype=np.float32)
                    for reader in self._readers.values():
                        mixed += reader.read(t, len(mixed))
                    np.clip(mixed, -1.0, 1.0, out=mixed)
                    p.play(mixed.reshape(-1, 1))
                    t += FEED_MS
                    self._anchor_ms = t
                    self._anchor_wall = time.perf_counter()
                else:
                    return  # 自然播完
                # 到这里说明有 seek：继续外层 while
                continue
