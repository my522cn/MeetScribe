"""双轨分段录音器：麦克风轨 + 系统声音回环轨，按固定时长切分 FLAC 文件。

设计要点：
- 两轨独立采集，样本按"距会议开始的样本数"换算时间，天然对齐同一条虚拟时间轴；
- 每段录制完成立即落盘并更新 session.json，崩溃最多丢一个分段的尾部；
- 便于后续：线上模式（mic=我 / loop=对方），线下模式（mic 再做声纹分离）。
"""
from __future__ import annotations

import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
import soundfile as sf

from .devices import default_input_device, find_loopback_source
from .manifest import Segment, SessionManifest, new_session_id
from .resample import StreamingResampler

SAMPLE_RATE = 16000
BLOCK = 1600  # 100ms


class DualTrackRecorder:
    def __init__(
        self,
        root: Path,
        *,
        segment_seconds: int = 300,
        samplerate: int = SAMPLE_RATE,
        mode: str = "online",
        title: str = "",
        source: str = "manual",
        mic_device: int | None = None,
        on_segment: Callable[[Segment], None] | None = None,
        on_audio: Callable[[str, np.ndarray], None] | None = None,
    ) -> None:
        self.samplerate = samplerate
        self.segment_samples = segment_seconds * samplerate
        self.on_segment = on_segment
        self.on_audio = on_audio
        self.mode = mode
        self.session_id = new_session_id()
        self.session_dir = Path(root) / f"{self.session_id}_{source}"
        self.manifest_path = self.session_dir / "session.json"

        self.manifest = SessionManifest(
            session_id=self.session_id,
            title=title or f"{datetime.now():%Y-%m-%d %H:%M} 会议",
            source=source,
            mode=mode,
            sample_rate=samplerate,
            started_at=datetime.now().isoformat(timespec="seconds"),
        )

        self._mic_device = mic_device
        self._queues: dict[str, queue.Queue] = {"mic": queue.Queue(maxsize=300), "loop": queue.Queue(maxsize=300)}
        self._buf: dict[str, list[np.ndarray]] = {"mic": [], "loop": []}
        self._buf_count: dict[str, int] = {"mic": 0, "loop": 0}
        self._written: dict[str, int] = {"mic": 0, "loop": 0}
        self._live: dict[str, int] = {"mic": 0, "loop": 0}
        self._seg_index: dict[str, int] = {"mic": 0, "loop": 0}
        self._level: dict[str, float] = {"mic": 0.0, "loop": 0.0}
        self._errors: list[str] = []
        self._mic_resampler: StreamingResampler | None = None
        self._running = False
        self._paused = False
        self._threads: list[threading.Thread] = []
        self._start_mono = 0.0

    # ---------- 生命周期 ----------
    def start(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "mic").mkdir(exist_ok=True)
        (self.session_dir / "loop").mkdir(exist_ok=True)
        self._running = True
        self._start_mono = time.monotonic()

        self._threads.append(threading.Thread(target=self._mic_worker, name="mic", daemon=True))
        self._threads.append(threading.Thread(target=self._loop_worker, name="loop", daemon=True))
        self._threads.append(threading.Thread(target=self._writer_worker, name="writer", daemon=True))
        for t in self._threads:
            t.start()
        self.manifest.save(self.manifest_path)

    def stop(self) -> SessionManifest:
        self._running = False
        for t in self._threads:
            t.join(timeout=3)
        self._flush_all()
        self.manifest.ended_at = datetime.now().isoformat(timespec="seconds")
        self.manifest.status = "completed" if not self._errors else "completed_with_errors"
        if self._errors:
            self.manifest.tracks["errors"] = self._errors
        self.manifest.save(self.manifest_path)
        return self.manifest

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    # ---------- 状态 ----------
    @property
    def elapsed_ms(self) -> int:
        """已录时长：取两轨中推进最快的（含尚未写盘的缓冲音频）。"""
        return int(max(self._live.values()) / self.samplerate * 1000)

    def levels(self) -> dict[str, float]:
        return dict(self._level)

    # ---------- 采集线程 ----------
    def _mic_worker(self) -> None:
        try:
            dev_index = self._mic_device
            if dev_index is None:
                info = default_input_device()
                dev_index = info.index if info else None

            stream, native_rate, channels = self._open_mic_stream(dev_index)
            self._mic_resampler = StreamingResampler(native_rate, self.samplerate)
            with stream:
                self.manifest.tracks["mic"] = {
                    "device": sd.query_devices(dev_index)["name"] if dev_index is not None else "default",
                    "native_rate": native_rate,
                    "channels": channels,
                    "sample_rate": self.samplerate,
                }
                block = int(native_rate * 0.1)
                while self._running:
                    data, _ = stream.read(block)
                    mono = data.mean(axis=1) if data.ndim > 1 else data
                    self._push("mic", self._mic_resampler.process(mono.reshape(-1).copy()))
            tail = self._mic_resampler.flush() if self._mic_resampler else None
            if tail is not None and len(tail):
                try:
                    self._queues["mic"].put_nowait(tail)
                except queue.Full:
                    pass
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"mic: {exc}")

    def _open_mic_stream(self, dev_index: int | None):
        """WASAPI 共享模式常拒绝 16kHz；且阵列麦用单声道打开会拿到错误/静音通道，
        因此优先按设备原生通道数打开，再在采集后自行降混为单声道。"""
        info = sd.query_devices(dev_index) if dev_index is not None else sd.query_devices(kind="input")
        native = int(info["default_samplerate"])
        max_ch = int(info["max_input_channels"])
        # 优先原生通道数，其次 2，最后才退回 1
        channel_opts: list[int] = []
        for ch in (max_ch, 2, 1):
            if 0 < ch <= max_ch and ch not in channel_opts:
                channel_opts.append(ch)
        rate_opts: list[int] = []
        for r in (native, 48000, 44100, self.samplerate):
            if r > 0 and r not in rate_opts:
                rate_opts.append(r)

        last_err: Exception | None = None
        for rate in rate_opts:
            for ch in channel_opts:
                try:
                    sd.check_input_settings(
                        device=dev_index, samplerate=rate, channels=ch, dtype="float32"
                    )
                    stream = sd.InputStream(
                        device=dev_index, samplerate=rate, channels=ch,
                        dtype="float32", blocksize=int(rate * 0.1),
                    )
                    return stream, rate, ch
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
        raise RuntimeError(f"无法打开麦克风: {last_err}")


    def _loop_worker(self) -> None:
        try:
            # WASAPI 走 COM，子线程必须自行初始化，否则报 0x800401f0
            try:
                import comtypes

                comtypes.CoInitialize()
            except Exception:  # noqa: BLE001
                pass
            src = find_loopback_source()
            if src is None:
                self._errors.append("loop: 未找到系统声音回环设备")
                return
            name, dev_id = src
            import soundcard as sc

            with sc.get_microphone(id=dev_id, include_loopback=True).recorder(
                samplerate=self.samplerate, channels=1
            ) as rec:
                self.manifest.tracks["loop"] = {"device": name, "sample_rate": self.samplerate}
                while self._running:
                    block = rec.record(numframes=BLOCK)
                    arr = np.asarray(block, dtype=np.float32).reshape(-1)
                    self._push("loop", arr)
        except Exception as exc:  # noqa: BLE001
            self._errors.append(f"loop: {exc}")

    def _push(self, track: str, block: np.ndarray) -> None:
        if self._paused or not self._running:
            return
        try:
            self._queues[track].put_nowait(block)
            self._live[track] += len(block)
        except queue.Full:
            pass  # 落盘跟不上时丢块，保证录音线程不被阻塞
        if self.on_audio is not None:
            try:
                self.on_audio(track, block)  # 旁路给实时识别，必须轻量（只入队）
            except Exception:  # noqa: BLE001
                pass

    # ---------- 写盘线程 ----------
    def _writer_worker(self) -> None:
        while self._running:
            time.sleep(0.05)
            self._drain()
            self._flush_ready()

    def _drain(self) -> None:
        for track, q in self._queues.items():
            while True:
                try:
                    block = q.get_nowait()
                except queue.Empty:
                    break
                self._buf[track].append(block)
                self._buf_count[track] += len(block)
                self._level[track] = float(np.sqrt(np.mean(block.astype(np.float32) ** 2) + 1e-12))

    def _flush_ready(self) -> None:
        for track in self._buf:
            while self._buf_count[track] >= self.segment_samples:
                data = np.concatenate(self._buf[track])
                self._write_segment(track, data[: self.segment_samples])
                rest = data[self.segment_samples:]
                self._buf[track] = [rest] if len(rest) else []
                self._buf_count[track] = len(rest)

    def _flush_all(self) -> None:
        self._drain()
        for track in self._buf:
            if self._buf_count[track] > 0:
                data = np.concatenate(self._buf[track])
                self._write_segment(track, data)
            self._buf[track] = []
            self._buf_count[track] = 0

    def _write_segment(self, track: str, data: np.ndarray) -> None:
        if len(data) == 0:
            return
        self._seg_index[track] += 1
        rel = f"{track}/seg_{self._seg_index[track]:04d}.flac"
        sf.write(
            self.session_dir / rel, data, self.samplerate,
            format="FLAC", subtype="PCM_16",
        )
        seg = Segment(
            track=track,
            index=self._seg_index[track],
            file=rel,
            start_ms=int(self._written[track] / self.samplerate * 1000),
            duration_ms=int(len(data) / self.samplerate * 1000),
            samples=len(data),
        )
        self._written[track] += len(data)
        self.manifest.add_segment(seg)
        self.manifest.save(self.manifest_path)
        if self.on_segment:
            self.on_segment(seg)
