"""离线二次识别（精修）。

流式模型为了低延迟牺牲了准确率；每句话定稿后，把这一句的原始音频交给离线大模型
重识别一遍，用更准的结果替换字幕。全程在后台线程，不阻塞录制与字幕。

实测（同一段中英混说音频）：
    流式 Paraformer  CER 3.7%   RTF 0.17
    离线 Paraformer  CER 0.0%   RTF 0.13   ← 默认精修模型
    SenseVoice       CER 3.7%   RTF 0.15
离线 Paraformer 对中文专名与英文术语都更稳，因此作为默认。
"""
from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

_TAG_RE = re.compile(r"<\|[^|]*\|>")
# 句尾静音裁剪：低于这个能量视为静音，最多保留 0.35s 尾巴
_TRIM_KEEP = 0.35
_TRIM_WIN = 0.02  # 20ms 窗


@dataclass
class RefineJob:
    track: str
    audio: np.ndarray
    raw_text: str
    start_ms: int
    end_ms: int


def trim_silence(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """裁掉首尾静音并做峰值归一化——离线模型对音量不敏感，但裁干净能少些幻觉。"""
    if audio.size < sr // 4:
        return audio
    win = max(1, int(sr * _TRIM_WIN))
    n = audio.size // win
    if n < 2:
        return audio
    frames = audio[: n * win].reshape(n, win)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    peak = float(rms.max())
    if peak <= 0:
        return audio
    loud = np.nonzero(rms > peak * 0.08)[0]
    if loud.size == 0:
        return audio
    start = max(0, int(loud[0] * win - sr * 0.12))
    end = min(audio.size, int((loud[-1] + 1) * win + sr * _TRIM_KEEP))
    out = audio[start:end]
    if out.size < sr // 4:
        return audio
    m = float(np.max(np.abs(out)))
    if m > 1e-3:
        out = out * (0.85 / m)
    return out.astype(np.float32, copy=False)


class RefineService:
    """后台离线重识别。kind: "paraformer"（默认，最准）或 "sense_voice"。"""

    def __init__(
        self,
        model_dir: Path,
        on_result: Callable[[RefineJob, str], None],
        num_threads: int = 2,
        kind: str = "paraformer",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.on_result = on_result
        self.num_threads = num_threads
        self.kind = kind
        self._queue: queue.Queue[RefineJob | None] = queue.Queue(maxsize=200)
        self._thread: threading.Thread | None = None
        self._running = False
        self._recognizer = None
        self.last_rtf: float = 0.0
        self.done = 0
        self.failed = 0

    @property
    def available(self) -> bool:
        return (self.model_dir / "model.int8.onnx").exists() and (self.model_dir / "tokens.txt").exists()

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if self._running or not self.available:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="refine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """等在途任务跑完再退出（最多 8s），避免丢最后一句的精修。"""
        self._running = False
        deadline = time.time() + 8
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.05)
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=4)

    def submit(self, job: RefineJob) -> None:
        if not self._running:
            return
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            pass  # 积压过多就放弃精修，优先保证不拖慢系统

    # ---------- 内部 ----------
    def _build(self):
        import sherpa_onnx

        model = str(self.model_dir / "model.int8.onnx")
        tokens = str(self.model_dir / "tokens.txt")
        if self.kind == "sense_voice":
            return sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model, tokens=tokens, num_threads=self.num_threads,
                use_itn=True, language="auto",
            )
        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=model, tokens=tokens, num_threads=self.num_threads,
            sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
        )

    def _loop(self) -> None:
        try:
            self._recognizer = self._build()
        except Exception:  # noqa: BLE001
            self._running = False
            return
        # 预热：第一次推理要把 onnx 权重读进内存，否则第一句延迟很高
        try:
            self._decode(np.zeros(16000, dtype=np.float32))
        except Exception:  # noqa: BLE001
            pass

        while self._running:
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                text = self._decode(job.audio)
            except Exception:  # noqa: BLE001
                self.failed += 1
                continue
            if text and text.strip():
                self.done += 1
                try:
                    self.on_result(job, text.strip())
                except Exception:  # noqa: BLE001
                    pass

    def _decode(self, audio: np.ndarray) -> str:
        rec = self._recognizer
        audio = trim_silence(audio)
        t0 = time.perf_counter()
        stream = rec.create_stream()
        stream.accept_waveform(16000, audio.astype(np.float32, copy=False))
        rec.decode_stream(stream)
        dur = len(audio) / 16000
        if dur > 0:
            self.last_rtf = (time.perf_counter() - t0) / dur
        text = stream.result.text
        return _TAG_RE.sub("", text).strip()
