"""离线识别器：进程内单例，供"句末精修"和"配置页测试识别"共用同一份权重。

两种模型：
- paraformer（默认）：sherpa-onnx-paraformer-zh-2023-09-14，中文+英文混说公认很稳
- sense_voice：多语种，但对英文术语/中文专名反而更易错，仅作备选

实测同段中英混说音频：paraformer CER 0.0%、sense_voice 3.7%，所以默认 paraformer。
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import numpy as np

_TAG_RE = re.compile(r"<\|[^|]*\|>")


def trim_silence(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """裁掉首尾静音并做峰值归一化——离线模型对音量不敏感，但裁干净能少些幻觉。"""
    if audio.size < sr // 4:
        return audio
    win = max(1, int(sr * 0.02))  # 20ms 窗
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
    end = min(audio.size, int((loud[-1] + 1) * win + sr * 0.35))
    out = audio[start:end]
    if out.size < sr // 4:
        return audio
    m = float(np.max(np.abs(out)))
    if m > 1e-3:
        out = out * (0.85 / m)
    return out.astype(np.float32, copy=False)


class OfflineASR:
    def __init__(self, model_dir: Path, kind: str = "paraformer", num_threads: int = 2) -> None:
        self.model_dir = Path(model_dir)
        self.kind = kind
        self.num_threads = num_threads
        self._lock = threading.Lock()
        self._rec = None
        self.last_rtf = 0.0
        self.load_seconds = 0.0

    @property
    def available(self) -> bool:
        return (self.model_dir / "model.int8.onnx").exists() and (self.model_dir / "tokens.txt").exists()

    def load(self) -> None:
        if self._rec is not None:
            return
        import sherpa_onnx

        t0 = time.perf_counter()
        model = str(self.model_dir / "model.int8.onnx")
        tokens = str(self.model_dir / "tokens.txt")
        if self.kind == "sense_voice":
            rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model, tokens=tokens, num_threads=self.num_threads,
                use_itn=True, language="auto",
            )
        else:
            rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=model, tokens=tokens, num_threads=self.num_threads,
                sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
            )
        self._rec = rec
        self.load_seconds = time.perf_counter() - t0

    def warmup(self) -> None:
        try:
            self.transcribe(np.zeros(16000, dtype=np.float32))
        except Exception:  # noqa: BLE001
            pass

    def transcribe(self, audio: np.ndarray, trim: bool = True) -> str:
        if self._rec is None:
            self.load()
        a = trim_silence(audio) if trim else audio
        if a.size == 0:
            return ""
        t0 = time.perf_counter()
        with self._lock:
            stream = self._rec.create_stream()
            stream.accept_waveform(16000, a.astype(np.float32, copy=False))
            self._rec.decode_stream(stream)
            text = stream.result.text
        dur = a.size / 16000
        if dur > 0:
            self.last_rtf = (time.perf_counter() - t0) / dur
        return _TAG_RE.sub("", text).strip()


_CACHE: dict[tuple[str, str, int], OfflineASR] = {}
_CACHE_LOCK = threading.Lock()


def get_offline(model_dir: Path, kind: str = "paraformer", num_threads: int = 2) -> OfflineASR:
    key = (str(Path(model_dir).resolve()), kind, num_threads)
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = OfflineASR(model_dir, kind, num_threads)
        return _CACHE[key]


def resolve_model_dir(models_root: Path, prefer: str = "auto") -> tuple[Path, str]:
    """按配置挑精修模型目录，返回 (目录, kind)。prefer=off 时返回 paraformer 目录但调用方不该用。"""
    order = [("paraformer_offline", "paraformer"), ("sense_voice", "sense_voice")]
    if prefer == "sense_voice":
        order.reverse()
    for folder, kind in order:
        d = models_root / folder
        if (d / "model.int8.onnx").exists() and (d / "tokens.txt").exists():
            return d, kind
    return models_root / "paraformer_offline", "paraformer"
