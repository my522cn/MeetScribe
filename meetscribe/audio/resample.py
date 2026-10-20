"""流式重采样：把设备原生采样率（通常 48kHz）实时转成 ASR 需要的 16kHz。"""
from __future__ import annotations

import numpy as np
import soxr


class StreamingResampler:
    def __init__(self, in_rate: int, out_rate: int = 16000, channels: int = 1) -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.channels = channels
        self._stream = soxr.ResampleStream(in_rate, out_rate, channels, dtype="float32")

    @property
    def needed(self) -> bool:
        return self.in_rate != self.out_rate

    def process(self, block: np.ndarray) -> np.ndarray:
        if not self.needed:
            return block.astype(np.float32, copy=False)
        out = self._stream.resample_chunk(block, last=False)
        return np.asarray(out, dtype=np.float32)

    def flush(self) -> np.ndarray:
        if not self.needed:
            return np.zeros(0, dtype=np.float32)
        out = self._stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
        return np.asarray(out, dtype=np.float32)
