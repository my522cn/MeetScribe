"""设备自检：录一小段音频回来做电平统计与试识别。

配置页的"测试麦克风 / 测试系统声音"用它——只采 5 秒，不落盘、不影响正在进行的录音。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .agc import AGC
from .devices import open_mic_stream
from .resample import StreamingResampler

SR = 16000


@dataclass
class ProbeResult:
    audio: np.ndarray
    seconds: float
    peak: float
    rms: float
    device: str

    @property
    def empty(self) -> bool:
        # 低于这个能量基本是环境底噪：识别只会得到幻觉句子，不如直接提示没声音
        return self.rms < 0.004

    @property
    def verdict(self) -> str:
        if self.empty:
            return "几乎没有声音：检查设备是否静音、是否选错通道，或系统隐私设置是否禁止了麦克风"
        if self.peak > 0.98:
            return "音量偏大，已经削顶：把系统或麦克风增益调低一些，否则识别会掉字"
        if self.rms < 0.012:
            return "音量偏小：建议提高输入增益，太小的声音识别准确率会明显下降"
        return "音量正常"


def _com_init() -> None:
    try:
        import comtypes

        comtypes.CoInitialize()
    except Exception:  # noqa: BLE001
        pass


def record_mic(
    dev_index: int | None, seconds: float = 5.0,
    on_progress: Callable[[float, float], None] | None = None,
) -> ProbeResult:
    """录一段麦克风音频（16k 单声道）。on_progress(已录秒数, 当前RMS)。"""
    stream, native, channels = open_mic_stream(dev_index, SR)
    rs = StreamingResampler(native, SR)
    agc = AGC()  # 与正式录音一致的增益，测试识别的结果才有代表性
    chunks: list[np.ndarray] = []
    peak = 0.0
    block = int(native * 0.1)
    t0 = time.perf_counter()
    with stream:
        while time.perf_counter() - t0 < seconds:
            data, _ = stream.read(block)
            mono = data.mean(axis=1) if data.ndim > 1 else data
            mono = np.asarray(mono, dtype=np.float32).reshape(-1)
            out = agc.process(rs.process(mono.copy()))
            peak = max(peak, float(np.max(np.abs(out))) if out.size else 0.0)
            if out.size:
                chunks.append(out)
            if on_progress:
                rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
                on_progress(time.perf_counter() - t0, rms)
    tail = rs.flush()
    if tail is not None and tail.size:
        chunks.append(tail)
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12)) if audio.size else 0.0
    name = "默认麦克风"
    try:
        import sounddevice as sd

        name = sd.query_devices(dev_index)["name"] if dev_index is not None else sd.query_devices(kind="input")["name"]
    except Exception:  # noqa: BLE001
        pass
    return ProbeResult(audio, len(audio) / SR, peak, rms, str(name))


def record_loopback(
    speaker_name: str | None, seconds: float = 5.0,
    on_progress: Callable[[float, float], None] | None = None,
) -> ProbeResult:
    """录一段系统声音（回环）。speaker_name 为空则用默认扬声器。"""
    _com_init()
    import soundcard as sc

    speaker = None
    if speaker_name:
        for sp in sc.all_speakers():
            if sp.name == speaker_name:
                speaker = sp
                break
    if speaker is None:
        speaker = sc.default_speaker()
    if speaker is None:
        raise RuntimeError("未找到可用的扬声器（回环）设备")

    chunks: list[np.ndarray] = []
    peak = 0.0
    t0 = time.perf_counter()
    with sc.get_microphone(id=str(speaker.id), include_loopback=True).recorder(
        samplerate=SR, channels=1
    ) as rec:
        while time.perf_counter() - t0 < seconds:
            block = np.asarray(rec.record(numframes=1600), dtype=np.float32).reshape(-1)
            peak = max(peak, float(np.max(np.abs(block))) if block.size else 0.0)
            chunks.append(block)
            if on_progress:
                rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2) + 1e-12))
                on_progress(time.perf_counter() - t0, rms)
    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12)) if audio.size else 0.0
    return ProbeResult(audio, len(audio) / SR, peak, rms, str(speaker.name))
