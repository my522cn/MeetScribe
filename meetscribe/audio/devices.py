"""音频设备发现：优先选择 WASAPI 端点，并定位系统声音回环源。"""
from __future__ import annotations

from dataclasses import dataclass

import sounddevice as sd


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    hostapi: str
    max_input: int
    max_output: int
    default_samplerate: float

    @property
    def is_wasapi(self) -> bool:
        return "WASAPI" in self.hostapi


def list_devices() -> list[DeviceInfo]:
    hostapis = sd.query_hostapis()
    items: list[DeviceInfo] = []
    for idx, dev in enumerate(sd.query_devices()):
        api = hostapis[dev["hostapi"]]["name"]
        items.append(
            DeviceInfo(
                index=idx,
                name=dev["name"],
                hostapi=api,
                max_input=dev["max_input_channels"],
                max_output=dev["max_output_channels"],
                default_samplerate=dev["default_samplerate"],
            )
        )
    return items


def _pick(devices: list[DeviceInfo], *, want_input: bool) -> DeviceInfo | None:
    candidates = [
        d for d in devices
        if d.is_wasapi and (d.max_input > 0 if want_input else d.max_output > 0)
    ]
    if not candidates:
        return None
    default_idx = sd.default.device[0 if want_input else 1]
    for d in candidates:
        if d.index == default_idx:
            return d
    return candidates[0]


def default_input_device() -> DeviceInfo | None:
    return _pick(list_devices(), want_input=True)


def default_output_device() -> DeviceInfo | None:
    return _pick(list_devices(), want_input=False)


def find_loopback_source() -> tuple[str, str] | None:
    """返回 (扬声器名称, 设备 id)，用于 WASAPI 回环采集系统声音。"""
    try:
        import soundcard as sc
    except Exception:  # noqa: BLE001
        return None

    speaker = sc.default_speaker()
    if speaker is None:
        return None
    return speaker.name, str(speaker.id)


def list_input_devices() -> list[DeviceInfo]:
    """可作为"麦克风"的设备：WASAPI 且有输入通道。"""
    return [d for d in list_devices() if d.is_wasapi and d.max_input > 0]


def list_speakers() -> list[tuple[str, str]]:
    """可作为回环源的扬声器（soundcard 视角，name/id）。"""
    try:
        import soundcard as sc
    except Exception:  # noqa: BLE001
        return []
    out: list[tuple[str, str]] = []
    try:
        for sp in sc.all_speakers():
            out.append((sp.name, str(sp.id)))
    except Exception:  # noqa: BLE001
        pass
    return out


def resolve_input_index(mic_name: str, mic_index: int | None) -> int | None:
    """设备索引会随插拔变化，先按名字找，找不到再用索引。"""
    devs = list_input_devices()
    if mic_name:
        for d in devs:
            if d.name == mic_name:
                return d.index
    if mic_index is not None and any(d.index == mic_index for d in devs):
        return mic_index
    return None


def open_mic_stream(dev_index: int | None, target_rate: int = 16000):
    """打开麦克风输入流，返回 (stream, 采样率, 通道数)。

    WASAPI 共享模式常拒绝 16kHz；且阵列麦用单声道打开会拿到错误/静音通道，
    因此优先按设备原生通道数打开，再在采集后自行降混为单声道。
    """
    info = sd.query_devices(dev_index) if dev_index is not None else sd.query_devices(kind="input")
    native = int(info["default_samplerate"])
    max_ch = int(info["max_input_channels"])
    channel_opts: list[int] = []
    for ch in (max_ch, 2, 1):
        if 0 < ch <= max_ch and ch not in channel_opts:
            channel_opts.append(ch)
    rate_opts: list[int] = []
    for r in (native, 48000, 44100, target_rate):
        if r > 0 and r not in rate_opts:
            rate_opts.append(r)

    last_err: Exception | None = None
    for rate in rate_opts:
        for ch in channel_opts:
            try:
                sd.check_input_settings(device=dev_index, samplerate=rate, channels=ch, dtype="float32")
                stream = sd.InputStream(
                    device=dev_index, samplerate=rate, channels=ch,
                    dtype="float32", blocksize=int(rate * 0.1),
                )
                return stream, rate, ch
            except Exception as exc:  # noqa: BLE001
                last_err = exc
    raise RuntimeError(f"无法打开麦克风: {last_err}")


def describe() -> str:
    lines = ["音频设备清单（仅 WASAPI）:"]
    for d in list_devices():
        if not d.is_wasapi:
            continue
        kind = []
        if d.max_input:
            kind.append(f"IN x{d.max_input}")
        if d.max_output:
            kind.append(f"OUT x{d.max_output}")
        lines.append(f"  [{d.index}] {d.name}  {d.default_samplerate:.0f}Hz  {' '.join(kind)}")
    loop = find_loopback_source()
    lines.append(f"回环源: {loop[0] if loop else '未找到'}")
    return "\n".join(lines)
