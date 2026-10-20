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
