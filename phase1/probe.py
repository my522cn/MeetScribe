"""阶段一 · 音频设备探测
列出所有音频设备，定位 WASAPI 回环（系统声音）源，并测试麦克风与回环是否可打开。
用法: python probe.py
"""
from __future__ import annotations

import sys
import traceback

import sounddevice as sd


def list_sounddevice() -> None:
    print("=" * 70)
    print("sounddevice (PortAudio) 设备列表")
    print("=" * 70)
    hostapis = sd.query_hostapis()
    default_in, default_out = sd.default.device
    print(f"默认输入设备索引: {default_in}  默认输出设备索引: {default_out}")
    for idx, dev in enumerate(sd.query_devices()):
        api = hostapis[dev["hostapi"]]["name"]
        if "WASAPI" not in api and "MME" not in api:
            continue
        flags = []
        if dev["max_input_channels"] > 0:
            flags.append(f"IN x{dev['max_input_channels']}")
        if dev["max_output_channels"] > 0:
            flags.append(f"OUT x{dev['max_output_channels']}")
        print(
            f"[{idx:>3}] {api:<18} {dev['name'][:52]:<52} "
            f"{dev['default_samplerate']:.0f}Hz  {' '.join(flags)}"
        )


def probe_loopback() -> bool:
    print()
    print("=" * 70)
    print("soundcard WASAPI 回环探测")
    print("=" * 70)
    try:
        import soundcard as sc
    except Exception as exc:  # noqa: BLE001
        print(f"soundcard 不可用: {exc}")
        return False

    speakers = sc.all_speakers()
    mics = sc.all_microphones(include_loopback=False)
    print("扬声器/输出设备:")
    for spk in speakers:
        print(f"  - {spk.name}  (id={spk.id})")
    print("麦克风/输入设备:")
    for mic in mics:
        print(f"  - {mic.name}  (id={mic.id})")

    default_spk = sc.default_speaker()
    if default_spk is None:
        print("!! 未找到默认扬声器")
        return False
    print(f"\n默认扬声器: {default_spk.name}")
    try:
        loop = sc.get_microphone(id=str(default_spk.name), include_loopback=True)
        print(f"回环采集设备: {loop.name}")
        with loop.recorder(samplerate=16000, channels=2) as rec:
            data = rec.record(numframes=16000)
        peak = float(abs(data).max()) if len(data) else 0.0
        print(f"回环采集 1 秒成功，帧数={len(data)}，峰值={peak:.4f}")
        return True
    except Exception:  # noqa: BLE001
        print("!! 回环采集失败:")
        traceback.print_exc()
        return False


def probe_microphone() -> bool:
    print()
    print("=" * 70)
    print("麦克风采集探测")
    print("=" * 70)
    try:
        import numpy as np

        dev = sd.query_devices(kind="input")
        print(f"默认输入: {dev['name']}  通道={dev['max_input_channels']}")
        with sd.InputStream(samplerate=16000, channels=1, dtype="float32") as stream:
            frames = []
            for _ in range(10):
                block, overflowed = stream.read(1600)
                frames.append(block.copy())
            data = np.concatenate(frames)
        print(f"麦克风采集成功，帧数={len(data)}，峰值={float(abs(data).max()):.4f}")
        return True
    except Exception:  # noqa: BLE001
        print("!! 麦克风采集失败（可能无麦克风或权限被拒）:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if sys.platform != "win32":
        print("本脚本仅针对 Windows")
    list_sounddevice()
    loop_ok = probe_loopback()
    mic_ok = probe_microphone()
    print()
    print("=" * 70)
    print(f"探测结论: 回环={'可用' if loop_ok else '不可用'}  麦克风={'可用' if mic_ok else '不可用'}")
    print("=" * 70)
