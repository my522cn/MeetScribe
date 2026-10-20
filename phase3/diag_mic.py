"""麦克风轨诊断：录制同一段测试语音（扬声器外放），逐通道分析并分别送识别。

输出：
1. 各通道 RMS / 通道间相关性（判断多麦阵列是否相位抵消）
2. 单声道 vs 多通道平均 的识别结果
3. 与系统声回环轨的识别结果对比（回环轨是"标准答案"参照）
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import sounddevice as sd
import soundcard as sc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.asr.engine import ASRConfig, StreamingASR  # noqa: E402
from meetscribe.audio.devices import default_input_device, default_output_device, find_loopback_source  # noqa: E402

DURATION = 14.0
SR = 16000


def play_async(path: Path) -> threading.Thread:
    audio, sr = sf.read(path, dtype="float32")
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    if sr != 48000:
        mono = soxr.resample(mono, sr, 48000)
        sr = 48000
    dev = default_output_device()

    def run() -> None:
        with sd.OutputStream(device=dev.index if dev else None, samplerate=sr, channels=1, dtype="float32") as out:
            step = int(sr * 0.05)
            for i in range(0, len(mono), step):
                out.write(mono[i:i + step].reshape(-1, 1))

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def record_mic(seconds: float, channels: int) -> tuple[np.ndarray, int]:
    dev = default_input_device()
    info = sd.query_devices(dev.index)
    rate = int(info["default_samplerate"])
    frames = int(rate * seconds)
    with sd.InputStream(device=dev.index, samplerate=rate, channels=channels,
                        dtype="float32", blocksize=int(rate * 0.1)) as stream:
        chunks = []
        got = 0
        while got < frames:
            data, _ = stream.read(int(rate * 0.1))
            chunks.append(data.copy())
            got += data.shape[0]
    return np.concatenate(chunks), rate


def record_loop(seconds: float) -> np.ndarray:
    import comtypes

    comtypes.CoInitialize()
    src = find_loopback_source()
    out = []
    with sc.get_microphone(id=src[1], include_loopback=True).recorder(samplerate=SR, channels=1) as rec:
        for _ in range(int(seconds * 10)):
            out.append(np.asarray(rec.record(numframes=1600), dtype=np.float32).reshape(-1))
    return np.concatenate(out)


def asr_text(audio: np.ndarray, sr: int, tag: str) -> str:
    if sr != SR:
        audio = soxr.resample(audio, sr, SR)
    engine = StreamingASR(ASRConfig(model_dir=ROOT / "models" / "paraformer_zh_en"))
    block = 1600
    for i in range(0, len(audio), block):
        engine.feed(tag, audio[i:i + block])
    engine.flush(tag, 1.0)
    return engine.recognizer.get_result(engine.stream(tag).stream)


def main() -> None:
    wav = ROOT / "phase2" / "test_zh_en_16k.wav"
    dev = default_input_device()
    info = sd.query_devices(dev.index)
    print(f"输入设备: {info['name']}  通道={info['max_input_channels']}  原生={info['default_samplerate']}Hz")

    # 先测驱动实际接受的通道数
    supported = []
    for ch in (1, 2, 3, 4):
        if ch > info["max_input_channels"]:
            break
        try:
            sd.check_input_settings(device=dev.index, samplerate=int(info["default_samplerate"]),
                                    channels=ch, dtype="float32")
            supported.append(ch)
        except Exception:  # noqa: BLE001
            pass
    print(f"可打开的通道数: {supported}")

    ch = min(supported) if supported else 1
    print(f"\n播放测试语音（外放），同时录制麦克风 {DURATION:.0f}s（{ch} 通道）+ 系统声…")

    results: dict[str, object] = {}

    def mic_job() -> None:
        results["mic"] = record_mic(DURATION, ch)

    def loop_job() -> None:
        results["loop"] = record_loop(DURATION)

    player = play_async(wav)
    t1 = threading.Thread(target=mic_job)
    t2 = threading.Thread(target=loop_job)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    player.join(timeout=3)

    mic, mic_rate = results["mic"]  # type: ignore[misc]
    loop = results["loop"]          # type: ignore[assignment]

    print(f"麦克风采集: {mic.shape} @ {mic_rate}Hz   回环采集: {loop.shape}")

    # 逐通道统计
    for c in range(mic.shape[1]):
        rms = float(np.sqrt((mic[:, c] ** 2).mean() + 1e-12))
        print(f"  通道{c}: RMS={rms:.4f}")
    if mic.shape[1] >= 2:
        corr = np.corrcoef(mic[: 16000 * 4].T)
        print("  通道相关性:")
        for i in range(corr.shape[0]):
            print("    " + "  ".join(f"{corr[i, j]:+.2f}" for j in range(corr.shape[1])))

    mono0 = mic[:, 0]
    mono_mean = mic.mean(axis=1)

    print("\n分别送识别：")
    for tag, audio in (("mic_ch0", mono0), ("mic_mean", mono_mean), ("loop", loop)):
        text = asr_text(audio, mic_rate if tag != "loop" else SR, tag)
        print(f"  [{tag}] {text}")

    sf.write(ROOT / "phase3" / "diag_mic_raw.wav", mono0, mic_rate)
    sf.write(ROOT / "phase3" / "diag_loop.wav", loop, SR)


if __name__ == "__main__":
    main()
