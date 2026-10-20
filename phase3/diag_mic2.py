"""麦克风深度诊断：双通道分别对比回环轨，定位是信号问题还是解码问题。"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.asr.engine import ASRConfig, StreamingASR  # noqa: E402
from meetscribe.audio.devices import default_input_device, default_output_device  # noqa: E402
from phase3.diag_mic import play_async, record_loop  # noqa: E402


def record_mic_ch(seconds: float, channels: int):
    dev = default_input_device()
    info = sd.query_devices(dev.index)
    rate = int(info["default_samplerate"])
    need = int(rate * seconds)
    with sd.InputStream(device=dev.index, samplerate=rate, channels=channels,
                        dtype="float32", blocksize=int(rate * 0.1)) as s:
        print(f"  流实际参数: {s.samplerate}Hz {s.channels}ch latency={s.latency}")
        chunks, got = [], 0
        while got < need:
            d, _ = s.read(int(rate * 0.1))
            chunks.append(d.copy())
            got += d.shape[0]
    return np.concatenate(chunks), rate


def asr(audio, sr, tag):
    if sr != 16000:
        audio = soxr.resample(audio, sr, 16000)
    e = StreamingASR(ASRConfig(model_dir=ROOT / "models" / "paraformer_zh_en"))
    for i in range(0, len(audio), 1600):
        e.feed(tag, audio[i:i + 1600])
    e.flush(tag, 1.0)
    return e.recognizer.get_result(e.stream(tag).stream)


def best_corr(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    n = min(len(a), len(b), 16000 * 10)
    a, b = a[:n], b[:n]
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    corr = np.correlate(a, b, mode="full") / n
    lag = int(np.argmax(np.abs(corr)) - (n - 1))
    return float(corr[np.argmax(np.abs(corr))]), lag


def main() -> None:
    wav = ROOT / "phase2" / "test_zh_en_16k.wav"
    res = {}

    def mic_job():
        res["mic"] = record_mic_ch(12, 2)

    def loop_job():
        res["loop"] = record_loop(12)

    player = play_async(wav)
    t1 = threading.Thread(target=mic_job)
    t2 = threading.Thread(target=loop_job)
    t1.start(); t2.start(); t1.join(); t2.join(); player.join(timeout=3)

    mic, rate = res["mic"]
    loop = res["loop"]
    print(f"mic {mic.shape} @{rate}   loop {loop.shape} @16000")

    for c in range(mic.shape[1]):
        ch = mic[:, c]
        rms = float(np.sqrt((ch ** 2).mean()))
        peak = float(np.abs(ch).max())
        ch16 = soxr.resample(ch, rate, 16000)
        corr, lag = best_corr(ch16, loop)
        print(f"通道{c}: RMS={rms:.4f} 峰值={peak:.3f} 与回环相关={corr:+.3f} 延迟={lag/16:.0f}ms")
        print(f"  识别: {asr(ch, rate, f'c{c}')[:70]}")

    # 拼接对齐后强制比较：把 mic 按最佳 lag 平移再识别
    ch16 = soxr.resample(mic[:, 0], rate, 16000)
    _, lag = best_corr(ch16, loop)
    if lag > 0:
        shifted = np.concatenate([np.zeros(lag, dtype=np.float32), ch16])
    else:
        shifted = ch16[-lag:]
    sf.write(ROOT / "phase3" / "mic16_shifted.wav", shifted[: len(loop)], 16000)
    sf.write(ROOT / "phase3" / "loop16.wav", loop, 16000)
    print("\n已保存 mic16_shifted.wav / loop16.wav 供对比")


if __name__ == "__main__":
    main()
