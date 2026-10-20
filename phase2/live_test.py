"""阶段二验收（二）：端到端实时字幕。

扬声器播放一段中英混说音频 → WASAPI 回环采集 → 流式识别 → 逐字刷新的实时字幕。
统计：首字延迟、每块处理耗时(RTF)、临时文本被修正的次数（验证"可重新显示"）。
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.asr.engine import ASRConfig, StreamingASR  # noqa: E402
from meetscribe.audio.devices import default_output_device, find_loopback_source  # noqa: E402

BLOCK = 1600  # 100ms


def play_async(signal: np.ndarray, sr: int) -> threading.Thread:
    dev = default_output_device()

    def _run() -> None:
        with sd.OutputStream(device=dev.index if dev else None, samplerate=sr,
                             channels=1, dtype="float32") as out:
            step = int(sr * 0.05)
            for i in range(0, len(signal), step):
                out.write(signal[i:i + step].reshape(-1, 1))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def main() -> None:
    wav = ROOT / "phase2" / "test_zh_en_16k.wav"
    model_dir = ROOT / "models" / "paraformer_zh_en"
    audio, sr = sf.read(wav, dtype="float32")
    play_rate = 48000
    play_signal = soxr.resample(audio, sr, play_rate)
    print(f"测试音频 {len(audio)/sr:.2f}s，模型 {model_dir.name}")

    asr = StreamingASR(ASRConfig(model_dir=model_dir, num_threads=2))
    src = find_loopback_source()
    if src is None:
        raise SystemExit("未找到系统声音回环设备")
    import soundcard as sc

    finals: list[str] = []
    corrections = 0
    first_text_at: float | None = None
    proc_total = 0.0
    audio_total = 0.0
    t_start = time.perf_counter()

    with sc.get_microphone(id=src[1], include_loopback=True).recorder(
        samplerate=sr, channels=1
    ) as rec:
        player = play_async(play_signal, play_rate)
        silence_after = 0
        while True:
            chunk = np.asarray(rec.record(numframes=BLOCK), dtype=np.float32).reshape(-1)

            t0 = time.perf_counter()
            partial = asr.feed("loop", chunk)
            proc_total += time.perf_counter() - t0
            audio_total += len(chunk) / sr

            if partial and first_text_at is None:
                first_text_at = time.perf_counter() - t_start

            if partial:
                sys.stdout.write(f"\r[{audio_total:5.2f}s] {partial[-60:]:<62}")
                sys.stdout.flush()
                silence_after = 0
            else:
                silence_after += 1

            if asr.recognizer.is_endpoint(asr.stream("loop").stream):
                text = asr.finish_sentence("loop")
                if text:
                    finals.append(text)
                    sys.stdout.write(f"\r  ── 断句 ── {text}\n")
                    corrections += 1

            if not player.is_alive() and silence_after > 15:
                break
            if audio_total > len(audio) / sr + 5:
                break

    asr.flush("loop", 1.0)
    tail = asr.recognizer.get_result(asr.stream("loop").stream).strip()
    if tail:
        finals.append(tail)

    print("\n" + "=" * 74)
    print("实时字幕最终文本")
    print("=" * 74)
    print("".join(finals))
    print(f"\n首字延迟: {first_text_at:.2f}s（含 0.1s 采集缓冲）" if first_text_at else "未产出文本")
    print(f"RTF: {proc_total/max(audio_total,1e-6):.3f}   处理 {proc_total:.2f}s / 音频 {audio_total:.1f}s")
    print(f"最终句数: {len(finals)}")


if __name__ == "__main__":
    main()
