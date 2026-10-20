"""阶段一验收脚本：双轨分段落盘 + 回环真实信号验证 + 资源占用测量。

做法：一边向默认扬声器播放测试音（440Hz 6s → 880Hz 6s → 静音 3s），
一边用麦克风轨 + 系统声音回环轨同时录制 15 秒，每 6 秒切一个 FLAC 分段，
最后核对：回环轨是否真的抓到声音、两轨样本漂移、磁盘占用、CPU 占用。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from meetscribe.audio.devices import default_output_device  # noqa: E402
from meetscribe.audio.recorder import DualTrackRecorder  # noqa: E402

OUT_RATE = 48000
DURATION = 15
SEGMENT_SECONDS = 6
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "sessions"


def build_tone() -> np.ndarray:
    total = OUT_RATE * DURATION
    t = np.arange(total) / OUT_RATE
    sig = np.zeros(total, dtype=np.float32)
    head = OUT_RATE * 6
    mid = OUT_RATE * 12
    sig[:head] = 0.25 * np.sin(2 * np.pi * 440 * t[:head])
    sig[head:mid] = 0.20 * np.sin(2 * np.pi * 880 * t[head:mid])
    return sig


def play(signal: np.ndarray) -> None:
    dev = default_output_device()
    step = int(OUT_RATE * 0.05)
    with sd.OutputStream(
        device=dev.index if dev else None,
        samplerate=OUT_RATE, channels=1, dtype="float32",
    ) as out:
        for i in range(0, len(signal), step):
            out.write(signal[i:i + step].reshape(-1, 1))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def main() -> None:
    proc = psutil.Process()
    rec = DualTrackRecorder(DATA_ROOT, segment_seconds=SEGMENT_SECONDS, title="阶段一采集验证")
    print(f"会话目录: {rec.session_dir}")
    proc.cpu_percent(interval=None)

    rec.start()
    player = threading.Thread(target=play, args=(build_tone(),), daemon=True)
    player.start()

    cpu_samples: list[float] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < DURATION:
        cpu_samples.append(proc.cpu_percent(interval=None))
        time.sleep(1.0)
    manifest = rec.stop()
    player.join(timeout=5)

    print("\n" + "=" * 72)
    print("采集结果")
    print("=" * 72)
    total_bytes = 0
    for seg in manifest.segments:
        path = rec.session_dir / seg.file
        size = path.stat().st_size
        total_bytes += size
        data, sr = sf.read(path, dtype="float32")
        print(
            f"{seg.track:<5} #{seg.index:<2} {seg.start_ms/1000:>6.1f}s ~ {seg.end_ms/1000:>6.1f}s  "
            f"{seg.samples:>7} samples  {size/1024:>7.1f} KB  RMS={rms(data):.4f}  峰值={abs(data).max():.3f}"
        )

    print("\n" + "=" * 72)
    print("校验")
    print("=" * 72)
    loop_rms = 0.0
    for seg in manifest.segments:
        if seg.track == "loop":
            data, _ = sf.read(rec.session_dir / seg.file, dtype="float32")
            loop_rms = max(loop_rms, rms(data))
    print(f"回环轨最大 RMS: {loop_rms:.4f}  → {'抓到了系统声音' if loop_rms > 0.01 else '未抓到有效声音'}")

    for track in ("mic", "loop"):
        segs = [s for s in manifest.segments if s.track == track]
        total = sum(s.samples for s in segs)
        span = max((s.end_ms for s in segs), default=0) / 1000
        if not segs:
            print(f"{track:<5} 无数据")
            continue
        drift = total / 16000 - span
        print(f"{track:<5} 总样本 {total:>8}  时间跨度 {span:>5.2f}s  漂移 {drift:+.3f}s")

    avg_cpu = sum(cpu_samples) / max(len(cpu_samples), 1)
    cores = psutil.cpu_count(logical=True) or 1
    print(f"\n进程 CPU: 平均 {avg_cpu:.1f}% (单核口径)，折合整机 {avg_cpu/cores:.1f}%")
    print(f"系统 CPU: 当前 {psutil.cpu_percent(interval=0.5):.1f}%   逻辑核数 {cores}")
    print(f"磁盘: 15 秒双轨共 {total_bytes/1024/1024:.2f} MB → 约 {total_bytes/1024/1024/DURATION*3600:.1f} MB/小时")

    # 编码格式体积对比
    print("\n" + "=" * 72)
    print("存储格式体积对比（同一段回环音频）")
    print("=" * 72)
    loop_seg = next((s for s in manifest.segments if s.track == "loop"), None)
    if loop_seg:
        data, sr = sf.read(rec.session_dir / loop_seg.file, dtype="float32")
        secs = len(data) / sr
        flac_size = (rec.session_dir / loop_seg.file).stat().st_size
        print(f"FLAC  {flac_size/1024:>8.1f} KB / {secs:.0f}s  → {flac_size*3600/secs/1024/1024:.1f} MB/小时")
        wav_path = Path(rec.session_dir) / "compare.wav"
        sf.write(wav_path, data, sr, format="WAV", subtype="PCM_16")
        print(f"WAV   {wav_path.stat().st_size/1024:>8.1f} KB  → {wav_path.stat().st_size*3600/secs/1024/1024:.1f} MB/小时")
        wav_path.unlink(missing_ok=True)
        ogg_path = Path(rec.session_dir) / "compare.ogg"
        try:
            sf.write(ogg_path, data, sr, format="OGG", subtype="OPUS")
            print(f"Opus  {ogg_path.stat().st_size/1024:>8.1f} KB  → {ogg_path.stat().st_size*3600/secs/1024/1024:.1f} MB/小时")
        except Exception as exc:  # noqa: BLE001
            print(f"Opus 不可用: {exc}")
        finally:
            ogg_path.unlink(missing_ok=True)

    print(f"\n清单文件: {rec.manifest_path}")
    print(f"分段数: {len(manifest.segments)}")


if __name__ == "__main__":
    main()
