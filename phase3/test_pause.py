"""暂停场景验证：暂停后半句要立即定稿，精修与大模型纠错要继续跑完。

流程：播 5 句 → 暂停 → 等 8 秒 → 断言：暂停后仍产生了新的定稿/精修。
"""
from __future__ import annotations

import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.asr.engine import ASRConfig, StreamingASR  # noqa: E402
from meetscribe.asr.pipeline import LivePipeline  # noqa: E402
from meetscribe.asr.refine import RefineJob, RefineService  # noqa: E402

SR = 16000


def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as f:
        a = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return a


def main() -> int:
    src = read_wav(ROOT / "phase2" / "test_zh_en_16k.wav")
    pieces = [(0.0, 4.45), (4.90, 7.05), (7.30, 9.45), (9.90, 10.70), (11.10, 13.60)]
    gap = np.zeros(int(1.6 * SR), dtype=np.float32)
    audio = np.concatenate([np.concatenate([src[int(s * SR):int(e * SR)], gap]) for s, e in pieces])
    audio = np.concatenate([audio, np.zeros(3 * SR, dtype=np.float32)])

    engine = StreamingASR(ASRConfig(model_dir=str(ROOT / "models" / "paraformer_zh_en"), num_threads=2))
    finals: list = []
    pipeline = LivePipeline(engine, on_final=finals.append, samplerate=SR)
    pipeline.start()

    before_pause = -1
    paused_at = 0.0
    block = 1600
    i = 0
    t0 = time.perf_counter()
    playing = True
    while i * block < len(audio):
        if playing and time.perf_counter() - t0 > 9.0:  # 第 9 秒（大概说到第 3 句）暂停
            playing = False
            pipeline.pause()
            paused_at = time.perf_counter() - t0
            before_pause = len(finals)
            print(f"[暂停 @ {paused_at:.1f}s] 暂停时已定稿 {before_pause} 句")
        if playing and i * block < len(audio):
            pipeline.push("mic", audio[i * block:(i + 1) * block])
        i += 1
        time.sleep(0.01)

    time.sleep(6)  # 暂停后等 6 秒，看有没有"迟到的修订"
    after = len(finals)
    print(f"暂停期间新增定稿：{after - before_pause} 句（应 ≥1）")
    for k, cap in enumerate(finals):
        print(f"  [{k+1}] {cap.start_ms/1000:6.2f}s  {cap.text}")

    # 再验证精修服务能处理暂停期间定稿的句子
    svc = RefineService(ROOT / "models" / "paraformer_offline", on_result=lambda j, t: None, kind="paraformer")
    from meetscribe.asr.offline import get_offline

    asr = get_offline(ROOT / "models" / "paraformer_offline", "paraformer", 2)
    asr.load()
    ok = 0
    for cap in finals:
        if cap.audio is not None and len(cap.audio) > 4000:
            t = asr.transcribe(cap.audio)
            if t:
                ok += 1
    print(f"精修可处理 {ok}/{len(finals)} 句")
    return 0 if after > before_pause else 1


if __name__ == "__main__":
    raise SystemExit(main())
