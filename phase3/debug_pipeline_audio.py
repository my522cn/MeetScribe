"""把真实会话音频喂给 LivePipeline + 真流式引擎，检查定稿句的 cap.audio。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from meetscribe.asr.engine import ASRConfig, StreamingASR
from meetscribe.asr.pipeline import LivePipeline

ROOT = Path(__file__).resolve().parents[1]
d = ROOT / "data/sessions/20261021_001629_manual"
audio, sr = sf.read(str(d / "mic/seg_0001.flac"), dtype="float32", always_2d=True)
audio = audio.mean(axis=1).astype(np.float32)
print("音频:", len(audio) / sr, "秒")

engine = StreamingASR(ASRConfig(model_dir=ROOT / "models/paraformer_zh_en", num_threads=2))

stats = {"n": 0, "none_audio": 0, "audio_lens": []}


def on_final(cap) -> None:
    stats["n"] += 1
    a = cap.audio
    if a is None:
        stats["none_audio"] += 1
        print(f"final#{stats['n']} {cap.start_ms}-{cap.end_ms} audio=None!!! text={cap.text[:20]}")
    else:
        stats["audio_lens"].append(len(a))
        print(f"final#{stats['n']} {cap.start_ms}-{cap.end_ms} audio={len(a)} 样本 text={cap.text[:20]}")


pipe = LivePipeline(engine, on_final=on_final)
pipe.start()
BLOCK = 1600
for i in range(0, len(audio), BLOCK):
    pipe.push("mic", audio[i : i + BLOCK])
import time

time.sleep(3)  # 等消费线程跑完
pipe.stop()

print("\n== 结果 ==")
print("定稿句数:", stats["n"], " audio=None:", stats["none_audio"],
      " 音频长度:", stats["audio_lens"])
