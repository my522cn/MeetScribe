"""用真实会话音频调试：特征提取为何失败、精修为何没跑。"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

d = "data/sessions/20261021_001629_manual"
recs = [json.loads(l) for l in open(d + "/transcript.jsonl", encoding="utf-8")]
print("记录字段:", sorted(recs[0].keys()))

segs = sorted(glob.glob(d + "/mic/*.flac"))
print("mic 分段:", [Path(s).name for s in segs])
audio, sr = sf.read(segs[0], dtype="float32", always_2d=True)
audio = audio.mean(axis=1)
print("sr=", sr, "len=", len(audio),
      "rms=", round(float(np.sqrt(np.mean(audio**2))), 5),
      "peak=", round(float(np.max(np.abs(audio))), 4))

from meetscribe.asr.diarize import _autocorr_f0, sentence_features

w = audio[int(5 * 16000):int(15 * 16000)]
feat = sentence_features(w)
print("真实语音 10s 窗口特征:", feat)

FRAME, HOP = 400, 160
f0s, nof0, silent = [], 0, 0
for i in range(0, len(w) - FRAME, HOP):
    seg = w[i:i + FRAME]
    e = float(np.mean(seg * seg))
    if e < 1e-6:
        silent += 1
        continue
    f0 = _autocorr_f0(seg, 16000)
    if f0:
        f0s.append(f0)
    else:
        nof0 += 1
print(f"非静音帧={len(f0s) + nof0}, 有F0帧={len(f0s)}, 无F0帧={nof0}, 静音帧={silent}")
if f0s:
    print("F0 中位数=", round(float(np.median(f0s)), 1), "Hz")
