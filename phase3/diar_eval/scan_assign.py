# -*- coding: utf-8 -*-
"""阈值扫描：新投票算法（孤儿集体立簇）在两份真实数据上的表现。

数据 A：用户实测会话（20261021_195849，麦克风回采 B 站视频，25.7s，6 句）
数据 B：B 站视频语料（audio.wav 173.5s，38 句，多角色配音+背景音乐的极端样本）

每档阈值都新建 registry 顺序跑 assign_sentence，输出说话人数、逐句标签与 mixed，
并用 F0 弱真值算一致率（B 有 F0 数据）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.asr.diarize import EmbeddingSpeakerRegistry  # noqa: E402

SR = 16000
MODEL = str(ROOT / "models/campplus_zh/campplus_zh.onnx")


def load_wav(path: str) -> np.ndarray:
    wav, sr = sf.read(path)
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        n = int(len(x) * SR / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x


def run(audio, sents, thresholds, tag):
    print(f"\n===== 数据 {tag}：{len(sents)} 句 =====")
    for th in thresholds:
        reg = EmbeddingSpeakerRegistry(MODEL, threshold=th)
        out = []
        for s in sents:
            seg = audio[int(s["start_ms"] * 16):int(s["end_ms"] * 16)]
            lab, mixed = reg.assign_sentence(seg if len(seg) >= 8000 else None)
            out.append((lab, mixed))
        n_spk = len({lab for lab, _ in out})
        print(f"[th={th:.2f}] 说话人数={n_spk}  " + "  ".join(
            f"{lab[-1]}:{mx:.2f}" for lab, mx in out))


def main() -> int:
    # 数据 A：用户实测
    sess = ROOT / "release/MeetScribe/data/sessions/20261021_195849_manual"
    rows = [json.loads(l) for l in open(sess / "transcript.jsonl", encoding="utf-8")]
    for track in ("mic", "loop"):
        audio = load_wav(str(sess / track / "seg_0001.flac"))
        sents = [r for r in rows if r["track"] == track]
        run(audio, sents, [0.30, 0.34, 0.38, 0.42, 0.45], f"A-{track}")

    # 数据 B：视频语料 + F0 弱真值一致率
    audio = load_wav(str(HERE / "audio.wav"))
    sents = json.load(open(HERE / "asr_sents.json", encoding="utf-8"))
    wc = json.load(open(HERE / "window_cluster.json", encoding="utf-8"))

    def f0_band_of(t0_ms, t1_ms):
        # 用滑窗 F0 的多数派频带做弱真值（低/中/高）；starts 是采样点
        vals = [f for st, f in zip(wc["starts"], wc["f0s"])
                if f is not None and t0_ms * 16 <= st < t1_ms * 16]
        if not vals:
            return None
        m = float(np.median(vals))
        return "L" if m < 165 else ("M" if m < 260 else "H")

    print(f"\n===== 数据 B-F0一致率 =====")
    for th in [0.30, 0.34, 0.38, 0.42, 0.45]:
        reg = EmbeddingSpeakerRegistry(MODEL, threshold=th)
        out = []
        for s in sents:
            seg = audio[int(s["start_ms"] * 16):int(s["end_ms"] * 16)]
            lab, mixed = reg.assign_sentence(seg)
            out.append((lab, f0_band_of(s["start_ms"], s["end_ms"])))
        # 每个频带内多数派标签占比
        agree, tot = 0.0, 0
        for band in "LMH":
            labs = [lab for lab, b in out if b == band]
            if labs:
                mb = max(set(labs), key=labs.count)
                agree += sum(1 for l in labs if l == mb) / len(labs)
                tot += 1
        n_spk = len({lab for lab, _ in out})
        print(f"[th={th:.2f}] 说话人数={n_spk} F0频带一致率={agree / max(1, tot):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
