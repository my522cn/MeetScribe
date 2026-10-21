"""滑窗嵌入分析（scipy 平均联结版）：检验窗口级嵌入能否分出说话人。

F0 双峰（男 ~<150Hz / 女 ~>150Hz）作为弱真值。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402

from meetscribe.asr.diarize import _autocorr_f0  # noqa: E402

SR = 16000
MODEL = ROOT / "models" / "campplus_zh" / "campplus_zh.onnx"
WIN = SR          # 1s
HOP = SR // 2     # 0.5s


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> int:
    audio, sr = sf.read(HERE / "audio.wav", dtype="float32")

    import sherpa_onnx
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    ext = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODEL), num_threads=2))

    wins, f0s = [], []
    for start in range(0, len(audio) - WIN, HOP):
        seg = audio[start:start + WIN]
        if float(np.sqrt(np.mean(seg ** 2))) < 0.01:
            continue
        vals = [_autocorr_f0(seg[k:k + 400], SR) for k in range(0, len(seg) - 400, 160)]
        vals = [v for v in vals if v is not None]
        f0s.append(float(np.median(vals)) if vals else None)
        wins.append((start, seg))
    n = len(wins)
    print(f"有声窗口 {n}")

    embs = []
    for _, seg in wins:
        stream = ext.create_stream()
        stream.accept_waveform(SR, seg)
        stream.input_finished()
        embs.append(np.asarray(ext.compute(stream), dtype=np.float32))

    sim = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim[i, j] = sim[j, i] = cos(embs[i], embs[j])

    tri = sim[np.triu_indices(n, 1)]
    print(f"相似度：p10={np.quantile(tri,.1):.2f} p50={np.quantile(tri,.5):.2f} p90={np.quantile(tri,.9):.2f}")

    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0, 2)
    Z = linkage(squareform(dist, checks=False), method="average")

    def band(f):
        return "?" if f is None else ("M" if f < 150 else "F")

    print("\n切簇距离(=1-相似度) -> 簇数 / 大小 / F0一致率 / 轮廓：")
    results = {}
    for cut_d in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        labels = fcluster(Z, t=cut_d, criterion="distance")
        k = len(set(labels))
        sizes = sorted([int((labels == v).sum()) for v in set(labels)], reverse=True)
        # F0 一致率（各簇多数带占比的均值）
        agree, tot_c = 0.0, 0
        for v in set(labels):
            bands = [band(f0s[i]) for i in range(n) if labels[i] == v and f0s[i] is not None]
            if bands:
                mb = max(set(bands), key=bands.count)
                agree += sum(1 for b in bands if b == mb) / len(bands)
                tot_c += 1
        agree = agree / max(1, tot_c)
        # 轮廓
        sils = []
        for i in range(n):
            inn = [dist[i, j] for j in range(n) if j != i and labels[j] == labels[i]]
            outs = [np.mean([dist[i, j] for j in range(n) if labels[j] == v])
                    for v in set(labels) if v != labels[i]]
            if inn and outs:
                sils.append((np.mean(outs) - np.mean(inn)) / max(np.mean(outs), np.mean(inn)))
        sil = float(np.mean(sils)) if sils else -1
        print(f"  cut={cut_d:.2f}(th={1-cut_d:.2f})  簇数={k} 大小={sizes[:6]} 轮廓={sil:.3f} F0一致率={agree:.2f}")
        results[cut_d] = (labels, sil, agree)

    # 输出最优切分的逐窗明细（前 40 个）
    best_cut = max(results, key=lambda d: results[d][1])
    labels, sil, agree = results[best_cut]
    print(f"\n轮廓最优 cut={best_cut}（th={1-best_cut:.2f} 轮廓={sil:.3f} F0一致率={agree:.2f}）")
    for i in range(min(n, 400)):
        t = wins[i][0] / SR
        print(f"  {t:6.1f}s  簇{labels[i]}  F0带={band(f0s[i])} ({f0s[i] and round(f0s[i])})")
    (HERE / "window_cluster.json").write_text(json.dumps({
        "best_cut": best_cut, "silhouette": sil, "f0_agree": agree,
        "labels": [int(x) for x in labels], "starts": [int(w[0]) for w in wins],
        "f0s": [None if f is None else float(f) for f in f0s],
    }, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
