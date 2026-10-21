"""阈值扫描：用全部句子的嵌入相似度矩阵做层次聚类，找自然簇数与最佳阈值。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402

SR = 16000
MODEL = ROOT / "models" / "campplus_zh" / "campplus_zh.onnx"


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> int:
    sents = json.loads((HERE / "asr_sents.json").read_text(encoding="utf-8"))
    audio, sr = sf.read(HERE / "audio.wav", dtype="float32")

    import sherpa_onnx

    ext = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(MODEL), num_threads=2))

    embs, idxs = [], []
    for i, st in enumerate(sents):
        seg = audio[int(st["start_ms"] * SR / 1000): int(st["end_ms"] * SR / 1000)]
        if len(seg) < int(SR * 0.3):
            continue
        stream = ext.create_stream()
        stream.accept_waveform(SR, seg)
        stream.input_finished()
        embs.append(np.asarray(ext.compute(stream), dtype=np.float32))
        idxs.append(i)

    n = len(embs)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = cos(embs[i], embs[j]) if i != j else 1.0

    # 相似度分布
    tri = sim[np.triu_indices(n, 1)]
    print(f"句子(可嵌入) {n}；相似度分位：p5={np.quantile(tri, .05):.2f} p25={np.quantile(tri, .25):.2f} "
          f"p50={np.quantile(tri, .5):.2f} p75={np.quantile(tri, .75):.2f} p95={np.quantile(tri, .95):.2f}")

    # 单连通层次聚类（贪心合并最高相似度对，直到全部低于阈值）——够用且无依赖
    def cluster(threshold: float) -> list[int]:
        labels = list(range(n))  # 每句自成一簇，再合并
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        pairs = sorted(((sim[i, j], i, j) for i in range(n) for j in range(i + 1, n)),
                       reverse=True)
        for s, i, j in pairs:
            if s < threshold:
                break
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
        return [find(i) for i in range(n)]

    print("\n阈值 -> 簇数 / 各簇大小：")
    for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        lab = cluster(th)
        roots = sorted(set(lab), key=lambda r: -lab.count(r))
        sizes = [lab.count(r) for r in roots]
        # 轮廓系数（余弦距离）
        if len(roots) > 1:
            sils = []
            for i in range(n):
                inn = [1 - sim[i, j] for j in range(n) if j != i and lab[j] == lab[i]]
                outs = []
                for r in set(lab):
                    if r == lab[i]:
                        continue
                    outs.append(np.mean([1 - sim[i, j] for j in range(n) if lab[j] == r]))
                if inn and outs:
                    sils.append((np.mean(outs) - np.mean(inn)) / max(np.mean(outs), np.mean(inn)))
            sil = float(np.mean(sils))
        else:
            sil = -1.0
        print(f"  th={th:.2f}  簇数={len(roots)}  大小={sizes}  轮廓={sil:.3f}")

    # 用轮廓最优的阈值输出逐句标签
    best_th, best_sil, best_lab = None, -2.0, None
    for th in [x / 100 for x in range(28, 66, 2)]:
        lab = cluster(th)
        roots = set(lab)
        if not (1 < len(roots) <= 5):
            continue
        sils = []
        for i in range(n):
            inn = [1 - sim[i, j] for j in range(n) if j != i and lab[j] == lab[i]]
            outs = [np.mean([1 - sim[i, j] for j in range(n) if lab[j] == r])
                    for r in set(lab) if r != lab[i]]
            if inn and outs:
                sils.append((np.mean(outs) - np.mean(inn)) / max(np.mean(outs), np.mean(inn)))
        sil = float(np.mean(sils)) if sils else -1
        if sil > best_sil:
            best_th, best_sil, best_lab = th, sil, lab
    print(f"\n最优阈值 th={best_th}（轮廓 {best_sil:.3f}）")

    root_to_name = {}
    out_rows = []
    for k, i in enumerate(idxs):
        r = best_lab[k]
        if r not in root_to_name:
            root_to_name[r] = f"V{len(root_to_name) + 1}"
        out_rows.append({"i": i, "spk": root_to_name[r], "start_ms": sents[i]["start_ms"],
                         "text": sents[i]["text"][:26]})
    for r in out_rows:
        print(f"[{r['start_ms']:>6}] {r['spk']}  {r['text']}")
    (HERE / "cluster_best.json").write_text(
        json.dumps({"threshold": best_th, "silhouette": best_sil, "rows": out_rows},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
