"""对比两种声纹方案在同一批句子上的聚类质量：

A. 现行方案：F0 + 谱质心，增量质心聚类（meetscribe/asr/diarize.py）
B. 新方案：CAM++ 说话人嵌入（sherpa-onnx），余弦相似度增量聚类

输出：
- diar_compare.json  每句两种方案的标签与特征
- 控制台打印逐句对照表 + 聚类质量指标

用法：../../.venv/Scripts/python.exe eval_diar.py
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

from meetscribe.asr.diarize import (  # noqa: E402
    SpeakerRegistry, sentence_features,
)

MODEL = ROOT / "models" / "campplus_zh" / "campplus_zh.onnx"
SR = 16000
EMB_MIN_SEC = 0.30     # 短于它的句子不做嵌入（太不可靠），沿用最近说话人
COS_ASSIGN = 0.55      # 嵌入余弦相似度阈值：>=它归入该说话人


# ---------- 嵌入 ----------
class Embedder:
    def __init__(self) -> None:
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(MODEL), num_threads=2
        )
        self.ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def __call__(self, audio: np.ndarray) -> np.ndarray | None:
        if len(audio) < int(SR * EMB_MIN_SEC):
            return None
        stream = self.ext.create_stream()
        stream.accept_waveform(SR, audio)
        stream.input_finished()
        return np.asarray(self.ext.compute(stream), dtype=np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------- 方案 B 的增量聚类（与生产逻辑对齐：质心滑动平均 + 阈值立新人） ----------
class EmbRegistry:
    def __init__(self, threshold: float = COS_ASSIGN, max_speakers: int = 4) -> None:
        self._cents: list[np.ndarray] = []
        self._counts: list[int] = []
        self._names: list[str] = []
        self.threshold = threshold
        self.max_speakers = max_speakers

    @property
    def n_speakers(self) -> int:
        return len(self._cents)

    def seed(self, name: str, emb: np.ndarray, count: int = 3) -> None:
        self._cents.append(emb)
        self._counts.append(max(1, count))
        self._names.append(name)

    def majority_label(self) -> str:
        if not self._counts:
            return "说话人A"
        return self._names[int(np.argmax(self._counts))]

    def assign(self, emb: np.ndarray) -> str:
        if not self._cents:
            self._cents.append(emb)
            self._counts.append(1)
            self._names.append("说话人A")
            return self._names[0]
        sims = [cos(emb, c) for c in self._cents]
        best = int(np.argmax(sims))
        if sims[best] >= self.threshold or len(self._cents) >= self.max_speakers:
            n = self._counts[best]
            self._cents[best] = (self._cents[best] * n + emb) / (n + 1)
            self._counts[best] = n + 1
            return self._names[best]
        self._cents.append(emb)
        self._counts.append(1)
        idx = len(self._cents) - 1
        self._names.append(f"说话人{'ABCD'[idx] if idx < 4 else idx + 1}")
        return self._names[idx]


# ---------- 全批聚类质量（不依赖在线顺序） ----------
def batch_metrics(veats: list, labels: list[int]) -> dict:
    """veats: 特征列表；labels: 每句的簇号。返回簇内/簇间平均相似度（余弦）。"""
    n = len(veats)
    same, diff = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if veats[i] is None or veats[j] is None:
                continue
            s = cos(veats[i], veats[j])
            (same if labels[i] == labels[j] else diff).append(s)
    return {
        "簇内平均相似度": round(float(np.mean(same)), 3) if same else None,
        "簇间平均相似度": round(float(np.mean(diff)), 3) if diff else None,
        "可分性(内-间)": round(float(np.mean(same) - np.mean(diff)), 3) if same and diff else None,
    }


def main() -> int:
    sents = json.loads((HERE / "asr_sents.json").read_text(encoding="utf-8"))
    audio, sr = sf.read(HERE / "audio.wav", dtype="float32")
    assert sr == SR
    emb = Embedder()
    print(f"句子 {len(sents)}，嵌入维度 {emb.ext.dim}")

    reg_old = SpeakerRegistry()
    reg_new = EmbRegistry()

    rows = []
    old_feats: list = []
    new_embs: list = []

    for i, st in enumerate(sents):
        a0, a1 = int(st["start_ms"] * SR / 1000), int(st["end_ms"] * SR / 1000)
        seg = audio[a0:a1]
        feat = sentence_features(seg)
        e = emb(seg)

        if feat is not None:
            lab_old = reg_old.assign(feat)
        else:
            lab_old = reg_old.majority_label()
        if e is not None:
            lab_new = reg_new.assign(e)
        else:
            lab_new = reg_new.majority_label()

        old_feats.append(np.array(feat) if feat is not None else None)
        new_embs.append(e)
        rows.append({
            "i": i, "start_ms": st["start_ms"], "end_ms": st["end_ms"],
            "dur_ms": st["end_ms"] - st["start_ms"],
            "text": st["text"], "old": lab_old, "new": lab_new,
            "f0": round(feat[0], 1) if feat else None,
            "cent": round(feat[1], 0) if feat else None,
            "emb": True if e is not None else False,
        })

    # 旧特征也用余弦形式可比：把 (f0, cent) 归一后视作向量
    olds = [np.array([f[0] / 45.0, f[1] / 900.0]) if f is not None else None for f in old_feats]
    # 转成“相似度”视角：1/(1+距离) 不合适，这里直接报告两两欧氏距离的簇内/簇间
    def label_idx(lab: str) -> int:
        if lab.startswith("说话人") and len(lab) > 3 and lab[-1] in "ABCD":
            return "ABCD".index(lab[-1])
        return 0

    old_labels = [label_idx(r["old"]) for r in rows]
    new_labels = [label_idx(r["new"]) for r in rows]

    def dist_metrics(feats, labels):
        n = len(feats)
        same, diff = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if feats[i] is None or feats[j] is None:
                    continue
                d = float(np.linalg.norm(feats[i] - feats[j]))
                (same if labels[i] == labels[j] else diff).append(d)
        return {
            "簇内平均距离": round(float(np.mean(same)), 3) if same else None,
            "簇间平均距离": round(float(np.mean(diff)), 3) if diff else None,
            "可分性(间-内)": round(float(np.mean(diff) - np.mean(same)), 3) if same and diff else None,
        }

    print("\n=== 聚类质量（在线增量聚类结果上统计） ===")
    print("旧特征(F0+质心, 欧氏):", dist_metrics(olds, old_labels))
    print("新嵌入(CAM++, 余弦)  :", batch_metrics(new_embs, new_labels))
    print(f"说话人数：旧={reg_old.n_speakers} 新={reg_new.n_speakers}")

    print("\n=== 逐句对照（供人工比对字幕轮替） ===")
    for r in rows:
        mark = " " if r["old"] == r["new"] else "|"
        f0s = f"F0={r['f0']}" if r["f0"] else "F0= —  "
        print(f"[{r['start_ms']:>6}] {r['dur_ms']:>4}ms {f0s} 旧:{r['old']} 新:{r['new']} {mark} {r['text'][:28]}")

    (HERE / "diar_compare.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
