"""嵌入版声纹（生产实现）回归：用真实视频音频切片验证。

1. 男/女声窗口（F0 弱真值）分别被聚成不同说话人
2. 同一人后续句子稳定归入同一标签
3. 男女拼接的"混说句"给出高多人比例
4. 档案 seed：预置嵌入后，对应说话人直接用真名
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402

from meetscribe.asr.diarize import EmbeddingSpeakerRegistry, embedding_model_path  # noqa: E402

SR = 16000
HERE = ROOT / "phase3" / "diar_eval"

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


def main() -> int:
    model = embedding_model_path(ROOT / "models")
    if model is None:
        print("[SKIP] 没有声纹模型 models/campplus_zh/campplus_zh.onnx")
        return 0
    wc = HERE / "window_cluster.json"
    if not wc.exists():
        print("[SKIP] 缺 window_cluster.json（先跑 scan_windows.py）")
        return 0
    info = json.loads(wc.read_text(encoding="utf-8"))
    audio, sr = sf.read(HERE / "audio.wav", dtype="float32")
    assert sr == SR

    labels, starts, f0s = info["labels"], info["starts"], info["f0s"]
    male = [i for i in range(len(starts)) if f0s[i] and f0s[i] < 120]
    female = [i for i in range(len(starts)) if f0s[i] and 220 < f0s[i] < 340]
    if len(male) < 2 or len(female) < 2:
        print(f"[SKIP] 男女声样本不足（男 {len(male)} 女 {len(female)}）")
        return 0

    def seg_of(widx: int) -> np.ndarray:
        s = starts[widx]
        return audio[s:s + SR]

    reg = EmbeddingSpeakerRegistry(model)

    # 1. 首个男声 → 说话人A
    lab0, _ = reg.assign_sentence(seg_of(male[0]))
    check("首个男声建立说话人A", lab0 == "说话人A", f"lab={lab0}")

    # 2. 女声 → 应立新说话人（而不是混进 A）
    lab1, _ = reg.assign_sentence(seg_of(female[0]))
    check("女声另立说话人", lab1 != lab0, f"lab={lab1}")

    # 3. 同类声音的后续句子稳定归入同一标签（各取 3 句）
    male_ok = sum(1 for w in male[1:4] if reg.assign_sentence(seg_of(w))[0] == lab0)
    female_ok = sum(1 for w in female[1:4] if reg.assign_sentence(seg_of(w))[0] == lab1)
    check("男声后续稳定归入", male_ok >= 2, f"{male_ok}/3")
    check("女声后续稳定归入", female_ok >= 2, f"{female_ok}/3")

    # 4. 男女拼接 = 混说句：多人比例应明显
    mixed_seg = np.concatenate([seg_of(male[0]), seg_of(female[0])])
    _, ratio = reg.assign_sentence(mixed_seg)
    check("拼接句检测到多人", ratio >= 0.3, f"mixed={ratio:.2f}")

    # 5. 档案：seed 真名后直接认出
    profs = reg.profiles()
    reg2 = EmbeddingSpeakerRegistry(model)
    reg2.seed("张三", np.asarray(profs[lab0]))
    lab_seed, _ = reg2.assign_sentence(seg_of(male[1]))
    check("档案预置后按真名识别", lab_seed == "张三", f"lab={lab_seed}")

    print()
    bad = [n for n, ok, _ in checks if not ok]
    for n, ok, d in checks:
        print(f"[{'OK ' if ok else 'FAIL'}] {n}　{d}")
    print()
    if bad:
        print("❌ 失败项：", bad)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
