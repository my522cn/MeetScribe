"""验证轻量声纹聚类：合成两个不同基频的"说话人"，交替说 8 句，
检查 SpeakerRegistry 是否能正确分开并稳定归类。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.asr.diarize import SpeakerRegistry, sentence_features  # noqa: E402

SR = 16000


def make_voice(f0: float, seconds: float, seed: int) -> np.ndarray:
    """合成带谐波+共振峰味道的语音状信号（不是纯音，接近人声频谱）。"""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    # 基频 + 3 个谐波，高次谐波衰减（模拟人声频谱包络）
    x = np.zeros(n, dtype=np.float32)
    for k, amp in ((1, 1.0), (2, 0.5), (3, 0.28), (4, 0.14)):
        x += amp * np.sin(2 * np.pi * f0 * k * t).astype(np.float32)
    # 叠加宽带气息噪声 + 句内幅度起伏
    x += 0.05 * rng.standard_normal(n).astype(np.float32)
    env = 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 3.1 * t))
    return (x * env * 0.12).astype(np.float32)


def main() -> int:
    registry = SpeakerRegistry()
    plan = [1, 0, 1, 0, 0, 1, 0, 1]  # 1=男声(120Hz) 0=女声(230Hz)
    # 第一个出现的说话人（男声）获得"说话人A"，女声为"说话人B"
    truth = ["说话人A" if p else "说话人B" for p in plan]
    got = []
    for i, p in enumerate(plan):
        f0 = 120.0 if p else 230.0
        audio = make_voice(f0, 2.0 + 0.1 * i, seed=100 + i)
        feat = sentence_features(audio)
        assert feat is not None, f"第{i}句特征提取失败"
        label = registry.assign(feat)
        got.append(label)
        print(f"句{i} 基频{f0:.0f}Hz → 特征(F0={feat[0]:.1f}Hz, 质心={feat[1]:.0f}Hz) → {label}")

    ok = got == truth
    print(f"\n说话人数 = {registry.n_speakers}（期望 2）")
    print("期望:", " ".join(truth))
    print("实际:", " ".join(got))
    if not ok:
        print("❌ 聚类结果与真值不一致")
        return 1
    print("✅ 两名说话人全部正确区分并稳定归类")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
