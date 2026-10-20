"""轻量说话人区分（线下会议模式）。

思路：不跑重型说话人模型，用每句音频的两个声学特征做增量聚类——
- F0 基频（自相关法）：男声 ~85-180Hz，女声 ~165-255Hz，差异最稳定；
- 谱质心：音色明亮度（同一人齿音/共振峰分布稳定）。

每句定稿时算一次特征，与已有说话人质心比距离：近则归入并微调质心，
远则立新说话人（最多 4 人）。纯 numpy，单句 <10ms，不影响实时性。
"""
from __future__ import annotations

import numpy as np

SR = 16000
FRAME = 400          # 25ms
HOP = 160            # 10ms
F0_MIN, F0_MAX = 60.0, 400.0

# 距离归一化尺度：F0 差 45Hz / 质心差 900Hz 视为"一个单位"
SCALE_F0 = 45.0
SCALE_CENT = 900.0
NEW_SPEAKER_THRESHOLD = 1.15   # 归一化距离超过它就认为是新说话人
MAX_SPEAKERS = 4

LABELS = ["说话人A", "说话人B", "说话人C", "说话人D"]


def sentence_features(audio: np.ndarray, sr: int = SR) -> tuple[float, float] | None:
    """返回 (f0中位数Hz, 谱质心中位数Hz)；音频太短或基本无声时返回 None。"""
    if audio is None or len(audio) < sr // 2:  # 至少 0.5 秒
        return None
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    rms = float(np.sqrt(np.mean(x * x) + 1e-9))
    if rms < 0.0012:  # 门限只挡纯底噪；低电平麦克风由录音端 AGC 归一，这里不再误杀
        return None

    n_frames = (len(x) - FRAME) // HOP + 1
    if n_frames < 5:
        return None

    f0s: list[float] = []
    cents: list[float] = []
    freqs = np.fft.rfftfreq(1024, d=1.0 / sr)

    for i in range(n_frames):
        seg = x[i * HOP : i * HOP + FRAME]
        e = float(np.mean(seg * seg))
        if e < 1e-6:  # 帧内静音，跳过
            continue
        cents.append(float(np.sum(freqs * np.abs(np.fft.rfft(seg, 1024))) / (np.sum(np.abs(np.fft.rfft(seg, 1024))) + 1e-9)))
        f0 = _autocorr_f0(seg, sr)
        if f0 is not None:
            f0s.append(f0)

    if not f0s or not cents:
        return None
    return float(np.median(f0s)), float(np.median(cents))


def _autocorr_f0(seg: np.ndarray, sr: int) -> float | None:
    """自相关法估单帧基频。"""
    seg = seg - seg.mean()
    energy = float(np.sum(seg * seg))
    if energy < 1e-5:
        return None
    corr = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
    lag_min, lag_max = int(sr / F0_MAX), int(sr / F0_MIN)
    if lag_max >= len(corr) - 1:
        return None
    window = corr[lag_min : lag_max + 1]
    peak = int(np.argmax(window)) + lag_min
    if corr[peak] < 0.35 * (energy + 1e-9):  # 清音/噪声帧没有明显周期性
        return None
    return sr / peak


class SpeakerRegistry:
    """增量式说话人聚类：录音会话内维护每个说话人的特征质心。"""

    def __init__(self, max_speakers: int = MAX_SPEAKERS) -> None:
        self._centroids: list[tuple[float, float]] = []
        self._counts: list[int] = []
        self.max_speakers = max_speakers

    @property
    def n_speakers(self) -> int:
        return len(self._centroids)

    def majority_label(self) -> str:
        """特征提取失败（太短/太轻）的句子归入当前说话最多的人。"""
        if not self._counts:
            return LABELS[0]
        return LABELS[int(np.argmax(self._counts))]

    def assign(self, feat: tuple[float, float]) -> str:
        """把一句的特征归到某个说话人，返回"说话人A"式标签。"""
        if not self._centroids:
            self._centroids.append(feat)
            self._counts.append(1)
            return LABELS[0]

        dists = [self._distance(feat, c) for c in self._centroids]
        best = int(np.argmin(dists))
        if dists[best] <= NEW_SPEAKER_THRESHOLD or len(self._centroids) >= self.max_speakers:
            # 归入最近的人，质心做滑动平均（适应音量/设备漂移）
            n = self._counts[best]
            f0, cent = self._centroids[best]
            self._centroids[best] = (
                (f0 * n + feat[0]) / (n + 1),
                (cent * n + feat[1]) / (n + 1),
            )
            self._counts[best] = n + 1
            return LABELS[best]
        self._centroids.append(feat)
        self._counts.append(1)
        return LABELS[len(self._centroids) - 1]

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        df = abs(a[0] - b[0]) / SCALE_F0
        dc = abs(a[1] - b[1]) / SCALE_CENT
        return float(np.hypot(df, dc))
