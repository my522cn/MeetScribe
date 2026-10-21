"""轻量说话人区分（线下会议模式）。

两代方案：
- F0 + 谱质心（SpeakerRegistry）：纯 numpy，单句 <10ms，模型缺失时的兜底；
- CAM++ 说话人嵌入（EmbeddingSpeakerRegistry）：sherpa-onnx 声纹模型，
  对同性别、音色接近的说话人远比 F0 可靠。

嵌入版按 1s 滑窗（0.5s 步进）给整句投票——一句里混着两个人（间隔极短甚至
同时出声）时，整句嵌入会被"平均"成第三个人，窗口投票 + 多数派标注能扛住
这种情况，并顺带给出"疑似多人同时说话"的比例。
"""
from __future__ import annotations

from pathlib import Path

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

# 嵌入版默认参数（阈值来自 diar_eval 在真实语料上的扫描；可被 settings 覆盖）
EMB_MODEL_REL = "campplus_zh/campplus_zh.onnx"
EMB_ASSIGN_THRESHOLD = 0.38
EMB_WIN_SEC = 1.0
EMB_HOP_SEC = 0.5
EMB_WIN_RMS = 0.008   # 窗口能量门限（AGC 之后正常说话远高于它）
EMB_MIN_NEW_WINDOWS = 2  # 一句里至少这么多个"不匹配"窗口才允许立新说话人（防噪声单窗立簇）

LABELS = [f"说话人{chr(65 + i)}" for i in range(26)]  # A..Z，max_speakers 上限由此决定


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
    """增量式说话人聚类：录音会话内维护每个说话人的特征质心。

    seed() 可预置"长期档案"里的已知说话人（上次会议命名过的人），
    本场会议第一句就能认出是谁，实现跨会话的声纹长期学习。
    """

    def __init__(self, max_speakers: int = MAX_SPEAKERS) -> None:
        self._centroids: list[tuple[float, float]] = []
        self._counts: list[int] = []
        self._names: list[str] = []
        self.max_speakers = max_speakers

    @property
    def n_speakers(self) -> int:
        return len(self._centroids)

    def seed(self, name: str, feat: tuple[float, float], count: int = 3) -> None:
        """用历史档案预置一个已知说话人（count 给大一点，让质心更稳）。"""
        self._centroids.append(feat)
        self._counts.append(max(1, count))
        self._names.append(name)

    def majority_label(self) -> str:
        """特征提取失败（太短/太轻）的句子归入当前说话最多的人。"""
        if not self._counts:
            return LABELS[0]
        return self._names[int(np.argmax(self._counts))]

    def assign(self, feat: tuple[float, float]) -> str:
        """把一句的特征归到某个说话人，返回标签（预置档案用真名，新人用说话人X）。"""
        if not self._centroids:
            self._centroids.append(feat)
            self._counts.append(1)
            self._names.append(LABELS[0])
            return self._names[0]

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
            return self._names[best]
        self._centroids.append(feat)
        self._counts.append(1)
        idx = len(self._centroids) - 1
        self._names.append(LABELS[idx] if idx < len(LABELS) else f"说话人{idx + 1}")
        return self._names[idx]

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        df = abs(a[0] - b[0]) / SCALE_F0
        dc = abs(a[1] - b[1]) / SCALE_CENT
        return float(np.hypot(df, dc))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class EmbeddingSpeakerRegistry:
    """CAM++ 声纹嵌入 + 滑窗投票的说话人聚类。

    - assign_sentence(audio) 返回 (标签, 疑似多人比例)。一句里两个人抢话时，
      不同窗口会投给不同说话人，多数派赢，少数派比例高就打"多人"标记。
    - 质心用"投给它"的窗口嵌入均值更新，混说窗口不污染质心。
    - profiles() 输出每个说话人的平均嵌入，供跨会话长期档案使用。
    """

    def __init__(
        self,
        model_path: Path,
        threshold: float = EMB_ASSIGN_THRESHOLD,
        max_speakers: int = MAX_SPEAKERS,
        win_sec: float = EMB_WIN_SEC,
        hop_sec: float = EMB_HOP_SEC,
        min_new_windows: int = EMB_MIN_NEW_WINDOWS,
    ) -> None:
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_path), num_threads=2
        )
        self._ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.min_new_windows = max(1, min_new_windows)
        self._win = int(win_sec * SR)
        self._hop = int(hop_sec * SR)
        self._cents: list[np.ndarray] = []
        self._counts: list[int] = []
        self._names: list[str] = []

    @property
    def n_speakers(self) -> int:
        return len(self._cents)

    @property
    def dim(self) -> int:
        """嵌入维度（CAM++ 为 192），供自检/档案校验使用。"""
        return self._ext.dim

    def seed(self, name: str, emb: np.ndarray, count: int = 3) -> None:
        """长期档案预置（emb 是历史会话积累的平均嵌入）。"""
        self._cents.append(np.asarray(emb, dtype=np.float32))
        self._counts.append(max(1, count))
        self._names.append(name)

    def majority_label(self) -> str:
        """整句提不出任何有效窗口时（太短/纯静音），归入当前说话最多的人。"""
        if not self._counts:
            return LABELS[0]
        return self._names[int(np.argmax(self._counts))]

    def profiles(self) -> dict[str, list[float]]:
        """每个说话人的平均嵌入（跨会话档案用）。"""
        return {self._names[i]: self._cents[i].tolist() for i in range(len(self._cents))}

    def _windows(self, audio: np.ndarray) -> list[np.ndarray]:
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        out = []
        for start in range(0, len(x) - self._win + 1, self._hop):
            seg = x[start:start + self._win]
            if float(np.sqrt(np.mean(seg ** 2))) < EMB_WIN_RMS:
                continue
            out.append(seg)
        return out

    def assign_sentence(self, audio: np.ndarray | None) -> tuple[str, float]:
        """整句标注。返回 (说话人, 疑似多人比例 0~1)。"""
        windows = self._windows(audio) if audio is not None else []
        if not windows:
            return self.majority_label(), 0.0

        embs: list[np.ndarray] = []
        for seg in windows:
            stream = self._ext.create_stream()
            stream.accept_waveform(SR, seg)
            stream.input_finished()
            embs.append(np.asarray(self._ext.compute(stream), dtype=np.float32))

        votes: list[int] = []
        if not self._cents:
            # 首句：直接建第一个说话人
            self._cents.append(np.mean(embs, axis=0))
            self._counts.append(len(embs))
            self._names.append(LABELS[0])
            return self._names[0], 0.0

        # 第一遍：逐窗找最像的质心；低于阈值的是"孤儿"（可能是新人/噪声）
        sims = [[_cos(e, c) for c in self._cents] for e in embs]
        best = [int(np.argmax(s)) for s in sims]
        best_sim = [sims[k][best[k]] for k in range(len(embs))]
        orphans = [k for k in range(len(embs)) if best_sim[k] < self.threshold]

        # 第二遍：立新说话人需要满足其一（防止信道噪声随机立簇）——
        #   a) 孤儿构成多数且 >= min_new_windows（整句基本是新人）
        #   b) 全部窗口都是孤儿、且最高相似度显著低于阈值（-0.10 余量）：
        #      短句（1s 单窗）也允许立簇，但要求"明显不像"而不是"差一点不像"
        new_idx = -1
        if len(self._cents) < self.max_speakers and orphans:
            majority = len(orphans) >= self.min_new_windows and len(orphans) * 2 > len(embs)
            all_orphan = len(orphans) == len(embs) and max(
                best_sim[k] for k in orphans
            ) < self.threshold - 0.10
            if majority or all_orphan:
                self._cents.append(np.mean([embs[k] for k in orphans], axis=0))
                self._counts.append(0)
                self._names.append(LABELS[len(self._cents) - 1])
                new_idx = len(self._cents) - 1

        for k in range(len(embs)):
            if new_idx >= 0 and k in orphans:
                votes.append(new_idx)
            else:
                votes.append(best[k])

        counts = np.bincount(votes, minlength=len(self._cents))
        winner = int(np.argmax(counts))
        # 质心用投给赢家且相似度达标的窗口均值更新（混说窗口不进来）
        members = [embs[k] for k, v in enumerate(votes) if v == winner and _cos(embs[k], self._cents[winner]) >= self.threshold]
        if members:
            n = max(1, self._counts[winner])
            self._cents[winner] = (self._cents[winner] * n + np.mean(members, axis=0)) / (n + len(members))
        self._counts[winner] += len(members)

        mixed_ratio = 1.0 - counts[winner] / len(votes)
        return self._names[winner], float(mixed_ratio)


def embedding_model_path(models_root: Path) -> Path | None:
    """声纹嵌入模型的位置；不存在返回 None（走 F0 兜底）。"""
    p = Path(models_root) / EMB_MODEL_REL
    return p if p.exists() else None
