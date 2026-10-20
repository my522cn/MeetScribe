"""录音自动增益（AGC）。

背景：部分麦克风（尤其阵列麦/智音技术处理链）录出来的电平极低
（整段 rms≈0.003、峰值≈0.024），连锁导致：
- 声纹特征被静音门限判为无声 → 线下模式分不出人；
- 离线精修模型识别不出内容 → 句子永远没有 ✓；
- 流式首遍识别质量差、错字多。

方案：在录音线程对每块做平滑增益，把语音电平归一到目标 rms，
写盘文件与实时识别用同一条放大后的信号，全链路一致生效。
"""
from __future__ import annotations

import numpy as np

TARGET_RMS = 0.06      # 归一目标电平（正常语音区间中位）
MAX_GAIN = 30.0        # 最大增益 30dB，避免把底噪抬成幻觉
SPEAK_RMS = 0.0015     # 高于它才调增益（静音段冻结，防止把底噪越拉越高）


class AGC:
    """单轨 AGC。attack 快（起音快速跟上）、release 慢（避免忽大忽小）。"""

    def __init__(self, target_rms: float = TARGET_RMS, max_gain: float = MAX_GAIN) -> None:
        self.target = target_rms
        self.max_gain = max_gain
        self.gain = 1.0

    def process(self, block: np.ndarray) -> np.ndarray:
        x = np.asarray(block, dtype=np.float32)
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        if rms > SPEAK_RMS:
            inst = min(self.max_gain, self.target / rms)
            if inst > self.gain:
                self.gain = 0.8 * self.gain + 0.2 * inst   # attack：~0.5s 跟上
            else:
                self.gain = 0.97 * self.gain + 0.03 * inst  # release：慢速回落
        out = (x * self.gain).astype(np.float32, copy=False)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.98:  # 防削顶
            out *= 0.98 / peak
        return out
