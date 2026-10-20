"""AGC 全链路验证：真实低电平录音 → AGC → pipeline 定稿（带 audio）→ 声纹特征 → 离线精修。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from meetscribe.audio.agc import AGC
from meetscribe.asr.engine import ASRConfig, StreamingASR
from meetscribe.asr.diarize import sentence_features
from meetscribe.asr.offline import OfflineASR
from meetscribe.asr.pipeline import LivePipeline

ROOT = Path(__file__).resolve().parents[1]
d = ROOT / "data/sessions/20261021_001629_manual"
audio, sr = sf.read(str(d / "mic/seg_0001.flac"), dtype="float32", always_2d=True)
audio = audio.mean(axis=1).astype(np.float32)

# 1. AGC 后电平
agc = AGC()
boosted = np.concatenate([agc.process(audio[i:i + 1600]) for i in range(0, len(audio), 1600)])
print(f"原始 rms={np.sqrt(np.mean(audio**2)):.4f} → AGC 后 rms={np.sqrt(np.mean(boosted**2)):.4f} "
      f"(gain 最终 {agc.gain:.1f}x)")

# 2. 声纹特征（修复前返回 None）
feat = sentence_features(boosted[int(5 * 16000):int(15 * 16000)])
print("AGC 后声纹特征:", feat)

# 3. 离线精修能否识别 AGC 后的音频
asr = OfflineASR.__new__(OfflineASR)
from meetscribe.asr.offline import get_offline
off = get_offline(ROOT / "models/paraformer_offline", "paraformer", 2)
off.load()
raw_text = off.transcribe(boosted[int(1.4 * 16000):int(20 * 16000)])
print("离线精修输出(AGC 后):", raw_text[:60])

# 4. pipeline 全链路：定稿句必须带 audio（慢速喂入模拟实时）
engine = StreamingASR(ASRConfig(model_dir=ROOT / "models/paraformer_zh_en", num_threads=2))
finals = []
pipe = LivePipeline(
    engine,
    on_final=lambda c: finals.append((c.start_ms, c.end_ms, c.audio, c.text)),
)
pipe.start()
BLOCK = 1600
for i in range(0, len(boosted), BLOCK):
    pipe.push("mic", boosted[i:i + BLOCK])
    time.sleep(0.004)  # 模拟实时节奏，避免测试里把队列挤爆
time.sleep(3)
pipe.stop()
print("\n== pipeline 定稿 ==")
for s, e, a, t in finals:
    print(f"{s}-{e}ms audio={'None!!' if a is None else len(a)} text={t[:24]}")
