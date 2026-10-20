"""阶段三验收（无界面部分）：模拟双轨音频驱动实时字幕流水线。

把测试音频拆成"麦克风轨"（前半段）和"系统声轨"（后半段），以 100ms 块并行推入，
检查：临时文本刷新、断句定稿、双泳道并行、时间戳是否随音频推进。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.asr.engine import ASRConfig, StreamingASR  # noqa: E402
from meetscribe.asr.pipeline import Caption, LivePipeline  # noqa: E402

partial_count = {"n": 0}
finals: list[Caption] = []


def on_partial(cap: Caption) -> None:
    partial_count["n"] += 1
    if partial_count["n"] % 8 == 0:
        print(f"  [{cap.start_ms/1000:5.1f}s] {cap.track:<4} 临时: {cap.text[-40:]}")


def on_final(cap: Caption) -> None:
    finals.append(cap)
    print(f"  ── 定稿 {cap.track:<4} {cap.start_ms/1000:5.1f}~{cap.end_ms/1000:5.1f}s  {cap.text}")


def on_overlap(active: bool) -> None:
    print(f"  ⚡ 重叠发言: {'开始' if active else '结束'}")


def main() -> None:
    audio, sr = sf.read(ROOT / "phase2" / "test_zh_en_16k.wav", dtype="float32")
    half = len(audio) // 2
    mic_audio = audio[:half]
    loop_audio = np.concatenate([np.zeros(16000 * 3, dtype=np.float32), audio[half:]])

    engine = StreamingASR(ASRConfig(model_dir=ROOT / "models" / "paraformer_zh_en"))
    pipe = LivePipeline(engine, on_partial=on_partial, on_final=on_final, on_overlap=on_overlap)
    pipe.start()

    block = 1600
    total = max(len(mic_audio), len(loop_audio))
    for i in range(0, total, block):
        if i < len(mic_audio):
            pipe.push("mic", mic_audio[i:i + block])
        if i < len(loop_audio):
            pipe.push("loop", loop_audio[i:i + block])
        time.sleep(0.02)  # 稍微快于实时，模拟积压

    time.sleep(1.5)
    pipe.stop()

    print("\n结果")
    print(f"临时文本刷新次数: {partial_count['n']}")
    print(f"定稿句数: {len(finals)}")
    for cap in finals:
        print(f"  {cap.track:<5} {cap.start_ms/1000:5.1f}~{cap.end_ms/1000:5.1f}s  {cap.text}")


if __name__ == "__main__":
    main()
