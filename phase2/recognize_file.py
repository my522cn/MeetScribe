"""阶段二验收（一）：文件流式识别 + 准确率 + 实时率。

把一个 16k 单声道 wav 按 100ms 块模拟实时送入识别器，
打印每块的临时文本变化，统计最终文本、字错率(CER)、处理实时率(RTF)。
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


def normalize(text: str) -> str:
    drop = set("，。、？！：；“”‘’（）()[]{},.?!:;\"' \n\t-—")
    return "".join(ch.lower() for ch in text if ch not in drop)


def edit_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main() -> None:
    wav = ROOT / "phase2" / "test_zh_en_16k.wav"
    if not wav.exists():
        raise SystemExit(f"缺少测试音频: {wav}")
    model_dir = ROOT / "models" / "paraformer_zh_en"

    audio, sr = sf.read(wav, dtype="float32")
    assert sr == 16000
    print(f"音频: {len(audio)/sr:.2f}s  模型: {model_dir.name}")

    t0 = time.perf_counter()
    asr = StreamingASR(ASRConfig(model_dir=model_dir, num_threads=2))
    print(f"模型加载耗时: {time.perf_counter()-t0:.2f}s")

    block = 1600
    finals: list[str] = []
    total_audio = 0.0
    total_proc = 0.0
    last_shown = ""
    first_text_at = None

    for i in range(0, len(audio), block):
        chunk = audio[i:i + block]
        start = time.perf_counter()
        partial = asr.feed("demo", chunk)
        total_proc += time.perf_counter() - start
        total_audio += len(chunk) / sr

        if partial and first_text_at is None:
            first_text_at = total_audio
        if partial and partial != last_shown:
            sys.stdout.write(f"\r[{total_audio:5.2f}s] {partial[:80]:<80}")
            sys.stdout.flush()
            last_shown = partial

        if asr.recognizer.is_endpoint(asr.stream("demo").stream):
            text = asr.finish_sentence("demo")
            if text:
                finals.append(text)
                sys.stdout.write(f"\n  ── 断句 ── {text}\n")
                last_shown = ""

    asr.flush("demo", 1.0)
    tail = asr.recognizer.get_result(asr.stream("demo").stream).strip()
    if tail:
        finals.append(tail)

    print("\n" + "=" * 74)
    print("识别结果")
    print("=" * 74)
    full = "".join(finals)
    print(full)
    print(f"\n首字延迟: {first_text_at:.2f}s   RTF: {total_proc/total_audio:.3f}   "
          f"处理总耗时 {total_proc:.2f}s / 音频 {total_audio:.1f}s")

    ref = "大家好，今天我们先过一遍 HARMONIC 项目的排期。I think the timeline is too tight, 联调至少要留两周时间。第二件事，客户那边希望下个月能看到 demo。"
    a, b = normalize(full), normalize(ref)
    dist = edit_distance(a, b)
    print(f"\n参考: {ref}")
    print(f"字错率 CER ≈ {dist/max(len(b),1)*100:.1f}%  (编辑距离 {dist} / 参考字数 {len(b)})")


if __name__ == "__main__":
    main()
