"""端到端验证"两遍识别"：流式首遍 vs 离线 Paraformer 精修。

做法：把原始测试音频按自然停顿切成 5 句，中间插入 1.6s 静音强制断句，
逐句比较首遍与精修结果的 CER，并统计耗时/RTF。
用法：python -m phase3.test_refine_e2e
"""
from __future__ import annotations

import re
import sys
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meetscribe.asr.engine import ASRConfig, StreamingASR  # noqa: E402
from meetscribe.asr.pipeline import LivePipeline  # noqa: E402
from meetscribe.asr.refine import RefineService  # noqa: E402
from meetscribe.text.postprocess import TextPostProcessor  # noqa: E402

WAV = ROOT / "phase2" / "test_zh_en_16k.wav"
SR = 16000

# (起, 止, 参考文本) —— 依据音频能量谷切分
PIECES = [
    (0.0, 4.45, "大家好，今天我们先过一遍 HARMONIC 项目的排期"),
    (4.90, 7.05, "I think the timeline is too tight"),
    (7.30, 9.45, "联调至少要留两周时间"),
    (9.90, 10.70, "第二件事"),
    (11.10, 13.60, "客户那边希望下个月能看到 demo"),
]
GAP = 1.6

_PUNC = re.compile(r"[\s，。、！？；：“”‘’（）《》…—\-_,.!?;:\"'()\[\]]+")


def norm(t: str) -> str:
    return _PUNC.sub("", unicodedata.normalize("NFKC", t).lower())


def ed(a: str, b: str) -> int:
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


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    return ed(r, h) / max(1, len(r))


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        data = f.readframes(f.getnframes())
        sr, ch = f.getframerate(), f.getnchannels()
    a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    assert sr == SR, sr
    return a


def build_multi(src: np.ndarray) -> np.ndarray:
    gap = np.zeros(int(GAP * SR), dtype=np.float32)
    out = []
    for s, e, _ in PIECES:
        out.append(src[int(s * SR) : int(e * SR)])
        out.append(gap)
    return np.concatenate(out)


def main() -> None:
    raw = read_wav(WAV)
    audio = build_multi(raw)
    audio = np.concatenate([audio, np.zeros(int(3 * SR), dtype=np.float32)])
    print(f"合成多句音频 {len(audio)/SR:.2f}s，共 {len(PIECES)} 句\n")

    engine = StreamingASR(ASRConfig(model_dir=str(ROOT / "models" / "paraformer_zh_en"), num_threads=2))
    finals: list = []
    pipeline = LivePipeline(engine, on_final=finals.append, samplerate=SR)
    pipeline.start()

    block = 1600
    t0 = time.perf_counter()
    for i in range(0, len(audio), block):
        pipeline.push("mic", audio[i : i + block])
        time.sleep(0.002)
    while not pipeline._queues["mic"].empty():
        time.sleep(0.05)
    time.sleep(0.4)
    pipeline.stop()
    stream_elapsed = time.perf_counter() - t0
    print(f"流式：{len(finals)} 句，音频 {len(audio)/SR:.1f}s，耗时 {stream_elapsed:.1f}s "
          f"(RTF {stream_elapsed/(len(audio)/SR):.2f})\n")

    post = TextPostProcessor(
        hotwords_file=ROOT / "config" / "hotwords.txt",
        replacements_file=ROOT / "config" / "replacements.txt",
        punct_model=ROOT / "models" / "punct_zh_en" / "model.int8.onnx",
    )
    svc = RefineService(ROOT / "models" / "paraformer_offline", on_result=lambda j, t: None, kind="paraformer")
    t_build = time.perf_counter()
    svc._recognizer = svc._build()
    print(f"精修模型加载 {time.perf_counter()-t_build:.2f}s")

    sum_first = sum_ref = 0.0
    n = 0
    print("-" * 78)
    for i, cap in enumerate(finals):
        first = post.process(cap.text)
        # 句序与 PIECES 一致（端点会延后触发，按时间就近匹配会错位）
        ref = PIECES[i][2] if i < len(PIECES) else ""
        refined = ""
        if cap.audio is not None and len(cap.audio) > 4000:
            refined = post.process(svc._decode(cap.audio))
        c1, c2 = cer(ref, first), cer(ref, refined) if refined else c1
        sum_first += c1
        sum_ref += c2
        n += 1
        print(f"[{i+1}] {cap.start_ms/1000:6.2f}s  ref: {ref}")
        print(f"    首遍 CER {c1*100:5.1f}%  {first}")
        print(f"    精修 CER {c2*100:5.1f}%  {refined}")
    if n:
        print("-" * 78)
        print(f"平均 CER：首遍 {sum_first/n*100:.1f}% → 精修 {sum_ref/n*100:.1f}%")
        print(f"精修单句 RTF：{svc.last_rtf:.3f}")


if __name__ == "__main__":
    main()
