"""三个模型同段音频识别对比：流式 Paraformer / 离线 Paraformer / SenseVoice。

目的：给"错别字"问题挑出最准的后台精修模型。
用法：python -m phase3.compare_models
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

REF = "大家好，今天我们先过一遍 HARMONIC 项目的排期。I think the timeline is too tight, 联调至少要留两周时间。第二件事，客户那边希望下个月能看到 demo。"

WAV = ROOT / "phase2" / "test_zh_en_16k.wav"

_PUNC = re.compile(r"[\s，。、！？；：“”‘’（）《》…—\-_,.!?;:\"'()\[\]]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    return _PUNC.sub("", text)


def _tags(text: str) -> str:
    return re.sub(r"<\|[^|]*\|>", "", text)


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


def cer(ref: str, hyp: str) -> tuple[float, int, int]:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 1.0, 0, 0
    d = edit_distance(r, h)
    return d / len(r), d, len(r)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        assert f.getframerate() == 16000, f.getframerate()
        assert f.getnchannels() == 1, f.getnchannels()
        data = f.readframes(f.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def show_diff(ref: str, hyp: str) -> str:
    """逐字对齐，把错的地方用 [] 标出来。"""
    import difflib

    r, h = normalize(ref), normalize(hyp)
    sm = difflib.SequenceMatcher(None, r, h)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(r[i1:i2])
        elif tag == "replace":
            out.append(f"[{r[i1:i2]}→{h[j1:j2]}]")
        elif tag == "delete":
            out.append(f"[-{r[i1:i2]}]")
        else:
            out.append(f"[+{h[j1:j2]}]")
    return "".join(out)


def main() -> None:
    import sherpa_onnx

    audio = read_wav(WAV)
    dur = len(audio) / 16000
    print(f"音频: {WAV.name}  {dur:.2f}s  {len(audio)} samples\n")

    results: list[tuple[str, str, float]] = []

    # 1) 流式 Paraformer
    import sys as _s

    _s.path.insert(0, str(ROOT))
    from meetscribe.asr.engine import ASRConfig, StreamingASR

    t0 = time.perf_counter()
    engine = StreamingASR(ASRConfig(model_dir=str(ROOT / "models" / "paraformer_zh_en"), num_threads=2))
    engine.stream("t")
    step = 1600
    for i in range(0, len(audio), step):
        engine.feed("t", audio[i : i + step])
    text = engine.flush("t", 1.2)
    el = time.perf_counter() - t0
    results.append(("流式 Paraformer（当前实时字幕）", text, el))

    # 2) 离线 Paraformer
    t0 = time.perf_counter()
    rec = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=str(ROOT / "models" / "paraformer_offline" / "model.int8.onnx"),
        tokens=str(ROOT / "models" / "paraformer_offline" / "tokens.txt"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
    )
    st = rec.create_stream()
    st.accept_waveform(16000, audio)
    rec.decode_stream(st)
    text = st.result.text
    el = time.perf_counter() - t0
    results.append(("离线 Paraformer（大型中文模型）", text, el))

    # 3) SenseVoice
    t0 = time.perf_counter()
    rec2 = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=str(ROOT / "models" / "sense_voice" / "model.int8.onnx"),
        tokens=str(ROOT / "models" / "sense_voice" / "tokens.txt"),
        num_threads=2,
        use_itn=True,
        language="auto",
    )
    st2 = rec2.create_stream()
    st2.accept_waveform(16000, audio)
    rec2.decode_stream(st2)
    text = _tags(st2.result.text)
    el = time.perf_counter() - t0
    results.append(("SenseVoice", text, el))

    print("=" * 78)
    print(f"参考文本：{REF}")
    print("=" * 78)
    for name, hyp, el in results:
        rate, d, n = cer(REF, hyp)
        rtf = el / dur
        print(f"\n【{name}】")
        print(f"  输出：{hyp}")
        print(f"  CER：{rate*100:.1f}%  (错 {d} / 共 {n} 字)   耗时 {el:.2f}s  RTF {rtf:.3f}")
        print(f"  差异：{show_diff(REF, hyp)}")

    print("\n" + "=" * 78)
    print("结论：CER 最低者最适合做后台精修模型")


if __name__ == "__main__":
    main()
