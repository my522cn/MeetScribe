"""构建声纹评测语料：B 站视频 → 音频 + OCR 硬字幕 + 自家管线切句。

产物（本目录）：
- audio.wav        16k 单声道
- subs.json        OCR 出的字幕行 [{start_ms,end_ms,text}]
- asr_sents.json   自家流式管线切出的句子 [{start_ms,end_ms,text,audio?}]
- report.txt       对齐与校对摘要

用法：../../.venv/Scripts/python.exe build_corpus.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

SR = 16000


# ---------- 1. 音频解码 ----------
def decode_audio() -> np.ndarray:
    import av

    src = HERE / "video.m4a"
    out = HERE / "audio.wav"
    if out.exists():
        import soundfile as sf

        audio, sr = sf.read(out, dtype="float32")
        assert sr == SR
        return audio

    resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
    chunks: list[np.ndarray] = []
    with av.open(str(src)) as container:
        for frame in container.decode(audio=0):
            for rf in resampler.resample(frame):
                arr = rf.to_ndarray()
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)
                chunks.append(arr.astype(np.float32) / 32768.0)
    audio = np.concatenate(chunks)
    import soundfile as sf

    sf.write(out, audio, SR)
    print(f"音频：{len(audio) / SR:.1f}s，峰值 {np.abs(audio).max():.3f}，RMS {np.sqrt((audio**2).mean()):.4f}")
    return audio


# ---------- 2. OCR 硬字幕 ----------
def ocr_subs() -> list[dict]:
    out = HERE / "subs.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))

    import av
    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()
    rows: list[tuple[float, str]] = []  # (t_sec, text)
    with av.open(str(HERE / "vid.mp4")) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        step = 0.25  # 每 0.25s 取一帧
        last_t = -1.0
        for frame in container.decode(video=0):
            t = float(frame.pts * tb)
            if t - last_t < step:
                continue
            last_t = t
            img = frame.to_ndarray(format="rgb24")
            h, w = img.shape[:2]
            strip = img[int(h * 0.78):, :]  # 底部字幕带
            result, _ = ocr(strip)
            if not result:
                continue
            text = "".join(r[1] for r in result).strip()
            if text:
                rows.append((t, text))

    # 相邻相同文本合并成行 [{start_ms,end_ms,text}]
    lines: list[dict] = []
    for t, text in rows:
        text = text.replace(" ", "")
        if lines and text == lines[-1]["text"] and (t - lines[-1]["_last"]) < 1.2:
            lines[-1]["end_ms"] = int((t + 0.25) * 1000)
            lines[-1]["_last"] = t
        else:
            lines.append({"start_ms": int(t * 1000), "end_ms": int((t + 0.25) * 1000),
                          "text": text, "_last": t})
    for ln in lines:
        ln.pop("_last")
    out.write_text(json.dumps(lines, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"OCR 字幕行：{len(lines)}")
    return lines


# ---------- 3. 自家管线切句 ----------
def asr_sentences() -> list[dict]:
    out = HERE / "asr_sents.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))

    import queue

    from meetscribe.asr.engine import ASRConfig, StreamingASR
    from meetscribe.asr.pipeline import LivePipeline
    from meetscribe.config import Settings

    import soundfile as sf

    audio, sr = sf.read(HERE / "audio.wav", dtype="float32")
    assert sr == SR

    s = Settings()
    engine = StreamingASR(ASRConfig(
        model_dir=ROOT / "models" / "paraformer_zh_en",
        num_threads=int(s.get("asr.num_threads", 2)),
        # 快节奏对话没有 1.2s 的空隙，用更短的断句静音（评测口径，生产仍用配置值）
        rule2_min_trailing_silence=0.6,
        rule1_min_trailing_silence=1.2,
    ))
    sents: list[dict] = []

    def on_final(cap):
        if cap.text.strip():
            sents.append({"start_ms": cap.start_ms, "end_ms": cap.end_ms, "text": cap.text.strip()})

    pipe = LivePipeline(engine, on_final=on_final)
    pipe.start()
    block = SR // 10
    import time

    for i in range(0, len(audio), block):
        blk = audio[i:i + block]
        # 背压：队列快满就等，否则 173s 音频 2 秒灌完，队列溢出丢掉绝大多数音频
        while pipe._queues["mic"].qsize() > 100:
            time.sleep(0.01)
        pipe.push("mic", blk)
    # 等队列清空、尾部半句走完端点
    deadline = time.time() + 30
    while time.time() < deadline and pipe._queues["mic"].qsize() > 0:
        time.sleep(0.05)
    time.sleep(2.0)
    pipe.stop()
    out.write_text(json.dumps(sents, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ASR 句子：{len(sents)}")
    return sents


# ---------- 4. 对齐校对 ----------
def cer(ref: str, hyp: str) -> float:
    # 编辑距离（滚动数组，最终距离为 dp[n]）
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return dp[n] / max(1, m)


def align_report(subs: list[dict], sents: list[dict]) -> str:
    lines = []
    lines.append(f"字幕行 {len(subs)}，ASR 句 {len(sents)}")
    # 简单贪心对齐：按时间重叠取字幕
    total_cer, n_pair = 0.0, 0
    for st in sents:
        mid = (st["start_ms"] + st["end_ms"]) / 2
        over = [sb["text"] for sb in subs
                if sb["start_ms"] - 400 <= (st["start_ms"] + st["end_ms"]) / 2 <= sb["end_ms"] + 800
                or sb["start_ms"] < st["end_ms"] and sb["end_ms"] > st["start_ms"]]
        if over:
            best_text = "".join(over)
            c = cer(best_text, st["text"])
            total_cer += c
            n_pair += 1
            lines.append(f"  [{st['start_ms']:>7}ms] ASR「{st['text']}」 vs 字幕「{best_text}」 CER={c:.2f}")
    if n_pair:
        lines.append(f"对齐 {n_pair} 对，平均 CER = {total_cer / n_pair:.3f}")
    # 句长分布（短句多 = 间隔短/抢话的体现）
    durs = sorted(st["end_ms"] - st["start_ms"] for st in sents)
    if durs:
        lines.append(f"句长 ms：min={durs[0]} 中位={durs[len(durs)//2]} max={durs[-1]}")
    return "\n".join(lines)


def main() -> int:
    audio = decode_audio()
    subs = ocr_subs()
    sents = asr_sentences()
    report = align_report(subs, sents)
    (HERE / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
