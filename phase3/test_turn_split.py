# -*- coding: utf-8 -*-
"""换人断句回归：A 说完 B 马上接，必须分成两句、标成两个人。

构造方式（不依赖录音）：从多人配音语料里取出两段不同人的语音，
中间只留很短的间隙拼起来，模拟"A 说完 B 立刻接话"。

验证两条防线：
  第一遍：LivePipeline 的"换人间隙提前断句"——间隙够长且有人接话就先定稿上一句
  第二遍：精修时 diarizer.find_switch 检出句内换人 → 拆成两句各标说话人
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.asr.diarize import (  # noqa: E402
    EMB_WIN_SEC, EmbeddingSpeakerRegistry, embedding_model_path,
)
from meetscribe.asr.engine import ASRConfig, StreamingASR  # noqa: E402
from meetscribe.asr.pipeline import Caption, LivePipeline  # noqa: E402

SR = 16000
CORPUS = ROOT / "bench" / "corpus" / "BV1fopY6QEmP.wav"

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))
    print(f"[{'OK ' if ok else 'FAIL'}] {name}　{detail}")


def pick_two_speakers() -> tuple[np.ndarray, np.ndarray]:
    """从语料里挑两段"连续高能、且明显不是同一个人"的语音（各 2.5 秒）。

    必须挑连续高能的段：语料带背景音乐，能量 VAD 在安静段会漂，
    挑到带内部静音的段会先被端点检测切开，就测不出"被并成一句"了。
    """
    audio, sr = sf.read(CORPUS, dtype="float32")
    if sr != SR:
        raise SystemExit(f"语料采样率 {sr} != 16000")
    model = embedding_model_path(ROOT / "models")
    if model is None:
        raise SystemExit("缺声纹模型")
    reg = EmbeddingSpeakerRegistry(model, threshold=0.38, max_speakers=4)

    seg = audio[int(15 * SR):int(180 * SR)]
    items = reg._embs(seg)   # [(起始采样点, 嵌入)]，已跳过静音窗
    if len(items) < 20:
        raise SystemExit("语料窗口太少")

    def unit(e: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(e)) or 1.0
        return e / n

    def cos(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.dot(unit(x), unit(y)))

    # 以每个窗口为种子，向后取"内部彼此都像"的最长连续段（>=5 窗 ≈ 3 秒）。
    # 直接按能量挑会挑到跨人片段（后半个窗变成另一个人），必须按声纹一致性挑。
    runs: list[tuple[int, int]] = []
    for i in range(len(items) - 5):
        n = 1
        j = i + 1
        while j < len(items) and cos(items[j][1], items[i][1]) >= 0.45:
            n += 1
            j += 1
        if n >= 5:
            runs.append((n, i))
    if len(runs) < 2:
        raise SystemExit("找不到足够长的单人连续段")
    runs.sort(reverse=True)

    def slice_of(i: int, n: int) -> np.ndarray:
        s = items[i][0]
        e = items[i + n - 1][0] + int(EMB_WIN_SEC * SR)
        return seg[s:min(e, s + int(3.0 * SR))]

    n_a, i_a = runs[0]
    # 在剩下的段里挑一个跟 A 最不像的
    best = min(runs[1:], key=lambda r: cos(items[r[1]][1], items[i_a][1]))
    n_b, i_b = best
    sim = cos(items[i_b][1], items[i_a][1])
    print(f"  两段余弦相似度={sim:.2f}（阈值 0.38 以上算同一人，越小越不像）")
    if sim > 0.2:
        raise SystemExit(f"两段太像（cos={sim:.2f}），换一段语料再测")
    return slice_of(i_a, n_a), slice_of(i_b, n_b)


def run_pipeline(audio: np.ndarray, gap_cut_ms: float) -> list[Caption]:
    """把音频按 100ms 块推进流水线，返回定稿句。"""
    engine = StreamingASR(ASRConfig(model_dir=ROOT / "models" / "paraformer_zh_en"))
    finals: list[Caption] = []
    pipe = LivePipeline(engine, on_final=finals.append, gap_cut_ms=gap_cut_ms)
    pipe.start()
    block = 1600
    for i in range(0, len(audio), block):
        pipe.push("mic", audio[i:i + block])
        time.sleep(0.004)
    time.sleep(1.2)
    pipe.stop()
    return finals


def test_worker_split() -> None:
    """精修回调层：一句拆成两条记录，第二条插在后面并发信号给界面加一行。"""
    import tempfile

    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from meetscribe.asr.refine import RefineJob
    from meetscribe.ui.worker import CaptureWorker

    app = QApplication.instance() or QApplication([])
    tmp = Path(tempfile.mkdtemp(prefix="ms_split_"))
    worker = CaptureWorker(ROOT / "models" / "paraformer_zh_en", tmp / "data", ROOT / "config")

    split_n = int(2.5 * SR)

    class StubDiarizer:
        def assign_sentence(self, audio):
            return ("说话人A" if len(audio) <= split_n else "说话人B"), 0.0

    worker.diarizer = StubDiarizer()
    worker._records = [{
        "track": "mic", "speaker": "说话人A", "start_ms": 1000, "end_ms": 6000,
        "text": "前半句加后半句的原始文本", "refined": False, "raw": "", "mixed": 0.0,
    }]
    got: list[tuple] = []
    worker.split_added.connect(lambda *a: got.append(a))

    job = RefineJob(track="mic", audio=np.zeros(int(7 * SR), dtype=np.float32),
                    raw_text="", start_ms=1000, end_ms=8000)
    worker._on_refined_split(job, "前半句是这个内容", "后半句是另一个内容", 2500)

    r = worker._records
    check("一条记录拆成两条", len(r) == 2, f"{len(r)} 条")
    if len(r) == 2:
        check("前半句缩短到切点", r[0]["end_ms"] == 3500 and r[0]["speaker"] == "说话人A",
              f"end={r[0]['end_ms']} {r[0]['speaker']}")
        check("后半句另起一句", r[1]["start_ms"] == 3500 and r[1]["end_ms"] == 8000
              and r[1]["speaker"] == "说话人B",
              f"{r[1]['start_ms']}~{r[1]['end_ms']} {r[1]['speaker']}")
        check("界面收到新增行信号", len(got) == 1 and got[0][2] == 3500, f"{got}")


def main() -> int:
    a, b = pick_two_speakers()
    print(f"语料：A {len(a)/SR:.1f}s  B {len(b)/SR:.1f}s")

    # ---------- 第一遍：换人间隙提前断句 ----------
    gap = np.zeros(int(0.35 * SR), dtype=np.float32)   # 350ms 间隙：端点(1.2s)等不到
    combo = np.concatenate([a, gap, b])

    off = run_pipeline(combo, gap_cut_ms=0)
    on = run_pipeline(combo, gap_cut_ms=300)
    print(f"  关闭换人间隙：{len(off)} 句")
    print(f"  开启换人间隙：{len(on)} 句")

    check("关闭时会被并成一句（复现问题）", len(off) == 1, f"{len(off)} 句")
    check("开启后按间隙切成两句", len(on) == 2, f"{len(on)} 句")
    if len(on) == 2:
        d = on[1].start_ms - on[0].end_ms
        check("切点在间隙附近", 0 <= d <= 900, f"两句间隔 {d}ms")
        check("两句都有文本", bool(on[0].text.strip()) and bool(on[1].text.strip()),
              f"{on[0].text[:12]}… / {on[1].text[:12]}…")

    # ---------- 第二遍：精修期检出句内换人 ----------
    model = embedding_model_path(ROOT / "models")
    reg = EmbeddingSpeakerRegistry(model, threshold=0.38, max_speakers=4)
    # 先让注册器认识这两个人
    reg.assign_sentence(a)
    reg.assign_sentence(b)
    check("注册器已认识两人", reg.n_speakers == 2, f"{reg.n_speakers} 人")

    at = reg.find_switch(combo)
    truth_at = len(a) + len(gap)
    check("检出句内换人", at is not None, f"切点={at}")
    if at is not None:
        err_ms = abs(at - truth_at) / SR * 1000
        check("切点误差 <600ms", err_ms < 600, f"误差 {err_ms:.0f}ms")

        s1, _ = reg.assign_sentence(combo[:at])
        s2, _ = reg.assign_sentence(combo[at:])
        check("两段判成不同说话人", s1 != s2, f"{s1} / {s2}")

    # 只有一个人说话时不该乱切
    single = np.concatenate([a, np.zeros(int(0.4 * SR), dtype=np.float32), a[len(a) // 2:]])
    check("同一人说话不误切", reg.find_switch(single) is None, "")

    # ---------- 拆分落到转写记录：一句变两句、各标各的说话人 ----------
    test_worker_split()

    print()
    bad = [n for n, ok, _ in checks if not ok]
    if bad:
        print("❌ 失败项：", bad)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
