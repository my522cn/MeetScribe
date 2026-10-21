# -*- coding: utf-8 -*-
"""转写质量基准：网络视频语料 + YouTube 字幕真值。

对比四条识别路径：
  stream  流式 paraformer（生产实时字幕那一路）
  offline 离线 paraformer 整段精修
  two_pass 流式切句 + 离线逐句重识别（生产"两遍识别"）
  sense   离线 sense_voice 整段

指标：内容 CER（去标点空格后编辑距离）、含标点 CER、句数、RTF、耗时。
用法：python bench/asr_bench.py [窗口秒数] [起始秒]
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import soundfile as sf  # noqa: E402

from meetscribe.asr.engine import ASRConfig, StreamingASR  # noqa: E402
from meetscribe.asr.offline import get_offline  # noqa: E402
from meetscribe.asr.pipeline import LivePipeline  # noqa: E402
from meetscribe.config import Settings  # noqa: E402

SR = 16000
HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
OUT = HERE / "report.json"

ITEMS = [
    {"id": "K2vBHUUqWoI", "name": "访谈（双人对话）"},
    {"id": "Rmp-1j7i9Ek", "name": "技术演讲（中英混读）"},
    {"id": "BV1fopY6QEmP", "name": "多人配音（带BGM）"},
]

_PUNCT = re.compile(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()\[\]【】…—\-~]+")


def norm(text: str) -> str:
    """去标点空白，只比内容（避免标点策略差异淹没内容准确率）。"""
    return _PUNCT.sub("", text)


def cer(ref: str, hyp: str) -> float:
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


def find_audio(vid: str) -> Path | None:
    for ext in ("m4a", "webm", "opus", "mp4", "wav"):
        p = CORPUS / f"{vid}.{ext}"
        if p.exists():
            return p
    return None


def find_subs(vid: str) -> Path | None:
    cands = sorted(CORPUS.glob(f"{vid}*")) + sorted(CORPUS.glob(f"{vid}.*"))
    zh = [p for p in cands if p.suffix in (".srt", ".vtt") and ("zh" in p.name)]
    if not zh:
        zh = [p for p in cands if p.suffix in (".srt", ".vtt")]
    # 简体优先
    for p in zh:
        if "zh-Hans" in p.name:
            return p
    return zh[0] if zh else None


def load_subs(vid: str) -> list[dict]:
    """字幕真值：优先 .json（B站 OCR 结果），其次 srt/vtt。"""
    j = CORPUS / f"{vid}.json"
    if j.exists():
        return json.loads(j.read_text(encoding="utf-8"))
    p = find_subs(vid)
    return parse_subs(p) if p else []


def to_simplified(text: str) -> str:
    """繁体字幕转简体（ASR 输出是简体，不转会把 CER 全吃掉）。"""
    try:
        from opencc import OpenCC

        global _CC
        if "_CC" not in globals():
            globals()["_CC"] = OpenCC("t2s")
        return globals()["_CC"].convert(text)
    except Exception:  # noqa: BLE001
        return text


def parse_subs(path: Path) -> list[dict]:
    """解析 srt / vtt 为 [{start_ms,end_ms,text}]，合并重复行。"""
    raw = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", raw.replace("\r\n", "\n"))
    out: list[dict] = []
    for b in blocks:
        m = re.search(
            r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})", b)
        if not m:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(x) for x in m.groups())
        start = ((h1 * 60 + m1) * 60 + s1) * 1000 + (ms1 if ms1 > 99 else ms1 * 10)
        end = ((h2 * 60 + m2) * 60 + s2) * 1000 + (ms2 if ms2 > 99 else ms2 * 10)
        lines = [l for l in b.split("\n")[1:] if l.strip() and "-->" not in l]
        text = to_simplified(re.sub(r"<[^>]+>", "", " ".join(lines)).strip())
        if text:
            out.append({"start_ms": start, "end_ms": end, "text": text})
    return out


def decode(path: Path) -> np.ndarray:
    """用 PyAV 解码到 16k 单声道 float32（本机没有 ffmpeg）。"""
    wav = path.with_suffix(".wav")
    if wav.exists():
        x, sr = sf.read(str(wav), dtype="float32")
        if sr == SR:
            return x if x.ndim == 1 else x.mean(axis=1)
    import av

    with av.open(str(path)) as c:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
        chunks = []
        for frame in c.decode(audio=0):
            for out_frame in resampler.resample(frame):
                chunks.append(np.asarray(out_frame.to_ndarray()).ravel())
        chunks += [np.asarray(f.to_ndarray()).ravel() for f in resampler.resample(None) or []]
    x = np.concatenate(chunks).astype(np.float32).ravel() / 32768.0
    sf.write(str(wav), x, SR)
    return x


def make_post(use_hotwords: bool = True):
    """生产同一套后处理：精确替换 + 热词同音纠正 + 标点。"""
    from meetscribe.text.postprocess import TextPostProcessor

    cfg = ROOT / "config"
    if not use_hotwords:
        # 用空热词表做 A/B（其余一致）
        empty = HERE / "corpus" / "_empty_hotwords.txt"
        empty.write_text("", encoding="utf-8")
        hw = empty
    else:
        hw = cfg / "hotwords.txt"
    return TextPostProcessor(
        hotwords_file=hw,
        replacements_file=cfg / "replacements.txt",
        punct_model=ROOT / "models" / "punct_zh_en" / "model.int8.onnx",
    )


def run_stream(audio: np.ndarray, settings: Settings, rule2: float) -> tuple[list[dict], float]:
    cfg = ASRConfig(
        model_dir=ROOT / "models" / "paraformer_zh_en",
        num_threads=int(settings.get("asr.num_threads", 4)),
        rule2_min_trailing_silence=rule2,
        rule1_min_trailing_silence=max(rule2 * 2, 1.2),
    )
    engine = StreamingASR(cfg)
    sents: list[dict] = []
    pipe = LivePipeline(engine, on_final=lambda cap: sents.append(
        {"start_ms": cap.start_ms, "end_ms": cap.end_ms, "text": cap.text.strip()})
        if cap.text.strip() else None)
    pipe.start()
    block = SR // 10
    t0 = time.perf_counter()
    for i in range(0, len(audio), block):
        while pipe._queues["mic"].qsize() > 100:
            time.sleep(0.01)
        pipe.push("mic", audio[i:i + block])
    deadline = time.time() + 60
    while time.time() < deadline and pipe._queues["mic"].qsize() > 0:
        time.sleep(0.05)
    time.sleep(2.0)
    pipe.stop()
    dt = time.perf_counter() - t0
    return sents, dt


def run_offline(audio: np.ndarray, kind: str, threads: int = 4) -> tuple[str, float]:
    from meetscribe.asr.offline import resolve_model_dir

    model_dir, real_kind = resolve_model_dir(ROOT / "models", prefer=kind)
    asr = get_offline(model_dir, kind=real_kind, num_threads=threads)
    t0 = time.perf_counter()
    text = asr.transcribe(audio)
    dt = time.perf_counter() - t0
    return text, dt


CAT_DIGIT = re.compile(r"\d")
CAT_LATIN = re.compile(r"[A-Za-z]{2,}")


def collect_pairs(sents: list[dict], subs: list[dict], base_ms: float = 0.0) -> list[dict]:
    """逐句明细：真值、识别、CER、类别标记（数字/英文/长句/短句）。"""
    out = []
    for st in sents:
        a0, a1 = st["start_ms"] + base_ms, st["end_ms"] + base_ms
        over = [sb["text"] for sb in subs if sb["start_ms"] < a1 and sb["end_ms"] > a0]
        if not over:
            continue
        ref = "".join(over)
        hyp = st["text"]
        item = {
            "ref": ref, "hyp": hyp,
            "cer": cer(norm(ref), norm(hyp)),
            "digit": bool(CAT_DIGIT.search(ref)),
            "latin": bool(CAT_LATIN.search(ref)),
            "long": len(norm(ref)) >= 25,
            "short": len(norm(ref)) <= 10,
            "has_end_punct": bool(re.search(r"[。？！?!]$", hyp.strip())),
        }
        out.append(item)
    return out


def align_cer(sents: list[dict], subs: list[dict], base_ms: float = 0.0) -> tuple[float, float, int]:
    """按时间重叠把 ASR 句与字幕真值配对：返回 (内容CER, 含标点CER, 配对数)。

    ASR 句的时间戳是相对窗口起点的，字幕是整片绝对时间，base_ms 用来对齐两者。
    """
    cers, raws = [], []
    for st in sents:
        st = {**st, "start_ms": st["start_ms"] + base_ms, "end_ms": st["end_ms"] + base_ms}
        over = [sb["text"] for sb in subs
                if sb["start_ms"] < st["end_ms"] and sb["end_ms"] > st["start_ms"]]
        if not over:
            continue
        ref = "".join(over)
        cers.append(cer(norm(ref), norm(st["text"])))
        raws.append(cer(ref.replace(" ", ""), st["text"].replace(" ", "")))
    if not cers:
        return 1.0, 1.0, 0
    return float(np.mean(cers)), float(np.mean(raws)), len(cers)


def main() -> int:
    win = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
    start_off = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    only = None
    append = False
    for arg in sys.argv[3:]:
        if arg.startswith("--only="):
            only = arg.split("=", 1)[1]
        if arg == "--append":
            append = True
    settings = Settings()
    results: list[dict] = []
    if append and OUT.exists():
        results = json.loads(OUT.read_text(encoding="utf-8"))
        results = [r for r in results if r["id"] != only]

    for item in ITEMS:
        vid = item["id"]
        if only and vid != only:
            continue
        afile = find_audio(vid)
        subs_all = load_subs(vid)
        if afile is None or not subs_all:
            print(f"[SKIP] {item['name']} 缺素材 audio={afile} 字幕={len(subs_all)} 行")
            continue
        audio = decode(afile)
        s0 = int(start_off * SR)
        s1 = int(min(len(audio), (start_off + win) * SR))
        seg = audio[s0:s1]
        subs = [s for s in load_subs(vid) if s["end_ms"] > start_off * 1000
                and s["start_ms"] < (start_off + win) * 1000]
        if len(seg) < SR * 30 or not subs:
            print(f"[SKIP] {item['name']} 窗口内素材不足")
            continue

        truth = norm("".join(s["text"] for s in subs))
        print(f"\n===== {item['name']}（{vid}）窗口 {start_off:.0f}~{start_off + win:.0f}s "
              f"真值 {len(truth)} 字 =====", flush=True)

        row = {"id": vid, "name": item["name"], "chars": len(truth), "sub_lines": len(subs)}

        # 1) 流式（生产默认断句 1.2s）
        sents, dt = run_stream(seg, settings, rule2=1.2)
        c_content, c_raw, n_pair = align_cer(sents, subs, start_off * 1000)
        row["stream_default"] = {"cer": c_content, "cer_raw": c_raw, "sents": len(sents),
                                 "pairs": n_pair, "rtf": dt / (len(seg) / SR)}
        print(f"  stream(1.2s断句) CER={c_content:.3f} 含标点CER={c_raw:.3f} "
              f"句数={len(sents)} RTF={row['stream_default']['rtf']:.3f}", flush=True)

        # 2) 流式（短断句 0.6s：快节奏/抢话场景）
        sents2, dt2 = run_stream(seg, settings, rule2=0.6)
        c2, c2r, n2 = align_cer(sents2, subs, start_off * 1000)
        row["stream_short"] = {"cer": c2, "cer_raw": c2r, "sents": len(sents2),
                               "pairs": n2, "rtf": dt2 / (len(seg) / SR)}
        print(f"  stream(0.6s断句) CER={c2:.3f} 含标点CER={c2r:.3f} "
              f"句数={len(sents2)} RTF={row['stream_short']['rtf']:.3f}", flush=True)

        # 2.5) 生产链路：流式句 + postprocess（热词开/关 A/B）
        try:
            for flag, tag in ((True, "post_hw"), (False, "post_nohw")):
                post = make_post(flag)
                psents = [{"start_ms": st["start_ms"], "end_ms": st["end_ms"],
                           "text": post.process(st["text"])} for st in sents]
                pc, _, pn = align_cer(psents, subs, start_off * 1000)
                row[tag] = {"cer": pc, "pairs": pn}
                if flag:
                    row["_detail"] = collect_pairs(psents, subs, start_off * 1000)
                    row["_punct_rate"] = round(
                        sum(1 for d in row["_detail"] if d["has_end_punct"]) / max(1, len(row["_detail"])), 3)
            print(f"  stream+后处理   热词开 CER={row['post_hw']['cer']:.3f} "
                  f"热词关 CER={row['post_nohw']['cer']:.3f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  stream+后处理 失败：{exc!r}")

        # 3) 离线整段（paraformer）
        try:
            txt, dt3 = run_offline(seg, "paraformer")
            row["offline_paraformer"] = {"cer": cer(truth, norm(txt)), "rtf": dt3 / (len(seg) / SR),
                                        "chars_out": len(norm(txt))}
            print(f"  offline整段     CER={row['offline_paraformer']['cer']:.3f} "
                  f"RTF={row['offline_paraformer']['rtf']:.3f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  offline整段 失败：{exc!r}")

        # 4) 两遍：流式句 + 离线逐句重识别
        try:
            from meetscribe.asr.offline import resolve_model_dir

            model_dir, kind = resolve_model_dir(ROOT / "models", prefer="paraformer")
            asr = get_offline(model_dir, kind=kind, num_threads=int(settings.get("asr.refine_threads", 4)))
            t0 = time.perf_counter()
            fixed = [asr.transcribe(seg[int(s["start_ms"]) * 16:int(s["end_ms"]) * 16])
                     for s in sents]
            dt4 = time.perf_counter() - t0
            pairs4 = []
            for st, fx in zip(sents, fixed):
                a0, a1 = st["start_ms"] + start_off * 1000, st["end_ms"] + start_off * 1000
                over = [sb["text"] for sb in subs
                        if sb["start_ms"] < a1 and sb["end_ms"] > a0]
                if over:
                    pairs4.append(cer(norm("".join(over)), norm(fx)))
            row["two_pass"] = {"cer": float(np.mean(pairs4)) if pairs4 else 1.0,
                               "rtf": (dt + dt4) / (len(seg) / SR), "pairs": len(pairs4)}
            print(f"  两遍识别       CER={row['two_pass']['cer']:.3f} "
                  f"RTF={row['two_pass']['rtf']:.3f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  两遍识别 失败：{exc!r}")

        # 5) sense_voice 整段
        try:
            txt5, dt5 = run_offline(seg, "sense_voice")
            row["offline_sense"] = {"cer": cer(truth, norm(txt5)), "rtf": dt5 / (len(seg) / SR)}
            print(f"  sense_voice整段 CER={row['offline_sense']['cer']:.3f} "
                  f"RTF={row['offline_sense']['rtf']:.3f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  sense_voice 失败：{exc!r}")

        results.append(row)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    # 分类细粒度汇总
    cats: dict[str, list[float]] = {}
    punct_rates = []
    for r in results:
        for d in r.get("_detail", []):
            for key, hit in (("含数字", d["digit"]), ("含英文", d["latin"]),
                             ("长句(≥25字)", d["long"]), ("短句(≤10字)", d["short"]), ("全部", True)):
                if hit:
                    cats.setdefault(key, []).append(d["cer"])
        if "_punct_rate" in r:
            punct_rates.append(r["_punct_rate"])
    print("\n===== 分类 CER（生产链路：流式+后处理） =====")
    cat_summary = {}
    for k, v in cats.items():
        if v:
            cat_summary[k] = {"n": len(v), "cer": round(float(np.mean(v)), 4)}
            print(f"  {k:<12} n={len(v):<4} CER={np.mean(v):.3f}")
    if punct_rates:
        print(f"  句末标点覆盖率 {np.mean(punct_rates):.2f}")
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n===== 汇总 =====")
    for r in results:
        print(f"{r['name']:>12} | " + "  ".join(
            f"{k}={v['cer']:.3f}" for k, v in r.items() if isinstance(v, dict) and "cer" in v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
