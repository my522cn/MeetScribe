# -*- coding: utf-8 -*-
"""纪要（summary）任务模型对比：结构完整度、思考泄漏、覆盖度、耗时。

用一份真实转写稿跑各模型，指标：
- 是否含 6 个固定标题（结论/待办等）
- 思考链泄漏（"我需要""让我想"等）与啰嗦度
- 覆盖度：原文高频实词出现在纪要中的比例
- 耗时与长度适中性
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meetscribe.llm.client import OllamaClient  # noqa: E402
from meetscribe.llm.summary import summarize  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "summary_report.json"

THINK_MARKS = ["让我想", "我需要", "我先看", "嗯，用户", "用户给", "我来分析",
               "首先，我", "作为AI", "作为一个", "好的，我"]
_HEAD = re.compile(r"^#+\s*(\S+)", re.M)
_STOP = set("的了是我在有和就不也都很要会说着没有这那个我们你们他们吧啊呢吗哦嗯对然后就是可以因为所以但是".split())


def load_transcript() -> tuple[str, str]:
    """优先用 bench/transcript.txt（长转写稿），否则找会话里最长的那份。"""
    tp = HERE / "transcript.txt"
    if tp.exists():
        return "产品访谈：创作焦虑与即时反馈", tp.read_text(encoding="utf-8")
    """找一份真实转写稿（优先 release 会话里最长的那份）。"""
    best, title, n = "", "会议记录", 0
    for base in (ROOT / "release/MeetScribe/data/sessions", ROOT / "data/sessions"):
        if not base.exists():
            continue
        for d in sorted(base.iterdir()):
            f = d / "transcript.jsonl"
            if not f.exists():
                continue
            rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            if len(rows) > n:
                n = len(rows)
                best = "\n".join(f"{r.get('speaker','')}：{r.get('text','')}" for r in rows)
                title = (d / "session.json")
                title = json.loads(title.read_text(encoding="utf-8")).get("title", "会议记录") \
                    if title.exists() else "会议记录"
    return title, best


def coverage(transcript: str, summary: str) -> float:
    words = [w for w in re.findall(r"[\u4e00-\u9fa5]{2,4}", transcript) if w not in _STOP]
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top = [w for w, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:30]]
    if not top:
        return 0.0
    return sum(1 for w in top if w in summary) / len(top)


def main() -> int:
    models = sys.argv[1:] or ["qwen2.5:7b-instruct", "qwen2.5:3b-instruct", "qwen3:8b", "gemma4:12b"]
    title, transcript = load_transcript()
    if not transcript:
        print("[SKIP] 没有真实转写稿")
        return 0
    # 控制长度：取前 4000 字，避免跑太久
    transcript = transcript[:4000]
    print(f"（用前 {len(transcript)} 字）", flush=True)
    print(f"转写稿 {len(transcript)} 字（{title}）", flush=True)

    client = OllamaClient(timeout=600)
    if not client.alive() and not client.ensure_server(wait=20):
        print("[FAIL] Ollama 不可用")
        return 1
    installed = client.models()
    results = []
    for m in models:
        if m not in installed:
            print(f"[SKIP] {m} 未安装")
            continue
        t0 = time.perf_counter()
        try:
            text = summarize(client, m, title, transcript)
        except Exception as exc:  # noqa: BLE001
            print(f"  {m}: 失败 {exc!r}")
            continue
        secs = time.perf_counter() - t0
        heads = _HEAD.findall(text)
        row = {
            "model": m, "seconds": round(secs, 1), "chars": len(text),
            "headings": len(set(heads)), "heading_names": list(dict.fromkeys(heads))[:8],
            "think_leak": sum(1 for k in THINK_MARKS if k in text),
            "coverage": round(coverage(transcript, text), 3),
            "preview": text[:180],
        }
        results.append(row)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {m:<22} {secs:>5.1f}s 长度 {len(text):>4} 标题 {row['headings']} "
              f"思考泄漏 {row['think_leak']} 覆盖 {row['coverage']:.2f}", flush=True)

    print("\n===== 纪要排序（覆盖度降序、耗时升序） =====")
    for r in sorted(results, key=lambda x: (-x["coverage"], x["seconds"])):
        print(f"{r['model']:<22} 覆盖 {r['coverage']:.2f} 标题 {r['headings']} "
              f"泄漏 {r['think_leak']} {r['seconds']:.0f}s {r['chars']}字")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
