# -*- coding: utf-8 -*-
"""本地大模型对比：转写纠错场景。

用生产同一套 Corrector，在同一批"ASR 句 + 字幕真值"上跑各模型，测：
- CER 改善（纠错前 → 纠错后，越小越好）
- 采纳率（模型给出的修改被 _accept 守卫接受的比例）
- 副作用（思考链混入、改坏、空输出、幻觉扩写）
- 速度（每条平均耗时、每批耗时）

结果写 bench/llm_report.json + 打印汇总，供"保留最优模型、卸载劣质模型"决策。
用法：python bench/llm_bench.py [每批条数] [模型1 模型2 ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.asr_bench import (ITEMS, cer, decode, find_audio, load_subs, norm,  # noqa: E402
                             run_stream)
from meetscribe.config import Settings  # noqa: E402
from meetscribe.llm.client import OllamaClient  # noqa: E402
from meetscribe.llm.correct import Corrector, FixRequest  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "llm_report.json"
SETS = HERE / "llm_set.json"   # 测试集缓存，保证各模型输入完全一致

THINK_MARKS = ["让我想", "我需要", "我先看", "嗯，用户", "用户给", "思考", "<think",
               "我来分析", "首先，我", "我需要先", "接下来我", "总结一下"]


def build_set(win: float = 120.0, start_off: float = 60.0, per_item: int = 40) -> list[dict]:
    """跑流式识别，和字幕真值配对，取有错且真值够长的句子做纠错测试集。"""
    if SETS.exists():
        return json.loads(SETS.read_text(encoding="utf-8"))
    settings = Settings()
    out: list[dict] = []
    for item in ITEMS:
        vid = item["id"]
        afile = find_audio(vid)
        subs = load_subs(vid)
        if afile is None or not subs:
            continue
        audio = decode(afile)
        seg = audio[int(start_off * 16000):int(min(len(audio), (start_off + win) * 16000))]
        sents, _ = run_stream(seg, settings, rule2=0.6)
        base = start_off * 1000
        for st in sents:
            a0, a1 = st["start_ms"] + base, st["end_ms"] + base
            over = [s["text"] for s in subs if s["start_ms"] < a1 and s["end_ms"] > a0]
            if not over:
                continue
            ref = "".join(over)
            if len(norm(ref)) < 8:
                continue
            c = cer(norm(ref), norm(st["text"]))
            if c < 0.03 or c > 0.8:      # 太准的没得改，太烂的（对齐问题）不算
                continue
            out.append({"corpus": item["name"], "key": f"{vid}:{st['start_ms']}",
                        "asr": st["text"], "truth": ref, "cer_before": round(c, 4)})
    # 每个语料取前 per_item 条，保证分布均衡
    picked: list[dict] = []
    for item in ITEMS:
        got = [x for x in out if x["corpus"] == item["name"]][:per_item]
        picked.extend(got)
    SETS.write_text(json.dumps(picked, ensure_ascii=False, indent=1), encoding="utf-8")
    return picked


def main() -> int:
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    models = sys.argv[2:] or ["qwen2.5:7b-instruct", "qwen2.5:3b-instruct", "qwen3:8b",
                              "qwen3:4b", "gemma4:12b"]
    data = build_set()
    print(f"测试集 {len(data)} 条，平均初始 CER={np.mean([d['cer_before'] for d in data]):.3f}", flush=True)

    client = OllamaClient()
    if not client.alive():
        print("Ollama 未运行，尝试拉起…")
        if not client.ensure_server(wait=20):
            print("[FAIL] 起不来 Ollama")
            return 1
    installed = client.models()
    results = []

    for model in models:
        if model not in installed:
            print(f"[SKIP] {model} 未安装")
            continue
        cor = Corrector(client, model)
        t0 = time.perf_counter()
        fixed: dict = {}
        batches = 0
        for i in range(0, len(data), batch):
            chunk = data[i:i + batch]
            items = [FixRequest((d["key"].split(":")[0], int(d["key"].split(":")[1])), d["asr"])
                     for d in chunk]
            try:
                got = cor.fix(items)
            except Exception as exc:  # noqa: BLE001
                got = {}
                print(f"  批次异常：{exc!r}")
            for (trk, start), text in got.items():
                fixed[f"{trk}:{start}"] = text
            batches += 1
        total = time.perf_counter() - t0

        cer_before, cer_after = [], []
        improved = worsened = unchanged = 0
        think_hits = 0
        for d in data:
            cb = d["cer_before"]
            new = fixed.get(d["key"])
            if new is None:
                unchanged += 1
                cer_before.append(cb)
                cer_after.append(cb)
                continue
            ca = cer(norm(d["truth"]), norm(new))
            cer_before.append(cb)
            cer_after.append(ca)
            if ca < cb - 0.005:
                improved += 1
            elif ca > cb + 0.005:
                worsened += 1
            else:
                unchanged += 1
            if any(m in new for m in THINK_MARKS):
                think_hits += 1

        row = {
            "model": model,
            "n": len(data),
            "cer_before": round(float(np.mean(cer_before)), 4),
            "cer_after": round(float(np.mean(cer_after)), 4),
            "gain": round(float(np.mean(cer_before) - np.mean(cer_after)), 4),
            "fixed": len(fixed),
            "fix_rate": round(len(fixed) / max(1, len(data)), 3),
            "improved": improved, "worsened": worsened, "unchanged": unchanged,
            "think_pollution": think_hits,
            "seconds_total": round(total, 1),
            "sec_per_item": round(total / max(1, len(data)), 2),
            "batches": batches,
        }
        results.append(row)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {model:<22} CER {row['cer_before']:.3f}→{row['cer_after']:.3f} "
              f"(改善 {row['gain']:+.3f}) 采纳 {row['fixed']}/{row['n']} "
              f"改好 {improved} 改坏 {worsened} 思考污染 {think_hits} "
              f"{row['sec_per_item']}s/条", flush=True)

    print("\n===== 排序（按 CER 改善降序、速度升序） =====")
    for r in sorted(results, key=lambda x: (-x["gain"], x["sec_per_item"])):
        print(f"{r['model']:<22} 改善 {r['gain']:+.3f}  采纳率 {r['fix_rate']:.2f}  "
              f"改坏 {r['worsened']:<3} 思考污染 {r['think_pollution']:<3} "
              f"{r['sec_per_item']:>5}s/条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
