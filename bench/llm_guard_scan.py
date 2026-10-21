# -*- coding: utf-8 -*-
"""纠错"接受守卫"阈值扫描：一次取回模型原始输出，离线扫不同门槛。

目的：找出能让纠错从"净负收益"变成"零伤害或正收益"的门槛组合。
门槛维度：字符相似度下限、允许的最大改动比例。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.asr_bench import cer, norm  # noqa: E402
from meetscribe.llm.client import OllamaClient  # noqa: E402
from meetscribe.llm.correct import Corrector, FixRequest, _similar, _norm  # noqa: E402

HERE = Path(__file__).parent
RAW = HERE / "llm_raw.json"


def fetch(model: str, data: list[dict], batch: int = 3) -> dict[str, str]:
    """跑一次模型，收集"未经 _accept 过滤"的候选修正。"""
    client = OllamaClient(timeout=180)
    cor = Corrector(client, model, batch=batch, think_off=True)
    raw: dict[str, str] = {}
    for i in range(0, len(data), batch):
        chunk = data[i:i + batch]
        items = [FixRequest((d["key"].split(":")[0], int(d["key"].split(":")[1])), d["asr"])
                 for d in chunk]
        lines = [f"{j+1}|{it.text}" for j, it in enumerate(items)]
        try:
            res = client.chat(model, Corrector.__init__.__doc__ or "", "")  # placeholder, replaced below
        except Exception:  # noqa: BLE001
            pass
        # 直接用 Corrector 的提示词与解析，但跳过 _accept
        try:
            from meetscribe.llm.correct import SYSTEM, NO_THINK

            res = client.chat(model, SYSTEM, NO_THINK + "待校稿句子：\n" + "\n".join(lines),
                              temperature=0.0, num_predict=64 * len(chunk) + 128)
        except Exception as exc:  # noqa: BLE001
            print("  请求失败", exc)
            continue
        got = cor._parse(res.text, items)
        for j, body in got.items():
            old = items[j].text
            if body != old:
                raw[f"{chunk[j]['key']}"] = body
    return raw


def evaluate(data: list[dict], raw: dict[str, str], sim_min: float, max_change: float) -> dict:
    before, after = [], []
    improved = worsened = adopted = 0
    for d in data:
        cb = cer(norm(d["truth"]), norm(d["asr"]))
        new = raw.get(d["key"])
        use = None
        if new:
            a, b = _norm(d["asr"]), _norm(new)
            change = abs(len(b) - len(a)) / max(1, len(a))
            if _similar(d["asr"], new) >= sim_min and change <= max_change and \
                    len(b) >= 0.7 * len(a) and len(b) <= 1.3 * len(a):
                use = new
        ca = cer(norm(d["truth"]), norm(use)) if use else cb
        before.append(cb)
        after.append(ca)
        if use:
            adopted += 1
            if ca < cb - 0.005:
                improved += 1
            elif ca > cb + 0.005:
                worsened += 1
    return {"sim_min": sim_min, "max_change": max_change, "adopted": adopted,
            "improved": improved, "worsened": worsened,
            "gain": round(float(np.mean(before) - np.mean(after)), 4)}


def main() -> int:
    data = json.loads((HERE / "llm_set.json").read_text(encoding="utf-8"))
    models = sys.argv[1:] or ["qwen2.5:7b-instruct", "qwen2.5:3b-instruct"]
    all_raw: dict[str, dict] = {}
    for m in models:
        if RAW.exists():
            all_raw = json.loads(RAW.read_text(encoding="utf-8"))
        if m in all_raw:
            print(f"[缓存] {m} {len(all_raw[m])} 条候选")
            continue
        print(f"[取回] {m} …", flush=True)
        all_raw[m] = fetch(m, data)
        RAW.write_text(json.dumps(all_raw, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {len(all_raw[m])} 条候选修正")

    print("\n===== 门槛扫描 =====")
    best = None
    for m in models:
        raw = all_raw.get(m, {})
        print(f"\n{m}（候选 {len(raw)} 条）")
        for sim in (0.75, 0.82, 0.88, 0.93, 0.97):
            for ch in (0.30, 0.15, 0.08):
                r = evaluate(data, raw, sim, ch)
                flag = ""
                if r["gain"] > 0 and r["worsened"] == 0:
                    flag = "  ★"
                print(f"  相似度≥{sim:.2f} 改动≤{ch:.2f}  采纳 {r['adopted']:>2} "
                      f"改好 {r['improved']} 改坏 {r['worsened']} 改善 {r['gain']:+.4f}{flag}")
                if best is None or (r["gain"] > best[0]["gain"]):
                    best = (r, m)
    if best:
        print(f"\n最佳：{best[1]} 相似度≥{best[0]['sim_min']} 改动≤{best[0]['max_change']} "
              f"改善 {best[0]['gain']:+.4f}（改好 {best[0]['improved']} 改坏 {best[0]['worsened']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
