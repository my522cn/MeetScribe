# -*- coding: utf-8 -*-
"""LLM 纠错提示词/格式 A/B：找出能让本地模型真出活的方案。

变体：
  A 现行（批量8，输出 编号|句子）
  B 小批量（批量3）
  C JSON（批量3，输出 {"1":"...","2":"..."}）
  D 小批量 + <arg_key:6124c78e> 指令（针对 qwen3 思考污染）

指标：采纳率（解析出可用修正的比例）、CER 改善、改坏数、重复退化次数、耗时。
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

from bench.asr_bench import cer, norm  # noqa: E402
from meetscribe.llm.client import OllamaClient  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "llm_ab.json"

SYSTEM_A = None  # 用生产 SYSTEM（从模块读）
SYS_JSON = (
    "你是中文转写校对助手。只修正明显听错的字词，禁止润色、改写、增删内容；"
    "不确定的一律不动。你必须只输出一个 JSON 对象，键是句子编号（字符串），值是改好的句子，"
    "不要输出任何解释、不要 markdown 代码块。"
)


def system_text() -> str:
    from meetscribe.llm.correct import SYSTEM

    return SYSTEM


def build_user(items: list[str], variant: str) -> str:
    if variant in ("C",):
        body = json.dumps({str(i + 1): t for i, t in enumerate(items)}, ensure_ascii=False)
        return "待校稿句子（JSON）：\n" + body
    lines = [f"{i+1}|{t}" for i, t in enumerate(items)]
    head = "<arg_key:6124c78e>\n" if variant == "D" else ""
    return head + "待校稿句子：\n" + "\n".join(lines)


def parse_out(text: str, n: int, variant: str) -> dict[int, str]:
    out: dict[int, str] = {}
    if variant in ("C",):
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                for k, v in obj.items():
                    i = int(re.sub(r"\D", "", str(k)) or 0) - 1
                    if 0 <= i < n and isinstance(v, str) and v.strip():
                        out[i] = v.strip()
            except Exception:  # noqa: BLE001
                pass
        return out
    for raw in text.splitlines():
        raw = raw.strip()
        if "|" not in raw:
            continue
        head, _, body = raw.partition("|")
        idx = re.sub(r"\D", "", head)
        if not idx:
            continue
        i = int(idx) - 1
        if 0 <= i < n and body.strip():
            out.setdefault(i, body.strip().strip("`"))
    return out


def main() -> int:
    data = json.loads((HERE / "llm_set.json").read_text(encoding="utf-8"))
    data = data[:12]
    models = sys.argv[1:] or ["qwen2.5:7b-instruct", "qwen2.5:3b-instruct", "qwen3:8b"]
    client = OllamaClient(timeout=180)
    if not client.alive() and not client.ensure_server(wait=20):
        print("[FAIL] Ollama 不可用")
        return 1
    installed = client.models()
    results = []

    for model in models:
        if model not in installed:
            print(f"[SKIP] {model}")
            continue
        for variant, batch in (("A", 8), ("B", 3), ("C", 3), ("D", 3)):
            sys_txt = SYS_JSON if variant == "C" else system_text()
            fixed: dict[int, str] = {}
            degenerate = 0
            t0 = time.perf_counter()
            for i in range(0, len(data), batch):
                chunk = data[i:i + batch]
                items = [c["asr"] for c in chunk]
                try:
                    res = client.chat(model, sys_txt, build_user(items, variant),
                                      temperature=0.0, num_predict=120 * len(items) + 128)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {model} {variant} 请求失败 {exc!r}")
                    continue
                got = parse_out(res.text, len(items), variant)
                vals = [got.get(k) for k in range(len(items))]
                if len(set(v for v in vals if v)) == 1 and len(items) > 1:
                    degenerate += 1
                for k, v in got.items():
                    if v and v != items[k]:
                        fixed[i + k] = v
            secs = time.perf_counter() - t0

            before, after = [], []
            improved = worsened = 0
            for idx, d in enumerate(data):
                cb = cer(norm(d["truth"]), norm(d["asr"]))
                new = fixed.get(idx)
                ca = cer(norm(d["truth"]), norm(new)) if new else cb
                before.append(cb)
                after.append(ca)
                if new:
                    if ca < cb - 0.005:
                        improved += 1
                    elif ca > cb + 0.005:
                        worsened += 1
            row = {
                "model": model, "variant": variant, "batch": batch,
                "cer_before": round(float(np.mean(before)), 4),
                "cer_after": round(float(np.mean(after)), 4),
                "gain": round(float(np.mean(before) - np.mean(after)), 4),
                "fixed": len(fixed), "improved": improved, "worsened": worsened,
                "degenerate_batches": degenerate, "seconds": round(secs, 1),
            }
            results.append(row)
            OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{model:<20} 变体{variant}(批{batch}) 采纳 {len(fixed):>2}/{len(data)} "
                  f"改善 {row['gain']:+.3f} 改好 {improved} 改坏 {worsened} "
                  f"退化批 {degenerate} {secs:.0f}s", flush=True)

    print("\n===== 最佳组合 =====")
    for r in sorted(results, key=lambda x: (-x["gain"], -x["fixed"], x["seconds"]))[:6]:
        print(f"{r['model']:<20} 变体{r['variant']} 改善 {r['gain']:+.3f} "
              f"采纳 {r['fixed']} 改坏 {r['worsened']} {r['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
