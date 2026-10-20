"""大模型纠错 / 纪要验证脚本。

用法：
  python phase3/test_llm.py            # 连接检测 + 纠错样例 + 纪要样例
  python phase3/test_llm.py --model qwen3:4b
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.config import Settings  # noqa: E402
from meetscribe.llm import Corrector, FixRequest, OllamaClient  # noqa: E402

# (ASR 原始输出, 期望结果)；期望留空表示"不该改动"
CASES = [
    ("联条至少要留两周时间", "联调至少要留两周时间"),
    ("我们先把排期过一遍", "我们先把排期过一遍"),
    ("i think the time mine is too tight", "i think the timeline is too tight"),
    ("harmonic 项目下周要交 demo", "harmonic 项目下周要交 demo"),
    ("研调工作还没开始", "联调工作还没开始"),
    ("客户希望下个月能看到 dem", "客户希望下个月能看到 demo"),
    ("这个方案的成本太搞了", "这个方案的成本太高了"),
    ("供应商报价还没回来", "供应商报价还没回来"),
]

TRANSCRIPT = """我 00:03 大家好今天我们先过一遍 harmonic 项目的排期
对方 00:11 i think the timeline is too tight 联条至少要留两周时间
我 00:25 第二件事客户那边希望下个月能看到 demo
对方 00:34 排期和联调时间能不能先冻下来
我 01:02 我这边需要供应商的报价才能确认
对方 01:15 争议点是验收标准谁来定
我 01:30 结论是本周五之前给出验收清单
对方 01:44 待办我整理报价单下周一二给""".strip()


def main() -> None:
    settings = Settings()
    base = settings.get("llm.base_url", "http://127.0.0.1:11434")
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else settings.get("llm.fix_model", "qwen3:4b")
    client = OllamaClient(base, timeout=120)

    print(f"Ollama: {base}")
    try:
        models = client.models()
        print("已安装模型:", "、".join(models) or "（无）")
    except Exception as exc:  # noqa: BLE001
        print("连接失败:", exc)
        return
    if model not in models:
        print(f"注意：{model} 未安装，先执行 ollama pull {model}")
        return

    hotwords = ["HARMONIC", "联调", "排期", "验收", "供应商", "demo", "timeline"]
    corr = Corrector(client, model, hotwords)

    print(f"\n===== 纠错测试（模型 {model}）=====")
    items = [FixRequest(("t", i), text) for i, (text, _) in enumerate(CASES)]
    t0 = time.perf_counter()
    fixed = corr.fix(items)
    elapsed = time.perf_counter() - t0

    ok = bad = miss = 0
    for i, (raw, want) in enumerate(CASES):
        got = fixed.get(("t", i), raw)
        if not want:
            status = "OK(未改动)" if got == raw else "!!误改"
            ok += got == raw
            bad += got != raw
        elif got == want:
            status = "OK"
            ok += 1
        elif got == raw:
            status = "漏改"
            miss += 1
        else:
            status = "!!改错"
            bad += 1
        print(f"  [{status}] {raw}\n         → {got}")
    print(f"\n通过 {ok} / 漏改 {miss} / 错改 {bad}，耗时 {elapsed:.1f}s（{len(CASES)} 句）")
    print("统计:", corr.last_stats)

    print("\n===== 纪要测试 =====")
    from meetscribe.llm.summary import summarize

    t1 = time.perf_counter()
    md = summarize(client, str(settings.get("llm.summary_model", "qwen3:8b")), "HARMONIC 项目周会", TRANSCRIPT)
    print(md)
    print(f"\n用时 {time.perf_counter()-t1:.1f}s")


if __name__ == "__main__":
    main()
