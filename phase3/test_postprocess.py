"""后处理验收：同音纠错 / 标点恢复 / 中英间距，逐条对比。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.text.postprocess import TextPostProcessor  # noqa: E402

SAMPLES = [
    "大家好今天我们先过一遍harmonic项目的排期",
    "i think the time mine is too tight联条至少要留两周时间",
    "第二件事客户那边希望下个月能看到dem",
    "这个接口我们下周就能开始联调前提是拿到测试账号",
    "我们现在要确认供应商报价然后更新排期表",
    "预算方面我记得是四十万左右具体要看报价",
]


def main() -> None:
    pp = TextPostProcessor(
        hotwords_file=ROOT / "config" / "hotwords.txt",
        replacements_file=ROOT / "config" / "replacements.txt",
        punct_model=ROOT / "models" / "punct_zh_en" / "model.int8.onnx",
    )
    print(f"标点模型: {'已加载' if pp.punctuator.available else '不可用'}   热词: {len(pp.hotwords)} 个")
    t0 = time.perf_counter()
    for raw in SAMPLES:
        out = pp.process(raw)
        print(f"\n原: {raw}\n后: {out}")
    n = len(SAMPLES)
    print(f"\n平均每句耗时 {(time.perf_counter()-t0)/n*1000:.1f} ms")


if __name__ == "__main__":
    main()
