"""在同一个进程里拉起 Ollama 服务、拉模型、验证纠错与纪要，跑完自动收尾。

沙箱环境会回收跨调用存活的子进程，所以把"起服务 → 干活 → 停服务"放进一次执行里。

用法：
  python phase3/with_ollama.py status
  python phase3/with_ollama.py pull qwen2.5:7b-instruct
  python phase3/with_ollama.py test            # 纠错 + 纪要
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.llm import OllamaClient  # noqa: E402


def locate() -> str:
    exe = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
    if os.path.exists(exe):
        return exe
    raise SystemExit("找不到 ollama.exe")


def start_server(port: int = 11434) -> subprocess.Popen:
    """普通子进程即可：本脚本活多久服务活多久，退出时一起收掉。"""
    proc = subprocess.Popen(
        [locate(), "serve"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
    )
    deadline = time.time() + 40
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/version", timeout=2) as r:
                r.read()
                return proc
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    raise SystemExit("服务启动超时")


def stop_server(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    proc = start_server()
    try:
        client = OllamaClient("http://127.0.0.1:11434", timeout=600)
        if cmd == "pull":
            model = sys.argv[2]
            print(f"拉取 {model} …", flush=True)
            t0 = time.time()
            client.pull(model)
            print(f"完成，用时 {time.time()-t0:.0f}s")
        elif cmd == "status":
            print("已装模型:", "、".join(client.models()) or "（无）")
        elif cmd == "test":
            run_tests(client)
        else:
            print(__doc__)
    finally:
        stop_server(proc)


def run_tests(client: OllamaClient) -> None:
    from meetscribe.config import Settings
    from meetscribe.llm import Corrector, FixRequest
    from meetscribe.llm.summary import summarize

    print("已装模型:", "、".join(client.models()))
    s = Settings()
    fix_model = str(s.get("llm.fix_model", "qwen2.5:3b-instruct"))
    sum_model = str(s.get("llm.summary_model", "qwen2.5:7b-instruct"))

    cases = [
        ("联条至少要留两周时间", "联调至少要留两周时间"),
        ("我们先把排期过一遍", ""),
        ("i think the time mine is too tight", "i think the timeline is too tight"),
        ("harmonic 项目下周要交 demo", ""),
        ("研调工作还没开始", "联调工作还没开始"),
        ("客户希望下个月能看到 dem", "客户希望下个月能看到 demo"),
        ("这个方案的成本太搞了", "这个方案的成本太高了"),
        ("供应商报价还没回来", ""),
    ]
    corr = Corrector(client, fix_model, ["HARMONIC", "联调", "排期", "demo", "timeline"])
    items = [FixRequest(("t", i), c[0]) for i, c in enumerate(cases)]
    t0 = time.perf_counter()
    fixed = corr.fix(items)
    el = time.perf_counter() - t0
    ok = miss = bad = 0
    print(f"\n===== 纠错（{fix_model}）=====")
    for i, (raw, want) in enumerate(cases):
        got = fixed.get(("t", i), raw)
        if not want:
            st = "OK(未改)" if got == raw else "!!误改"
            ok += got == raw
            bad += got != raw
        elif got == want:
            st = "OK"
            ok += 1
        elif got == raw:
            st = "漏改"
            miss += 1
        else:
            st = "!!改错"
            bad += 1
        print(f"  [{st}] {raw} → {got}")
    print(f"通过 {ok} / 漏改 {miss} / 错改 {bad}，{el:.1f}s（{len(cases)} 句）")

    transcript = """我 00:03 大家好今天我们先过一遍 harmonic 项目的排期
对方 00:11 i think the timeline is too tight 联调至少要留两周时间
我 00:25 第二件事客户那边希望下个月能看到 demo
对方 00:34 排期和联调时间能不能先冻下来
我 01:02 我这边需要供应商的报价才能确认
对方 01:15 争议点是验收标准谁来定
我 01:30 结论是本周五之前给出验收清单
对方 01:44 待办我整理报价单下周一二给"""
    print(f"\n===== 纪要（{sum_model}）=====")
    t1 = time.perf_counter()
    md = summarize(client, sum_model, "HARMONIC 项目周会", transcript)
    print(md)
    print(f"—— {time.perf_counter()-t1:.1f}s")


if __name__ == "__main__":
    main()
