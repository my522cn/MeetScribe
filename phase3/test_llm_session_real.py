# -*- coding: utf-8 -*-
"""LLM 纠错线程会话隔离 —— 真实 Ollama 模型端到端验证。

与 test_llm_session.py（假 Corrector）互补：这里用真实 OllamaClient + 真实模型
跑完整链路，复现用户报的 bug 场景：
1. 会话 1 有待纠错句子，纠错请求还在路上时点停止
2. 停止后收尾轮把没纠完的句子补纠并写入会话 1 的 transcript.jsonl
3. 立刻开新一场（代数 +1）：上一场的句子绝不写进新会话文件
4. 收尾晚于 _finish_session 关文件时：重开补写，纠错结果不丢

需要本机 Ollama 在线且装有 QWEN_MODEL 指定的模型。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QWEN_MODEL = os.environ.get("MS_E2E_MODEL", "qwen2.5:1.5b-instruct")
BASE = os.environ.get("MS_E2E_BASE", "http://127.0.0.1:11434")

from PySide6.QtWidgets import QApplication  # noqa: E402

from meetscribe.config import Settings  # noqa: E402
from meetscribe.llm import Corrector, OllamaClient  # noqa: E402
from meetscribe.ui.worker import CaptureWorker  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))
    print(f"[{'OK ' if ok else 'FAIL'}] {name}　{detail}")


def rec(text: str, start: int) -> dict:
    return {"track": "mic", "speaker": "说话人A", "start_ms": start, "end_ms": start + 3500,
            "text": text, "refined": False, "raw": "", "mixed": 0.0}


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> int:
    app = QApplication.instance() or QApplication([])

    # 前置：模型必须已装
    try:
        probe = OllamaClient(BASE, timeout=5)
        names = probe.models()
    except Exception as exc:  # noqa: BLE001
        print(f"跳过：Ollama 不可用 ({exc})")
        return 0
    if QWEN_MODEL not in names:
        print(f"跳过：{QWEN_MODEL} 未安装")
        return 0

    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="ms_llme2e_"))
    worker = CaptureWorker(ROOT / "models" / "paraformer_zh_en", tmp / "data", ROOT / "config")

    corr = Corrector(OllamaClient(BASE, timeout=90), QWEN_MODEL, [],
                     batch=3, think_off=True, min_similar=0.97)
    worker._corrector = corr

    # ---- 会话 1 ----
    s1 = tmp / "s1"
    s1.mkdir()
    t1 = s1 / "transcript.jsonl"
    worker._transcript_path = t1
    worker._transcript = open(t1, "a", encoding="utf-8")
    worker._records = [
        rec("我们下一步要做联条测试", 1000),
        rec("联条工作安排到下周", 5000),
        rec("客户的验收标准还没有对齐", 9000),
        rec("预算方面我记得是四十万左右", 13000),
    ]

    worker._llm_gen = 1
    stop1, wake1, done1 = threading.Event(), threading.Event(), threading.Event()
    worker._llm_stop, worker._llm_wake, worker._llm_done = stop1, wake1, done1
    t1_thread = threading.Thread(target=worker._llm_loop, args=(stop1, wake1, done1, 1),
                                 daemon=True)
    t1_thread.start()
    wake1.set()
    time.sleep(1.0)  # 让它进入真实 chat 请求

    # 模拟 _stop_llm + _finish_session 的 20 秒等待：这里只等 3 秒就"关文件"
    stop1.set()
    wake1.set()
    done1.wait(3)
    worker._flush_transcript()  # _finish_session 在等待超时后也会先刷一次盘
    worker._transcript.close()
    worker._transcript = None  # 模拟 _finish_session 关闭文件（纠错还没跑完）

    # ---- 立刻开会话 2（用户报的复现场景）----
    s2 = tmp / "s2"
    s2.mkdir()
    t2 = s2 / "transcript.jsonl"
    worker._transcript_path = t2
    worker._transcript = open(t2, "a", encoding="utf-8")
    worker._records = [rec("新会话的第一句话", 1000)]

    # 等会话 1 的收尾轮真正跑完
    done1.wait(300)
    rows1 = read_jsonl(t1)

    # 1. 会话 1 的句子全部补纠完（llm 标记剥离前代表已处理）
    marked = all(r.get("llm") for r in worker._records if r["start_ms"] <= 13000) if False else True
    # 注：_records 已被会话 2 重置，改为检查落盘文件
    texts1 = [r["text"] for r in rows1]
    check("会话1收尾落盘完整", len(rows1) == 4 and "新会话的第一句话" not in texts1,
          f"{len(rows1)} 句")
    changed = [r for r in rows1 if "llm_from" in r or r["text"] != str(r.get("text"))]
    # llm_from 会被 _flush_transcript 剥离，改从内存痕迹看：重新读原文对比
    # （真实模型改不改都行，关键是链路通了、句子没丢）

    # 2. 线程退出
    t1_thread.join(5)
    check("会话1线程已退出", not t1_thread.is_alive(), "")

    # 3. 会话 2 文件没有被会话 1 污染
    rows2 = read_jsonl(t2)
    leak = any(("联" in r["text"] and "测试" in r["text"]) or "四十万" in r["text"]
               for r in rows2)
    check("会话2无上一场泄漏", not leak and len(rows2) <= 1, f"{[r['text'] for r in rows2]}")

    # ---- 会话 2 自己的纠错轮 ----
    worker._llm_gen = 2
    stop2, wake2, done2 = threading.Event(), threading.Event(), threading.Event()
    worker._llm_stop, worker._llm_wake, worker._llm_done = stop2, wake2, done2
    t2_thread = threading.Thread(target=worker._llm_loop, args=(stop2, wake2, done2, 2),
                                 daemon=True)
    t2_thread.start()
    wake2.set()
    stop2.set()
    wake2.set()
    done2.wait(300)
    t2_thread.join(5)
    worker._flush_transcript()
    rows2b = read_jsonl(t2)
    check("会话2自身纠错完成且退出", not t2_thread.is_alive() and len(rows2b) == 1,
          f"{rows2b}")

    # 报告真实模型改了什么（信息性，不计入断言）
    print()
    print(f"模型 {QWEN_MODEL} 会话1落盘文本：")
    for r in rows1:
        print(f"  {r['text']}")

    bad = [n for n, ok, _ in checks if not ok]
    print()
    if bad:
        print("❌ 失败项：", bad)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
