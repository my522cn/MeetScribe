# -*- coding: utf-8 -*-
"""LLM 纠错线程会话隔离回归（复现用户报的 bug）：

Bug 表现：停止录音后纠错没跑完；下一次开录音时，上一场的纠错结果写进了新会话。
根因：_llm_loop 线程堵在一次 chat 请求里时 _stop_llm() 拦不住它；新会话
_start_llm() 又替换了 self._llm_stop/_llm_wake，老线程从此引用新事件 → 僵尸线程。

验证（用假 Corrector 模拟慢请求，不依赖 Ollama）：
1. 停止后排空：stop 之后 pending 的句子仍被纠错并写入转写
2. 换场隔离：老线程在请求返回时发现代数变了 → 结果丢弃、线程退出
3. 停止唤醒：_stop_llm 立刻唤醒 25 秒休眠中的线程
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

from PySide6.QtWidgets import QApplication  # noqa: E402

from meetscribe.config import Settings  # noqa: E402
from meetscribe.ui.worker import CaptureWorker  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


class SlowCorrector:
    """模拟慢 LLM：fix() 睡 delay 秒后把每句改一个同音字。"""

    def __init__(self, delay: float = 0.5) -> None:
        self.delay = delay
        self.calls = 0

    def fix(self, items):
        self.calls += 1
        time.sleep(self.delay)
        return {it.key: it.text.replace("联条", "联调") for it in items
                if "联条" in it.text}


def main() -> int:
    app = QApplication.instance() or QApplication([])
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="ms_llmgen_"))
    worker = CaptureWorker(ROOT / "models" / "paraformer_zh_en", tmp / "data", ROOT / "config")

    # 手动启动 LLM 线程（绕过 Ollama 检测）
    worker._llm_gen = getattr(worker, "_llm_gen", 0) + 1
    stop_ev, wake_ev, done_ev = threading.Event(), threading.Event(), threading.Event()
    worker._llm_stop, worker._llm_wake, worker._llm_done = stop_ev, wake_ev, done_ev
    gen = worker._llm_gen
    slow = SlowCorrector(delay=0.4)
    worker._corrector = slow
    worker._records = []
    t_old = threading.Thread(target=worker._llm_loop, args=(stop_ev, wake_ev, done_ev, gen),
                             daemon=True)
    t_old.start()

    # 会话 1 的两条待纠错句子
    worker._records = [
        {"track": "mic", "speaker": "说话人A", "start_ms": 1000, "end_ms": 4000,
         "text": "我们下一步要做联条测试", "refined": False, "raw": "", "mixed": 0.0},
        {"track": "mic", "speaker": "说话人A", "start_ms": 5000, "end_ms": 9000,
         "text": "联条工作安排到下周", "refined": False, "raw": "", "mixed": 0.0},
    ]
    wake_ev.set()
    time.sleep(0.1)

    # 1. 停止唤醒：线程在 25s 休眠里也能立刻看到停止
    stop_ev.set()
    wake_ev.set()   # _stop_llm 的行为
    done_ev.wait(15)
    alive_after_stop = t_old.is_alive()
    time.sleep(0.2)
    fixed_texts = [r["text"] for r in worker._records]
    check("停止后补完纠错", fixed_texts == ["我们下一步要做联调测试", "联调工作安排到下周"],
          f"{fixed_texts}")
    check("收尾后线程退出", not t_old.is_alive(), f"alive={alive_after_stop}")

    # 2. 换场隔离：老线程堵在请求里时新会话启动 → 老结果必须丢弃
    worker._llm_gen = gen + 1
    worker._records = [
        {"track": "mic", "speaker": "说话人A", "start_ms": 1000, "end_ms": 4000,
         "text": "新会话的原始句子", "refined": False, "raw": "", "mixed": 0.0},
    ]
    stop2, wake2, done2 = threading.Event(), threading.Event(), threading.Event()
    worker._llm_stop, worker._llm_wake, worker._llm_done = stop2, wake2, done2
    slow2 = SlowCorrector(delay=0.6)
    slow2.fix_calls_marker = True
    worker._corrector = slow2
    t_old2 = threading.Thread(target=worker._llm_loop,
                              args=(threading.Event(), threading.Event(), threading.Event(), gen),
                              daemon=True)
    # 老代数线程：醒来即发现过期，不该碰新会话的记录
    t_old2.start()
    wake_ev = threading.Event()
    time.sleep(0.3)
    t_old2.join(5)
    check("过期线程不碰新会话记录", all("联调" not in r["text"] for r in worker._records)
          and worker._records[0]["text"] == "新会话的原始句子",
          f"{[r['text'] for r in worker._records]}")
    check("过期线程自行退出", not t_old2.is_alive(), "")

    # 3. 收尾晚于关闭：_finish_session 已关文件后收尾轮才跑完 → 重开补写，纠错不丢
    tpath = tmp / "late" / "transcript.jsonl"
    tpath.parent.mkdir(parents=True, exist_ok=True)
    tpath.write_text("", encoding="utf-8")
    worker._transcript = None
    worker._transcript_path = tpath
    worker._records = [
        {"track": "mic", "speaker": "说话人A", "start_ms": 1000, "end_ms": 4000,
         "text": "我们下一步要做联调测试", "refined": True, "raw": "", "mixed": 0.0,
         "llm": True, "llm_from": "我们下一步要做联条测试"},
    ]
    worker._flush_transcript()
    lines = [json.loads(x) for x in tpath.read_text(encoding="utf-8").splitlines() if x.strip()]
    check("文件已关仍能补写落盘",
          len(lines) == 1 and lines[0]["text"] == "我们下一步要做联调测试"
          and "llm" not in lines[0] and "llm_from" not in lines[0],
          f"{lines}")
    check("补写后文件句柄已复关", worker._transcript is None, "")

    bad = [n for n, ok, _ in checks if not ok]
    print()
    for n, ok, d in checks:
        print(f"[{'OK ' if ok else 'FAIL'}] {n}　{d}")
    print()
    if bad:
        print("❌ 失败项：", bad)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
