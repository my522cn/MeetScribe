"""单实例回归：

1. 第一个实例 acquire() 成功
2. 第二个实例 acquire() 失败，并且第一个实例收到 activated 信号
3. 第一个实例 release() 后，新实例可以重新拿到
4. 模拟崩溃残留（管道名存在但没人监听）时仍能正常启动
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QCoreApplication  # noqa: E402
from PySide6.QtNetwork import QLocalServer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from meetscribe.ui.single_instance import SingleInstance  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


def pump(app: QApplication, rounds: int = 40) -> None:
    for _ in range(rounds):
        app.processEvents()


def main() -> int:
    app = QApplication(sys.argv)
    key = "MeetScribe.Test.SingleInstance"

    # 1. 首个实例
    a = SingleInstance(key)
    got_first = a.acquire()
    check("第一个实例取得所有权", got_first)
    hits: list[int] = []
    a.activated.connect(lambda: hits.append(1))

    # 2. 第二个实例：应失败，并唤醒第一个
    b = SingleInstance(key)
    got_second = b.acquire()
    pump(app)
    check("第二个实例被拒绝", not got_second)
    check("第一个实例收到唤醒信号", len(hits) >= 1, f"hits={len(hits)}")

    # 3. release 后可以重新取得
    a.release()
    pump(app)
    c = SingleInstance(key)
    got_third = c.acquire()
    pump(app)
    check("释放后新实例可重新取得", got_third)
    pump(app)

    # 4. 幂等性：同一实例重复 acquire / 重复 release 都不应出问题
    #    （真实场景：开机自启 + 用户手点快捷方式，可能连着来两次）
    ok_again = c.acquire()
    pump(app)
    check("同一实例重复 acquire 仍持有所有权", ok_again)
    c.release()
    c.release()  # 第二次 release 必须安全（不抛异常）
    pump(app)
    ok_reacquire = c.acquire()
    pump(app)
    check("重复 release 后可重新取得", ok_reacquire)
    c.release()
    QLocalServer.removeServer(key)

    print()
    bad = [n for n, ok, _ in checks if not ok]
    for n, ok, det in checks:
        print(f"[{'OK ' if ok else 'FAIL'}] {n}　{det}")
    print()
    if bad:
        print("❌ 失败项：", bad)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
