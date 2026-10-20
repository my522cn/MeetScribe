"""配置页测试链路验证：测试 5 秒 → 电平 → 流式/离线两版识别。

用法：python phase3/test_settings_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.ui.main_window import MainWindow  # noqa: E402
from meetscribe.ui.theme import QSS  # noqa: E402

LOG: list[str] = []


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.btn_settings.click()
    page = win.settings_page

    page.probe.level.connect(lambda v: None)
    page.probe.finished.connect(
        lambda r: LOG.append(f"finished: {r.device} rms={r.rms:.4f} empty={r.empty}")
    )
    page.probe.recognized.connect(
        lambda a, b, rtf, ls: LOG.append(
            f"recognized: 流式={a[:40]!r} 离线={b[:40]!r} rtf={rtf:.3f} load={ls:.1f}s"
        )
    )
    page.probe.failed.connect(lambda m: LOG.append(f"failed: {m}"))

    print("麦克风下拉项:", [page.cmb_mic.itemText(i) for i in range(page.cmb_mic.count())])
    print("选中索引:", page._selected_mic_index())
    page._test_device("mic")

    def check() -> None:
        for line in LOG:
            print(" ", line)
        if not LOG or not any(l.startswith(("recognized", "failed")) or "empty=True" in l for l in LOG):
            QTimer.singleShot(1500, check)
        else:
            app.quit()
            return 0

    QTimer.singleShot(1000, check)
    rc = app.exec()
    for line in LOG:
        print(" ", line)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
