"""设置页多尺寸渲染 + 录音页多尺寸渲染，检查自适应。"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.ui.main_window import MainWindow  # noqa: E402
from meetscribe.ui.theme import QSS  # noqa: E402

SIZES = [(1000, 680), (1400, 920)]


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    win.btn_settings.click()
    out = []

    def step(i: int = 0) -> None:
        if i >= len(SIZES):
            win.btn_record.click()
            QTimer.singleShot(300, lambda: shot_record(0))
            return
        w, h = SIZES[i]
        win.resize(w, h)
        QTimer.singleShot(400, lambda: shoot(f"ui_set_{w}.png", lambda: step(i + 1)))

    def shot_record(i: int = 0) -> None:
        if i >= len(SIZES):
            app.quit()
            print("saved:", ", ".join(out))
            return
        w, h = SIZES[i]
        win.resize(w, h)
        QTimer.singleShot(400, lambda: shoot(f"ui_rec_{w}.png", lambda: shot_record(i + 1)))

    def shoot(name: str, nxt) -> None:
        p = ROOT / "phase3" / name
        win.grab().save(str(p))
        out.append(name)
        nxt()

    QTimer.singleShot(2200, step)
    app.exec()


if __name__ == "__main__":
    main()
