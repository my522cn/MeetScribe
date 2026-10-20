"""配置页截图：检查版式（不触发录音）。"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.ui.main_window import MainWindow  # noqa: E402
from meetscribe.ui.theme import QSS  # noqa: E402


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.resize(1180, 900)
    win.show()
    win.btn_settings.click()

    def shoot() -> None:
        win.grab().save(str(ROOT / "phase3" / "ui_settings.png"))
        app.quit()

    QTimer.singleShot(2500, shoot)  # 等硬件探测线程回来
    app.exec()
    print("saved ui_settings.png")


if __name__ == "__main__":
    main()
