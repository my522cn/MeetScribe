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
from run_app import _light_palette  # noqa: E402


def main() -> None:
    app = QApplication(sys.argv)
    # 与 run_app.py 保持一致：Fusion + 浅色调色板，否则截出来是深色渗透的旧样子
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.resize(1180, 900)
    win.show()
    win.btn_settings.click()

    def shoot() -> None:
        win.grab().save(str(ROOT / "phase3" / "ui_settings.png"))
        win.btn_record.click()

        def shoot2() -> None:
            win.grab().save(str(ROOT / "phase3" / "ui_record.png"))
            app.quit()

        QTimer.singleShot(600, shoot2)

    QTimer.singleShot(2500, shoot)  # 等硬件探测线程回来
    app.exec()
    print("saved ui_settings.png / ui_record.png")


if __name__ == "__main__":
    main()
