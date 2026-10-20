"""MeetScribe 启动入口：python run_app.py"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from meetscribe.ui.main_window import MainWindow, make_icon  # noqa: E402
from meetscribe.ui.theme import QSS  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("会议转写助手")
    app.setWindowIcon(make_icon())
    app.setQuitOnLastWindowClosed(False)
    app.setStyleSheet(QSS)

    try:
        win = MainWindow()
    except Exception as exc:  # noqa: BLE001
        QMessageBox.critical(None, "启动失败", f"{exc}\n\n{traceback.format_exc()}")
        return 1
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
