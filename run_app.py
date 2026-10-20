"""MeetScribe 启动入口：python run_app.py"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QMessageBox

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from meetscribe.ui.main_window import MainWindow, make_icon  # noqa: E402
from meetscribe.ui.theme import QSS  # noqa: E402


def _light_palette() -> QPalette:
    """强制浅色调色板。

    根因：Windows 处于深色模式时，系统调色板（Window=#1e1e1e）会渗透进
    输入类控件，QLineEdit/QSpinBox 看起来是灰色（像被禁用），但实际能输入。
    同时 PySide6 在 Win11 上默认用 windows11 原生样式，它对 QSS 的
    background 支持有缺陷，所以这里强制 Fusion + 浅色 palette 双保险。
    """
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#F5F6F8"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#1F2329"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#FFFFFF"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#F2F3F7"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#1F2329"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#FFFFFF"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#1F2329"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#4F46E5"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor("#B3B8C0"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#23262D"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#FFFFFF"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#B3B8C0"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#B3B8C0"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor("#F2F3F5"))
    return pal


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("会议转写助手")
    app.setWindowIcon(make_icon())
    app.setQuitOnLastWindowClosed(False)
    # 顺序很重要：先样式，再调色板，最后样式表
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
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
