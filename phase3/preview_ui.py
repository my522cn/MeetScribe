"""界面预览：启动主窗口，渲染若干帧后截图保存，便于检查版式。"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.ui.main_window import MainWindow  # noqa: E402
from meetscribe.ui.theme import QSS  # noqa: E402

OUT = ROOT / "phase3"


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()

    # 造几条假字幕，检查字幕区排版
    cap = win.record_page.captions
    cap.update_partial("mic", "我觉得整体排期还是偏紧，联调至少要留两周", 12000)
    cap.finalize("loop", "预算方面我记得是四十万左右，具体要看供应商报价。", 3000, 11000)
    cap.finalize("mic", "大家好，今天我们先过一遍 HARMONIC 项目的排期。", 12000, 11800)
    cap.update_final_text("mic", 12000, "大家好，今天我们先过一遍 HARMONIC 项目的排期。I think the timeline is too tight。")
    cap.update_partial("loop", "但是我们到现在还没拿到测试账号，这个要", 21000)
    cap.set_overlap(True)
    win.record_page.set_elapsed(12 * 60 * 1000 + 34000)
    win.record_page.set_state("recording")
    win.record_page.set_caption_hint(f"实时识别中 · 已二次校对 {cap.refined_count} 句")

    win.float_bar.place_top_right()
    win.float_bar.set_elapsed(12 * 60 * 1000 + 34000)
    win.float_bar.show()

    def shoot() -> None:
        win.grab().save(str(OUT / "ui_main.png"))
        win.float_bar.grab().save(str(OUT / "ui_float.png"))
        app.quit()

    QTimer.singleShot(900, shoot)
    app.exec()
    print("saved:", OUT / "ui_main.png")


if __name__ == "__main__":
    main()
