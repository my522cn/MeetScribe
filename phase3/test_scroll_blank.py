"""字幕滚动回归：

1. 自动贴底后视口底部必须紧贴最后一行字幕（不能出现大段空白）
2. 用户上滚 → 停止自动跟随，右下角浮出“回到底部”
3. 点“回到底部” → 回到最新并恢复自动跟随
4. 用户手动滚回底部 → 同样恢复自动跟随
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from run_app import _light_palette  # noqa: E402
from meetscribe.ui.caption_view import CaptionView  # noqa: E402
from meetscribe.ui.theme import QSS  # noqa: E402

META = {"mic": ("我", "ChipMic"), "loop": ("对方", "ChipLoop")}

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


def feed(view: CaptionView, app: QApplication, n: int) -> None:
    for i in range(n):
        for track in ("mic", "loop"):
            view.update_partial(track, f"第{i}句：这是一段比较长的会议发言内容，" * 3, i * 5000)
            app.processEvents()
            view.finalize(track, f"第{i}句：这是一段比较长的会议发言内容，" * 3,
                          i * 5000, i * 5000 + 4800)
            app.processEvents()


def bottom_blank(view: CaptionView) -> int:
    """视口底部与最后一行底部之间的空白像素。"""
    body = view.widget()
    lay = body.layout()
    last_bottom = 0
    for k in range(lay.count()):
        w = lay.itemAt(k).widget()
        if w is not None and not w.isHidden():
            last_bottom = max(last_bottom, w.geometry().bottom())
    bar = view.verticalScrollBar()
    return (bar.value() + view.viewport().height()) - last_bottom


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    app.setStyleSheet(QSS)

    view = CaptionView()
    view.set_speaker_meta(META)
    view.resize(900, 300)
    view.show()
    for _ in range(5):
        app.processEvents()

    # ---------- 1. 长会自动贴底：底部不允许有肉眼可见的空白 ----------
    feed(view, app, 30)
    app.processEvents()
    bar = view.verticalScrollBar()
    check(
        "自动贴底（滚动条在最大值）",
        bar.value() == bar.maximum() and bar.maximum() > 0,
        f"value={bar.value()} max={bar.maximum()}",
    )
    blank = bottom_blank(view)
    check("贴底后视口底部无空白", blank <= 16, f"空白 {blank}px（修复前约 961px）")

    # ---------- 2. 用户上滚 → 停自动跟随 + 浮出按钮 ----------
    bar.setValue(max(0, bar.maximum() - 600))
    app.processEvents()
    check(
        "用户上滚后停止自动跟随",
        not view._auto and view.goto_btn.isVisible(),
        f"auto={view._auto} btn_visible={view.goto_btn.isVisible()}",
    )
    # 此时再来新句子，视口必须原地不动（不能强行拉回底部）
    pos_before = bar.value()
    view.finalize("mic", "新句子：用户正在看历史，不该被拉走", 200000, 200400)
    app.processEvents()
    check(
        "浏览历史时新字幕不拉走视口",
        bar.value() == pos_before,
        f"value={bar.value()}（前 {pos_before}）",
    )

    # ---------- 3. 点“回到底部” → 回最新并恢复跟随 ----------
    view.goto_btn.click()
    app.processEvents()
    check(
        "点按钮回到底部并恢复自动跟随",
        view._auto and not view.goto_btn.isVisible() and bar.value() == bar.maximum(),
        f"auto={view._auto} value={bar.value()} max={bar.maximum()}",
    )
    # 恢复后再来新句子，必须继续贴底
    view.finalize("mic", "恢复跟随后的新句子", 201000, 201400)
    app.processEvents()
    check("恢复后继续自动贴底", bar.value() == bar.maximum(), f"value={bar.value()} max={bar.maximum()}")

    # ---------- 4. 滚轮上滚同样触发，手动滚回底部恢复 ----------
    from PySide6.QtCore import QPoint

    evt = QWheelEvent(QPointF(10, 10), QPointF(10, 10),
                      QPoint(0, 240), QPoint(0, 240),  # 正值 = 向上滚（离开底部）
                      Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                      Qt.ScrollPhase.NoScrollPhase, False)
    val_before = bar.value()
    view.wheelEvent(evt)
    app.processEvents()
    check(
        "滚轮上滚停止跟随",
        not view._auto and view.goto_btn.isVisible() and bar.value() < val_before,
        f"auto={view._auto} value={bar.value()}（前 {val_before}）",
    )
    bar.setValue(bar.maximum())  # 手动滚回底部
    app.processEvents()
    check(
        "手动滚回底部恢复跟随",
        view._auto and not view.goto_btn.isVisible(),
        f"auto={view._auto}",
    )

    # ---------- 5. 清空后回到初始自动状态 ----------
    view.clear()
    app.processEvents()
    check("清空后恢复自动跟随", view._auto and not view.goto_btn.isVisible(),
          f"auto={view._auto}")

    print()
    bad = [n for n, ok, _ in checks if not ok]
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
