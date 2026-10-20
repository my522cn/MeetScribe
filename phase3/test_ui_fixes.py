"""验证本轮 UI 修复（offscreen 渲染）：

1. Fusion + 浅色 palette + QSS 后，设置页输入类控件中心像素应为纯白 #FFFFFF
   （修复"输入框灰色像禁用"）
2. 麦克风下拉与系统声音下拉宽度一致，测试后不再变长
3. 录音中点击另一个模式按钮，两个按钮保持互斥（单选）
4. 字幕行：时间在 chip 之后、内容之前（行首）
5. finalize 带 speaker 参数时 chip 会更新为"说话人A"并换配色
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from meetscribe.ui.theme import QSS  # noqa: E402
from run_app import _light_palette  # noqa: E402


def center_pixel(widget) -> tuple[int, int, int]:
    pm = widget.grab()
    img = pm.toImage()
    c = img.pixelColor(img.width() // 2, img.height() // 2)
    return c.red(), c.green(), c.blue()


def hexs(t: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*t)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    app.setStyleSheet(QSS)

    from meetscribe.config import Settings
    from meetscribe.ui.record_page import RecordPage
    from meetscribe.ui.settings_page import SettingsPage

    failures: list[str] = []

    # ---------- 1. 输入框白底 ----------
    page = SettingsPage(Settings())
    page.resize(1100, 900)
    page.show()
    for _ in range(10):  # 等布局与首帧绘制完成
        app.processEvents()
    app.sendPostedEvents(None, 0)
    for name, w in (
        ("cmb_mic", page.cmb_mic),
        ("cmb_loop", page.cmb_loop),
        ("spin_segment", page.spin_segment),
        ("edit_root", page.edit_root),
        ("cmb_fix_model(editable)", page.cmb_fix_model),
        ("spin_threads", page.spin_threads),
    ):
        px = center_pixel(w)
        ok = px == (255, 255, 255)
        print(f"[{'OK ' if ok else 'FAIL'}] {name} 中心像素 = {hexs(px)}")
        if not ok:
            failures.append(f"{name} 底色 {hexs(px)} != #FFFFFF")

    # ---------- 2. 两个下拉等宽 ----------
    page.refresh_devices()
    app.processEvents()
    w1, w2 = page.cmb_mic.width(), page.cmb_loop.width()
    ok = abs(w1 - w2) <= 2
    print(f"[{'OK ' if ok else 'FAIL'}] 麦克风框宽 {w1}px vs 系统声框宽 {w2}px")
    if not ok:
        failures.append(f"下拉宽度不一致: {w1} vs {w2}")

    # ---------- 3. 模式按钮互斥 ----------
    rp = RecordPage()
    rp.show()
    app.processEvents()
    rp.set_state("recording")
    rp.offline_btn.click()
    app.processEvents()
    ok = rp.online_btn.isChecked() and not rp.offline_btn.isChecked()
    print(f"[{'OK ' if ok else 'FAIL'}] 录音中点'线下会议' → 仍保持线上={rp.online_btn.isChecked()}, 线下={rp.offline_btn.isChecked()}")
    if not ok:
        failures.append("录音中模式按钮没有保持互斥")
    rp.set_state("idle")
    rp.offline_btn.click()
    ok = rp.offline_btn.isChecked() and not rp.online_btn.isChecked()
    print(f"[{'OK ' if ok else 'FAIL'}] 待机中可正常切换到线下={ok}")
    if not ok:
        failures.append("待机时无法切换模式")

    # ---------- 4/5. 字幕行时间前置 + 说话人更新 ----------
    cv = rp.captions
    cv.set_speaker_meta({"mic": ("本地", "ChipOffline"), "loop": ("远端", "ChipLoop")})
    cv.update_partial("mic", "这是测试", 65_000)
    cv.finalize("mic", "这是测试。", 65_000, 68_000, "说话人A")
    row = cv._finals[("mic", 65_000)]
    lay = row.layout()
    order = [lay.itemAt(i).widget() for i in range(lay.count())]
    ok = order.index(row.time) == 1 and order.index(row.chip) == 0 and order.index(row.text) == 2
    print(f"[{'OK ' if ok else 'FAIL'}] 行内顺序 chip→time→text：{[w.objectName() for w in order]}")
    if not ok:
        failures.append("字幕行顺序不是 chip|time|text")
    ok = row.time.text() == "01:05" and row.chip.text() == "说话人A"
    print(f"[{'OK ' if ok else 'FAIL'}] 时间={row.time.text()} 角色={row.chip.text()} 样式={row.chip.objectName()}")
    if not ok:
        failures.append("时间/说话人显示不正确")

    print()
    if failures:
        print("存在失败项：")
        for f in failures:
            print(" -", f)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
