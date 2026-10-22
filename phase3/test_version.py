# -*- coding: utf-8 -*-
"""版本号 + 设置页推荐模型回归。

版本规则：大版本.小版本.修订版.日期（月日，3~4 位），每次更新递增。
验证：
1. 版本号格式合法（4 段、最后一段 3~4 位数字）
2. 主窗口标题带版本号；侧边栏与托盘提示也带
3. 设置页显示版本号
4. 设置页"推荐模型"行给出最优组合，且能一键应用
5. 改了模型后提示"与推荐不一致"，应用推荐后恢复一致
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from meetscribe import (  # noqa: E402
    RECOMMEND_FIX_MODEL, RECOMMEND_SUMMARY_MODEL, __version__, app_title,
)
from meetscribe.config import Settings  # noqa: E402
from meetscribe.ui.settings_page import SettingsPage  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


def main() -> int:
    app = QApplication.instance() or QApplication([])

    # 1. 版本号格式
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)\.(\d{3,4})", __version__)
    check("版本号格式合法", m is not None, f"v{__version__}")
    if m:
        date_part = m.group(4)
        check("日期段 3~4 位且是合法月日",
              len(date_part) in (3, 4)
              and 101 <= int(date_part) <= 1231,
              f"日期段={date_part}")

    # 2. 主窗口标题 / 托盘
    from meetscribe.ui.main_window import MainWindow

    win = MainWindow()
    # 测试只关心版本展示：停掉会议检测轮询，避免它在后台误触发自动录音
    win.detect_timer.stop()
    win.auto_record = False
    check("主窗口标题带版本", __version__ in win.windowTitle(), win.windowTitle())
    check("托盘提示带版本", __version__ in win.tray.toolTip(), win.tray.toolTip())
    from PySide6.QtWidgets import QLabel

    ver_labels = [lbl.text() for lbl in win.findChildren(QLabel)
                  if lbl.text().startswith("v") and "." in lbl.text()]
    check("侧边栏显示版本", any(__version__ in t for t in ver_labels), f"{ver_labels}")
    win.close()

    # 3. 设置页
    s = Settings()
    page = SettingsPage(s)
    page_labels = [lbl.text() for lbl in page.findChildren(QLabel)]
    check("设置页显示版本", any(t == f"v{__version__}" for t in page_labels), "")

    # 4. 推荐模型行
    rec = page.rec_text.text()
    check("推荐模型含纠错推荐", RECOMMEND_FIX_MODEL in rec, "")
    check("推荐模型含纪要推荐", RECOMMEND_SUMMARY_MODEL in rec, "")

    # 5. 改模型 → 提示不一致；点应用推荐 → 恢复一致
    page.cmb_fix_model.setCurrentText("qwen2.5:7b-instruct")
    check("偏离推荐时提示不一致", "不一致" in page.rec_text.text()
          and page.btn_rec_apply.isEnabled(), "")
    page.cmb_fix_model.setCurrentText(RECOMMEND_FIX_MODEL)
    page.cmb_sum_model.setCurrentText(RECOMMEND_SUMMARY_MODEL)
    check("回到推荐时按钮置灰", not page.btn_rec_apply.isEnabled()
          and "已是推荐" in page.rec_text.text(), page.rec_text.text().splitlines()[-1])

    page.cmb_sum_model.setCurrentText("qwen2.5:3b-instruct")
    page.btn_rec_apply.click()
    check("一键应用推荐写入两个下拉",
          page.cmb_fix_model.currentText() == RECOMMEND_FIX_MODEL
          and page.cmb_sum_model.currentText() == RECOMMEND_SUMMARY_MODEL,
          f"{page.cmb_fix_model.currentText()} / {page.cmb_sum_model.currentText()}")

    # 6. 900px 下不出现横向滚动（新增控件不能把页面撑宽）
    page.resize(900, 800)
    page.show()
    app.processEvents()
    from PySide6.QtWidgets import QScrollArea

    areas = page.findChildren(QScrollArea)
    max_width = max([a.horizontalScrollBar().maximum() for a in areas] or [0])
    check("900px 无横向滚动条", max_width == 0, f"maximum={max_width}")

    print()
    for n, ok, d in checks:
        print(f"[{'OK ' if ok else 'FAIL'}] {n}　{d}")
    print()
    bad = [n for n, ok, _ in checks if not ok]
    if bad:
        print("❌ 失败项：", bad)
        return 1
    print(f"全部通过 ✅　当前版本 v{__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
