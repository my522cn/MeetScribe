"""会议管理页回归：会话扫描、详情展示、转写渲染。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from meetscribe.config import Settings  # noqa: E402
from meetscribe.ui.manage_page import ManagePage, SessionInfo  # noqa: E402
from run_app import _light_palette  # noqa: E402
from meetscribe.ui.theme import QSS  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    app.setStyleSheet(QSS)

    settings = Settings()
    page = ManagePage(settings)
    page.resize(1180, 780)
    page.show()
    for _ in range(10):
        app.processEvents()
    page.refresh()

    sessions = page._sessions
    print(f"扫描到 {len(sessions)} 个会话")
    if not sessions:
        print("❌ 没有扫描到任何会话（data/sessions 下应有测试会话）")
        return 1

    # 详情面板跟随选中项
    cur = page.list.currentItem()
    info: SessionInfo = cur.data(0x0100)  # Qt.UserRole
    page._on_select(cur, None)
    ok_title = page.lbl_title.text() == info.title
    print(f"[{'OK ' if ok_title else 'FAIL'}] 标题：{page.lbl_title.text()}（期望 {info.title}）")

    txt = page.txt.toPlainText()
    if info.n_lines:
        ok_txt = bool(txt.strip())
        print(f"[{'OK ' if ok_txt else 'FAIL'}] 转写渲染 {len(txt)} 字符，共 {info.n_lines} 句")
        print("      首行示例：", txt.splitlines()[0][:80])
    else:
        ok_txt = "没有可显示" in txt
        print(f"[{'OK ' if ok_txt else 'FAIL'}] 无转写会话正确显示占位提示")

    # 纪要未启用 LLM 时按钮提示
    page.btn_minutes.setEnabled(True)
    page._gen_minutes()
    s1 = page.status.text()
    ok_llm = "启用本地大模型" in s1 or "生成" in s1
    print(f"[{'OK ' if ok_llm else 'FAIL'}] 生成纪要入口响应：{s1}")

    # 截图留档
    page.grab().save(str(ROOT / "phase3" / "ui_manage.png"))
    print("saved ui_manage.png")

    failures = [n for n, ok in (("title", ok_title), ("transcript", ok_txt), ("minutes", ok_llm)) if not ok]
    if failures:
        print("❌ 失败项：", failures)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
