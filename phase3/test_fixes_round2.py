"""第二轮修复回归：

设置页：
1. 测试按钮在识别完成后恢复可点（_release bug）
2. 页面无横向滚动条（动态适应窗口宽度）
3. LLM 纠错/纪要模型下拉存在且可见
会议管理页：
4. 多选批量删除（在临时目录伪造会话，不动真实数据）
5. 纪要状态按会话跟踪：生成中禁按钮、完成后恢复 + 状态随会话切换
6. 回放停止后进度归零
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from meetscribe.config import Settings  # noqa: E402
from meetscribe.ui.manage_page import ManagePage  # noqa: E402
from meetscribe.ui.settings_page import SettingsPage  # noqa: E402
from meetscribe.ui.theme import QSS  # noqa: E402
from run_app import _light_palette  # noqa: E402

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK ' if ok else 'FAIL'}] {name}" + (f"：{detail}" if detail else ""))
    if not ok:
        failures.append(name)


def make_fake_session(root: Path, name: str, segs: int = 1) -> Path:
    d = root / name
    (d / "mic").mkdir(parents=True)
    manifest = {
        "session_id": name, "title": name, "source": "manual", "mode": "online",
        "sample_rate": 16000, "started_at": "2026-10-20T10:00:00",
        "ended_at": "2026-10-20T10:01:00", "status": "completed",
        "tracks": {}, "duration_ms": 60000,
        "segments": [
            {"track": "mic", "index": i, "file": f"mic/seg_{i:04d}.flac",
             "start_ms": i * 30000, "duration_ms": 30000, "samples": 480000}
            for i in range(segs)
        ],
    }
    (d / "session.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "transcript.jsonl").write_text(
        '{"track":"mic","speaker":"我","start_ms":0,"end_ms":2000,"text":"测试句子"}\n',
        encoding="utf-8",
    )
    return d


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    app.setStyleSheet(QSS)

    # ============ 设置页 ============
    page = SettingsPage(Settings())
    page.resize(1000, 900)
    page.show()
    for _ in range(10):
        app.processEvents()

    # 1. 识别完成后测试按钮恢复
    page.btn_speak_test.setEnabled(False)
    page.btn_test_mic.setEnabled(False)
    page.btn_test_loop.setEnabled(False)
    page._on_recognized("你好", "你好", 0.1, 2.0)
    check("识别完成后测试按钮恢复可点",
          page.btn_speak_test.isEnabled() and page.btn_test_mic.isEnabled()
          and page.btn_test_loop.isEnabled())

    # 2. 无横向滚动条
    for w in (900, 1000, 1200):
        page.resize(w, 900)
        for _ in range(10):
            app.processEvents()
        hmax = page.findChildren(type(page).__mro__ and __import__("PySide6.QtWidgets", fromlist=["QScrollArea"]).QScrollArea)
        from PySide6.QtWidgets import QScrollArea
        bars = [s.horizontalScrollBar().maximum() for s in page.findChildren(QScrollArea)]
        check(f"窗口宽 {w}px 时无横向滚动条", all(b == 0 for b in bars), f"maximum={bars}")

    # 3. LLM 模型下拉存在且可见
    check("纠错模型下拉可见", page.cmb_fix_model.isVisible() and page.cmb_fix_model.height() >= 20,
          f"h={page.cmb_fix_model.height()}, editable={page.cmb_fix_model.isEditable()}")
    check("纪要模型下拉可见", page.cmb_sum_model.isVisible() and page.cmb_sum_model.height() >= 20,
          f"h={page.cmb_sum_model.height()}, editable={page.cmb_sum_model.isEditable()}")
    page.grab().save(str(ROOT / "phase3" / "ui_settings_r2.png"))
    # 滚到设置页底部再截一张（LLM 卡片在页面下方）
    from PySide6.QtWidgets import QScrollArea
    sa = page.findChild(QScrollArea)
    sa.verticalScrollBar().setValue(sa.verticalScrollBar().maximum())
    for _ in range(10):
        app.processEvents()
    page.grab().save(str(ROOT / "phase3" / "ui_settings_bottom.png"))

    # ============ 会议管理页 ============
    tmp = Path(tempfile.mkdtemp(prefix="ms_manage_test_"))
    s = Settings()
    s.set("storage.session_root", str(tmp))
    mp = ManagePage(s)
    mp.resize(1180, 780)
    mp.show()
    for _ in range(10):
        app.processEvents()

    p1 = make_fake_session(tmp, "20261020_100000_测试一")
    p2 = make_fake_session(tmp, "20261020_100100_测试二")
    p3 = make_fake_session(tmp, "20261020_100200_测试三")
    mp.refresh()
    check("扫描到 3 个伪造会话", len(mp._sessions) == 3, f"实际 {len(mp._sessions)}")

    # 4. 批量删除
    mp.list.selectAll()
    orig = QMessageBox.question
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    try:
        mp._delete_sessions()
    finally:
        QMessageBox.question = orig
    for _ in range(10):
        app.processEvents()
    left = [d for d in tmp.iterdir() if d.is_dir()]
    check("批量删除后目录清空", len(left) == 0, f"剩余 {left}")

    # 5. 纪要状态按会话跟踪
    make_fake_session(tmp, "20261020_100300_甲")
    make_fake_session(tmp, "20261020_100400_乙")
    mp.refresh()
    mp.list.setCurrentRow(0)  # 甲
    for _ in range(5):
        app.processEvents()
    cur = mp._current
    mp._minute_state[str(cur.path)] = "generating"
    mp._apply_minute_ui()
    check("生成中：纪要按钮禁用", not mp.btn_minutes.isEnabled())
    mp.list.setCurrentRow(1)  # 乙
    for _ in range(5):
        app.processEvents()
    check("切到其他会议：按钮可点且状态不串", mp.btn_minutes.isEnabled() and "正在" not in mp.status.text())
    mp.list.setCurrentRow(0)  # 回到甲
    for _ in range(5):
        app.processEvents()
    check("切回生成中的会议：仍显示生成中", not mp.btn_minutes.isEnabled() and mp.btn_minutes.text() == "生成中…")
    mp._on_minute_finished(str(cur.path), True, "纪要已生成")
    for _ in range(5):
        app.processEvents()
    check("完成后按钮恢复", mp.btn_minutes.isEnabled())
    check("完成后状态显示成功", "纪要已生成" in mp.status.text())

    # 6. 回放停止归零
    mp.slider.setValue(12345)
    mp._player = None
    mp._stop_play()
    check("停止后进度归零", mp.slider.value() == 0 and mp.lbl_playpos.text().startswith("00:00"))

    print()
    if failures:
        print("❌ 失败项：", failures)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
