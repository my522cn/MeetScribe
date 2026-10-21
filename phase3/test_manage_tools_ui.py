"""管理页 UI 回归：删除按钮可用性 / 会后整理工具按钮 / 编辑转写往返 / 按钮配色区分。

覆盖用户反馈过的两个点：
1. “删除会议一直是禁用状态” —— 曾因 self.btn_delete 被一个未加入布局的重复按钮覆盖，
   标题栏那个永远 disabled。这里断言实际可见的按钮是 enabled 的。
2. “生成纪要按钮颜色和播放按钮一样，播放中变暂停颜色也一样” —— 断言 objectName 互不相同。

全部操作在临时 session_root 下进行，不触碰真实会话。
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from run_app import _light_palette  # noqa: E402
from meetscribe.ui.manage_page import ManagePage  # noqa: E402
from meetscribe.ui.theme import QSS  # noqa: E402


class StubSettings:
    """只提供 ManagePage 需要的接口。"""

    def __init__(self, root: Path) -> None:
        self.session_root = root

    def get(self, key, default=None):
        return default


def make_session(
    root: Path, name: str, status: str, lines: list[str], audio: bool = False
) -> Path:
    d = root / name
    (d / "mic").mkdir(parents=True, exist_ok=True)
    (d / "loop").mkdir(parents=True, exist_ok=True)
    segments = []
    if audio:
        wav = d / "mic" / "seg_0001.wav"
        with wave.open(str(wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(b"".join(struct.pack("<h", 0) for _ in range(16000)))
        segments.append(
            {"track": "mic", "file": "mic/seg_0001.wav", "start_ms": 0, "duration_ms": 1000}
        )
    (d / "session.json").write_text(
        json.dumps(
            {
                "title": f"测试会议 {name}",
                "started_at": "2026-10-21 10:00:00",
                "ended_at": "2026-10-21 10:30:00",
                "duration_ms": 1800_000,
                "mode": "offline",
                "status": status,
                "segments": segments,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    recs = [
        {"start_ms": i * 3000, "end_ms": i * 3000 + 2800, "speaker": spk, "text": txt}
        for i, (spk, txt) in enumerate(lines)
    ]
    # 真实格式是 transcript.jsonl（每行一条 JSON）
    (d / "transcript.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8"
    )
    return d


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    app.setStyleSheet(QSS)

    tmp = Path(tempfile.mkdtemp(prefix="ms_manage_"))
    root = tmp / "sessions"
    root.mkdir()
    make_session(
        root,
        "20261021_100000_manual",
        "completed",
        [("说话人A", "今天先对一下进度。"), ("说话人B", "我这边测试通过了。")],
        audio=True,
    )
    make_session(
        root,
        "20261021_110000_manual",
        "recording",
        [("说话人A", "异常退出的会话。")],
    )

    page = ManagePage(StubSettings(root))
    page.resize(1180, 800)
    page.show()
    for _ in range(10):
        app.processEvents()
    page.refresh()

    def select(name: str) -> None:
        """按会话目录名选中（列表按时间倒序，不能靠固定行号）。"""
        for i in range(page.list.count()):
            it = page.list.item(i)
            if name in page.list.item(i).text():
                page.list.setCurrentRow(i)
                break
        for _ in range(20):
            app.processEvents()

    checks: list[tuple[str, bool, str]] = []

    # 1. 扫描到两个会话
    checks.append(("扫描到 2 个会话", len(page._sessions) == 2, f"实际 {len(page._sessions)}"))

    # 2. 删除按钮：必须真的在界面上且可用（回归“一直禁用”）
    select("20261021_100000")
    checks.append(
        (
            "删除按钮可见且可用",
            page.btn_delete.isVisible() and page.btn_delete.isEnabled(),
            f"visible={page.btn_delete.isVisible()} enabled={page.btn_delete.isEnabled()}",
        )
    )
    # 3. 删除按钮挂在标题栏（紧邻刷新按钮），不是孤立控件
    head_ok = page.btn_delete.parent() is page and page.btn_delete.y() < page.list.y()
    checks.append(("删除按钮位于标题栏一行", head_ok, f"y={page.btn_delete.y()} list_y={page.list.y()}"))

    # 4. 会后整理工具按钮在选中会话后全部可用
    tools = [page.btn_edit, page.btn_fix, page.btn_names, page.btn_compress]
    checks.append(
        ("整理工具按钮全部可用", all(b.isEnabled() for b in tools),
         ", ".join(f"{b.text()}={b.isEnabled()}" for b in tools))
    )

    # 5. 已完成会话不显示“标记为已结束”
    checks.append(("已完成会话隐藏修复按钮", not page.btn_repair.isVisible(),
                   f"status={page._current.status} visible={page.btn_repair.isVisible()}"))

    # 6. 异常会话显示“标记为已结束”
    select("20261021_110000")
    checks.append(("异常会话显示修复按钮", page.btn_repair.isVisible(),
                   f"status={page._current.status} visible={page.btn_repair.isVisible()}"))

    # 7. 编辑转写 → 可写；再点 → 保存并回读
    select("20261021_100000")
    checks.append(("默认只读", page.txt.isReadOnly(), f"readOnly={page.txt.isReadOnly()}"))
    page.btn_edit.setChecked(True)
    page._toggle_edit()
    checks.append(("进入编辑可写", not page.txt.isReadOnly() and page.btn_edit.text() == "保存转写",
                   f"readOnly={page.txt.isReadOnly()} text={page.btn_edit.text()}"))
    original = page.txt.toPlainText()
    page.txt.setPlainText(original.replace("测试通过了", "测试全流程通过了"))
    page.btn_edit.setChecked(False)
    page._toggle_edit()
    saved = (page._current.path / "transcript.jsonl").read_text(encoding="utf-8")
    checks.append(("保存转写写回文件", "测试全流程通过了" in saved and page.txt.isReadOnly(),
                   f"命中={'测试全流程通过了' in saved} readOnly={page.txt.isReadOnly()}"))

    # 8. 三个关键按钮配色互不相同
    names = (page.btn_minutes.objectName(), page.btn_play.objectName(), page.btn_delete.objectName())
    checks.append(("纪要/播放/删除配色区分", len(set(names)) == 3, f"{names}"))

    # 9. 播放按钮三态：开始播放 → 白底“⏸ 暂停”(Ghost)；停止 → 绿底“▶ 播放”(Success)
    if page._current.n_segments > 0:
        page.btn_play.click()
        for _ in range(20):
            app.processEvents()
            time.sleep(0.03)
        playing_ok = page.btn_play.objectName() == "Ghost" and "暂停" in page.btn_play.text()
        checks.append(
            (
                "播放中按钮变白底“暂停”",
                playing_ok,
                f"style={page.btn_play.objectName()} text={page.btn_play.text()}",
            )
        )
        page._stop_play()
        for _ in range(3):
            app.processEvents()
        checks.append(
            (
                "停止后按钮回绿底“播放”",
                page.btn_play.objectName() == "Success" and "播放" in page.btn_play.text()
                and page.slider.value() == 0,
                f"style={page.btn_play.objectName()} text={page.btn_play.text()} pos={page.slider.value()}",
            )
        )
    else:
        checks.append(("播放配色切换", False, "测试会话缺少音频分段"))

    page.grab().save(str(ROOT / "phase3" / "ui_manage_tools.png"))
    print("saved ui_manage_tools.png")

    print()
    bad = []
    for name, ok, detail in checks:
        print(f"[{'OK ' if ok else 'FAIL'}] {name}　{detail}")
        if not ok:
            bad.append(name)
    print()
    if bad:
        print("❌ 失败项：", bad)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
