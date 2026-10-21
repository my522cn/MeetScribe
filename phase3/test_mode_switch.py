"""线上线下模式切换回归：

1. 待机/已结束状态可自由切换（老 bug：录过一次就永远选不了）
2. 录音中/暂停同样可以切换（新需求：热切换），信号正确发出且无重复
3. 重复点同一个模式不重复发信号
4. worker.set_mode：未录音时只记模式；录音中切换会更新 manifest、
   切到线下时热建声纹聚类器并打开 diarize.log，切回线上保留聚类器
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from run_app import _light_palette  # noqa: E402
from meetscribe.config import Settings  # noqa: E402
from meetscribe.ui.record_page import RecordPage, SPEAKER_OFFLINE, SPEAKER_ONLINE  # noqa: E402
from meetscribe.ui.theme import QSS  # noqa: E402
from meetscribe.ui.worker import CaptureWorker  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


class FakeManifest:
    def __init__(self) -> None:
        self.mode = "online"


class FakeRecorder:
    def __init__(self, session_dir: Path) -> None:
        self.manifest = FakeManifest()
        self.session_dir = session_dir


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    app.setStyleSheet(QSS)

    # ---------- 录音页 ----------
    rp = RecordPage()
    fired: list[str] = []
    rp.mode_changed.connect(fired.append)
    rp.show()
    for _ in range(5):
        app.processEvents()

    # 1. 待机切换
    rp._set_mode("offline")
    check("待机可切换到线下", fired == ["offline"] and rp._mode == "offline"
          and rp.offline_btn.isChecked() and not rp.online_btn.isChecked(),
          f"fired={fired} mode={rp._mode}")
    check("字幕说话人标注随模式更新", rp.captions._meta == SPEAKER_OFFLINE)

    # 2. 已结束（stopped）状态切换 —— 老bug回归点
    rp.set_state("stopped")
    rp._set_mode("online")
    check("已结束状态可切回线上", fired == ["offline", "online"] and rp._mode == "online"
          and rp.online_btn.isChecked(),
          f"fired={fired}")

    # 3. 录音中热切换 —— 本次修复的核心
    rp.set_state("recording")
    rp._set_mode("offline")
    check("录音中可切到线下", fired == ["offline", "online", "offline"] and rp._mode == "offline"
          and rp.offline_btn.isChecked() and not rp.online_btn.isChecked(),
          f"fired={fired}")
    rp.set_state("paused")
    rp._set_mode("online")
    check("暂停中同样可切换", fired[-1] == "online" and rp._mode == "online")

    # 4. 重复点击不重复发信号（前 4 次真实切换：off/on/off/on）
    rp._set_mode("online")
    check("重复点同一模式不发信号", fired[-1] == "online" and len(fired) == 4,
          f"fired={fired}")

    # ---------- worker ----------
    tmp = Path(tempfile.mkdtemp(prefix="ms_mode_"))
    sess = tmp / "session"
    (sess / "mic").mkdir(parents=True)
    worker = CaptureWorker(ROOT / "models" / "paraformer_zh_en", tmp / "data", ROOT / "config")
    worker.mode = "online"

    # 5. 未录音时切换：只记模式，不建聚类器（start() 会按新模式初始化）
    worker.set_mode("offline")
    check("未录音切换只记模式", worker.mode == "offline" and worker.diarizer is None)

    # 6. 录音中切到线下：manifest 更新 + 热建聚类器 + 打开 diarize.log
    worker.recorder = FakeRecorder(sess)
    worker.set_mode("online")  # 先回到线上再切，验证完整路径
    worker.set_mode("offline")
    log_exists = (sess / "diarize.log").exists()
    check("录音中切线下：manifest/聚类器/日志齐备",
          worker.recorder.manifest.mode == "offline" and worker.diarizer is not None and log_exists,
          f"manifest={worker.recorder.manifest.mode} diarizer={worker.diarizer is not None} log={log_exists}")

    # 7. 切回线上：聚类器保留（再切回线下可沿用本场聚类结果）
    worker.set_mode("online")
    check("切回线上保留聚类器", worker.mode == "online" and worker.diarizer is not None
          and worker.recorder.manifest.mode == "online")

    # 8. 重复设置同一模式为无操作
    d_before = worker.diarizer
    worker.set_mode("online")
    check("重复设置同一模式无操作", worker.diarizer is d_before)

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
