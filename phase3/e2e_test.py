"""阶段三端到端验收：完整应用内跑一遍 自动录音→实时识别→结束保存。"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sounddevice as sd  # noqa: E402

from meetscribe.audio.devices import default_output_device  # noqa: E402
from meetscribe.ui.main_window import MainWindow  # noqa: E402
from meetscribe.ui.theme import QSS  # noqa: E402

stats = {"partial": 0, "final": 0, "overlap": 0, "segments": 0}


def play_audio(path: Path) -> None:
    audio, sr = sf.read(path, dtype="float32")
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    out_rate = 48000
    if sr != out_rate:
        mono = soxr.resample(mono, sr, out_rate)
        sr = out_rate
    dev = default_output_device()
    with sd.OutputStream(device=dev.index if dev else None, samplerate=sr, channels=1, dtype="float32") as out:
        step = int(sr * 0.05)
        for i in range(0, len(mono), step):
            out.write(mono[i:i + step].reshape(-1, 1))


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = MainWindow()
    win.show()
    win.record_page.captions.clear()

    win.worker.partial.connect(lambda *a: stats.__setitem__("partial", stats["partial"] + 1))
    win.worker.final.connect(lambda *a: stats.__setitem__("final", stats["final"] + 1))
    win.worker.overlap.connect(lambda *a: stats.__setitem__("overlap", stats["overlap"] + 1))
    win.worker.segment_count.connect(lambda *a: stats.__setitem__("segments", stats["segments"] + 1))

    phase = {"n": 0}
    session_dir = {"path": ""}

    def on_state(state: str) -> None:
        if state == "recording" and phase["n"] == 0:
            phase["n"] = 1
            threading.Thread(target=play_audio, args=(ROOT / "phase2" / "test_zh_en_16k.wav",), daemon=True).start()
            QTimer.singleShot(20000, win._stop)
        elif state == "stopped":
            phase["n"] = 2
            session_dir["path"] = win.worker.recorder.session_dir.as_posix() if win.worker.recorder else ""
            finish()

    def finish() -> None:
        print("===== 端到端结果 =====")
        print(f"临时文本刷新: {stats['partial']} 次")
        print(f"定稿句: {stats['final']}")
        print(f"重叠事件: {stats['overlap']}")
        print(f"写盘分段: {stats['segments']}")
        d = Path(session_dir["path"])
        if d.exists():
            flacs = list(d.rglob("*.flac"))
            print(f"音频文件: {len(flacs)} 个, 共 {sum(f.stat().st_size for f in flacs)/1024:.0f} KB")
            print(f"清单: {(d / 'session.json').exists()}")
        app.quit()

    win.worker.state_changed.connect(on_state)
    win._start("online", source="e2e", title="端到端验收")
    QTimer.singleShot(13000, lambda: win.grab().save(str(ROOT / "phase3" / "ui_live.png")))
    QTimer.singleShot(60000, finish)  # 兜底超时
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
