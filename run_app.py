"""MeetScribe 启动入口：python run_app.py"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QMessageBox

ROOT = Path(__file__).resolve().parent
if not getattr(sys, "frozen", False):  # 打包后模块已进 PYZ，不需要改 path
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


def selftest() -> int:
    """`MeetScribe.exe --selftest`：验证打包后模型/配置/原生库都能用。

    窗口程序没有控制台，结果同时写进 exe 旁边的 selftest.txt，
    打包脚本读它判断 exe 是否真的能用（能起窗口 ≠ 能录音识别）。
    """
    from meetscribe.paths import project_root

    root = project_root()
    lines: list[str] = []
    ok = True

    def check(name: str, fn) -> None:
        nonlocal ok
        try:
            detail = fn()
            lines.append(f"[OK ] {name}　{detail}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            lines.append(f"[FAIL] {name}　{exc!r}")

    lines.append(f"frozen={getattr(sys, 'frozen', False)} root={root}")

    def models() -> str:
        need = [
            "paraformer_zh_en/encoder.int8.onnx",
            "paraformer_zh_en/decoder.int8.onnx",
            "paraformer_zh_en/tokens.txt",
            "paraformer_offline/model.int8.onnx",
            "punct_zh_en/model.int8.onnx",
            "campplus_zh/campplus_zh.onnx",
        ]
        missing = [p for p in need if not (root / "models" / p).exists()]
        if missing:
            raise FileNotFoundError(f"缺模型：{missing}")
        n_emb = 0
        try:
            from meetscribe.asr.diarize import EmbeddingSpeakerRegistry

            n_emb = EmbeddingSpeakerRegistry(root / "models" / "campplus_zh" / "campplus_zh.onnx").dim
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"声纹嵌入不可用：{exc!r}")
        return f"{len(need)} 个文件齐全（声纹嵌入 {n_emb} 维）"

    def settings() -> str:
        from meetscribe.config import Settings

        s = Settings(root / "config" / "settings.json")
        return f"session_root={s.session_root}"

    def asr() -> str:
        import time

        import numpy as np

        from meetscribe.asr.engine import ASRConfig, StreamingASR

        t0 = time.time()
        eng = StreamingASR(ASRConfig(model_dir=root / "models" / "paraformer_zh_en", num_threads=2))
        eng.feed("t", np.zeros(8000, dtype=np.float32))
        eng.finish_sentence("t")
        return f"流式识别可用 {time.time() - t0:.1f}s"

    def offline() -> str:
        from meetscribe.asr.offline import get_offline

        a = get_offline(root / "models" / "paraformer_offline", "paraformer", 2)
        a.load()
        return f"离线精修模型可用 rtf={a.last_rtf:.2f}"

    def audio() -> str:
        import sounddevice as sd

        ins = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
        import soundcard as sc

        spk = sc.default_speaker()
        if not ins:
            raise RuntimeError("没有可用输入设备")
        return f"{len(ins)} 个输入设备，回环={spk.name}"

    check("模型文件", models)
    check("配置读取", settings)
    check("流式识别", asr)
    check("离线精修", offline)
    check("音频设备", audio)

    report = root / "selftest.txt"
    report.write_text("\n".join(lines) + f"\n结论：{'通过' if ok else '失败'}\n", encoding="utf-8")
    return 0 if ok else 1


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()

    app = QApplication(sys.argv)
    app.setApplicationName("会议转写助手")
    app.setWindowIcon(make_icon())
    app.setQuitOnLastWindowClosed(False)
    # 顺序很重要：先样式，再调色板，最后样式表
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    app.setStyleSheet(QSS)

    # 单实例：已经开着一个就把它的窗口抬到最前，自己静默退出
    from meetscribe.ui.single_instance import SingleInstance

    guard = SingleInstance()
    if not guard.acquire():
        return 0

    try:
        win = MainWindow()
    except Exception as exc:  # noqa: BLE001
        QMessageBox.critical(None, "启动失败", f"{exc}\n\n{traceback.format_exc()}")
        return 1
    guard.activated.connect(win.bring_to_front)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
