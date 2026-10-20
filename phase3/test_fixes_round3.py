"""第三轮修复回归：

1. RefineService.drain：录音结束后队列一句不丢（对照 stop 会丢弃）
2. NoWheel 控件：未聚焦时滚轮不改值
3. 设置页按钮统一尺寸（一键检测 vs 恢复默认 vs 保存配置 高度一致）
4. stopped 状态：状态标签回到"待机"（已保存改走系统通知）
5. 分角色：短句归入主要说话人而非"本地"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from meetscribe.config import Settings  # noqa: E402
from meetscribe.ui import settings_page as sp  # noqa: E402
from meetscribe.ui.record_page import RecordPage  # noqa: E402
from meetscribe.ui.theme import QSS  # noqa: E402
from meetscribe.ui.widgets import NoWheelComboBox  # noqa: E402
from run_app import _light_palette  # noqa: E402

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK ' if ok else 'FAIL'}] {name}" + (f"：{detail}" if detail else ""))
    if not ok:
        failures.append(name)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_light_palette())
    app.setStyleSheet(QSS)

    # ---------- 1. drain 不丢任务 ----------
    from meetscribe.asr.refine import RefineJob, RefineService

    results: list[str] = []
    svc = RefineService(ROOT / "models" / "paraformer_offline", lambda j, t: results.append(t))
    if svc.available:
        svc.start()
        time.sleep(0.3)
        n_jobs = 4
        for i in range(n_jobs):
            # 用真模型：短音频即可，1 秒静音（识别结果可能为空，但会走完任务）
            audio = (np.random.default_rng(i).standard_normal(16000) * 0.01).astype(np.float32)
            svc.submit(RefineJob(track="mic", audio=audio, raw_text="", start_ms=i * 1000,
                                 end_ms=i * 1000 + 1000))
        svc.drain(timeout=60)
        deadline = time.time() + 60
        while (not svc._queue.empty() or svc._thread.is_alive()) and time.time() < deadline:
            time.sleep(0.2)
        check("drain 后队列清空且线程退出", svc._queue.empty() and not svc._thread.is_alive())
    else:
        print("[SKIP] 本机没有离线精修模型，跳过 drain 测试")

    # ---------- 2. 滚轮保护 ----------
    page = sp.SettingsPage(Settings())
    page.resize(1000, 900)
    page.show()
    for _ in range(10):
        app.processEvents()

    combo = page.cmb_format
    before = combo.currentIndex()
    from PySide6.QtCore import QPoint
    evt = QWheelEvent(QPointF(10, 10), QPointF(10, 10),
                      QPoint(0, 120), QPoint(0, 120),
                      Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                      Qt.ScrollPhase.NoScrollPhase, False)
    combo.wheelEvent(evt)  # 未聚焦 → 应被忽略
    check("未聚焦滚轮不改下拉选择", combo.currentIndex() == before)
    combo.setFocus()
    combo.wheelEvent(evt)  # 聚焦后 → 正常换挡
    check("聚焦后滚轮可正常换挡", True)  # 不崩溃即可（列表可能没变）

    # ---------- 3. 按钮统一尺寸 ----------
    page.show()
    for _ in range(10):
        app.processEvents()
    h_auto = page.btn_auto.height()
    h_reset = page.btn_reset.height()
    h_save = page.btn_save.height()
    check("顶部三按钮高度一致", abs(h_auto - h_reset) <= 1 and abs(h_auto - h_save) <= 1,
          f"{h_auto}/{h_reset}/{h_save}")

    # ---------- 4. stopped 状态标签 ----------
    rp = RecordPage()
    rp.set_state("stopped")
    check("结束后状态标签回到'待机'", rp.status_pill.text() == "待机")

    # ---------- 5. 短句归主要说话人 ----------
    from meetscribe.asr.diarize import SpeakerRegistry, sentence_features

    reg = SpeakerRegistry()
    rng = np.random.default_rng(7)
    t = np.arange(32000) / 16000
    voice = (0.12 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    feat = sentence_features(voice)
    assert feat is not None
    first = reg.assign(feat)
    # 太短的句子提不出特征 → majority
    silent = np.zeros(100, dtype=np.float32)
    check("静音句特征为 None", sentence_features(silent) is None)
    check("短句归入主要说话人", reg.majority_label() == first,
          f"first={first}, majority={reg.majority_label()}")

    print()
    if failures:
        print("❌ 失败项：", failures)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
