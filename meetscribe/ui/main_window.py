"""主窗口：左侧导航（录音 / 会议管理）+ 系统托盘 + 会议软件自动检测。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QHBoxLayout, QLabel, QMainWindow, QMenu,
    QMessageBox, QPushButton, QStackedWidget, QSystemTrayIcon, QVBoxLayout, QWidget,
)

from ..audio.devices import default_input_device, find_loopback_source
from ..config import Settings
from ..system.detect import MeetingDetector, capability
from .caption_view import fmt_ms
from .float_bar import FloatBar
from ..paths import project_root
from .manage_page import ManagePage
from .record_page import RecordPage
from .settings_page import SettingsPage
from .worker import CaptureWorker

PROJECT_ROOT = project_root()
SESSION_ROOT = PROJECT_ROOT / "data" / "sessions"
MODEL_DIR = PROJECT_ROOT / "models" / "paraformer_zh_en"


def make_icon() -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor("#534AB7"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(4, 4, 56, 56, 14, 14)
    p.setPen(QColor("#FFFFFF"))
    f = p.font()
    f.setPixelSize(30)
    f.setBold(True)
    p.setFont(f)
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "M")
    p.end()
    return QIcon(pm)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("会议转写助手")
        self.setWindowIcon(make_icon())
        self.resize(1180, 780)
        self.setMinimumSize(980, 640)

        self.settings = Settings()
        self.worker = CaptureWorker(MODEL_DIR, SESSION_ROOT, settings=self.settings)
        self.detector = MeetingDetector()
        self.auto_record = bool(self.settings.get("ui.auto_record", True))
        self._tray_hinted = False
        self._source = "manual"
        self._draining = False

        self._build_ui()
        self._build_tray()
        self._wire()

        self.ui_timer = QTimer(self)
        self.ui_timer.setInterval(200)
        self.ui_timer.timeout.connect(self._tick)
        self.ui_timer.start()

        self.detect_timer = QTimer(self)
        self.detect_timer.setInterval(3000)
        self.detect_timer.timeout.connect(self._poll_meetings)
        self.detect_timer.start()

        self.detector.prime()
        self._show_devices()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(176)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(12, 18, 12, 14)
        sl.setSpacing(6)

        brand = QLabel("会议转写助手")
        brand.setStyleSheet("font-size:14px; font-weight:500; padding:0 8px 10px 8px;")
        sl.addWidget(brand)

        self.nav_group = QButtonGroup(self)
        self.btn_record = self._nav_button("录音")
        self.btn_manage = self._nav_button("会议管理")
        self.btn_settings = self._nav_button("设置")
        self.btn_record.setChecked(True)
        sl.addWidget(self.btn_record)
        sl.addWidget(self.btn_manage)
        sl.addWidget(self.btn_settings)
        sl.addStretch(1)

        self.detect_label = QLabel(f"会议检测\n{capability()}")
        self.detect_label.setObjectName("Hint")
        self.detect_label.setWordWrap(True)
        sl.addWidget(self.detect_label)
        layout.addWidget(sidebar)

        self.stack = QStackedWidget()
        self.record_page = RecordPage()
        self.stack.addWidget(self.record_page)

        self.manage_page = ManagePage(self.settings)
        self.stack.addWidget(self.manage_page)

        self.settings_page = SettingsPage(self.settings, self.worker)
        self.stack.addWidget(self.settings_page)
        layout.addWidget(self.stack, 1)

        self.setCentralWidget(root)
        self.float_bar = FloatBar()
        self._navs = [self.btn_record, self.btn_manage, self.btn_settings]

    def _nav_button(self, text: str) -> QPushButton:
        btn = QPushButton(text)
        btn.setObjectName("NavButton")
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.nav_group.addButton(btn)
        btn.clicked.connect(self._on_nav)
        return btn

    def _on_nav(self) -> None:
        for i, btn in enumerate(self._navs):
            if btn.isChecked():
                self.stack.setCurrentIndex(i)
                if i == 1:  # 切到会议管理页时重扫会话列表
                    self.manage_page.refresh()
                return

    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(make_icon(), self)
        self.tray.setToolTip("会议转写助手")
        menu = QMenu()
        self.act_show = QAction("显示主窗口", self)
        self.act_show.triggered.connect(self._restore)
        self.act_pause = QAction("暂停录音", self)
        self.act_pause.triggered.connect(self._toggle_pause)
        self.act_stop = QAction("结束录音", self)
        self.act_stop.triggered.connect(self._stop)
        self.act_auto = QAction("自动检测会议并录音", self)
        self.act_auto.setCheckable(True)
        self.act_auto.setChecked(True)
        self.act_auto.toggled.connect(self._set_auto_record)
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(self._quit)
        for a in (self.act_show, self.act_pause, self.act_stop):
            menu.addAction(a)
        menu.addSeparator()
        menu.addAction(self.act_auto)
        menu.addSeparator()
        menu.addAction(act_quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda r: self._restore() if r == QSystemTrayIcon.ActivationReason.Trigger else None)
        self.tray.show()

    def _wire(self) -> None:
        rp = self.record_page
        rp.start_requested.connect(self._start)
        rp.pause_requested.connect(self.worker.pause)
        rp.resume_requested.connect(self.worker.resume)
        rp.stop_requested.connect(self._stop)
        rp.float_requested.connect(self._toggle_float)

        self.worker.partial.connect(rp.captions.update_partial)
        self.worker.final.connect(rp.captions.finalize)
        self.worker.refined.connect(rp.captions.update_final_text)
        self.worker.live_refined.connect(rp.captions.update_live_text)
        self.worker.overlap.connect(rp.captions.set_overlap)
        self.worker.state_changed.connect(self._on_state)
        self.worker.error.connect(self._on_error)
        self.worker.segment_count.connect(
            lambda n: rp.footer.setText(f"已保存 {n} 个音频分段 · 意外关闭后可恢复")
        )
        self.worker.session_saved.connect(
            lambda p: rp.footer.setText(f"已保存到 {p}")
        )
        self.worker.session_saved.connect(lambda _p: self.manage_page.refresh())
        self.worker.session_saved.connect(self._notify_saved)

        self.float_bar.pause_clicked.connect(self._toggle_pause)
        self.float_bar.stop_clicked.connect(self._stop)
        self.float_bar.expand_clicked.connect(self._restore)

        self.settings_page.saved.connect(self._on_settings_saved)

    def _on_settings_saved(self) -> None:
        """配置改了以后：精修模型可能换了，重建服务；识别器参数由 _ensure_engine 自己判断。"""
        if not self.worker.is_active:
            self.worker.refiner = self.worker._make_refiner()
            self.record_page.footer.setText("配置已更新，下次录音生效")

    # ---------- 录音控制 ----------
    def _start(self, mode: str, source: str = "manual", title: str = "") -> None:
        if self.worker.is_active:
            return
        self._source = source
        self.record_page.clear_captions()
        self.record_page.set_state("recording")
        self.float_bar.reset()
        self.float_bar.place_top_right()
        self.float_bar.show()
        self.worker.start(mode=mode, title=title or "会议记录", source=source)

    def _stop(self) -> None:
        if not self.worker.is_active:
            return
        self.worker.stop()
        self.float_bar.hide()

    def _toggle_pause(self) -> None:
        if not self.worker.is_active:
            return
        if self.record_page._state == "recording":
            self.worker.pause()
        else:
            self.worker.resume()

    def _on_state(self, state: str) -> None:
        self.record_page.set_state(state)
        if state == "stopped":
            self.float_bar.hide()
            self.act_pause.setText("暂停录音")
        else:
            self.float_bar.set_paused(state == "paused")
            self.act_pause.setText("继续录音" if state == "paused" else "暂停录音")

    def _notify_saved(self, path: str) -> None:
        """录音结束用系统托盘气泡通知，不再挤在页面状态标签里。"""
        self.tray.showMessage(
            "录音已保存",
            f"会议文件在 {path}，错别字正在后台继续校对。",
            QSystemTrayIcon.MessageIcon.Information, 5000,
        )

    def _on_error(self, msg: str) -> None:
        self.record_page.set_state("idle")
        self.float_bar.hide()
        QMessageBox.warning(self, "出错了", msg)

    # ---------- 定时刷新 ----------
    def _tick(self) -> None:
        if not self.worker.is_active:
            # 录音已结束但精修队列还没跑完：显示后台校对进度
            if self.record_page._state == "stopped" and self.worker.refiner is not None:
                n = self.worker.refiner.pending
                if n > 0:
                    self._draining = True
                    self.record_page.set_caption_hint(f"后台校对错别字中… 还剩 {n} 句")
                elif self._draining:
                    self._draining = False
                    self.record_page.set_caption_hint("后台校对完成，全部句子已用离线模型修正")
            return
        ms = self.worker.elapsed_ms
        self.record_page.set_elapsed(ms)
        self.float_bar.set_elapsed(ms)
        mic, loop = self.worker.levels()
        self.record_page.set_levels(mic, loop)
        # 已二次校对（离线模型重识别）的句数，让用户看到纠错在后台跑
        n = self.record_page.captions.refined_count
        if self.record_page._state == "paused":
            self.record_page.set_caption_hint(f"已暂停 · 后台仍在校对，已修订 {n} 句")
        elif self.record_page._state == "recording":
            hint = f"实时识别中 · 已二次校对 {n} 句" if n else "实时识别中 · 后台正在校对错别字"
            if self.worker.diarizer is not None:
                hint += f" · 已区分 {self.worker.diarizer.n_speakers} 位说话人"
            self.record_page.set_caption_hint(hint)

    def _poll_meetings(self) -> None:
        app = self.detector.poll()
        if not app:
            return
        self.detect_label.setText(f"会议检测\n{capability()}\n最近: {app.app_name}")
        if not self.auto_record or self.worker.is_active:
            return
        has_audio_api = not capability().startswith("仅进程")
        if app.audio_active or not has_audio_api:
            self._start("online", source=app.process.replace(".exe", ""), title=f"{app.app_name} 会议")
            self.tray.showMessage(
                "已开始自动录音",
                f"检测到 {app.app_name} 正在进行语音通话",
                QSystemTrayIcon.MessageIcon.Information, 4000,
            )

    def _set_auto_record(self, on: bool) -> None:
        self.auto_record = on

    def _toggle_float(self) -> None:
        if self.float_bar.isVisible():
            self.float_bar.hide()
        else:
            self.float_bar.place_top_right()
            self.float_bar.show()

    # ---------- 设备信息 ----------
    def _show_devices(self) -> None:
        mic = default_input_device()
        loop = find_loopback_source()
        self.record_page.set_device_hint(
            f"麦克风: {mic.name if mic else '未找到'}    "
            f"系统声音: {loop[0] if loop else '未找到'}    "
            f"检测方式: {capability()}"
        )

    # ---------- 窗口行为 ----------
    def _restore(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:
        if self.worker.is_active:
            reply = QMessageBox.question(
                self, "还在录音", "录音仍在进行。要最小化到托盘继续录音吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply == QMessageBox.StandardButton.Yes:
                event.ignore()
                self.hide()
                self.tray.showMessage("仍在录音", "程序已最小化到托盘", QSystemTrayIcon.MessageIcon.Information, 2500)
                return
            self._stop()
        event.accept()

    def _quit(self) -> None:
        if self.worker.is_active:
            self._stop()
        self.float_bar.close()
        self.tray.hide()
        QApplication.quit()
