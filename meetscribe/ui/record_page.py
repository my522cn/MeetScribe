"""录音页：只放录制相关控件 + 只读实时字幕。"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from .caption_view import CaptionView, fmt_ms

SPEAKER_ONLINE = {"mic": ("我", "ChipMic"), "loop": ("对方", "ChipLoop")}
SPEAKER_OFFLINE = {"mic": ("本地", "ChipOffline"), "loop": ("远端", "ChipLoop")}


class RecordPage(QWidget):
    start_requested = Signal(str)      # mode
    pause_requested = Signal()
    resume_requested = Signal()
    stop_requested = Signal()
    float_requested = Signal()
    mode_changed = Signal(str)         # 录音中也能切换，worker 收到后热切换标注方式

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._state = "idle"
        self._mode = "online"
        self._build()

    # ---------- 构建 ----------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 18)
        root.setSpacing(14)

        # 标题行
        title_row = QHBoxLayout()
        title = QLabel("录音")
        title.setStyleSheet("font-size:19px; font-weight:500;")
        self.status_pill = QLabel("待机")
        self.status_pill.setObjectName("PillOff")
        title_row.addWidget(title)
        title_row.addSpacing(10)
        title_row.addWidget(self.status_pill)
        title_row.addStretch(1)
        self.float_btn = QPushButton("置顶小窗")
        self.float_btn.setObjectName("Ghost")
        self.float_btn.clicked.connect(self.float_requested.emit)
        title_row.addWidget(self.float_btn)
        root.addLayout(title_row)

        # 控制卡片
        card = QFrame()
        card.setObjectName("Card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 16, 20, 16)
        cl.setSpacing(14)

        top = QHBoxLayout()
        top.setSpacing(10)
        self.start_btn = QPushButton("● 开始录音")
        self.start_btn.setObjectName("Danger")  # 红色=录音语义；Primary 靛蓝留给普通主操作
        self.start_btn.clicked.connect(lambda: self.start_requested.emit(self._mode))
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setObjectName("Ghost")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause)
        self.stop_btn = QPushButton("结束")
        self.stop_btn.setObjectName("Ghost")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_requested.emit)

        self.timer = QLabel("00:00:00")
        self.timer.setObjectName("Timer")

        top.addWidget(self.start_btn)
        top.addWidget(self.pause_btn)
        top.addWidget(self.stop_btn)
        top.addSpacing(18)
        top.addWidget(self.timer)
        top.addStretch(1)
        cl.addLayout(top)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.online_btn = QPushButton("线上会议")
        self.online_btn.setObjectName("SegmentLeft")
        self.online_btn.setCheckable(True)
        self.online_btn.setChecked(True)
        self.offline_btn = QPushButton("线下会议")
        self.offline_btn.setObjectName("SegmentRight")
        self.offline_btn.setCheckable(True)
        self.online_btn.clicked.connect(lambda: self._set_mode("online"))
        self.offline_btn.clicked.connect(lambda: self._set_mode("offline"))
        bottom.addWidget(self.online_btn)
        bottom.addWidget(self.offline_btn)

        bottom.addSpacing(20)
        bottom.addWidget(self._level_label("麦克风"))
        self.mic_bar = self._level_bar()
        bottom.addWidget(self.mic_bar)
        bottom.addSpacing(10)
        bottom.addWidget(self._level_label("系统声"))
        self.loop_bar = self._level_bar()
        bottom.addWidget(self.loop_bar)
        bottom.addStretch(1)
        cl.addLayout(bottom)

        self.device_hint = QLabel("设备检测中…")
        self.device_hint.setObjectName("Hint")
        cl.addWidget(self.device_hint)
        root.addWidget(card)

        # 字幕区
        cap_card = QFrame()
        cap_card.setObjectName("Card")
        ccl = QVBoxLayout(cap_card)
        ccl.setContentsMargins(16, 12, 12, 12)
        ccl.setSpacing(8)
        head = QHBoxLayout()
        cap_title = QLabel("实时字幕")
        cap_title.setStyleSheet("font-weight:500;")
        self.caption_hint = QLabel("未开始录音")
        self.caption_hint.setObjectName("Hint")
        head.addWidget(cap_title)
        head.addSpacing(8)
        head.addWidget(self.caption_hint)
        head.addStretch(1)
        ccl.addLayout(head)

        self.captions = CaptionView()
        self.captions.set_speaker_meta(SPEAKER_ONLINE)
        ccl.addWidget(self.captions, 1)
        root.addWidget(cap_card, 1)

        self.footer = QLabel("录音文件分段保存，意外关闭后可恢复")
        self.footer.setObjectName("Hint")
        root.addWidget(self.footer)

    def _level_label(self, text: str) -> QLabel:
        lb = QLabel(text)
        lb.setObjectName("Hint")
        return lb

    def _level_bar(self) -> QProgressBar:
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedSize(90, 6)
        bar.setStyleSheet(
            "QProgressBar{background:#EEEDE8;border:none;border-radius:3px;}"
            "QProgressBar::chunk{background:#1D9E75;border-radius:3px;}"
        )
        return bar

    # ---------- 交互 ----------
    def _set_mode(self, mode: str) -> None:
        """任何状态（待机/录音中/暂停/已结束）都允许切换。

        之前的两版 bug 都出在拦截上：先是无条件拦截（录过一次就永远选不了），
        后来改成录音中拦截——但线上/线下都一直在录双轨，切换只是改"之后的
        句子怎么标注说话人"，随时可切。这里不再有任何拦截。
        """
        changed = mode != self._mode
        self._mode = mode
        self.online_btn.setChecked(mode == "online")
        self.offline_btn.setChecked(mode == "offline")
        self.captions.set_speaker_meta(SPEAKER_ONLINE if mode == "online" else SPEAKER_OFFLINE)
        if changed:
            self.mode_changed.emit(mode)

    def _on_pause(self) -> None:
        if self._state == "recording":
            self.pause_requested.emit()
        else:
            self.resume_requested.emit()

    # ---------- 状态更新 ----------
    def set_state(self, state: str) -> None:
        self._state = state
        if state == "recording":
            self.status_pill.setText("● 录音中")
            self.status_pill.setObjectName("PillOn")
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
            self.pause_btn.setText("暂停")
            self.stop_btn.setEnabled(True)
            self.caption_hint.setText("实时识别中")
        elif state == "paused":
            self.status_pill.setText("已暂停")
            self.status_pill.setObjectName("PillOff")
            self.pause_btn.setText("继续")
            self.caption_hint.setText("已暂停 · 后台仍在校对已识别的文字")
        elif state == "stopped":
            # 已保存不占状态标签（改用系统托盘通知），页面回到待机观感
            self.status_pill.setText("待机")
            self.status_pill.setObjectName("PillOff")
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText("暂停")
            self.stop_btn.setEnabled(False)
            self.caption_hint.setText("已结束")
        else:
            self.status_pill.setText("待机")
            self.status_pill.setObjectName("PillOff")
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText("暂停")
            self.stop_btn.setEnabled(False)
            self.caption_hint.setText("未开始录音")
        self.status_pill.style().unpolish(self.status_pill)
        self.status_pill.style().polish(self.status_pill)

    def set_elapsed(self, ms: int) -> None:
        self.timer.setText(fmt_ms(ms))

    def set_levels(self, mic: float, loop: float) -> None:
        self.mic_bar.setValue(min(100, int(mic * 400)))
        self.loop_bar.setValue(min(100, int(loop * 400)))

    def set_device_hint(self, text: str) -> None:
        self.device_hint.setText(text)

    def set_caption_hint(self, text: str) -> None:
        self.caption_hint.setText(text)

    def clear_captions(self) -> None:
        self.captions.clear()
