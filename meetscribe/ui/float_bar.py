"""右上角置顶小窗：只保留计时、暂停、结束、回到主窗。"""
from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QGuiApplication, QMouseEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from .caption_view import fmt_ms


class FloatBar(QWidget):
    pause_clicked = Signal()
    stop_clicked = Signal()
    expand_clicked = Signal()

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setObjectName("FloatBar")
        self.setFixedSize(268, 46)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self._drag_offset: QPoint | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 10, 0)
        layout.setSpacing(8)

        self.dot = QLabel("●")
        self.dot.setStyleSheet("color:#E24B4A; font-size:12px;")
        self.timer = QLabel("00:00:00")
        self.timer.setStyleSheet("color:#FFFFFF; font-family:Consolas,monospace; font-size:14px;")

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setObjectName("FloatBtn")
        self.pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_btn.clicked.connect(self.pause_clicked.emit)

        self.stop_btn = QPushButton("结束")
        self.stop_btn.setObjectName("FloatBtn")
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)

        self.expand_btn = QPushButton("展开")
        self.expand_btn.setObjectName("FloatBtn")
        self.expand_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.expand_btn.clicked.connect(self.expand_clicked.emit)

        layout.addWidget(self.dot)
        layout.addWidget(self.timer)
        layout.addStretch(1)
        layout.addWidget(self.pause_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.expand_btn)

    # ---------- 位置 ----------
    def place_top_right(self, margin: int = 24) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        self.move(geo.right() - self.width() - margin, geo.top() + margin)

    # ---------- 状态 ----------
    def set_elapsed(self, ms: int) -> None:
        self.timer.setText(fmt_ms(ms))

    def set_paused(self, paused: bool) -> None:
        self.pause_btn.setText("继续" if paused else "暂停")
        self.dot.setStyleSheet("color:#EF9F27;" if paused else "color:#E24B4A;")

    def reset(self) -> None:
        self.set_elapsed(0)
        self.set_paused(False)

    # ---------- 拖动 ----------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_offset = None
