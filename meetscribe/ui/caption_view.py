"""实时字幕视图：按说话人分行，临时文本灰色斜体，定稿后转为正常色。

两条轨同时有未定稿文本时（重叠发言），顶部显示"重叠"标记，两行并列显示互不覆盖。
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)


def fmt_ms(ms: int) -> str:
    total = max(0, int(ms // 1000))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


class CaptionRow(QFrame):
    def __init__(self, speaker: str, chip_style: str, time_text: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("CaptionRow")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        self.chip = QLabel(speaker)
        self.chip.setObjectName(chip_style)
        self.chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chip.setFixedWidth(64)

        self.text = QLabel("")
        self.text.setObjectName("CaptionText")
        self.text.setWordWrap(True)

        self.time = QLabel(time_text)
        self.time.setObjectName("CaptionTime")
        self.time.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        self.time.setFixedWidth(58)

        # 离线模型精修过的句子打个小勾，让用户看得见"错别字已被二次校正"
        self.mark = QLabel("✓")
        self.mark.setObjectName("RefineMark")
        self.mark.setVisible(False)
        self.mark.setToolTip("已用离线模型二次识别校正")

        layout.addWidget(self.chip, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.text, 1)
        layout.addWidget(self.mark, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.time, 0, Qt.AlignmentFlag.AlignTop)

    def set_text(self, text: str, final: bool) -> None:
        self.text.setText(text)
        if final:
            self.text.setStyleSheet("font-size:14px; color:#2C2C2A;")
        else:
            self.text.setStyleSheet("font-size:14px; color:#888780; font-style:italic;")

    def set_refined(self) -> None:
        self.mark.setVisible(True)

    def set_speaker(self, speaker: str, chip_style: str) -> None:
        self.chip.setText(speaker)
        self.chip.setObjectName(chip_style)
        self.chip.style().unpolish(self.chip)
        self.chip.style().polish(self.chip)


class CaptionView(QScrollArea):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._body = QWidget()
        self._body.setStyleSheet("background: transparent;")
        self._layout = QVBoxLayout(self._body)
        self._layout.setContentsMargins(0, 0, 6, 0)
        self._layout.setSpacing(8)
        self._layout.addStretch(1)
        self.setWidget(self._body)

        self._live: dict[str, CaptionRow] = {}
        self._finals: dict[tuple[str, int], CaptionRow] = {}
        self.refined_count = 0
        self.overlap_badge = QLabel("⚡ 重叠发言")
        self.overlap_badge.setObjectName("OverlapBadge")
        self.overlap_badge.setVisible(False)
        self._layout.insertWidget(0, self.overlap_badge, 0, Qt.AlignmentFlag.AlignLeft)

    # ---------- 对外 ----------
    def set_speaker_meta(self, meta: dict[str, tuple[str, str]]) -> None:
        self._meta = meta

    def update_partial(self, track: str, text: str, start_ms: int) -> None:
        row = self._live.get(track)
        if row is None:
            row = self._new_row(track, start_ms)
            self._live[track] = row
        row.set_text(text, final=False)
        self._scroll_bottom()

    def update_live_text(self, track: str, start_ms: int, text: str) -> None:
        """句中滑窗预校正结果：替换临时字幕（保持"未定稿"的灰色样式）。"""
        row = self._live.get(track)
        if row is not None:
            row.set_text(text, final=False)
            self._scroll_bottom()

    def finalize(self, track: str, text: str, start_ms: int, end_ms: int) -> None:
        row = self._live.pop(track, None)
        if row is None:
            row = self._new_row(track, start_ms)
        row.set_text(text, final=True)
        row.time.setText(fmt_ms(start_ms)[3:])
        self._finals[(track, start_ms)] = row
        self._scroll_bottom()

    def update_final_text(self, track: str, start_ms: int, text: str) -> None:
        """离线精修回来后原地替换该句文字。"""
        row = self._finals.get((track, start_ms))
        if row is not None:
            row.set_text(text, final=True)
            row.set_refined()
            self.refined_count += 1

    def set_overlap(self, active: bool) -> None:
        self.overlap_badge.setVisible(active)

    def clear(self) -> None:
        for row in list(self._live.values()):
            row.setParent(None)
        self._live.clear()
        self._finals.clear()
        self.refined_count = 0
        while self._layout.count() > 2:
            item = self._layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()

    # ---------- 内部 ----------
    def _new_row(self, track: str, start_ms: int) -> CaptionRow:
        speaker, chip = self._meta.get(track, (track, "ChipMic"))
        row = CaptionRow(speaker, chip, fmt_ms(start_ms)[3:])
        self._layout.insertWidget(self._layout.count() - 1, row)
        return row

    def _scroll_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())
