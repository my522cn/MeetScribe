"""实时字幕视图：按说话人分行，临时文本灰色斜体，定稿后转为正常色。

两条轨同时有未定稿文本时（重叠发言），顶部显示"重叠"标记，两行并列显示互不覆盖。

滚动模型（修复过两个坑）：
- 自动跟随：内容变化后始终贴底。不能用“尾部 stretch + widgetResizable”实现，
  wordWrap QLabel 的 sizeHint 是虚高的（比实际换行后的高度高），QScrollArea 会按
  sizeHint 把内容拉高，滚到底就是一片空白。正确做法是布局 AlignTop + 按最后一行
  的实际 geometry 撑内容高度。
- 用户优先：录音中用户向上滚动时立即停掉自动跟随，右下角浮出“回到底部”按钮；
  点按钮（或自己滚回底部）恢复自动跟随。
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)


def fmt_ms(ms: int) -> str:
    total = max(0, int(ms // 1000))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _speaker_chip(speaker: str) -> str:
    """按说话人身份挑 chip 颜色：线下分出的角色 A/B/C/D 各一色。"""
    if speaker == "我":
        return "ChipMic"
    if speaker == "对方" or speaker == "远端":
        return "ChipLoop"
    if speaker == "本地":
        return "ChipOffline"
    for i in "ABCD":
        if speaker.endswith(i):
            return ["ChipMic", "ChipLoop", "ChipSpkC", "ChipSpkD"]["ABCD".index(i)]
    return "ChipMic"


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

        # 句子开始时间放在行首（chip 之后、内容之前），扫一眼就能对上时间轴
        self.time = QLabel(time_text)
        self.time.setObjectName("CaptionTime")
        self.time.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.time.setFixedWidth(58)

        self.text = QLabel("")
        self.text.setObjectName("CaptionText")
        self.text.setWordWrap(True)

        # 离线模型精修过的句子打个小勾，让用户看得见"错别字已被二次校正"
        self.mark = QLabel("✓")
        self.mark.setObjectName("RefineMark")
        self.mark.setVisible(False)
        self.mark.setToolTip("已用离线模型二次识别校正")

        # 声纹检测到一句里有多人抢话时打个 ⚡（文字仍按多数说话人归属）
        self.multi = QLabel("⚡多人")
        self.multi.setObjectName("MixedBadge")
        self.multi.setVisible(False)
        self.multi.setToolTip("这句话里检测到多人同时说话，文字可能有交叠")

        layout.addWidget(self.chip, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.time, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.text, 1)
        layout.addWidget(self.mark, 0, Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.multi, 0, Qt.AlignmentFlag.AlignTop)

    def set_text(self, text: str, final: bool) -> None:
        self.text.setText(text)
        if final:
            self.text.setStyleSheet("font-size:14px; color:#2C2C2A;")
        else:
            self.text.setStyleSheet("font-size:14px; color:#888780; font-style:italic;")

    def set_refined(self) -> None:
        self.mark.setVisible(True)

    def set_mixed(self) -> None:
        self.multi.setVisible(True)

    def set_speaker(self, speaker: str, chip_style: str) -> None:
        self.chip.setText(speaker)
        self.chip.setObjectName(chip_style)
        self.chip.style().unpolish(self.chip)
        self.chip.style().polish(self.chip)


class CaptionView(QScrollArea):
    # 距底部多少像素以内仍算“贴底”（避免 1px 抖动误判成用户在看历史）
    BOTTOM_TOLERANCE = 12

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._body = QWidget()
        # 同 settings_page：透明规则必须限定在自身，否则会覆盖后代输入控件的白底
        self._body.setObjectName("CapBody")
        self._body.setStyleSheet("QWidget#CapBody { background: transparent; }")
        self._layout = QVBoxLayout(self._body)
        self._layout.setContentsMargins(0, 0, 6, 0)
        self._layout.setSpacing(8)
        # 行靠顶排（以前靠尾部 stretch 实现，但 stretch 会把滚动内容撑出一截空白）
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setWidget(self._body)

        self._auto = True            # 是否自动跟随到底部
        self._live: dict[str, CaptionRow] = {}
        self._finals: dict[tuple[str, int], CaptionRow] = {}
        self.refined_count = 0
        self.overlap_badge = QLabel("⚡ 重叠发言")
        self.overlap_badge.setObjectName("OverlapBadge")
        self.overlap_badge.setVisible(False)
        self._layout.insertWidget(0, self.overlap_badge, 0, Qt.AlignmentFlag.AlignLeft)

        # “回到底部”浮动按钮：用户上滚后出现，点一下回到最新并恢复自动跟随
        self.goto_btn = QPushButton("⏬ 回到底部")
        self.goto_btn.setObjectName("GotoEnd")
        self.goto_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.goto_btn.setVisible(False)
        self.goto_btn.clicked.connect(self.goto_end)
        self.goto_btn.setParent(self)
        self.goto_btn.raise_()

        bar = self.verticalScrollBar()
        bar.sliderMoved.connect(self._on_user_scroll)
        bar.actionTriggered.connect(self._on_scroll_action)
        bar.valueChanged.connect(self._on_value_changed)

    # ---------- 对外 ----------
    def set_speaker_meta(self, meta: dict[str, tuple[str, str]]) -> None:
        self._meta = meta

    def update_partial(self, track: str, text: str, start_ms: int) -> None:
        row = self._live.get(track)
        if row is None:
            row = self._new_row(track, start_ms)
            self._live[track] = row
        row.set_text(text, final=False)
        self._after_change()

    def update_live_text(self, track: str, start_ms: int, text: str) -> None:
        """句中滑窗预校正结果：替换临时字幕（保持"未定稿"的灰色样式）。"""
        row = self._live.get(track)
        if row is not None:
            row.set_text(text, final=False)
            self._after_change()

    def finalize(self, track: str, text: str, start_ms: int, end_ms: int,
                 speaker: str | None = None, mixed: bool = False) -> None:
        row = self._live.pop(track, None)
        if row is None:
            row = self._new_row(track, start_ms)
        row.set_text(text, final=True)
        row.time.setText(fmt_ms(start_ms)[3:])
        if speaker is not None and speaker != row.chip.text():
            row.set_speaker(speaker, _speaker_chip(speaker))
        if mixed:
            row.set_mixed()
        self._finals[(track, start_ms)] = row
        self._after_change()

    def insert_final_after(self, track: str, text: str, start_ms: int, end_ms: int,
                           speaker: str, after_start_ms: int) -> None:
        """精修检出句内换人：后半句单独成行，插在原句后面（不是追加到末尾）。"""
        anchor = self._finals.get((track, after_start_ms))
        row = self._new_row(track, start_ms)
        row.set_text(text, final=True)
        row.time.setText(fmt_ms(start_ms)[3:])
        if speaker:
            row.set_speaker(speaker, _speaker_chip(speaker))
        row.set_refined()
        if anchor is not None:
            idx = self._layout.indexOf(anchor)
            if idx >= 0:
                self._layout.removeWidget(row)
                self._layout.insertWidget(idx + 1, row)
        self._finals[(track, start_ms)] = row
        self.refined_count += 1
        self._after_change()

    def update_final_text(self, track: str, start_ms: int, text: str) -> None:
        """离线精修回来后原地替换该句文字。"""
        row = self._finals.get((track, start_ms))
        if row is not None:
            row.set_text(text, final=True)
            row.set_refined()
            self.refined_count += 1
            self._after_change()

    def set_overlap(self, active: bool) -> None:
        self.overlap_badge.setVisible(active)

    def at_bottom(self) -> bool:
        bar = self.verticalScrollBar()
        return bar.value() >= bar.maximum() - self.BOTTOM_TOLERANCE

    def goto_end(self) -> None:
        """一键回到底部并恢复自动跟随。"""
        self._auto = True
        self._after_change()
        self._place_goto_btn()

    def clear(self) -> None:
        for row in list(self._live.values()):
            row.setParent(None)
        self._live.clear()
        self._finals.clear()
        self.refined_count = 0
        while self._layout.count() > 1:
            item = self._layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        self._auto = True
        self._after_change()

    # ---------- 内部 ----------
    def _new_row(self, track: str, start_ms: int) -> CaptionRow:
        speaker, chip = self._meta.get(track, (track, "ChipMic"))
        row = CaptionRow(speaker, chip, fmt_ms(start_ms)[3:])
        self._layout.insertWidget(self._layout.count(), row)
        return row

    # ---------- 滚动控制 ----------
    def _after_change(self) -> None:
        """内容变化后的统一收尾：先按真实行高撑内容，再决定是否贴底。"""
        self._update_height()
        if self._auto:
            bar = self.verticalScrollBar()
            bar.setValue(bar.maximum())
        self._place_goto_btn()

    def _update_height(self) -> None:
        """内容高度 = max(视口高, 最后一行的实际底部 + 边距)。

        不能依赖 sizeHint：wordWrap QLabel 的 sizeHint 虚高，会被
        widgetResizable 拉成一长截空白。行几何在宽度确定后是准确的。
        """
        lay = self._layout
        lay.activate()  # 先强制按当前宽度重排，geometry 才是最新的
        bottom = 0
        for i in range(lay.count()):
            w = lay.itemAt(i).widget()
            if w is not None and not w.isHidden():
                bottom = max(bottom, w.geometry().bottom())
        need = max(self.viewport().height(), bottom + 8)
        if self._body.height() != int(need):
            self._body.setFixedHeight(int(need))

    def _on_user_scroll(self, _value: int) -> None:
        self._sync_auto_state()

    def _on_scroll_action(self, _action: int) -> None:
        # 点滚动条空白/箭头也算用户操作
        self._sync_auto_state()

    def _on_value_changed(self, _value: int) -> None:
        # 自动贴底时同步按钮状态；程序滚动也会走这里，但状态一致不会误弹
        self._sync_auto_state()

    def _sync_auto_state(self) -> None:
        if self.at_bottom():
            self._auto = True
        else:
            self._auto = False
        self._place_goto_btn()

    def _place_goto_btn(self) -> None:
        show = not self._auto and self._body.height() > self.viewport().height()
        self.goto_btn.setVisible(show)
        if show:
            self._position_goto_btn()

    def _position_goto_btn(self) -> None:
        self.goto_btn.adjustSize()
        vp = self.viewport()
        self.goto_btn.move(
            vp.width() - self.goto_btn.width() - 18,
            vp.height() - self.goto_btn.height() - 16,
        )
        self.goto_btn.raise_()

    def wheelEvent(self, event) -> None:  # noqa: N802
        super().wheelEvent(event)
        # 滚轮是明确的用户意图：离开底部就停自动跟随，滚回底部就恢复
        self._sync_auto_state()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_height()
        if self._auto:
            bar = self.verticalScrollBar()
            bar.setValue(bar.maximum())
        self._position_goto_btn()
        self._place_goto_btn()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(640, 360)
