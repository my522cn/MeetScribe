"""带滚轮禁用的输入控件。

问题：在设置页用鼠标滚轮翻页时，光标扫过下拉框/数字框会误改其值。
按用户要求彻底禁用这些控件的滚轮事件——改值只能通过点击/键盘，
滚轮事件一律上交，让页面正常滚动。
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()  # 上交滚轮，页面继续滚动；改值只能点开下拉


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()
