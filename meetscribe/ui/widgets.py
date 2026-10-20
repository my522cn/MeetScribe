"""带滚轮保护的输入控件。

问题：在设置页用鼠标滚轮翻页时，光标扫过下拉框/数字框就会误改其值。
解决：只有控件已被激活（点击获得焦点）后才响应滚轮；否则把滚轮事件
交还给父级滚动区域继续翻页。
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()  # 交给上层滚动区域翻页


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event) -> None:  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()
