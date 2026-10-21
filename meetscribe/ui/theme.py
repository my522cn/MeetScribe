"""浅色主题 v2：冷灰底 + 白卡片 + 靛蓝强调色。

设计原则：
- 三层信息层级：页面底 #F5F6F8 → 白卡片 → 控件；靠留白和描边区分，不堆颜色；
- 强调色只用于"可操作"的东西：选中态、焦点、主按钮，一眼能看出哪里能点；
- 分段选择按钮选中时用实心主色+白字，状态一目了然；
- 所有输入控件 focus 有主色描边，禁用态灰化明显，不会"像禁用其实能输"。
"""

QSS = """
* { font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif; font-size: 13px; color: #1F2329; }

QMainWindow, QWidget#Root { background: #F5F6F8; }

/* ---------- 侧边导航 ---------- */
QWidget#Sidebar { background: #FFFFFF; border-right: 1px solid #E6E8EF; }
QPushButton#NavButton {
    text-align: left; padding: 11px 16px; border: none; border-radius: 10px;
    color: #646A73; background: transparent; font-size: 14px;
}
QPushButton#NavButton:hover { background: #F2F3F7; color: #1F2329; }
QPushButton#NavButton:checked { background: #EEEDFD; color: #4F46E5; font-weight: 600; }

/* ---------- 卡片 ---------- */
QFrame#Card { background: #FFFFFF; border: 1px solid #E6E8EF; border-radius: 14px; }

/* ---------- 按钮 ---------- */
QPushButton#Primary {
    background: #4F46E5; color: #FFFFFF; border: none; border-radius: 9px;
    padding: 9px 24px; font-weight: 600; min-height: 18px;
}
QPushButton#Primary:hover { background: #4338CA; }
QPushButton#Primary:pressed { background: #3730A3; }
QPushButton#Primary:disabled { background: #C7C9D4; color: #FFFFFF; }
QPushButton#Danger {
    background: #E5484D; color: #FFFFFF; border: none; border-radius: 9px;
    padding: 9px 24px; font-weight: 600; min-height: 18px;
}
QPushButton#Danger:hover { background: #D93840; }
QPushButton#Danger:disabled { background: #EFD3D4; color: #FFFFFF; }
QPushButton#Success {
    background: #17A673; color: #FFFFFF; border: none; border-radius: 9px;
    padding: 9px 24px; font-weight: 600; min-height: 18px;
}
QPushButton#Success:hover { background: #128F63; }
QPushButton#Success:disabled { background: #CBE7DD; color: #FFFFFF; }
QPushButton#Ghost {
    background: #FFFFFF; border: 1px solid #D9DDE3; border-radius: 9px;
    padding: 8px 18px; color: #40464F; min-height: 18px;
}
QPushButton#Ghost:hover { background: #F2F3F7; border-color: #C3C8D1; }
QPushButton#Ghost:pressed { background: #E8EAF0; }
QPushButton#Ghost:disabled { color: #B3B8C0; border-color: #E6E8EF; background: #FAFAFC; }

/* ---------- 分段选择（线上/线下会议）：选中=实心主色+白字 ---------- */
QPushButton#SegmentLeft, QPushButton#SegmentRight {
    background: #FFFFFF; border: 1px solid #D9DDE3; padding: 8px 22px;
    color: #646A73; min-height: 18px;
}
QPushButton#SegmentLeft { border-top-left-radius: 9px; border-bottom-left-radius: 9px; }
QPushButton#SegmentRight { border-top-right-radius: 9px; border-bottom-right-radius: 9px; }
QPushButton#SegmentLeft:hover, QPushButton#SegmentRight:hover { background: #F2F3F7; }
QPushButton#SegmentLeft:checked, QPushButton#SegmentRight:checked {
    background: #4F46E5; color: #FFFFFF; border-color: #4F46E5; font-weight: 600;
}

/* ---------- 录音页 ---------- */
QLabel#Timer { font-family: "Cascadia Mono", Consolas, monospace; font-size: 28px; font-weight: 600; color: #1F2329; }
QLabel#Hint { color: #8A9099; font-size: 12px; }
QLabel#PillOn { background: #FDEBEC; color: #C4333A; border-radius: 11px; padding: 4px 12px; font-size: 12px; font-weight: 600; }
QLabel#PillOff { background: #F0F1F4; color: #646A73; border-radius: 11px; padding: 4px 12px; font-size: 12px; }

/* ---------- 滚动条 ---------- */
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 8px; background: transparent; margin: 2px; }
QScrollBar::handle:vertical { background: #D5D8DF; border-radius: 4px; min-height: 36px; }
QScrollBar::handle:vertical:hover { background: #C0C4CD; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar:horizontal { height: 8px; background: transparent; margin: 2px; }
QScrollBar::handle:horizontal { background: #D5D8DF; border-radius: 4px; min-width: 36px; }

/* ---------- 实时字幕 ---------- */
QFrame#CaptionRow { background: #FFFFFF; border: 1px solid #EBEDF2; border-radius: 11px; }
QFrame#CaptionRow:hover { border-color: #E0E3EA; }
QLabel#ChipMic { background: #E8F0FE; color: #1A5FD0; border-radius: 10px; padding: 3px 10px; font-size: 12px; font-weight: 600; }
QLabel#ChipLoop { background: #E4F5EE; color: #0E7A5F; border-radius: 10px; padding: 3px 10px; font-size: 12px; font-weight: 600; }
QLabel#ChipOffline { background: #FCEFDC; color: #96620A; border-radius: 10px; padding: 3px 10px; font-size: 12px; font-weight: 600; }
QLabel#ChipSpkC { background: #EEECFE; color: #4F46E5; border-radius: 10px; padding: 3px 10px; font-size: 12px; font-weight: 600; }
QLabel#ChipSpkD { background: #FCE9F1; color: #B0296A; border-radius: 10px; padding: 3px 10px; font-size: 12px; font-weight: 600; }
QLabel#CaptionText { font-size: 14px; }
QLabel#CaptionTime { color: #A6ABB4; font-size: 11px; font-family: "Cascadia Mono", Consolas, monospace; }
QLabel#RefineMark { color: #0E7A5F; font-size: 11px; }
QLabel#OverlapBadge { background: #FCEFDC; color: #96620A; border-radius: 9px; padding: 3px 10px; font-size: 11px; }

/* 字幕区右下角的“回到底部”浮动按钮（用户上滚后出现） */
QPushButton#GotoEnd {
    background: #4F46E5; color: #FFFFFF; border: none; border-radius: 14px;
    padding: 6px 14px; font-size: 12px;
}
QPushButton#GotoEnd:hover { background: #4338CA; }

/* ---------- 置顶小窗 ---------- */
QWidget#FloatBar { background: #23262D; border-radius: 12px; }
QWidget#FloatBar QLabel { color: #FFFFFF; }
QPushButton#FloatBtn {
    background: rgba(255,255,255,0.12); border: none; border-radius: 7px;
    color: #FFFFFF; padding: 5px 10px;
}
QPushButton#FloatBtn:hover { background: rgba(255,255,255,0.22); }

/* ---------- 菜单 / 弹窗 ---------- */
QMenu { background: #FFFFFF; border: 1px solid #E6E8EF; border-radius: 10px; padding: 6px; }
QMenu::item { padding: 7px 26px; border-radius: 7px; }
QMenu::item:selected { background: #F2F3F7; }
QMessageBox, QDialog { background: #FFFFFF; }
QMessageBox QLabel, QDialog QLabel { color: #1F2329; background: transparent; }
QMessageBox QPushButton, QDialog QPushButton {
    background: #FFFFFF; border: 1px solid #D9DDE3; border-radius: 8px;
    padding: 7px 22px; color: #1F2329; min-width: 68px;
}
QMessageBox QPushButton:hover, QDialog QPushButton:hover { background: #F2F3F7; }
QMessageBox QPushButton:default, QDialog QPushButton:default {
    background: #4F46E5; color: #FFFFFF; border-color: #4F46E5;
}
QToolTip { background: #23262D; color: #FFFFFF; border: none; padding: 5px 9px; border-radius: 6px; }

/* ---------- 设置页 ---------- */
QLabel#PageTitle { font-size: 20px; font-weight: 600; }
QLabel#SectionTitle { font-size: 13px; font-weight: 600; color: #40464F; }
QLabel#FieldLabel { color: #646A73; }
QLabel#Mono { font-family: "Cascadia Mono", Consolas, monospace; font-size: 12px; color: #5A6068; }
QLabel#Good { color: #0E7A5F; }
QLabel#Warn { color: #96620A; }
QLabel#Bad { color: #C4333A; }

/* ---------- 输入控件 ---------- */
QComboBox, QAbstractSpinBox, QLineEdit {
    background: #FFFFFF; border: 1px solid #D9DDE3; border-radius: 8px;
    padding: 6px 11px; min-height: 22px; color: #1F2329;
    selection-background-color: #C9C5F5; selection-color: #1F2329;
}
QSpinBox, QDoubleSpinBox {
    background: #FFFFFF; border: 1px solid #D9DDE3; border-radius: 8px;
    padding: 6px 11px; min-height: 22px; color: #1F2329;
    selection-background-color: #C9C5F5; selection-color: #1F2329;
}
QComboBox:hover, QAbstractSpinBox:hover, QLineEdit:hover,
QSpinBox:hover, QDoubleSpinBox:hover { border-color: #C3C8D1; }
QComboBox:focus, QAbstractSpinBox:focus, QLineEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #4F46E5; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow {
    image: none; width: 0; height: 0; margin-right: 8px;
    border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 6px solid #646A73;
}
QComboBox QLineEdit { background: transparent; border: none; padding: 0; min-height: 0; }
QComboBox QAbstractItemView {
    background: #FFFFFF; border: 1px solid #E6E8EF; border-radius: 8px;
    selection-background-color: #EEEDFD; selection-color: #4F46E5; padding: 4px; outline: 0;
}
QComboBox:disabled, QAbstractSpinBox:disabled, QLineEdit:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled {
    background: #F2F3F5; color: #B3B8C0; border-color: #E6E8EF;
}
/* 数字框上下按钮：浅灰底 + 三角箭头 */
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
    background: #EEF0F3; border: none; width: 20px;
}
QAbstractSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; }
QAbstractSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: #E2E5EA; }
QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed { background: #D3D7DE; }
QAbstractSpinBox::up-arrow {
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 5px solid #5A6068;
}
QAbstractSpinBox::down-arrow {
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid #5A6068;
}

/* ---------- 复选框 / 文本域 / 进度条 ---------- */
QCheckBox { spacing: 8px; color: #40464F; }
QCheckBox::indicator {
    width: 17px; height: 17px; border: 1.5px solid #C3C8D1; border-radius: 5px; background: #FFFFFF;
}
QCheckBox::indicator:hover { border-color: #4F46E5; }
QCheckBox::indicator:checked { background: #4F46E5; border-color: #4F46E5; }
QTextEdit {
    background: #FAFAFC; border: 1px solid #E6E8EF; border-radius: 10px; padding: 9px;
    color: #1F2329; selection-background-color: #C9C5F5;
}
QTextEdit:focus { border: 1px solid #C9C5F5; }
QProgressBar {
    background: #EEF0F3; border: none; border-radius: 4px; height: 8px; text-align: center;
}
QProgressBar::chunk { background: #4F46E5; border-radius: 4px; }
QProgressBar#LevelBar::chunk { background: #17A673; }
"""
