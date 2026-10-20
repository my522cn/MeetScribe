"""浅色主题样式表。"""

QSS = """
* { font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif; font-size: 13px; color: #2C2C2A; }
QMainWindow, QWidget#Root { background: #F7F6F3; }
QWidget#Sidebar { background: #FFFFFF; border-right: 1px solid #E4E2DC; }
QPushButton#NavButton {
    text-align: left; padding: 10px 16px; border: none; border-radius: 8px;
    color: #5F5E5A; background: transparent;
}
QPushButton#NavButton:hover { background: #F1EFE8; }
QPushButton#NavButton:checked { background: #EEEDFE; color: #534AB7; font-weight: 500; }

QFrame#Card { background: #FFFFFF; border: 1px solid #E4E2DC; border-radius: 12px; }

QPushButton#Primary {
    background: #E24B4A; color: #FFFFFF; border: none; border-radius: 8px;
    padding: 9px 22px; font-weight: 500;
}
QPushButton#Primary:hover { background: #C93E3D; }
QPushButton#Primary:disabled { background: #D3D1C7; color: #FFFFFF; }
QPushButton#Ghost {
    background: #FFFFFF; border: 1px solid #D3D1C7; border-radius: 8px; padding: 8px 18px;
}
QPushButton#Ghost:hover { background: #F1EFE8; }
QPushButton#Ghost:disabled { color: #B4B2A9; border-color: #E4E2DC; }

QPushButton#SegmentLeft, QPushButton#SegmentRight {
    background: #FFFFFF; border: 1px solid #D3D1C7; padding: 6px 16px; color: #5F5E5A;
}
QPushButton#SegmentLeft { border-top-left-radius: 8px; border-bottom-left-radius: 8px; }
QPushButton#SegmentRight { border-top-right-radius: 8px; border-bottom-right-radius: 8px; }
QPushButton#SegmentLeft:checked, QPushButton#SegmentRight:checked {
    background: #EEEDFE; color: #534AB7; border-color: #AFA9EC; font-weight: 500;
}

QLabel#Timer { font-family: "Cascadia Mono", Consolas, monospace; font-size: 26px; color: #2C2C2A; }
QLabel#Hint { color: #888780; font-size: 12px; }
QLabel#PillOn { background: #FBE9E9; color: #A32D2D; border-radius: 10px; padding: 3px 10px; font-size: 12px; }
QLabel#PillOff { background: #F1EFE8; color: #5F5E5A; border-radius: 10px; padding: 3px 10px; font-size: 12px; }

QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 8px; background: transparent; }
QScrollBar::handle:vertical { background: #D3D1C7; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }

QFrame#CaptionRow { background: #FFFFFF; border: 1px solid #EEEDE8; border-radius: 10px; }
QLabel#ChipMic { background: #E6F1FB; color: #185FA5; border-radius: 9px; padding: 2px 10px; font-size: 12px; }
QLabel#ChipLoop { background: #E1F5EE; color: #0F6E56; border-radius: 9px; padding: 2px 10px; font-size: 12px; }
QLabel#ChipOffline { background: #FAEEDA; color: #854F0B; border-radius: 9px; padding: 2px 10px; font-size: 12px; }
QLabel#CaptionText { font-size: 14px; }
QLabel#CaptionTime { color: #B4B2A9; font-size: 11px; }
QLabel#RefineMark { color: #0F6E56; font-size: 11px; }
QLabel#OverlapBadge { background: #FAEEDA; color: #BA7517; border-radius: 8px; padding: 2px 8px; font-size: 11px; }

QWidget#FloatBar { background: #2C2C2A; border-radius: 12px; }
QWidget#FloatBar QLabel { color: #FFFFFF; }
QPushButton#FloatBtn {
    background: rgba(255,255,255,0.12); border: none; border-radius: 7px;
    color: #FFFFFF; padding: 5px 10px;
}
QPushButton#FloatBtn:hover { background: rgba(255,255,255,0.22); }
QMenu { background: #FFFFFF; border: 1px solid #E4E2DC; padding: 4px; }
QMenu::item { padding: 6px 24px; border-radius: 6px; }
QMenu::item:selected { background: #F1EFE8; }

QMessageBox, QDialog { background: #FFFFFF; }
QMessageBox QLabel, QDialog QLabel { color: #2C2C2A; background: transparent; }
QMessageBox QPushButton, QDialog QPushButton {
    background: #FFFFFF; border: 1px solid #D3D1C7; border-radius: 8px;
    padding: 6px 20px; color: #2C2C2A; min-width: 64px;
}
QMessageBox QPushButton:hover, QDialog QPushButton:hover { background: #F1EFE8; }
QMessageBox QPushButton:default, QDialog QPushButton:default {
    background: #534AB7; color: #FFFFFF; border-color: #534AB7;
}
QToolTip { background: #2C2C2A; color: #FFFFFF; border: none; padding: 4px 8px; }

/* ---------- 设置页 ---------- */
QLabel#PageTitle { font-size: 19px; font-weight: 500; }
QLabel#SectionTitle { font-size: 13px; font-weight: 500; color: #2C2C2A; }
QLabel#FieldLabel { color: #5F5E5A; }
QLabel#Mono { font-family: "Cascadia Mono", Consolas, monospace; font-size: 12px; color: #5F5E5A; }
QLabel#Good { color: #0F6E56; }
QLabel#Warn { color: #A15C00; }
QLabel#Bad { color: #A32D2D; }

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #FFFFFF; border: 1px solid #D3D1C7; border-radius: 8px;
    padding: 6px 10px; min-height: 20px;
}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus { border-color: #AFA9EC; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #FFFFFF; border: 1px solid #E4E2DC; selection-background-color: #EEEDFE;
    selection-color: #534AB7; padding: 4px;
}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {
    background: #F7F6F3; color: #B4B2A9;
}
QCheckBox { spacing: 8px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #C6C4BB; border-radius: 4px; background: #FFFFFF;
}
QCheckBox::indicator:checked { background: #534AB7; border-color: #534AB7; }
QTextEdit {
    background: #FFFFFF; border: 1px solid #E4E2DC; border-radius: 10px; padding: 8px;
}
QProgressBar {
    background: #F1EFE8; border: none; border-radius: 4px; height: 8px; text-align: center;
}
QProgressBar::chunk { background: #534AB7; border-radius: 4px; }
QProgressBar#LevelBar::chunk { background: #1D9E75; }
"""
