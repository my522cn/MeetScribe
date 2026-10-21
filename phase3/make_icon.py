"""生成 exe 图标 assets/app.ico（与窗口内 make_icon 同一视觉：靛蓝圆角方块 + M）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def render(size: int) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = size / 64
    p.setBrush(QColor("#534AB7"))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(int(4 * s), int(4 * s), int(56 * s), int(56 * s), int(14 * s), int(14 * s))
    p.setPen(QColor("#FFFFFF"))
    f = p.font()
    f.setPixelSize(max(8, int(30 * s)))
    f.setBold(True)
    p.setFont(f)
    p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "M")
    p.end()
    return pm


def main() -> int:
    app = QApplication(sys.argv)
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    # .ico 用最大尺寸的一帧即可（Windows 会按需缩放）；多尺寸经 QPixmap 逐一保存更稳
    big = render(256)
    ico = out / "app.ico"
    if not big.save(str(ico), "ICO"):
        print("❌ 保存 ico 失败")
        return 1
    png = out / "app.png"
    big.save(str(png), "PNG")
    app.quit()
    print(f"已生成 {ico} ({ico.stat().st_size} B) 与 {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
