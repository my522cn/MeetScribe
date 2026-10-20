"""会议管理页：会话列表、转写查看、一键生成纪要、打开文件夹、删除会话。

数据来源：session_root/<session_id>/ 下的
- session.json     会话清单（SessionManifest.save 生成）
- transcript.jsonl 每行一条 {"track","speaker","start_ms","end_ms","text",...}
- mic/ loop/       分段音频（seg_0001.flac ...）

回放（跨分段连续播放、时间对应会议时间）在本页 v2 实现。
"""
from __future__ import annotations

import json
import shutil
import subprocess  # noqa: S404
import threading
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QTextEdit, QVBoxLayout, QWidget,
)

from ..config import Settings


def fmt_duration(ms: int) -> str:
    s = max(0, int(ms // 1000))
    if s >= 3600:
        return f"{s // 3600}小时{s % 3600 // 60:02d}分"
    return f"{s // 60}分{s % 60:02d}秒" if s >= 60 else f"{s}秒"


class SessionInfo:
    """一次扫描得到的会话摘要。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.title = path.name
        self.mode = ""
        self.status = ""
        self.started = ""
        self.duration_ms = 0
        self.n_segments = 0
        self.n_lines = 0
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads((self.path / "session.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        self.title = data.get("title") or self.path.name
        self.mode = data.get("mode", "")
        self.status = data.get("status", "")
        self.started = (data.get("started_at") or "").replace("T", " ")
        self.duration_ms = int(data.get("duration_ms") or 0)
        self.n_segments = len(data.get("segments") or [])
        try:
            with open(self.path / "transcript.jsonl", encoding="utf-8") as fh:
                self.n_lines = sum(1 for line in fh if line.strip())
        except Exception:  # noqa: BLE001
            pass

    @property
    def label(self) -> str:
        mode = "线上" if self.mode == "online" else "线下"
        return f"{self.started[:16]}\n{self.title} · {mode} · {fmt_duration(self.duration_ms)}"


class ManagePage(QWidget):
    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._sessions: list[SessionInfo] = []
        self._current: SessionInfo | None = None
        self._generating = False
        self._build()

    # ---------- 构建 ----------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 18)
        root.setSpacing(14)

        head = QHBoxLayout()
        title = QLabel("会议管理")
        title.setObjectName("PageTitle")
        head.addWidget(title)
        head.addStretch(1)
        self.btn_refresh = QPushButton("刷新列表")
        self.btn_refresh.setObjectName("Ghost")
        self.btn_refresh.clicked.connect(self.refresh)
        head.addWidget(self.btn_refresh)
        root.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(14)

        # 左：会话列表
        self.list = QListWidget()
        self.list.setFixedWidth(300)
        self.list.currentItemChanged.connect(self._on_select)
        body.addWidget(self.list)

        # 右：详情
        from PySide6.QtWidgets import QFrame

        card = QFrame()
        card.setObjectName("Card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 16, 20, 16)
        cl.setSpacing(10)

        self.lbl_title = QLabel("选择左侧会议查看详情")
        self.lbl_title.setStyleSheet("font-size:16px; font-weight:500;")
        self.lbl_meta = QLabel("")
        self.lbl_meta.setObjectName("Hint")
        self.lbl_meta.setWordWrap(True)
        cl.addWidget(self.lbl_title)
        cl.addWidget(self.lbl_meta)

        btns = QHBoxLayout()
        self.btn_minutes = QPushButton("生成会议纪要")
        self.btn_minutes.setObjectName("Primary")
        self.btn_minutes.clicked.connect(self._gen_minutes)
        self.btn_folder = QPushButton("打开文件夹")
        self.btn_folder.setObjectName("Ghost")
        self.btn_folder.clicked.connect(self._open_folder)
        self.btn_delete = QPushButton("删除会议")
        self.btn_delete.setObjectName("Ghost")
        self.btn_delete.clicked.connect(self._delete_session)
        self.btn_minutes.setEnabled(False)
        self.btn_folder.setEnabled(False)
        self.btn_delete.setEnabled(False)
        btns.addWidget(self.btn_minutes)
        btns.addWidget(self.btn_folder)
        btns.addWidget(self.btn_delete)
        btns.addStretch(1)
        cl.addLayout(btns)

        self.status = QLabel("纪要保存在会议文件夹的 minutes.md，可用任意编辑器查看。")
        self.status.setObjectName("Hint")
        self.status.setWordWrap(True)
        cl.addWidget(self.status)

        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setPlaceholderText("会议转写内容（说话人 · 时间 · 内容）")
        cl.addWidget(self.txt, 1)
        body.addWidget(card, 1)

        root.addLayout(body, 1)

    # ---------- 列表 ----------
    def refresh(self) -> None:
        """重扫会话目录（切到本页或点刷新时调用）。"""
        self.list.blockSignals(True)
        self.list.clear()
        self._sessions = self._scan()
        for s in self._sessions:
            item = QListWidgetItem(s.label)
            item.setData(Qt.ItemDataRole.UserRole, s)
            self.list.addItem(item)
        self.list.blockSignals(False)
        if self._sessions:
            self.list.setCurrentRow(0)
        else:
            self._show_empty()

    def _scan(self) -> list[SessionInfo]:
        root_dir = self.settings.session_root
        out: list[SessionInfo] = []
        if not root_dir.exists():
            return out
        for d in sorted(root_dir.iterdir(), reverse=True):
            if d.is_dir() and (d / "session.json").exists():
                info = SessionInfo(d)
                if info.status:  # session.json 可解析
                    out.append(info)
        return out

    def _show_empty(self) -> None:
        self._current = None
        self.lbl_title.setText("还没有会议记录")
        self.lbl_meta.setText("回到「录音」页开始第一次录音，结束后会自动出现在这里。")
        self.txt.clear()
        for b in (self.btn_minutes, self.btn_folder, self.btn_delete):
            b.setEnabled(False)

    # ---------- 详情 ----------
    def _on_select(self, cur, _prev) -> None:
        if cur is None:
            self._show_empty()
            return
        info: SessionInfo = cur.data(Qt.ItemDataRole.UserRole)
        self._current = info
        mode = "线上会议" if info.mode == "online" else "线下会议"
        st = {"completed": "已完成", "recording": "录音中/未正常收尾"}.get(info.status, info.status)
        self.lbl_title.setText(info.title)
        self.lbl_meta.setText(
            f"{info.started}　·　{mode}　·　时长 {fmt_duration(info.duration_ms)}　·　"
            f"{info.n_segments} 个音频分段　·　{info.n_lines} 句转写　·　{st}"
        )
        self.txt.setPlainText(self._load_transcript(info))
        for b in (self.btn_minutes, self.btn_folder, self.btn_delete):
            b.setEnabled(True)

    @staticmethod
    def _load_transcript(info: SessionInfo) -> str:
        lines: list[str] = []
        path = info.path / "transcript.jsonl"
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = fmt_duration(int(rec.get("start_ms", 0)))
                lines.append(f"{rec.get('speaker', '?')}　{ts}　{rec.get('text', '')}")
        except Exception as exc:  # noqa: BLE001
            return f"（读取转写失败：{exc}）"
        return "\n".join(lines) if lines else "（该会议没有可显示的转写内容）"

    # ---------- 操作 ----------
    def _open_folder(self) -> None:
        if not self._current:
            return
        subprocess.Popen(["explorer", str(self._current.path)], shell=False)  # noqa: S603,S607

    def _delete_session(self) -> None:
        if not self._current:
            return
        info = self._current
        if QMessageBox.question(
            self, "删除会议",
            f"确定删除「{info.title}」？\n\n将永久删除其音频、转写与纪要：\n{info.path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            shutil.rmtree(info.path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "删除失败", str(exc))
            return
        self.refresh()

    def _gen_minutes(self) -> None:
        """后台调用本地大模型生成纪要，写入 minutes.md。"""
        if not self._current or self._generating:
            return
        if not self.settings.get("llm.enabled", False):
            self.status.setText("未启用大模型：请到「设置」勾选「启用本地大模型」后再生成纪要。")
            return
        info = self._current
        self._generating = True
        self.btn_minutes.setEnabled(False)
        self.status.setText("正在生成纪要…（长会议需要几分钟，期间可继续使用其他页面）")

        def work() -> None:
            from ..llm import OllamaClient
            from ..llm.summary import summarize

            base = str(self.settings.get("llm.base_url", "http://127.0.0.1:11434"))
            model = str(self.settings.get("llm.summary_model", "qwen2.5:7b-instruct"))
            client = OllamaClient(base, timeout=float(self.settings.get("llm.timeout", 90)))
            try:
                if not client.alive():
                    self._set_status("正在启动 Ollama 服务…")
                    if not client.ensure_server():
                        self._set_status("连不上 Ollama，请确认已安装并运行后再试。")
                        return
                transcript = ManagePage._load_transcript(info)
                result = summarize(client, model, info.title, transcript)
                out = info.path / "minutes.md"
                out.write_text(result.text, encoding="utf-8")
                sec = f"{result.seconds:.0f}s" if hasattr(result, "seconds") else ""
                self._set_status(f"纪要已生成：{out}（{model}，用时 {sec}）")
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"生成失败：{exc}")
            finally:
                self._generating = False
                self._re_enable()

        threading.Thread(target=work, name="minutes", daemon=True).start()

    # 线程里不能直接碰控件，统一通过这个入口投递到 UI 线程
    def _set_status(self, text: str) -> None:
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, lambda: self.status.setText(text))

    def _re_enable(self) -> None:
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, lambda: self.btn_minutes.setEnabled(True))
