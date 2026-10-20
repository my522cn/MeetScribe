"""会议管理页：会话列表（可多选批量删除）、转写/纪要查看、一键生成纪要（带进度）、回放。

数据来源：session_root/<session_id>/ 下的
- session.json     会话清单（SessionManifest.save 生成）
- transcript.jsonl 每行一条 {"track","speaker","start_ms","end_ms","text",...}
- minutes.md       生成纪要（本页写入）
- mic/ loop/       分段音频（seg_0001.flac ...）
"""
from __future__ import annotations

import json
import shutil
import subprocess  # noqa: S404
import threading
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QSlider, QTextEdit, QVBoxLayout, QWidget, QSizePolicy,
)

from ..audio.playback import SessionPlayer
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
    # 后台纪要线程 → UI 线程（线程里禁止直接碰控件，统一走信号）
    minute_progress = Signal(str, str)   # 会话路径, 进度文本
    minute_finished = Signal(str, bool, str)  # 会话路径, 是否成功, 消息

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._sessions: list[SessionInfo] = []
        self._current: SessionInfo | None = None
        # 纪要生成状态按会话跟踪：path -> "generating" / "done" / "failed:xxx"
        self._minute_state: dict[str, str] = {}
        self._player: SessionPlayer | None = None
        self._slider_drag = False
        self._build()
        self._poll = QTimer(self)
        self._poll.setInterval(200)
        self._poll.timeout.connect(self._tick_player)
        self._poll.start()
        self.minute_progress.connect(self._on_minute_progress)
        self.minute_finished.connect(self._on_minute_finished)

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

        # 左：会话列表（多选 → 支持批量删除）
        self.list = QListWidget()
        self.list.setFixedWidth(300)
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
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
        self.lbl_meta.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
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
        self.btn_delete.clicked.connect(self._delete_sessions)
        self.btn_minutes.setEnabled(False)
        self.btn_folder.setEnabled(False)
        self.btn_delete.setEnabled(False)
        btns.addWidget(self.btn_minutes)
        btns.addWidget(self.btn_folder)
        btns.addWidget(self.btn_delete)
        btns.addStretch(1)
        cl.addLayout(btns)

        self.status = QLabel("")
        self.status.setObjectName("Hint")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        cl.addWidget(self.status)

        # ---------- 回放控制 ----------
        pb = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setObjectName("Primary")
        self.btn_play.setFixedWidth(110)
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("Ghost")
        self.btn_stop.clicked.connect(self._stop_play)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderPressed.connect(lambda: setattr(self, "_slider_drag", True))
        self.slider.sliderReleased.connect(self._on_seek)
        self.lbl_playpos = QLabel("00:00 / 00:00")
        self.lbl_playpos.setObjectName("Mono")
        pb.addWidget(self.btn_play)
        pb.addWidget(self.btn_stop)
        pb.addSpacing(8)
        pb.addWidget(self.slider, 1)
        pb.addWidget(self.lbl_playpos)
        cl.addLayout(pb)
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.slider.setEnabled(False)

        # ---------- 转写 / 纪要视图 ----------
        views = QHBoxLayout()
        self.btn_view_trans = QPushButton("转写稿")
        self.btn_view_trans.setObjectName("SegmentLeft")
        self.btn_view_trans.setCheckable(True)
        self.btn_view_trans.setChecked(True)
        self.btn_view_min = QPushButton("纪要预览")
        self.btn_view_min.setObjectName("SegmentRight")
        self.btn_view_min.setCheckable(True)
        self.btn_view_trans.clicked.connect(lambda: self._switch_view("trans"))
        self.btn_view_min.clicked.connect(lambda: self._switch_view("min"))
        views.addWidget(self.btn_view_trans)
        views.addWidget(self.btn_view_min)
        views.addStretch(1)
        cl.addLayout(views)
        self._view = "trans"

        self.txt = QTextEdit()
        self.txt.setReadOnly(True)
        self.txt.setPlaceholderText("会议转写内容（说话人 · 时间 · 内容）")
        cl.addWidget(self.txt, 1)
        body.addWidget(card, 1)

        root.addLayout(body, 1)

    # ---------- 列表 ----------
    def refresh(self) -> None:
        """重扫会话目录（切到本页或点刷新时调用）。"""
        selected = {i.data(Qt.ItemDataRole.UserRole).path for i in self.list.selectedItems()
                    if i.data(Qt.ItemDataRole.UserRole)}
        self.list.blockSignals(True)
        self.list.clear()
        self._sessions = self._scan()
        restore_row = 0
        for n, s in enumerate(self._sessions):
            item = QListWidgetItem(s.label)
            item.setData(Qt.ItemDataRole.UserRole, s)
            self.list.addItem(item)
            if s.path in selected:
                item.setSelected(True)
                restore_row = n
        self.list.blockSignals(False)
        if self._sessions:
            self.list.setCurrentRow(restore_row)
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
        self._stop_play()
        self.lbl_title.setText("还没有会议记录")
        self.lbl_meta.setText("回到「录音」页开始第一次录音，结束后会自动出现在这里。")
        self.txt.clear()
        for b in (self.btn_minutes, self.btn_folder, self.btn_delete):
            b.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.slider.setEnabled(False)

    # ---------- 详情 ----------
    def _on_select(self, cur, _prev) -> None:
        if cur is None:
            self._show_empty()
            return
        info: SessionInfo = cur.data(Qt.ItemDataRole.UserRole)
        if self._current is not info:
            self._stop_play()  # 换会话先停掉旧回放
        self._current = info
        mode = "线上会议" if info.mode == "online" else "线下会议"
        st = {"completed": "已完成", "recording": "录音中/未正常收尾"}.get(info.status, info.status)
        self.lbl_title.setText(info.title)
        self.lbl_meta.setText(
            f"{info.started}　·　{mode}　·　时长 {fmt_duration(info.duration_ms)}　·　"
            f"{info.n_segments} 个音频分段　·　{info.n_lines} 句转写　·　{st}"
        )
        for b in (self.btn_minutes, self.btn_folder, self.btn_delete):
            b.setEnabled(True)
        self._apply_minute_ui()
        # 回放控件
        has_audio = info.n_segments > 0
        self.btn_play.setEnabled(has_audio)
        self.slider.setEnabled(has_audio)
        self.slider.setRange(0, info.duration_ms)
        self.slider.setValue(0)
        self.lbl_playpos.setText(f"00:00 / {fmt_duration(info.duration_ms)}")
        self._render_view()

    # ---------- 转写 / 纪要视图 ----------
    def _switch_view(self, view: str) -> None:
        self._view = view
        self.btn_view_trans.setChecked(view == "trans")
        self.btn_view_min.setChecked(view == "min")
        self._render_view()

    def _render_view(self) -> None:
        if not self._current:
            self.txt.clear()
            return
        if self._view == "min":
            self.txt.setPlainText(self._load_minutes(self._current))
        else:
            self.txt.setPlainText(self._load_transcript(self._current))

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

    @staticmethod
    def _load_minutes(info: SessionInfo) -> str:
        path = info.path / "minutes.md"
        if not path.exists():
            return "（还没有生成纪要。点上方「生成会议纪要」，完成后这里会显示全文，文件保存在会议文件夹的 minutes.md）"
        try:
            return path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return f"（读取纪要失败：{exc}）"

    # ---------- 纪要生成 ----------
    def _gen_minutes(self) -> None:
        if not self._current:
            return
        info = self._current
        state = self._minute_state.get(str(info.path))
        if state == "generating":
            QMessageBox.information(self, "正在生成", "这个会议的纪要正在后台生成，请稍候。")
            return
        if not self.settings.get("llm.enabled", False):
            self.status.setText("未启用大模型：请到「设置」勾选「启用本地大模型」后再生成纪要。")
            return
        self._minute_state[str(info.path)] = "generating"
        self._apply_minute_ui()
        threading.Thread(target=self._minutes_work, args=(info,), name="minutes", daemon=True).start()

    def _minutes_work(self, info: SessionInfo) -> None:
        """后台线程：起服务 → summarize（带进度回调）→ 写 minutes.md。"""
        from ..llm import OllamaClient
        from ..llm.summary import summarize

        path = str(info.path)
        try:
            base = str(self.settings.get("llm.base_url", "http://127.0.0.1:11434"))
            model = str(self.settings.get("llm.summary_model", "qwen2.5:7b-instruct"))
            client = OllamaClient(base, timeout=float(self.settings.get("llm.timeout", 90)))
            if not client.alive():
                self.minute_progress.emit(path, "正在启动 Ollama 服务…")
                if not client.ensure_server():
                    raise RuntimeError("连不上 Ollama，请确认已安装并运行后再试")
            transcript = self._load_transcript(info)
            result = summarize(
                client, model, info.title, transcript,
                on_progress=lambda text: self.minute_progress.emit(path, text),
            )
            out = info.path / "minutes.md"
            out.write_text(result, encoding="utf-8")
            self.minute_finished.emit(path, True, f"纪要已生成：{out}")
        except Exception as exc:  # noqa: BLE001
            self.minute_finished.emit(path, False, f"生成失败：{exc}")

    def _on_minute_progress(self, path: str, text: str) -> None:
        if self._current and str(self._current.path) == path:
            self.status.setText(text)

    def _on_minute_finished(self, path: str, ok: bool, message: str) -> None:
        self._minute_state[path] = "done" if ok else f"failed:{message}"
        if self._current and str(self._current.path) == path:
            self.status.setText(message)
            if ok and self._view == "min":
                self._render_view()  # 完成后刷新纪要预览
            if ok and self._view == "trans":
                self.status.setText(message + "　（点下方「纪要预览」查看全文）")
        self._apply_minute_ui()

    def _apply_minute_ui(self) -> None:
        """按当前选中会话的纪要状态更新按钮与提示（切换会话时状态跟着走）。"""
        if not self._current:
            return
        state = self._minute_state.get(str(self._current.path))
        generating = state == "generating"
        self.btn_minutes.setEnabled(not generating)
        self.btn_minutes.setText("生成中…" if generating else "生成会议纪要")
        if state is None:
            self.status.setText("")
        elif generating:
            if not self.status.text().startswith(("正在", "纪要已")):
                self.status.setText("正在生成纪要…")
        elif state and state.startswith("failed:"):
            self.status.setText(state[len("failed:"):])
        elif state == "done":
            self.status.setText("纪要已生成（点下方「纪要预览」查看全文）")

    # ---------- 回放 ----------
    def _toggle_play(self) -> None:
        if not self._current:
            return
        if self._player is None or self._player.session_dir != self._current.path:
            self._player = SessionPlayer(self._current.path)
        if self._player.active and not self._player.paused:
            self._player.pause()
        else:
            self._player.play(self.slider.value())

    def _stop_play(self) -> None:
        if self._player is not None:
            self._player.stop()
            self._player = None
        self.btn_play.setText("▶ 播放")
        self.btn_stop.setEnabled(False)
        # 停止 = 回到起始位置（与暂停不同：暂停保留进度，停止归零）
        self.slider.setValue(0)
        if self._current:
            self.lbl_playpos.setText(f"00:00 / {fmt_duration(self._current.duration_ms)}")

    def _on_seek(self) -> None:
        self._slider_drag = False
        if self._player is not None and self._player.active:
            self._player.play(self.slider.value())

    def _tick_player(self) -> None:
        p = self._player
        if p is None:
            return
        if not p.active:
            self.btn_play.setText("▶ 播放")
            self.btn_stop.setEnabled(False)
            return
        self.btn_stop.setEnabled(True)
        self.btn_play.setText("⏸ 暂停" if not p.paused else "▶ 继续")
        if not self._slider_drag:
            self.slider.setValue(p.position_ms())
        self.lbl_playpos.setText(
            f"{fmt_duration(p.position_ms())} / {fmt_duration(p.duration_ms)}"
        )

    # ---------- 操作 ----------
    def _open_folder(self) -> None:
        if not self._current:
            return
        subprocess.Popen(["explorer", str(self._current.path)], shell=False)  # noqa: S603,S607

    def _delete_sessions(self) -> None:
        """删除当前选中的所有会话（支持 Ctrl/Shift 多选批量删除）。"""
        items = [i for i in self.list.selectedItems() if i.data(Qt.ItemDataRole.UserRole)]
        if not items:
            return
        infos: list[SessionInfo] = [i.data(Qt.ItemDataRole.UserRole) for i in items]
        names = "\n".join(f"· {x.title}（{x.started[:16]}）" for x in infos[:8])
        if len(infos) > 8:
            names += f"\n· …等共 {len(infos)} 个"
        if QMessageBox.question(
            self, "删除会议",
            f"确定删除选中的 {len(infos)} 个会议？\n\n将永久删除其音频、转写与纪要：\n{names}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        # 正在回放/生成纪要的会话先停掉
        doomed = {str(x.path) for x in infos}
        if self._player is not None and str(self._player.session_dir) in doomed:
            self._stop_play()
        failed: list[str] = []
        for x in infos:
            try:
                shutil.rmtree(x.path)
                self._minute_state.pop(str(x.path), None)
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{x.title}: {exc}")
        if failed:
            QMessageBox.warning(self, "部分删除失败", "\n".join(failed))
        self.refresh()

    def _set_status(self, text: str) -> None:  # 兼容旧调用
        self.status.setText(text)
