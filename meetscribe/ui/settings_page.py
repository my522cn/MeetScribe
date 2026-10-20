"""配置页：硬件概况、音频设备选择与自检、识别参数、存储路径、大模型接入。

设计原则：
- 所有耗时操作（枚举设备、探测硬件、录音测试、识别）都放到后台线程，界面不卡；
- 改完点"保存配置"才写入 settings.json，避免误触；
- "测试"录 5 秒音频，直接给出电平判断 + 流式/离线两版识别结果，用来对比两遍识别的差距。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from ..audio import probe
from ..audio.devices import list_input_devices, list_speakers
from ..config import PROJECT_ROOT, Settings
from ..system import hardware


# ---------------------------------------------------------------- 后台探测
class DeviceProbe(QObject):
    level = Signal(float)
    progress = Signal(float)
    finished = Signal(object)          # ProbeResult
    recognized = Signal(str, str, float, float)   # 流式文本, 离线文本, 离线RTF, 加载秒
    failed = Signal(str)
    stage = Signal(str)                # 阶段提示（录音中 x.x 秒 / 正在识别…）

    def __init__(self, settings: Settings, worker=None) -> None:
        super().__init__()
        self.settings = settings
        self.worker = worker    # 复用主工作对象的流式识别器，避免重复加载模型
        self.device_label = ""
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def start_mic(self, index: int | None, seconds: float) -> None:
        self._run(lambda: probe.record_mic(index, seconds, self._on_level))

    def start_loopback(self, name: str | None, seconds: float) -> None:
        self._run(lambda: probe.record_loopback(name, seconds, self._on_level))

    def _on_level(self, seconds: float, rms: float) -> None:
        self.level.emit(rms)
        self.progress.emit(seconds)
        # 录音过程给出实时反馈，否则 5 秒里界面毫无动静，看起来像"没反应"
        self.stage.emit(f"正在录音 {seconds:.1f}s / 5s…（现在对着麦克风说话）")

    def _run(self, fn) -> None:
        if self._busy:
            return
        self._busy = True
        threading.Thread(target=self._work, args=(fn,), name="probe", daemon=True).start()

    def _work(self, fn) -> None:
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001
            self._busy = False
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)
        # 有声音才做识别，否则纯噪声识别出来是幻觉，反而误导
        if result.empty:
            self._busy = False
            return
        try:
            self.stage.emit("正在流式识别（实时字幕同款引擎）…")
            t_load = time.perf_counter()
            stream_text = self._streaming(result.audio)
            from ..asr.offline import get_offline, resolve_model_dir

            self.stage.emit("正在离线精修识别（首次需加载模型，约 5~15 秒）…")
            model_dir, kind = resolve_model_dir(PROJECT_ROOT / "models", self.settings.refine_model)
            asr = get_offline(model_dir, kind, int(self.settings.get("asr.refine_threads", 2)))
            asr.load()
            load_s = time.perf_counter() - t_load
            off_text = asr.transcribe(result.audio)
            self.recognized.emit(stream_text, off_text, asr.last_rtf, load_s)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"识别失败：{exc}")
        finally:
            self._busy = False

    def _streaming(self, audio) -> str:
        import numpy as np

        from ..asr.engine import ASRConfig, StreamingASR

        engine = getattr(self.worker, "engine", None) if self.worker else None
        own = False
        if engine is None:
            engine = StreamingASR(ASRConfig(model_dir=PROJECT_ROOT / "models" / "paraformer_zh_en",
                                            num_threads=int(self.settings.get("asr.num_threads", 2))))
            own = True
        key = "probe"
        engine.stream(key)
        for i in range(0, len(audio), 1600):
            engine.feed(key, np.asarray(audio[i : i + 1600], dtype=np.float32))
        text = engine.flush(key, 0.8)
        engine.finish_sentence(key)
        if not own and self.worker is not None and self.worker.engine is None:
            self.worker.engine = engine
        engine._streams.pop(key, None)
        return (text or "").strip()


# ---------------------------------------------------------------- 页面
def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
    card = QFrame()
    card.setObjectName("Card")
    lay = QVBoxLayout(card)
    lay.setContentsMargins(18, 14, 18, 16)
    lay.setSpacing(10)
    if title:
        t = QLabel(title)
        t.setObjectName("SectionTitle")
        lay.addWidget(t)
    return card, lay


def _row(label: str, widget: QWidget, hint: str = "", stretch: bool = True,
         hint_width: int | None = None) -> QWidget:
    """一行 = 标签 + 控件 [+ 提示]。

    stretch=True：控件随窗口拉宽（设备下拉、路径输入这类长文本）；
    stretch=False：控件保持固有宽度（数字框这类短内容），剩余空间留白，
    否则一个 4 位数的数字框会被拉到 600px 宽，窗口一大就很难看。
    hint_width：给提示标签固定宽度。两个设备行的提示一长一短时会把
    两个下拉挤得不一样宽（用户反馈"麦克风框和系统声音框不一样长"），
    固定宽度后两行完全对称。
    """
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(12)
    lab = QLabel(label)
    lab.setObjectName("FieldLabel")
    lab.setFixedWidth(96)
    h.addWidget(lab, 0, Qt.AlignmentFlag.AlignVCenter)
    h.addWidget(widget, 1 if stretch else 0)
    if not stretch:
        h.addSpacing(8)
        h.addStretch(1)
    if hint:
        hh = QLabel(hint)
        hh.setObjectName("Hint")
        if hint_width:
            hh.setFixedWidth(hint_width)
        h.addWidget(hh, 0)
    return w


class SettingsPage(QWidget):
    saved = Signal()
    hw_ready = Signal()            # 后台线程探测完硬件
    llm_result = Signal(str, str)  # 文本, 样式名（Good/Warn/Bad）

    def __init__(self, settings: Settings, worker=None, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.worker = worker
        self.probe = DeviceProbe(settings, worker)
        self._hw: hardware.Hardware | None = None
        self._mic_devices = []
        self._speakers = []
        self._build()
        self._wire()
        self._load_from_settings()
        self.refresh_devices()
        self.refresh_hardware()

    # ---------- 构建 ----------
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 22, 28, 18)
        outer.setSpacing(14)

        head = QHBoxLayout()
        title = QLabel("设置")
        title.setObjectName("PageTitle")
        head.addWidget(title)
        head.addStretch(1)
        self.btn_reset = QPushButton("恢复默认")
        self.btn_reset.setObjectName("Ghost")
        self.btn_save = QPushButton("保存配置")
        self.btn_save.setObjectName("Primary")
        head.addWidget(self.btn_reset)
        head.addWidget(self.btn_save)
        outer.addLayout(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # 内容始终随窗口宽度收缩，永远不出横向滚动条
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        # 注意：直接写 setStyleSheet("background: transparent") 会作用于所有后代，
        # 把应用级样式表里输入框的白底也覆盖掉（离控件越近的样式表优先级越高），
        # 输入框就变成透明灰底、看起来像禁用——这是"设置页输入框发灰"的根因。
        # 这里用 objectName 选择器把规则限定在容器自身。
        body.setObjectName("SettingsBody")
        body.setStyleSheet("QWidget#SettingsBody { background: transparent; }")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(14)

        lay.addWidget(self._build_hardware())
        lay.addWidget(self._build_audio())
        lay.addWidget(self._build_asr())
        lay.addWidget(self._build_test())
        lay.addWidget(self._build_storage())
        lay.addWidget(self._build_llm())
        lay.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

    def _build_hardware(self) -> QFrame:
        card, lay = _card("本机硬件与建议")
        self.hw_text = QLabel("检测中…")
        self.hw_text.setObjectName("Mono")
        # 长文本必须换行且不参与最小宽度计算，否则会把整页撑出横向滚动条
        self.hw_text.setWordWrap(True)
        self.hw_text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.hw_text)
        self.hw_verdict = QLabel("")
        self.hw_verdict.setObjectName("Good")
        self.hw_verdict.setWordWrap(True)
        lay.addWidget(self.hw_verdict)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(1, 16)
        self.spin_threads.setFixedWidth(110)
        self.spin_refine_threads = QSpinBox()
        self.spin_refine_threads.setRange(1, 16)
        self.spin_refine_threads.setFixedWidth(110)
        h.addWidget(QLabel("流式识别线程"))
        h.addWidget(self.spin_threads)
        h.addSpacing(16)
        h.addWidget(QLabel("精修线程"))
        h.addWidget(self.spin_refine_threads)
        h.addStretch(1)
        self.btn_hw = QPushButton("重新检测")
        self.btn_hw.setObjectName("Ghost")
        h.addWidget(self.btn_hw)
        lay.addWidget(row)
        self.btn_apply_hw = QPushButton("按建议填充线程数")
        self.btn_apply_hw.setObjectName("Ghost")
        lay.addWidget(self.btn_apply_hw, 0, Qt.AlignmentFlag.AlignLeft)
        return card

    def _build_audio(self) -> QFrame:
        card, lay = _card("音频设备")
        self.cmb_mic = QComboBox()
        self.cmb_loop = QComboBox()
        # 默认策略会按最长设备名把框撑长（测试/刷新后突然变长、两框不一致），
        # 改为固定内容长度策略：宽度交给布局，两个下拉永远等宽
        for box in (self.cmb_mic, self.cmb_loop):
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(16)
        self.btn_test_mic = QPushButton("测试 5 秒")
        self.btn_test_mic.setObjectName("Ghost")
        self.btn_test_loop = QPushButton("测试 5 秒")
        self.btn_test_loop.setObjectName("Ghost")

        mic_row = QWidget()
        mh = QHBoxLayout(mic_row)
        mh.setContentsMargins(0, 0, 0, 0)
        mh.setSpacing(8)
        mh.addWidget(self.cmb_mic, 1)
        mh.addWidget(self.btn_test_mic)
        loop_row = QWidget()
        lh = QHBoxLayout(loop_row)
        lh.setContentsMargins(0, 0, 0, 0)
        lh.setSpacing(8)
        lh.addWidget(self.cmb_loop, 1)
        lh.addWidget(self.btn_test_loop)

        lay.addWidget(_row("麦克风", mic_row, "录你的声音", hint_width=170))
        lay.addWidget(_row("系统声音", loop_row, "录会议里对方的声音", hint_width=170))

        self.level_bar = QProgressBar()
        self.level_bar.setObjectName("LevelBar")
        self.level_bar.setRange(0, 100)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(8)
        self.level_status = QLabel("未测试")
        self.level_status.setObjectName("Hint")
        lv = QWidget()
        lv_l = QVBoxLayout(lv)
        lv_l.setContentsMargins(0, 0, 0, 0)
        lv_l.setSpacing(6)
        lv_l.addWidget(self.level_bar)
        lv_l.addWidget(self.level_status)
        lay.addWidget(_row("电平", lv))

        self.spin_segment = QSpinBox()
        self.spin_segment.setRange(1, 60)
        self.spin_segment.setSuffix(" 分钟")
        self.spin_segment.setFixedWidth(150)
        lay.addWidget(_row("分段时长", self.spin_segment, "崩溃最多丢一段", stretch=False))

        self.cmb_format = QComboBox()
        self.cmb_format.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_format.setMinimumContentsLength(16)
        self.cmb_format.addItem("FLAC 无损 · 约 30 MB/小时", "flac")
        self.cmb_format.addItem("OPUS 压缩 · 约 13 MB/小时（识别同样可用）", "opus")
        self.cmb_format.setMaximumWidth(430)
        lay.addWidget(_row("录音格式", self.cmb_format, stretch=False))
        return card

    def _build_asr(self) -> QFrame:
        card, lay = _card("识别与纠错")
        self.chk_two_pass = QCheckBox("句末二次精修：每句说完后用离线大模型重识别，修正错别字")
        self.chk_two_pass.setChecked(True)
        lay.addWidget(self.chk_two_pass)

        self.cmb_refine = QComboBox()
        self.cmb_refine.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_refine.setMinimumContentsLength(16)
        self.cmb_refine.addItem("自动（推荐：离线 Paraformer）", "auto")
        self.cmb_refine.addItem("强制离线 Paraformer", "paraformer")
        self.cmb_refine.addItem("SenseVoice（多语种，英文术语较弱）", "sense_voice")
        self.cmb_refine.addItem("关闭精修（省内存）", "off")
        self.cmb_refine.setMaximumWidth(430)
        lay.addWidget(_row("精修模型", self.cmb_refine, stretch=False))

        self.chk_live_refine = QCheckBox("长句滑窗预校正：说话超过 6 秒后，遇到句中停顿就提前校正一次（不必等断句）")
        lay.addWidget(self.chk_live_refine)

        self.spin_live_after = QDoubleSpinBox()
        self.spin_live_after.setRange(3.0, 30.0)
        self.spin_live_after.setSingleStep(0.5)
        self.spin_live_after.setSuffix(" 秒")
        self.spin_live_after.setFixedWidth(150)
        lay.addWidget(_row("触发时机", self.spin_live_after, "本句说满多少秒后，句中停顿 ≥0.6 秒时校正", stretch=False))

        self.spin_endpoint = QDoubleSpinBox()
        self.spin_endpoint.setRange(0.6, 4.0)
        self.spin_endpoint.setSingleStep(0.1)
        self.spin_endpoint.setSuffix(" 秒")
        self.spin_endpoint.setFixedWidth(150)
        lay.addWidget(_row("断句静音", self.spin_endpoint, "越小断句越碎，越大整段越长", stretch=False))
        return card

    def _build_test(self) -> QFrame:
        card, lay = _card("测试识别")
        hint = QLabel("按住说话或直接点按钮，录 5 秒后同时给出「流式首遍」和「离线精修」两版结果，可直接看出差别。")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self.btn_speak_test = QPushButton("录 5 秒并识别（用上面选中的麦克风）")
        self.btn_speak_test.setObjectName("Ghost")
        lay.addWidget(self.btn_speak_test, 0, Qt.AlignmentFlag.AlignLeft)

        self.txt_stream = QTextEdit()
        self.txt_stream.setReadOnly(True)
        self.txt_stream.setFixedHeight(58)
        self.txt_stream.setPlaceholderText("流式首遍（实时字幕看到的内容）")
        self.txt_offline = QTextEdit()
        self.txt_offline.setReadOnly(True)
        self.txt_offline.setFixedHeight(58)
        self.txt_offline.setPlaceholderText("离线精修（二次识别后的内容）")
        lay.addWidget(self.txt_stream)
        lay.addWidget(self.txt_offline)
        self.test_meta = QLabel("")
        self.test_meta.setObjectName("Hint")
        self.test_meta.setWordWrap(True)
        self.test_meta.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.test_meta)
        return card

    def _build_storage(self) -> QFrame:
        card, lay = _card("存储")
        self.edit_root = QLineEdit()
        btn = QPushButton("打开文件夹")
        btn.setObjectName("Ghost")
        btn.clicked.connect(self._open_root)
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        h.addWidget(self.edit_root, 1)
        h.addWidget(btn)
        lay.addWidget(_row("会议存放", w))
        return card

    def _build_llm(self) -> QFrame:
        card, lay = _card("大模型（可选，用于错别字上下文纠错与会议纪要）")
        self.chk_llm = QCheckBox("启用本地大模型（需要本机运行 Ollama）")
        lay.addWidget(self.chk_llm)

        self.edit_llm_url = QLineEdit()
        lay.addWidget(_row("Ollama 地址", self.edit_llm_url))

        self.cmb_fix_model = QComboBox()
        self.cmb_fix_model.setEditable(True)
        self.cmb_sum_model = QComboBox()
        self.cmb_sum_model.setEditable(True)
        for box, items in (
            (self.cmb_fix_model, ["qwen2.5:3b-instruct", "qwen2.5:7b-instruct", "qwen3:4b"]),
            (self.cmb_sum_model, ["qwen2.5:7b-instruct", "qwen2.5:3b-instruct", "qwen3:8b"]),
        ):
            box.addItems(items)
        for box in (self.cmb_fix_model, self.cmb_sum_model):
            box.setFixedWidth(260)
        lay.addWidget(_row("纠错模型", self.cmb_fix_model, stretch=False))
        lay.addWidget(_row("纪要模型", self.cmb_sum_model, stretch=False))

        self.btn_llm_test = QPushButton("检测连接与模型")
        self.btn_llm_test.setObjectName("Ghost")
        self.llm_status = QLabel("未检测")
        self.llm_status.setObjectName("Hint")
        self.llm_status.setWordWrap(True)
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        h.addWidget(self.btn_llm_test)
        h.addWidget(self.llm_status, 1)
        lay.addWidget(_row("连接", w))
        return card

    # ---------- 信号 ----------
    def _wire(self) -> None:
        self.btn_save.clicked.connect(self._save)
        self.btn_reset.clicked.connect(self._reset)
        self.btn_hw.clicked.connect(lambda: self.refresh_hardware())
        self.btn_apply_hw.clicked.connect(self._apply_hw)
        self.btn_test_mic.clicked.connect(lambda: self._test_device("mic"))
        self.btn_test_loop.clicked.connect(lambda: self._test_device("loop"))
        self.btn_speak_test.clicked.connect(self._test_speak)
        self.btn_llm_test.clicked.connect(self._test_llm)
        self.probe.level.connect(self._on_level)
        self.probe.finished.connect(self._on_probe_done)
        self.probe.recognized.connect(self._on_recognized)
        self.probe.failed.connect(self._on_probe_failed)
        self.probe.stage.connect(self._on_stage)
        self.hw_ready.connect(self._apply_hw_ui)
        self.llm_result.connect(self._on_llm_result)

    # ---------- 数据加载 ----------
    def _load_from_settings(self) -> None:
        s = self.settings
        self.spin_threads.setValue(int(s.get("asr.num_threads", 2)))
        self.spin_refine_threads.setValue(int(s.get("asr.refine_threads", 2)))
        self.spin_segment.setValue(int(s.get("audio.segment_minutes", 5)))
        idx = max(0, self.cmb_format.findData(str(s.get("audio.format", "flac"))))
        self.cmb_format.setCurrentIndex(idx)
        self.chk_two_pass.setChecked(bool(s.get("asr.two_pass", True)))
        idx = max(0, self.cmb_refine.findData(str(s.get("asr.refine_model", "auto"))))
        self.cmb_refine.setCurrentIndex(idx)
        self.chk_live_refine.setChecked(bool(s.get("asr.live_refine", True)))
        self.spin_live_after.setValue(float(s.get("asr.live_refine_after", 6.0)))
        self.spin_endpoint.setValue(float(s.get("asr.endpoint_silence", 1.2)))
        self.edit_root.setText(str(s.get("storage.session_root", "data/sessions")))
        self.chk_llm.setChecked(bool(s.get("llm.enabled", False)))
        self.edit_llm_url.setText(str(s.get("llm.base_url", "http://127.0.0.1:11434")))
        self.cmb_fix_model.setCurrentText(str(s.get("llm.fix_model", "qwen3:4b")))
        self.cmb_sum_model.setCurrentText(str(s.get("llm.summary_model", "qwen3:8b")))

    def refresh_devices(self) -> None:
        self.cmb_mic.blockSignals(True)
        self.cmb_mic.clear()
        self.cmb_mic.addItem("自动（系统默认麦克风）", None)
        try:
            self._mic_devices = list_input_devices()
        except Exception:  # noqa: BLE001
            self._mic_devices = []
        for d in self._mic_devices:
            self.cmb_mic.addItem(f"{d.name}  ·  {d.default_samplerate:.0f}Hz", d.name)
        want = str(self.settings.get("audio.mic_name", "") or "")
        hit = self.cmb_mic.findData(want) if want else 0
        self.cmb_mic.setCurrentIndex(max(0, hit))
        self.cmb_mic.blockSignals(False)

        self.cmb_loop.blockSignals(True)
        self.cmb_loop.clear()
        self.cmb_loop.addItem("自动（默认扬声器回环）", None)
        try:
            self._speakers = list_speakers()
        except Exception:  # noqa: BLE001
            self._speakers = []
        for name, _id in self._speakers:
            self.cmb_loop.addItem(name, name)
        want = self.settings.get("audio.loopback_name") or ""
        hit = self.cmb_loop.findData(want) if want else 0
        self.cmb_loop.setCurrentIndex(max(0, hit))
        self.cmb_loop.blockSignals(False)

    def refresh_hardware(self) -> None:
        self.hw_text.setText("检测中…")
        self.btn_hw.setEnabled(False)

        def work() -> None:
            self._hw = hardware.profile(force=True)
            self.hw_ready.emit()

        threading.Thread(target=work, name="hw", daemon=True).start()

    def _apply_hw_ui(self) -> None:
        hw = self._hw
        self.btn_hw.setEnabled(True)
        if hw is None:
            self.hw_text.setText("检测失败")
            return
        self.hw_text.setText(hw.summary())
        self.hw_verdict.setObjectName("Good" if hw.can_two_pass else "Warn")
        self.hw_verdict.style().unpolish(self.hw_verdict)
        self.hw_verdict.style().polish(self.hw_verdict)
        self.hw_verdict.setText(
            f"建议：{hw.verdict}　（流式 {hw.asr_threads} 线程 / 精修 {hw.refine_threads} 线程，"
            f"大模型可用规模 {hw.llm_tier}）"
        )

    def _apply_hw(self) -> None:
        hw = self._hw or hardware.profile()
        self.spin_threads.setValue(hw.asr_threads)
        self.spin_refine_threads.setValue(hw.refine_threads)
        self.chk_two_pass.setChecked(hw.can_two_pass)
        self.chk_live_refine.setChecked(hw.can_live_refine)

    # ---------- 设备测试 ----------
    def _selected_mic_index(self) -> int | None:
        """下拉里存的是设备名（索引会随插拔变化），测试与录音统一走名字解析。"""
        name = self.cmb_mic.currentData()
        if not name:
            return None
        for d in self._mic_devices:
            if d.name == name:
                return d.index
        return None

    def _busy_guard(self) -> bool:
        if self.worker is not None and getattr(self.worker, "is_active", False):
            QMessageBox.information(self, "正在录音", "录音过程中不能测试设备，请先结束当前录音。")
            return False
        if self.probe.busy:
            return False
        self.btn_test_mic.setEnabled(False)
        self.btn_test_loop.setEnabled(False)
        self.btn_speak_test.setEnabled(False)
        self.level_status.setText("正在采集…")
        return True

    def _test_device(self, kind: str) -> None:
        if not self._busy_guard():
            return
        if kind == "mic":
            self.probe.start_mic(self._selected_mic_index(), 5.0)
        else:
            self.probe.start_loopback(self.cmb_loop.currentData(), 5.0)

    def _test_speak(self) -> None:
        if not self._busy_guard():
            return
        self.txt_stream.setPlainText("")
        self.txt_offline.setPlainText("")
        self.test_meta.setText("请对着麦克风连续说一句话…")
        self.probe.start_mic(self._selected_mic_index(), 5.0)

    def _release(self) -> None:
        self.btn_test_mic.setEnabled(True)
        self.btn_test_loop.setEnabled(True)
        self.btn_speak_test.setEnabled(True)

    def _on_level(self, rms: float) -> None:
        self.level_bar.setValue(min(100, int(rms * 400)))

    def _on_probe_done(self, result) -> None:
        self.level_bar.setValue(min(100, int(result.rms * 400)))
        cls = "Good" if not result.empty else "Bad"
        self.level_status.setObjectName(cls)
        self.level_status.style().unpolish(self.level_status)
        self.level_status.style().polish(self.level_status)
        self.level_status.setText(
            f"{result.device} · 录到 {result.seconds:.1f}s · 峰值 {result.peak:.2f} · 均值 {result.rms:.3f} —— {result.verdict}"
        )
        if result.empty:
            # 没声音就不做识别（识别出来也是幻觉），但要把原因写进结果框，不能看起来像"没反应"
            self.txt_stream.setPlainText("（没有识别：没听到声音）")
            self.txt_offline.setPlainText("（没有识别：没听到声音）")
            self.test_meta.setText("请检查麦克风是否选对、系统设置里是否允许应用使用麦克风，然后再试一次。")
            self._release()

    def _on_stage(self, text: str) -> None:
        """识别过程的阶段提示，让每一步都看得见（否则像卡死/无效）。"""
        self.test_meta.setText(text)

    def _on_recognized(self, stream_text: str, off_text: str, rtf: float, load_s: float) -> None:
        self._release()  # 修复：识别完成后必须恢复测试按钮，否则之后再点"测试"毫无反应
        self.txt_stream.setPlainText(stream_text or "（流式无输出）")
        self.txt_offline.setPlainText(off_text or "（离线无输出）")
        same = (stream_text or "").replace(" ", "") == (off_text or "").replace(" ", "")
        self.test_meta.setText(
            ("两版完全一致，说明这段音频流式就够准。" if same else "两版不一致 → 精修改掉了首遍的错误。")
            + f"　离线耗时约为音频的 {rtf:.0%}，模型加载 {load_s:.1f}s"
        )

    def _on_probe_failed(self, msg: str) -> None:
        self._release()
        self.level_status.setObjectName("Bad")
        self.level_status.style().unpolish(self.level_status)
        self.level_status.style().polish(self.level_status)
        self.level_status.setText(f"测试失败：{msg}")

    # ---------- 保存 ----------
    def _save(self) -> None:
        s = self.settings
        s.set("asr.num_threads", self.spin_threads.value())
        s.set("asr.refine_threads", self.spin_refine_threads.value())
        s.set("audio.segment_minutes", self.spin_segment.value())
        s.set("audio.format", self.cmb_format.currentData())
        s.set("asr.two_pass", self.chk_two_pass.isChecked())
        s.set("asr.refine_model", self.cmb_refine.currentData())
        s.set("asr.live_refine", self.chk_live_refine.isChecked())
        s.set("asr.live_refine_after", self.spin_live_after.value())
        s.set("asr.endpoint_silence", self.spin_endpoint.value())
        s.set("storage.session_root", self.edit_root.text().strip() or "data/sessions")
        s.set("llm.enabled", self.chk_llm.isChecked())
        s.set("llm.base_url", self.edit_llm_url.text().strip() or "http://127.0.0.1:11434")
        s.set("llm.fix_model", self.cmb_fix_model.currentText().strip())
        s.set("llm.summary_model", self.cmb_sum_model.currentText().strip())

        mic_name = self.cmb_mic.currentData() or ""
        s.set("audio.mic_name", mic_name)
        s.set("audio.mic_index", next((d.index for d in self._mic_devices if d.name == mic_name), None))
        s.set("audio.loopback_name", self.cmb_loop.currentData())
        s.save()
        self.saved.emit()
        self.level_status.setObjectName("Good")
        self.level_status.style().unpolish(self.level_status)
        self.level_status.style().polish(self.level_status)
        self.level_status.setText("配置已保存，下次开始录音时生效。")

    def _reset(self) -> None:
        if QMessageBox.question(self, "恢复默认", "确定把所有配置恢复为默认值吗？") != QMessageBox.StandardButton.Yes:
            return
        from ..config import DEFAULTS
        import copy

        self.settings.data = copy.deepcopy(DEFAULTS)
        self.settings.save()
        self._load_from_settings()
        self.refresh_devices()
        self.saved.emit()

    def _open_root(self) -> None:
        import os
        import subprocess

        p = Path(self.edit_root.text().strip() or "data/sessions")
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(p)], shell=False)  # noqa: S603,S607

    # ---------- 大模型 ----------
    def _test_llm(self) -> None:
        self.llm_status.setText("检测中…")

        def work() -> None:
            from ..llm import OllamaClient

            url = self.edit_llm_url.text().strip() or "http://127.0.0.1:11434"
            fix_name = self.cmb_fix_model.currentText().strip()
            sum_name = self.cmb_sum_model.currentText().strip()
            client = OllamaClient(url, timeout=8)
            try:
                if not client.alive():
                    self.llm_result.emit("服务未运行，正在尝试启动 ollama serve…", "Warn")
                    if not client.ensure_server():
                        self.llm_result.emit(
                            "连不上 Ollama。请确认已安装并运行（开始菜单里启动 Ollama，或执行 ollama serve）",
                            "Bad",
                        )
                        return
                names = client.models()
                msg = f"已连接，本机有 {len(names)} 个模型"
                if names:
                    msg += "：" + "、".join(names[:5])
                missing = [n for n in (fix_name, sum_name) if n and n not in names]
                if missing:
                    msg += "\n未下载：" + "、".join(missing) + f"（执行 ollama pull {missing[0]}）"
                self.llm_result.emit(msg, "Warn" if missing else "Good")
            except Exception as exc:  # noqa: BLE001
                self.llm_result.emit(f"连接失败：{exc}", "Bad")

        threading.Thread(target=work, name="llmtest", daemon=True).start()

    def _on_llm_result(self, text: str, cls: str) -> None:
        self.llm_status.setText(text)
        self.llm_status.setObjectName(cls)
        self.llm_status.style().unpolish(self.llm_status)
        self.llm_status.style().polish(self.llm_status)
