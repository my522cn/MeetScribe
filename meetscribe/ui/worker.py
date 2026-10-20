"""录音 + 识别的 Qt 工作对象：把音频线程、识别线程与界面用信号连接起来。"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..asr.engine import ASRConfig, StreamingASR
from ..asr.pipeline import Caption, LivePipeline
from ..asr.refine import RefineJob, RefineService
from ..audio.manifest import SessionManifest
from ..audio.recorder import DualTrackRecorder
from ..text.postprocess import TextPostProcessor, has_punctuation

SPEAKER_ONLINE = {"mic": "我", "loop": "对方"}
SPEAKER_OFFLINE = {"mic": "本地", "loop": "远端"}

# 精修模型优先级：离线 Paraformer 实测 CER 0.0%，明显优于 SenseVoice（3.7%）
REFINE_CANDIDATES = [("paraformer_offline", "paraformer"), ("sense_voice", "sense_voice")]


def _normalize(text: str) -> str:
    drop = set("，。、？！：；“”‘’（）()[]{},.?!:;\"' \n\t-—")
    return "".join(ch.lower() for ch in text if ch not in drop)


class CaptureWorker(QObject):
    partial = Signal(str, str, int)          # track, text, start_ms
    final = Signal(str, str, int, int)       # track, text, start_ms, end_ms
    refined = Signal(str, int, str)          # track, start_ms, 精修后文本
    overlap = Signal(bool)
    state_changed = Signal(str)              # recording / paused / stopped
    error = Signal(str)
    session_saved = Signal(str)
    segment_count = Signal(int)

    def __init__(self, model_dir: Path, session_root: Path, config_dir: Path | None = None) -> None:
        super().__init__()
        self.model_dir = Path(model_dir)
        self.session_root = Path(session_root)
        root = Path(__file__).resolve().parents[2]
        cfg = Path(config_dir) if config_dir else root / "config"
        self.post = TextPostProcessor(
            hotwords_file=cfg / "hotwords.txt",
            replacements_file=cfg / "replacements.txt",
            punct_model=root / "models" / "punct_zh_en" / "model.int8.onnx",
        )
        self.refiner = self._make_refiner(root / "models")
        self.mode = "online"
        self.engine: StreamingASR | None = None
        self.pipeline: LivePipeline | None = None
        self.recorder: DualTrackRecorder | None = None
        self.manifest: SessionManifest | None = None
        self._segment_total = 0
        self._transcript = None
        self._records: list[dict] = []
        self._lock = threading.Lock()

    def _make_refiner(self, models_root: Path) -> RefineService:
        for folder, kind in REFINE_CANDIDATES:
            d = models_root / folder
            if (d / "model.int8.onnx").exists() and (d / "tokens.txt").exists():
                return RefineService(d, on_result=self._on_refined, kind=kind)
        # 一个都没有：返回一个不可用的服务，submit 会被静默丢弃
        return RefineService(models_root / "paraformer_offline", on_result=self._on_refined)

    # ---------- 对外 ----------
    @property
    def is_active(self) -> bool:
        return self.recorder is not None

    @property
    def elapsed_ms(self) -> int:
        return self.recorder.elapsed_ms if self.recorder else 0

    def levels(self) -> tuple[float, float]:
        if not self.recorder:
            return 0.0, 0.0
        lv = self.recorder.levels()
        return lv.get("mic", 0.0), lv.get("loop", 0.0)

    def start(self, *, mode: str, title: str, source: str) -> None:
        if self.recorder is not None:
            return
        self.mode = mode
        try:
            if self.engine is None:
                self.engine = StreamingASR(ASRConfig(model_dir=self.model_dir, num_threads=2))
            self.pipeline = LivePipeline(
                self.engine,
                on_partial=lambda c: self.partial.emit(c.track, c.text, c.start_ms),
                on_final=self._on_final,
                on_overlap=lambda b: self.overlap.emit(b),
            )
            self._segment_total = 0
            self.recorder = DualTrackRecorder(
                self.session_root,
                segment_seconds=300,
                mode=mode,
                title=title,
                source=source,
                on_audio=lambda t, b: self.pipeline.push(t, b),
                on_segment=self._on_segment,
            )
            self.pipeline.start()
            self.recorder.start()
            self.refiner.start()
            # 目录由 recorder.start() 创建，转写文件必须在之后再打开
            self._records = []
            self._transcript = open(  # noqa: SIM115
                self.recorder.session_dir / "transcript.jsonl", "a", encoding="utf-8"
            )
            self.state_changed.emit("recording")
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"启动录音失败: {exc}")
            self._teardown()

    def pause(self) -> None:
        if self.recorder:
            self.recorder.pause()
            self.state_changed.emit("paused")

    def resume(self) -> None:
        if self.recorder:
            self.recorder.resume()
            self.state_changed.emit("recording")

    def stop(self) -> None:
        if not self.recorder:
            return
        session_dir = str(self.recorder.session_dir)
        try:
            self.manifest = self.recorder.stop()
            if self.pipeline:
                self.pipeline.stop()
            self.refiner.stop()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"停止录音时出错: {exc}")
        finally:
            self.state_changed.emit("stopped")
            if self.manifest:
                self.session_saved.emit(session_dir)
            self._teardown()

    # ---------- 内部 ----------
    def _on_final(self, cap: Caption) -> None:
        text = self.post.process(cap.text)
        if not text:
            return
        speaker = (SPEAKER_ONLINE if self.mode == "online" else SPEAKER_OFFLINE).get(cap.track, cap.track)
        record = {
            "track": cap.track,
            "speaker": speaker,
            "start_ms": cap.start_ms,
            "end_ms": cap.end_ms,
            "text": text,
            "refined": False,
            "raw": cap.text,
        }
        with self._lock:
            self._records.append(record)
            self._flush_transcript()
        self.final.emit(cap.track, text, cap.start_ms, cap.end_ms)

        # 交给后台用离线模型重识别同一句音频（两遍识别）
        if cap.audio is not None and len(cap.audio) > 8000:  # 至少 0.5 秒
            self.refiner.submit(
                RefineJob(track=cap.track, audio=cap.audio, raw_text=cap.text,
                          start_ms=cap.start_ms, end_ms=cap.end_ms)
            )

    def _on_refined(self, job: RefineJob, text: str) -> None:
        polished = self.post.process(text, punctuate=not has_punctuation(text))
        if not polished:
            return
        with self._lock:
            for rec in reversed(self._records):
                if rec["track"] == job.track and rec["start_ms"] == job.start_ms:
                    if rec["text"] == polished:
                        return
                    if not self._worth_replacing(rec["text"], polished):
                        rec["refine_rejected"] = polished  # 留档，方便排查
                        return
                    rec["refine_from"] = rec["text"]
                    rec["text"] = polished
                    rec["refined"] = True
                    self._flush_transcript()
                    break
            else:
                return
        self.refined.emit(job.track, job.start_ms, polished)

    @staticmethod
    def _worth_replacing(old: str, new: str) -> bool:
        """离线结果明显偏离流式结果时（多半是切音频切歪了），宁可保留原文。"""
        a, b = _normalize(old), _normalize(new)
        if not a or not b:
            return False
        if len(b) < 0.65 * len(a) or len(b) > 1.8 * len(a):
            return False
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a, b).ratio() >= 0.55

    @staticmethod
    def _worth_replacing(old: str, new: str) -> bool:
        """离线结果明显偏离流式结果时（多半是切音频切歪了），宁可保留原文。"""
        a, b = _normalize(old), _normalize(new)
        if not a or not b:
            return False
        if len(b) < 0.65 * len(a) or len(b) > 1.8 * len(a):
            return False
        from difflib import SequenceMatcher

        return SequenceMatcher(None, a, b).ratio() >= 0.55

    def _flush_transcript(self) -> None:
        if self._transcript is None:
            return
        try:
            self._transcript.seek(0)
            self._transcript.truncate()
            for rec in self._records:
                out = {k: v for k, v in rec.items() if not k.startswith("refine_")}
                self._transcript.write(json.dumps(out, ensure_ascii=False) + "\n")
            self._transcript.flush()
        except Exception:  # noqa: BLE001
            pass

    def _on_segment(self, seg) -> None:
        self._segment_total += 1
        self.segment_count.emit(self._segment_total)

    def _teardown(self) -> None:
        if self._transcript is not None:
            try:
                self._transcript.close()
            except Exception:  # noqa: BLE001
                pass
            self._transcript = None
        self.pipeline = None
        self.recorder = None
        # 识别模型保留复用，避免二次加载
