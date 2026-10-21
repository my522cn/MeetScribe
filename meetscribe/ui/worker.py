"""录音 + 识别的 Qt 工作对象：把音频线程、识别线程与界面用信号连接起来。"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..asr.engine import ASRConfig, StreamingASR
from ..asr.diarize import EMB_ASSIGN_THRESHOLD as EMB_THRESHOLD_DEFAULT
from ..asr.diarize import SpeakerRegistry, sentence_features
from ..asr.pipeline import Caption, LivePipeline
from ..asr.refine import RefineJob, RefineService
from ..audio.devices import resolve_input_index
from ..audio.manifest import SessionManifest
from ..audio.recorder import DualTrackRecorder
from ..config import Settings
from ..paths import project_root
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
    final = Signal(str, str, int, int, str, bool)  # track, text, start_ms, end_ms, speaker, 疑似多人
    refined = Signal(str, int, str)          # track, start_ms, 精修后文本
    live_refined = Signal(str, int, str)     # track, start_ms, 句中滑窗校正文本
    overlap = Signal(bool)
    state_changed = Signal(str)              # recording / paused / stopped
    error = Signal(str)
    session_saved = Signal(str)
    segment_count = Signal(int)

    def __init__(
        self,
        model_dir: Path,
        session_root: Path,
        config_dir: Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__()
        self.model_dir = Path(model_dir)
        self.session_root = Path(session_root)
        root = project_root()
        cfg = Path(config_dir) if config_dir else root / "config"
        self.settings = settings or Settings(cfg / "settings.json")
        self._models_root = root / "models"
        self.post = TextPostProcessor(
            hotwords_file=cfg / "hotwords.txt",
            replacements_file=cfg / "replacements.txt",
            punct_model=root / "models" / "punct_zh_en" / "model.int8.onnx",
        )
        self.refiner = self._make_refiner()
        self.mode = "online"
        self.engine: StreamingASR | None = None
        self._engine_sig: tuple | None = None
        self.pipeline: LivePipeline | None = None
        self.recorder: DualTrackRecorder | None = None
        self.manifest: SessionManifest | None = None
        self._segment_total = 0
        self._transcript = None
        self._records: list[dict] = []
        self._lock = threading.Lock()
        self._corrector = None
        self._llm_stop: threading.Event | None = None
        self.diarizer: SpeakerRegistry | None = None

    def _make_refiner(self) -> RefineService:
        """按配置挑精修模型；prefer=off 或没有模型时返回不可用实例（submit 会被丢弃）。"""
        prefer = self.settings.refine_model
        order = list(REFINE_CANDIDATES)
        if prefer == "sense_voice":
            order.reverse()
        for folder, kind in order:
            d = self._models_root / folder
            if (d / "model.int8.onnx").exists() and (d / "tokens.txt").exists():
                return RefineService(
                    d, on_result=self._on_refined, kind=kind,
                    num_threads=int(self.settings.get("asr.refine_threads", 2)),
                )
        return RefineService(self._models_root / "paraformer_offline", on_result=self._on_refined)

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
        # 麦克风轨做说话人区分（线上线下都启用）：有声纹模型用嵌入聚类，没有就退回 F0+质心
        self.diarizer = None
        self._emb_profiles: dict[str, list[float]] = {}
        from ..asr.diarize import EmbeddingSpeakerRegistry, SpeakerRegistry, embedding_model_path

        model_path = embedding_model_path(self._models_root)
        if model_path is not None:
            try:
                self.diarizer = EmbeddingSpeakerRegistry(
                    model_path,
                    threshold=float(self.settings.get("asr.speaker_threshold", EMB_THRESHOLD_DEFAULT)),
                    max_speakers=max(1, min(26, int(self.settings.get("asr.speaker_max", 4)))),
                )
            except Exception:  # noqa: BLE001
                self.diarizer = None
        if self.diarizer is None:
            self.diarizer = SpeakerRegistry()
            self._seed_f0_profiles()
        else:
            self._seed_emb_profiles()
        self._diarize_log = None  # 特征调试日志（写入会话目录 diarize.log）
        s = self.settings
        try:
            self._ensure_engine()
            live_refine = bool(s.get("asr.live_refine", True)) and s.two_pass
            self.pipeline = LivePipeline(
                self.engine,
                on_partial=lambda c: self.partial.emit(c.track, c.text, c.start_ms),
                on_final=self._on_final,
                on_overlap=lambda b: self.overlap.emit(b),
                on_live_job=self._on_live_job if live_refine else None,
                live_refine_after=float(s.get("asr.live_refine_after", 6.0)),
            )
            self._segment_total = 0
            mic_name = str(s.get("audio.mic_name", "") or "")
            self.recorder = DualTrackRecorder(
                s.session_root,
                segment_seconds=s.segment_seconds,
                mode=mode,
                title=title,
                source=source,
                mic_device=resolve_input_index(mic_name, s.get("audio.mic_index")),
                loopback_name=s.get("audio.loopback_name"),
                audio_format=str(s.get("audio.format", "flac")),
                on_audio=lambda t, b: self.pipeline.push(t, b),
                on_segment=self._on_segment,
            )
            self.pipeline.start()
            self.recorder.start()
            if s.two_pass:
                self.refiner.start()
            self._start_llm()
            # 目录由 recorder.start() 创建，转写文件必须在之后再打开
            self._records = []
            self._transcript = open(  # noqa: SIM115
                self.recorder.session_dir / "transcript.jsonl", "a", encoding="utf-8"
            )
            if self.diarizer is not None:
                self._diarize_log = open(  # noqa: SIM115
                    self.recorder.session_dir / "diarize.log", "a", encoding="utf-8"
                )
            self.state_changed.emit("recording")
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"启动录音失败: {exc}")
            self._teardown()

    def _ensure_engine(self) -> None:
        """识别器只在参数变化时重建，避免每次录音都重新加载 237MB 模型。"""
        sig = (
            int(self.settings.get("asr.num_threads", 2)),
            float(self.settings.get("asr.endpoint_silence", 1.2)),
        )
        if self.engine is not None and self._engine_sig == sig:
            return
        self.engine = StreamingASR(
            ASRConfig(
                model_dir=self.model_dir,
                num_threads=sig[0],
                rule2_min_trailing_silence=sig[1],
                rule1_min_trailing_silence=max(2.4, sig[1] * 2),
            )
        )
        self._engine_sig = sig

    def _seed_f0_profiles(self) -> None:
        """声纹长期学习（F0 兜底版）：用历史档案预置已知说话人。"""
        try:
            from ..session_tools import load_global_profiles

            for name, feat in load_global_profiles().items():
                self.diarizer.seed(name, feat)
        except Exception:  # noqa: BLE001
            pass

    def _seed_emb_profiles(self) -> None:
        """声纹长期学习（嵌入版）：加载跨会话积累的平均嵌入。"""
        try:
            from ..session_tools import load_speaker_embeddings

            self._emb_profiles = load_speaker_embeddings()
            for name, vec in self._emb_profiles.items():
                self.diarizer.seed(name, np.asarray(vec, dtype=np.float32))
        except Exception:  # noqa: BLE001
            self._emb_profiles = {}

    def _save_emb_profiles(self) -> None:
        """本场每个说话人的平均嵌入并进长期档案（滑动平均，越用越准）。"""
        if getattr(self.diarizer, "profiles", None) is None:
            return
        try:
            from ..session_tools import merge_speaker_embeddings

            self._emb_profiles = merge_speaker_embeddings(self.diarizer.profiles())
        except Exception:  # noqa: BLE001
            pass

    def pause(self) -> None:
        if self.recorder:
            self.recorder.pause()
            if self.pipeline:
                # 先把没说完的半句定稿，精修/纠错才能继续处理它
                self.pipeline.pause()
            self.state_changed.emit("paused")
            ev = getattr(self, "_llm_wake", None)
            if ev is not None:
                ev.set()  # 立即跑一轮大模型纠错，不必等 25 秒

    def set_mode(self, mode: str) -> None:
        """录音中热切换线上线下。

        双轨（mic+loop）本来就一直在录，切换只改变"之后的句子怎么标注说话人"：
        - 两种模式下麦克风轨都按声纹聚类拆成说话人A/B/C/D（聚类器 start() 时已建好）
        - 线上：扬声器回采标"对方"；线下：回采轨标"本地回声"语义由 SPEAKER_OFFLINE 决定
        已写盘的历史句子不改标注——转写里能看出模式切换的"分界线"。
        """
        if mode == self.mode:
            return
        self.mode = mode
        if self.recorder is None:
            return  # 未在录音：下次 start() 会按新参数初始化
        self.recorder.manifest.mode = mode  # 管理页按最终使用的模式显示
        if self.diarizer is None:
            self.diarizer = SpeakerRegistry()
            try:
                from ..session_tools import load_global_profiles

                for name, feat in load_global_profiles().items():
                    self.diarizer.seed(name, feat)
            except Exception:  # noqa: BLE001
                pass
        if self.recorder is not None and getattr(self, "_diarize_log", None) is None:
            try:
                self._diarize_log = open(  # noqa: SIM115
                    self.recorder.session_dir / "diarize.log", "a", encoding="utf-8"
                )
            except Exception:  # noqa: BLE001
                pass

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
            self._stop_llm()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"停止录音时出错: {exc}")
        finally:
            self.state_changed.emit("stopped")
            if self.manifest:
                self.session_saved.emit(session_dir)
            self._teardown()
            # 队列里没精修完的句子一句不丢：后台继续跑，全部完成后才关转写文件
            if self.refiner.running:
                self.refiner.drain(on_done=self._finish_session)
            else:
                self._finish_session()

    def _finish_session(self) -> None:
        """所有精修完成后的收尾：把最终结果刷盘、关闭文件、释放精修线程。"""
        with self._lock:
            self._flush_transcript()
        if self._transcript is not None:
            try:
                self._transcript.close()
            except Exception:  # noqa: BLE001
                pass
            self._transcript = None
        if self._diarize_log is not None:
            try:
                self._diarize_log.close()
            except Exception:  # noqa: BLE001
                pass
            self._diarize_log = None
        self._save_emb_profiles()  # 声纹档案：本场结果并进跨会话长期档案
        self.refiner.stop()

    # ---------- 大模型纠错 ----------
    def _start_llm(self) -> None:
        """启用时每 25 秒用本地小模型把最近几句过一遍，改同音错字。"""
        self._llm_stop = threading.Event()
        self._corrector = None
        if not self.settings.get("llm.enabled", False):
            return
        try:
            from ..llm import Corrector, OllamaClient
        except Exception:  # noqa: BLE001
            return
        base = str(self.settings.get("llm.base_url", "http://127.0.0.1:11434"))
        model = str(self.settings.get("llm.fix_model", "qwen3:4b"))
        hotwords = []
        hw_file = self.settings.path.parent / "hotwords.txt"
        if hw_file.exists():
            hotwords = [ln.strip() for ln in hw_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self._corrector = Corrector(
            OllamaClient(base, timeout=float(self.settings.get("llm.timeout", 90))), model, hotwords
        )
        self._llm_wake = threading.Event()
        threading.Thread(target=self._prepare_llm, name="llmprep", daemon=True).start()

    def _prepare_llm(self) -> None:
        """后台确认服务可用（必要时拉起 ollama serve），再开始周期纠错。"""
        try:
            self._corrector.client.ensure_server()
        except Exception:  # noqa: BLE001
            pass
        self._llm_loop()

    def _stop_llm(self) -> None:
        ev = getattr(self, "_llm_stop", None)
        if ev is not None:
            ev.set()

    def _llm_loop(self) -> None:
        from ..llm import FixRequest

        while True:
            self._llm_wake.wait(25)  # 录音中每 25 秒一轮；暂停/新句定稿会被提前唤醒
            if self._llm_stop.is_set():
                return
            self._llm_wake.clear()
            with self._lock:
                pending = [
                    r for r in self._records
                    if not r.get("llm") and len(_normalize(r["text"])) >= 4
                ][:8]  # 按时间顺序处理，保证早晚能追上
            if not pending or self._corrector is None:
                continue
            items = [FixRequest((r["track"], r["start_ms"]), r["text"]) for r in pending]
            fixed = self._corrector.fix(items)
            with self._lock:
                for r in pending:
                    r["llm"] = True
                for rec in self._records:
                    key = (rec["track"], rec["start_ms"])
                    if key in fixed:
                        rec["llm_from"] = rec["text"]
                        rec["text"] = fixed[key]
                self._flush_transcript()
            for (track, start), new in fixed.items():
                self.refined.emit(track, start, new)

    # ---------- 内部 ----------
    def _is_echo_of_loop(self, track: str, start_ms: int, end_ms: int, text: str) -> bool:
        """扬声器播放声被麦克风重新拾音（声学回声）会整句复刻回采轨内容。

        判定：与某条回采轨句子时间重叠（±1.5s）且文本高度相似 → 这句是回声，
        丢弃，避免同一段话被标成两个人。只在麦克风轨上做（回声只会从喇叭进麦克风）。
        """
        if track != "mic":
            return False
        import difflib

        with self._lock:
            loop_rows = [
                r for r in self._records
                if r["track"] == "loop" and r["start_ms"] < end_ms + 1500 and start_ms < r["end_ms"] + 1500
            ]
        a = _normalize(text)
        if not a:
            return False
        for r in loop_rows:
            b = _normalize(r["text"])
            if not b:
                continue
            if difflib.SequenceMatcher(None, a, b).ratio() >= 0.6:
                return True
        return False

    def _on_final(self, cap: Caption) -> None:
        text = self.post.process(cap.text)
        if not text:
            return
        if self._is_echo_of_loop(cap.track, cap.start_ms, cap.end_ms, text):
            return  # 扬声器回声：回采轨已有这句，麦克风拾到的副本不重复入库
        base = SPEAKER_ONLINE if self.mode == "online" else SPEAKER_OFFLINE
        speaker = base.get(cap.track, cap.track)
        mixed = 0.0
        # 麦克风轨不分"本地/我"，按声纹聚类拆成说话人A/B/C/D（线上线下都启用）；
        # 扬声器回采轨保持"对方"（远端一路，无需拆分）
        if cap.track == "mic" and self.diarizer is not None:
            if getattr(self.diarizer, "assign_sentence", None) is not None:
                # 嵌入版：滑窗投票，能识别"一句里两个人抢话"并给出比例
                speaker, mixed = self.diarizer.assign_sentence(cap.audio)
                if self._diarize_log is not None:
                    try:
                        self._diarize_log.write(
                            f"{cap.start_ms}\temb\t{mixed:.2f}\t{speaker}\n"
                        )
                        self._diarize_log.flush()
                    except Exception:  # noqa: BLE001
                        pass
            else:
                feat = sentence_features(cap.audio)
                if feat is not None:
                    speaker = self.diarizer.assign(feat)
                else:
                    # 句子太短/太轻提不出特征：归入当前说话最多的人，而不是笼统的"本地"
                    speaker = self.diarizer.majority_label()
                if self._diarize_log is not None:
                    try:
                        f0s, cents = feat if feat is not None else (-1, -1)
                        self._diarize_log.write(
                            f"{cap.start_ms}\t{f0s:.1f}\t{cents:.1f}\t{speaker}\n"
                        )
                        self._diarize_log.flush()
                    except Exception:  # noqa: BLE001
                        pass
        record = {
            "track": cap.track,
            "speaker": speaker,
            "start_ms": cap.start_ms,
            "end_ms": cap.end_ms,
            "text": text,
            "refined": False,
            "raw": cap.text,
            "mixed": round(mixed, 2),
        }
        with self._lock:
            self._records.append(record)
            self._flush_transcript()
        self.final.emit(cap.track, text, cap.start_ms, cap.end_ms, speaker, mixed > 0.34)
        ev = getattr(self, "_llm_wake", None)
        if ev is not None:
            ev.set()  # 有新句子，尽快安排一轮纠错

        # 交给后台用离线模型重识别同一句音频（两遍识别）
        if cap.audio is not None and len(cap.audio) > 8000:  # 至少 0.5 秒
            self.refiner.submit(
                RefineJob(track=cap.track, audio=cap.audio, raw_text=cap.text,
                          start_ms=cap.start_ms, end_ms=cap.end_ms)
            )

    def _on_live_job(self, job: RefineJob) -> None:
        """句中滑窗预校正：结果只更新临时字幕，不进转写记录（定稿时还会再精修一次）。"""
        if self.refiner.running:
            self.refiner.submit(job)

    def _on_refined(self, job: RefineJob, text: str) -> None:
        polished = self.post.process(text, punctuate=not has_punctuation(text))
        if not polished:
            return
        if job.live:
            # 滑窗结果：比流式长才值得替换（离线模型常把没说完的部分补全）
            self.live_refined.emit(job.track, job.start_ms, polished)
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

    def _flush_transcript(self) -> None:
        if self._transcript is None:
            return
        try:
            self._transcript.seek(0)
            self._transcript.truncate()
            for rec in self._records:
                out = {k: v for k, v in rec.items()
                       if k not in ("refine_from", "refine_rejected", "llm_from", "llm")}
                self._transcript.write(json.dumps(out, ensure_ascii=False) + "\n")
            self._transcript.flush()
        except Exception:  # noqa: BLE001
            pass

    def _on_segment(self, seg) -> None:
        self._segment_total += 1
        self.segment_count.emit(self._segment_total)

    def _teardown(self) -> None:
        # 注意：transcript 在 _finish_session 里关闭（精修全部完成后才落盘收尾）
        self.pipeline = None
        self.recorder = None
        # 识别模型保留复用，避免二次加载
