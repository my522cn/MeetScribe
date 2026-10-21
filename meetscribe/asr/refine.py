"""离线二次识别（精修）。

流式模型为了低延迟牺牲了准确率；每句话定稿后，把这一句的原始音频交给离线大模型
重识别一遍，用更准的结果替换字幕。全程在后台线程，不阻塞录制与字幕。

实测（同一段中英混说音频）：
    流式 Paraformer  CER 3.7%   RTF 0.17
    离线 Paraformer  CER 0.0%   RTF 0.13   ← 默认精修模型
    SenseVoice       CER 3.7%   RTF 0.15
离线 Paraformer 对中文专名与英文术语都更稳，因此作为默认。
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .offline import OfflineASR, get_offline


@dataclass
class RefineJob:
    track: str
    audio: np.ndarray
    raw_text: str
    start_ms: int
    end_ms: int
    live: bool = False   # True = 句中滑窗预校正（只更新临时字幕，不动定稿记录）


class RefineService:
    """后台离线重识别。kind: "paraformer"（默认，最准）或 "sense_voice"。"""

    def __init__(
        self,
        model_dir: Path,
        on_result: Callable[[RefineJob, str], None],
        num_threads: int = 2,
        kind: str = "paraformer",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.on_result = on_result
        self.num_threads = num_threads
        self.kind = kind
        self._queue: queue.Queue[RefineJob | None] = queue.Queue(maxsize=200)
        self._thread: threading.Thread | None = None
        self._running = False
        self._asr: OfflineASR | None = None
        self.done = 0
        self.failed = 0

    @property
    def available(self) -> bool:
        return (self.model_dir / "model.int8.onnx").exists() and (self.model_dir / "tokens.txt").exists()

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def last_rtf(self) -> float:
        return self._asr.last_rtf if self._asr else 0.0

    def start(self) -> None:
        if self._running or not self.available:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="refine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """等在途任务跑完再退出（最多 8s），避免丢最后一句的精修。"""
        self._running = False
        deadline = time.time() + 8
        while not self._queue.empty() and time.time() < deadline:
            time.sleep(0.05)
        if self._thread and self._thread.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=4)
        # 消费线程已退出：残留的哨兵/积压不会再被取走，必须清空，
        # 否则 pending 永远显示 1 句（界面会一直停在“还剩 1 句”）。
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def drain(self, timeout: float = 600.0, on_done=None) -> None:
        """录音结束后用：队列里的句子一句不丢，全部精修完再收尾。

        stop() 只等 8 秒，积压多时会把没跑的句子全部扔掉——这就是
        "录音结束后半天不精修"的原因。drain 起一个看护线程等队列清空
        （默认上限 10 分钟），然后才真正 stop 并回调 on_done。
        """
        def watcher() -> None:
            deadline = time.time() + timeout
            while self._running and not self._queue.empty() and time.time() < deadline:
                time.sleep(0.1)
            self.stop()
            if on_done is not None:
                try:
                    on_done()
                except Exception:  # noqa: BLE001
                    pass

        threading.Thread(target=watcher, name="refine-drain", daemon=True).start()

    def submit(self, job: RefineJob) -> None:
        if not self._running:
            return
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            pass  # 积压过多就放弃精修，优先保证不拖慢系统

    # ---------- 内部 ----------
    def _loop(self) -> None:
        try:
            # 与配置页"测试识别"共用同一个识别器实例，避免重复占用内存
            self._asr = get_offline(self.model_dir, self.kind, self.num_threads)
            self._asr.load()
        except Exception:  # noqa: BLE001
            self._running = False
            return
        self._asr.warmup()  # 首次推理要把权重读进内存，否则第一句延迟很高

        while self._running:
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                break
            try:
                text = self._asr.transcribe(job.audio)
            except Exception:  # noqa: BLE001
                self.failed += 1
                continue
            if text and text.strip():
                self.done += 1
                try:
                    self.on_result(job, text.strip())
                except Exception:  # noqa: BLE001
                    pass
