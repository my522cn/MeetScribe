"""实时识别流水线：把两条音轨的音频块喂给识别器，输出"临时句 / 定稿句"。

设计：
- 单线程轮询两条轨（识别器内部不保证跨流并发安全），RTF 约 0.1，两轨合计仍远低于实时；
- 每条轨独立维护"当前句"：临时文本持续覆盖，断句后一次性定稿；
- 时间戳按该轨已消费的音频时长计算，天然与录音文件对齐，两轨共用一条时间轴。
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .engine import StreamingASR

TRACKS = ("mic", "loop")

# 静音门限：不用固定阈值（不同麦克风底噪差很多），而是跟踪本机噪声底线。
# 超过"噪声底线 × 1.6"才算说话；停止说话 1.5 秒后不再喂模型，避免在噪声上产生幻觉。
NOISE_FLOOR_MIN = 0.002
SPEECH_RATIO = 1.8
SILENCE_HANGOVER_MS = 2500


@dataclass
class Caption:
    track: str
    text: str
    start_ms: int
    end_ms: int
    final: bool
    audio: np.ndarray | None = None   # 定稿句对应的原始音频，供离线二次识别


class LivePipeline:
    def __init__(
        self,
        engine: StreamingASR,
        *,
        on_partial: Callable[[Caption], None] | None = None,
        on_final: Callable[[Caption], None] | None = None,
        on_overlap: Callable[[bool], None] | None = None,
        samplerate: int = 16000,
        max_queue: int = 200,
    ) -> None:
        self.engine = engine
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_overlap = on_overlap
        self.samplerate = samplerate
        self._queues = {t: queue.Queue(maxsize=max_queue) for t in TRACKS}
        self._consumed = {t: 0 for t in TRACKS}      # 已消费样本数（= 会议时间）
        self._sentence_start = {t: None for t in TRACKS}
        self._silence_ms = {t: 0.0 for t in TRACKS}
        self._noise = {t: 0.003 for t in TRACKS}
        self._last_speech_ms = {t: 0 for t in TRACKS}
        self._sentence_sample = {t: 0 for t in TRACKS}   # 本句起始样本位（用于切音频）
        # 滚动音频缓存：用于把定稿句的原始音频切出来做离线精修
        self._audio_buf: dict[str, deque] = {t: deque() for t in TRACKS}
        self._audio_samples = {t: 0 for t in TRACKS}
        self._max_cache_samples = 180 * samplerate
        self._spoken = {t: 0 for t in TRACKS}        # 本句已送入的有声块数
        self._last_final = {t: "" for t in TRACKS}
        self._running = False
        self._thread: threading.Thread | None = None
        self._overlap = False
        self.last_text = {t: "" for t in TRACKS}

    # ---------- 生命周期 ----------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="asr", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        # 收尾：仅对"确实还在说话"的轨补静音定稿，避免在纯静音上产生幻觉文本
        for track in TRACKS:
            if self._spoken[track] <= 0:
                continue
            tail = self.engine.flush(track, 1.0)
            if tail and tail.strip() and tail.strip() != self._last_final[track]:
                self._emit_final(track, tail.strip())

    def push(self, track: str, block: np.ndarray) -> None:
        try:
            self._queues[track].put_nowait(block)
        except queue.Full:
            pass

    # ---------- 消费线程 ----------
    def _loop(self) -> None:
        while self._running:
            got = False
            for track in TRACKS:
                try:
                    block = self._queues[track].get_nowait()
                except queue.Empty:
                    continue
                got = True
                self._consume(track, block)
            if not got:
                time.sleep(0.005)

    def _consume(self, track: str, block: np.ndarray) -> None:
        start_sample = self._consumed[track]
        self._consumed[track] += len(block)
        now_ms = int(self._consumed[track] / self.samplerate * 1000)
        self._remember_audio(track, start_sample, block)

        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2) + 1e-12))
        floor = self._noise[track]
        threshold = max(NOISE_FLOOR_MIN, floor * SPEECH_RATIO)
        speaking = rms > threshold

        if speaking:
            self._last_speech_ms[track] = now_ms
            if self._sentence_start[track] is None:
                # 用"第一块语音"的起点作为本句音频起点，比按文本出现时间倒推更准
                self._sentence_start[track] = max(0, int(start_sample / self.samplerate * 1000))
                self._sentence_sample[track] = max(0, start_sample - int(0.2 * self.samplerate))
            self._spoken[track] += 1
            # 噪声底线只向下慢速跟随，避免把持续人声算进底噪
            if rms < floor:
                self._noise[track] = 0.9 * floor + 0.1 * rms
        elif rms < floor * 1.2:
            self._noise[track] = 0.95 * floor + 0.05 * rms

        # 说完了并且静音超过 hangover：停止喂模型（端点检测已在 1.2s 静音时触发）
        if not speaking and (now_ms - self._last_speech_ms[track]) > SILENCE_HANGOVER_MS:
            return

        text = self.engine.feed(track, block)

        if text:
            self.last_text[track] = text
            if self.on_partial:
                self.on_partial(
                    Caption(track, text, self._sentence_start[track] or 0, now_ms, False)
                )

        self._check_overlap()

        if self.engine.recognizer.is_endpoint(self.engine.stream(track).stream):
            sentence_start_ms = self._sentence_start[track] or max(0, now_ms - 3000)
            audio = self._slice_audio(track, self._sentence_sample[track], self._consumed[track])
            final = self.engine.finish_sentence(track)
            # 有效语音不足 400ms 判定为噪声误触发，丢弃这句（避免"我国的。"这类幻觉）
            if self._spoken[track] >= 4 and final.strip():
                self._emit_final(track, final.strip(), audio, sentence_start_ms)
            self._sentence_start[track] = None
            self.last_text[track] = ""
            self._spoken[track] = 0

    def _remember_audio(self, track: str, start_sample: int, block: np.ndarray) -> None:
        self._audio_buf[track].append((start_sample, block))
        self._audio_samples[track] += len(block)
        while self._audio_samples[track] > self._max_cache_samples and self._audio_buf[track]:
            _, dropped = self._audio_buf[track].popleft()
            self._audio_samples[track] -= len(dropped)

    def _slice_audio(self, track: str, start_sample: int, end_sample: int) -> np.ndarray | None:
        picked = []
        for s, block in self._audio_buf[track]:
            if s + len(block) <= start_sample:
                continue
            if s >= end_sample:
                break
            lo = max(0, start_sample - s)
            hi = min(len(block), end_sample - s)
            if hi > lo:
                picked.append(block[lo:hi])
        if not picked:
            return None
        return np.concatenate(picked).astype(np.float32, copy=False)

    def _emit_final(self, track: str, text: str, audio: np.ndarray | None = None,
                    start_ms: int | None = None) -> None:
        if not text or text == self._last_final[track]:
            return
        self._last_final[track] = text
        end_ms = int(self._consumed[track] / self.samplerate * 1000)
        start = start_ms if start_ms is not None else (self._sentence_start[track] or max(0, end_ms - 3000))
        if self.on_final:
            self.on_final(Caption(track, text, start, end_ms, True, audio))

    def _check_overlap(self) -> None:
        """两条轨同时有未定稿文本 → 视为重叠发言，前端用双泳道高亮。"""
        both = all(self.last_text[t] for t in TRACKS)
        if both != self._overlap:
            self._overlap = both
            if self.on_overlap:
                self.on_overlap(both)
