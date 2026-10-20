"""流式语音识别引擎封装（sherpa-onnx）。

支持两类模型：
- Paraformer（中英双语流式，推荐）：encoder.int8.onnx + decoder.int8.onnx + tokens.txt
- Zipformer Transducer：encoder/decoder/joiner onnx + tokens.txt

对外只暴露三个动作：feed(音频块) → 拿临时文本；is_endpoint() → 断句；reset() → 开始下一句。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sherpa_onnx


@dataclass
class ASRConfig:
    model_dir: Path
    num_threads: int = 2
    sample_rate: int = 16000
    enable_endpoint: bool = True
    # 断句参数：尾部静音超过 1.2s 判定一句话结束（会议场景偏保守，避免切碎）
    rule1_min_trailing_silence: float = 2.4
    rule2_min_trailing_silence: float = 1.2
    rule3_min_utterance_length: float = 20.0


@dataclass
class StreamState:
    stream: object
    text: str = ""
    last_partial: str = ""


class StreamingASR:
    def __init__(self, cfg: ASRConfig) -> None:
        self.cfg = cfg
        self.recognizer = self._build()
        self._streams: dict[str, StreamState] = {}

    # ---------- 构建 ----------
    def _build(self):
        d = Path(self.cfg.model_dir)
        tokens = d / "tokens.txt"
        if not tokens.exists():
            raise FileNotFoundError(f"缺少 tokens.txt: {d}")

        enc = self._pick(d, ["encoder.int8.onnx", "encoder.onnx"])
        para_dec = self._pick(d, ["decoder.int8.onnx", "decoder.onnx"])
        joiner = self._pick(d, ["joiner.int8.onnx", "joiner.onnx"], required=False)

        common = dict(
            tokens=str(tokens),
            num_threads=self.cfg.num_threads,
            sample_rate=self.cfg.sample_rate,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=self.cfg.enable_endpoint,
            rule1_min_trailing_silence=self.cfg.rule1_min_trailing_silence,
            rule2_min_trailing_silence=self.cfg.rule2_min_trailing_silence,
            rule3_min_utterance_length=self.cfg.rule3_min_utterance_length,
        )
        if joiner is not None:
            return sherpa_onnx.OnlineRecognizer.from_transducer(
                encoder=str(enc), decoder=str(para_dec), joiner=str(joiner), **common
            )
        return sherpa_onnx.OnlineRecognizer.from_paraformer(
            encoder=str(enc), decoder=str(para_dec), **common
        )

    @staticmethod
    def _pick(root: Path, names: list[str], required: bool = True) -> Path | None:
        for name in names:
            p = root / name
            if p.exists():
                return p
        if required:
            raise FileNotFoundError(f"{root} 下找不到 {names}")
        return None

    # ---------- 流管理 ----------
    def stream(self, key: str):
        if key not in self._streams:
            self._streams[key] = StreamState(stream=self.recognizer.create_stream())
        return self._streams[key]

    def feed(self, key: str, samples: np.ndarray) -> str:
        """送入音频块，返回当前这句的临时文本（可能被后续修正）。"""
        st = self.stream(key)
        st.stream.accept_waveform(self.cfg.sample_rate, samples.astype(np.float32))
        while self.recognizer.is_ready(st.stream):
            self.recognizer.decode_stream(st.stream)
        st.text = self.recognizer.get_result(st.stream)
        st.last_partial = st.text
        return st.text

    def is_endpoint(self, key: str, tail_silence: float = 0.0) -> bool:
        st = self.stream(key)
        if not self.recognizer.is_endpoint(st.stream):
            return False
        return True

    def finish_sentence(self, key: str) -> str:
        """取走当前句的最终文本并重置该流。"""
        st = self.stream(key)
        text = self.recognizer.get_result(st.stream).strip()
        self.recognizer.reset(st.stream)
        st.text = ""
        st.last_partial = ""
        return text

    def flush(self, key: str, seconds: float = 1.0) -> str:
        """补一段静音，把模型内部的尾音吐出来（否则句尾一两个字会被吞掉）。"""
        pad = np.zeros(int(self.cfg.sample_rate * seconds), dtype=np.float32)
        return self.feed(key, pad)


@dataclass
class FinalSegment:
    text: str
    start_ms: int
    end_ms: int
    extra: dict = field(default_factory=dict)
