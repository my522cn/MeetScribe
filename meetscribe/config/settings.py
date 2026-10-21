"""应用配置：一个 settings.json 管住设备、识别、存储、模型四类参数。

设计：不做严格 schema 校验，缺字段就用默认值补齐（升级后老配置不会炸）；
路径统一存相对项目根的写法，换机器也能用。
"""
from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any

from ..paths import project_root

# 打包成 exe 后必须取 exe 所在目录（见 meetscribe/paths.py）
PROJECT_ROOT = project_root()
DEFAULT_PATH = PROJECT_ROOT / "config" / "settings.json"

DEFAULTS: dict[str, Any] = {
    "audio": {
        # sounddevice 设备索引；None = 用系统默认输入
        "mic_index": None,
        # 人类可读名，索引变化时按名字回退（设备插拔后索引会变）
        "mic_name": "",
        # soundcard 回环设备名；None = 用默认扬声器
        "loopback_name": None,
        "segment_minutes": 5,
        "format": "flac",            # flac（无损，30MB/h）| opus（12.8MB/h）
        "sample_rate": 16000,
    },
    "asr": {
        "two_pass": True,            # 句末离线重识别（精修）
        "refine_model": "auto",      # auto | paraformer | sense_voice | off
        "num_threads": 2,            # 流式识别线程数
        "refine_threads": 2,         # 精修线程数
        "endpoint_silence": 1.2,     # 句尾静音超过多少秒断句
        "live_refine": True,         # 长句滑窗预校正（说话中途就纠错）
        "live_refine_after": 6.0,    # 本句持续多少秒后开始滑窗校正
        "live_refine_every": 2.5,    # 滑窗间隔
    },
    "storage": {
        "session_root": "data/sessions",
    },
    "llm": {
        # 默认开启：只做"纠错"这一件事，服务没装/没开时自动跳过，不会影响录音
        "enabled": True,
        "base_url": "http://127.0.0.1:11434",
        # 纠错要快：qwen2.5:3b-instruct 不做思考链，8 句约 9 秒，实测 0 误改
        "fix_model": "qwen2.5:3b-instruct",
        # 纪要要质量：7b 能压住长会议，CPU 上一份纪要 1~2 分钟，后台跑不抢实时资源
        "summary_model": "qwen2.5:7b-instruct",
        "timeout": 90,
    },
    "ui": {
        "float_margin": 24,
        "auto_record": True,
    },
}


def _merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Settings:
    """点号路径读写：s.get("audio.segment_minutes")。"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_PATH
        self._lock = threading.RLock()
        self.data: dict[str, Any] = copy.deepcopy(DEFAULTS)
        self.load()

    # ---------- 读写 ----------
    def load(self) -> None:
        with self._lock:
            if self.path.exists():
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                    self.data = _merge(DEFAULTS, raw)
                except Exception:  # noqa: BLE001
                    self.data = copy.deepcopy(DEFAULTS)

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            cur: Any = self.data
            for part in key.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return default
                cur = cur[part]
            return cur

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            parts = key.split(".")
            cur = self.data
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = value

    # ---------- 便捷属性 ----------
    @property
    def session_root(self) -> Path:
        p = Path(self.get("storage.session_root", "data/sessions"))
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def segment_seconds(self) -> int:
        return max(1, int(self.get("audio.segment_minutes", 5))) * 60

    @property
    def refine_model(self) -> str:
        return str(self.get("asr.refine_model", "auto"))

    @property
    def two_pass(self) -> bool:
        return bool(self.get("asr.two_pass", True)) and self.refine_model != "off"
