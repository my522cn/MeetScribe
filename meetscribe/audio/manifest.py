"""会议会话清单：分段音频索引 + 崩溃恢复所需的元数据。"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Segment:
    track: str
    index: int
    file: str
    start_ms: int
    duration_ms: int
    samples: int

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_ms


@dataclass
class SessionManifest:
    session_id: str
    title: str
    source: str = "manual"
    mode: str = "online"
    sample_rate: int = 16000
    started_at: str = ""
    ended_at: str = ""
    status: str = "recording"
    tracks: dict = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return max((s.end_ms for s in self.segments), default=0)

    def add_segment(self, seg: Segment) -> None:
        self.segments.append(seg)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration_ms"] = self.duration_ms
        return data

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def load(cls, path: Path) -> "SessionManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        segs = [Segment(**s) for s in data.pop("segments", [])]
        data.pop("duration_ms", None)
        return cls(segments=segs, **data)


def new_session_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")
