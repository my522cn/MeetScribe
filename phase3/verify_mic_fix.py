"""修复验证：走完整录音器（双轨分段落盘），再分别识别两轨，确认麦克风轨恢复正常。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.audio.recorder import DualTrackRecorder  # noqa: E402
from meetscribe.asr.engine import ASRConfig, StreamingASR  # noqa: E402
from phase3.diag_mic import play_async  # noqa: E402


def asr_segments(session_dir: Path, track: str, engine: StreamingASR) -> str:
    files = sorted((session_dir / track).glob("*.flac"))
    parts = []
    for f in files:
        audio, sr = sf.read(f, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        for i in range(0, len(audio), 1600):
            engine.feed(track, audio[i:i + 1600])
        engine.flush(track, 1.0)
        parts.append(engine.finish_sentence(track))
    return "".join(p for p in parts if p)


def main() -> None:
    rec = DualTrackRecorder(ROOT / "data" / "sessions", segment_seconds=30, title="麦克风修复验证")
    rec.start()
    play_async(ROOT / "phase2" / "test_zh_en_16k.wav")
    time.sleep(17)
    manifest = rec.stop()
    print(f"会话: {rec.session_dir}")
    for t in ("mic", "loop"):
        info = manifest.tracks.get(t, {})
        print(f"  {t}: {info}")

    engine = StreamingASR(ASRConfig(model_dir=ROOT / "models" / "paraformer_zh_en"))
    for track in ("mic", "loop"):
        text = asr_segments(rec.session_dir, track, engine)
        print(f"\n[{track}] {text}")


if __name__ == "__main__":
    main()
