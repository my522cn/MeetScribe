"""阶段一验收（二）：崩溃恢复验证。

启动一个子进程录音，录到一半直接强杀（模拟软件被意外关闭/断电），
然后重新读取 session.json，检查已落盘的分段是否完整可用。
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.audio.manifest import SessionManifest  # noqa: E402

CHILD = r"""
import sys, time
from pathlib import Path
sys.path.insert(0, r"{root}")
from meetscribe.audio.recorder import DualTrackRecorder
rec = DualTrackRecorder(Path(r"{root}") / "data" / "sessions", segment_seconds=3, title="崩溃恢复测试")
rec.start()
print("SESSION_DIR=" + str(rec.session_dir), flush=True)
while True:
    time.sleep(0.5)
"""


def main() -> None:
    code = CHILD.format(root=str(ROOT))
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=str(ROOT),
    )
    session_dir = None
    t0 = time.time()
    while time.time() - t0 < 10:
        line = proc.stdout.readline()
        if line.startswith("SESSION_DIR="):
            session_dir = Path(line.strip().split("=", 1)[1])
            break
    if session_dir is None:
        proc.kill()
        raise SystemExit("未能启动录音子进程")

    print(f"录制中会话目录: {session_dir}")
    print("录制 9 秒后强制杀死进程（模拟意外关闭）...")
    time.sleep(9)
    proc.kill()
    proc.wait(timeout=5)

    manifest_path = session_dir / "session.json"
    print(f"\n进程已强杀，检查清单: {manifest_path.exists()}")
    manifest = SessionManifest.load(manifest_path)
    print(f"会话状态: {manifest.status}")
    print(f"已落盘时长: {manifest.duration_ms/1000:.1f}s，分段数: {len(manifest.segments)}")

    import soundfile as sf

    ok = 0
    for seg in manifest.segments:
        path = session_dir / seg.file
        if not path.exists():
            print(f"  缺失: {seg.file}")
            continue
        data, sr = sf.read(path)
        ok += 1
        print(f"  {seg.track:<5} #{seg.index} {seg.start_ms/1000:>5.1f}s  {len(data)/sr:>4.1f}s  可解码")
    print(f"\n结论: {ok}/{len(manifest.segments)} 个分段完整可用，"
          f"{'可直接续录/继续转写' if ok == len(manifest.segments) and ok > 0 else '存在损坏需重建索引'}")


if __name__ == "__main__":
    main()
