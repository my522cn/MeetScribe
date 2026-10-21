"""回放模块回归：分段读取衔接、虚拟时间轴推进、暂停/停止。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.audio.playback import SessionPlayer, _TrackReader  # noqa: E402
from meetscribe.config import Settings  # noqa: E402


def find_session() -> Path | None:
    root_dir = Settings().session_root
    best = None
    for d in sorted(root_dir.iterdir(), reverse=True):
        if d.is_dir() and (d / "session.json").exists():
            best = d
    return best


def main() -> int:
    session = find_session()
    if session is None:
        print("❌ 找不到任何会话")
        return 1
    print(f"测试会话：{session.name}")

    # ---------- 1. TrackReader 跨段衔接 ----------
    player = SessionPlayer(session)
    total_chunks = sum(len(r.chunks) for r in player._readers.values())
    print(f"[{'OK ' if total_chunks else 'FAIL'}] 分段清单加载：{total_chunks} 个分段，"
          f"时长 {player.duration_ms}ms")
    if not total_chunks:
        print("❌ 该会话没有音频分段，换一个会话再试")
        return 1

    mic = player._readers["mic"]
    failures = []
    if mic.chunks:
        start = mic.chunks[0][0] + 500  # 第一段中间
        buf = mic.read(start, 16000)    # 1 秒
        rms = float(np.sqrt(np.mean(buf * buf) + 1e-12))
        print(f"[{'OK ' if rms > 1e-4 else 'WARN'}] 第 1 秒音频 RMS={rms:.4f}（期望非静音）")

    # ---------- 2. 播放推进 ----------
    player.play(0)
    # 开播前要打开 WASAPI 设备（冷启动可达数百毫秒），等位置真正开始推进再计时，
    # 否则断言会随设备状态抖动
    deadline = time.time() + 5
    while time.time() < deadline and (not player.active or player.position_ms() < 50):
        time.sleep(0.05)
    time.sleep(1.0)
    pos = player.position_ms()
    ok = 700 <= pos <= player.duration_ms
    print(f"[{'OK ' if ok else 'FAIL'}] 开播 1s 后虚拟位置 = {pos}ms")
    if not ok:
        failures.append("播放推进")

    # ---------- 3. 暂停冻结 ----------
    player.pause()
    p1 = player.position_ms()
    time.sleep(0.6)
    p2 = player.position_ms()
    ok = p1 == p2 and player.paused
    print(f"[{'OK ' if ok else 'FAIL'}] 暂停后位置冻结：{p1} → {p2}")
    if not ok:
        failures.append("暂停冻结")
    player.resume()
    time.sleep(0.8)
    pos3 = player.position_ms()
    ok = pos3 > p2
    print(f"[{'OK ' if ok else 'FAIL'}] 继续后位置推进：{pos3}ms")
    if not ok:
        failures.append("继续推进")

    # ---------- 4. 停止 ----------
    player.stop()
    time.sleep(0.5)
    ok = not player.active
    print(f"[{'OK ' if ok else 'FAIL'}] stop 后线程退出：active={player.active}")
    if not ok:
        failures.append("停止")

    if not failures:
        print("全部通过 ✅")
        return 0
    print("❌ 失败项：", failures)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
