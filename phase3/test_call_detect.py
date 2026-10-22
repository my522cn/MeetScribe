# -*- coding: utf-8 -*-
"""会议自动检测回归：重点覆盖"常驻软件（微信）通话"这个场景。

Bug：微信/钉钉/企微常驻后台，进程一直在，"进程从无到有"永远不成立 →
自动录音从不触发。修复后按"通话状态 false → true"触发。

验证：
1. 常驻微信一直在时，静默不触发
2. 微信开始通话（音频转活跃，连续 2 次）→ 触发一次
3. 通话持续期间不重复触发
4. 挂断后再次通话能再次触发
5. 只响 1 次轮询（提示音）不触发 —— 去抖生效
6. 没有 pycaw 时靠"通话窗口标题"也能触发
7. 启动时正在通话（prime）不会立刻触发
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.system import detect as D  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


def install_scan(state: dict[str, dict]) -> None:
    """用可控的假 scan 替换真实扫描：state = {应用名: {"audio":bool,"title":str}}。"""

    def fake_scan():
        out = []
        for name, st in state.items():
            out.append(D.MeetingApp(
                process=name, app_name=name, pid=1,
                audio_active=bool(st.get("audio")), title=str(st.get("title", "")),
                pids=[1],
            ))
        return out

    D.scan = fake_scan  # type: ignore[assignment]


def main() -> int:
    state: dict[str, dict] = {"微信": {"audio": False, "title": ""}}
    install_scan(state)

    # 1~2. 常驻但没通话 → 不触发；转通话 → 触发一次
    det = D.MeetingDetector(confirm_polls=2)
    det.prime()
    got = [det.poll() for _ in range(3)]
    check("常驻静默时不触发", all(g is None for g in got), f"{[g and g.app_name for g in got]}")

    state["微信"]["audio"] = True          # 开始通话
    r1 = det.poll()                        # 第 1 次活跃：去抖中
    r2 = det.poll()                        # 第 2 次活跃：确认通话
    check("首次活跃不立刻触发（去抖）", r1 is None, "")
    check("连续两次活跃才触发", r2 is not None and r2.app_name == "微信",
          f"{r2 and r2.app_name}")

    # 3. 通话中持续活跃：不重复触发
    again = [det.poll() for _ in range(4)]
    check("通话中不重复触发", all(g is None for g in again), "")

    # 4. 挂断后再打 → 能再触发
    state["微信"]["audio"] = False
    for _ in range(3):
        det.poll()
    state["微信"]["audio"] = True
    seq = [det.poll() for _ in range(3)]
    check("挂断后可再次触发", seq[1] is not None and seq[1].app_name == "微信",
          f"{[g and g.app_name for g in seq]}")

    # 5. 只活跃 1 次（提示音）→ 不触发
    det2 = D.MeetingDetector(confirm_polls=2)
    det2.prime()
    state["微信"]["audio"] = False
    for _ in range(3):
        det2.poll()
    state["微信"]["audio"] = True
    one = det2.poll()
    state["微信"]["audio"] = False
    two = det2.poll()
    check("提示音只响一次不触发", one is None and two is None, "")

    # 6. 没 pycaw 但有"通话"窗口标题 → 立即触发（强证据，无需去抖）
    state["微信"] = {"audio": False, "title": ""}
    det3 = D.MeetingDetector(confirm_polls=2)
    det3.prime()
    for _ in range(2):
        det3.poll()
    state["微信"]["title"] = "张三 语音通话"
    r3 = det3.poll()
    check("通话窗口标题立即触发", r3 is not None and "通话" in r3.title, f"{r3 and r3.title}")

    # 7. 启动时已在通话 → prime 后不触发（避免一开软件就录）
    state["微信"] = {"audio": False, "title": "和李四 视频通话"}
    det4 = D.MeetingDetector(confirm_polls=2)
    det4.prime()
    check("启动时已在通话不立刻触发", det4.poll() is None, "")

    # 8. 关键词设计：群名不能误判
    check("群名『项目会议群』不算通话", not D._is_call_title("项目会议群"), "")
    check("『会议中』算通话", D._is_call_title("产品周会 会议中"), "")
    check("英文 Meeting 标题算通话", D._is_call_title("Zoom Meeting"), "")

    print()
    for n, ok, d in checks:
        print(f"[{'OK ' if ok else 'FAIL'}] {n}　{d}")
    print()
    bad = [n for n, ok, _ in checks if not ok]
    if bad:
        print("❌ 失败项：", bad)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
