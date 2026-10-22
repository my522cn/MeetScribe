# -*- coding: utf-8 -*-
"""微信/钉钉/企微通话检测现场诊断（给用户实测用）。

用法：python phase3/diag_call_detect.py
然后去打一个微信语音/视频电话，观察输出：
  - 「音频活跃」是否变 True（通话必然占用音频设备）
  - 「通话窗口」是否能枚举到标题
  - 「判定：通话中」何时出现（出现 = 修复后的自动录音会在这一刻触发）

每 2 秒刷新一次，Ctrl+C 退出。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.system import detect as D  # noqa: E402

INTERVAL = 2.0


def main() -> int:
    print(f"监控中（每 {INTERVAL:.0f} 秒一次）：{sorted(set(D.WATCHED.values()))}", flush=True)
    print("现在去打一个微信电话，观察下面的变化。Ctrl+C 退出。\n")
    try:
        while True:
            apps = D.scan()
            if not apps:
                print("  （没检测到任何被监控的软件在运行）")
            for a in apps:
                state = "通话中" if a.in_call else "空闲"
                print(f"  {a.app_name:6s} 进程x{len(a.pids):<2d} "
                      f"音频活跃={'是' if a.audio_active else '否'}  "
                      f"通话窗口={'「' + a.title + '」' if a.title else '无'}  "
                      f"判定：{state}")
            print("  " + "-" * 60, flush=True)
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\n结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
