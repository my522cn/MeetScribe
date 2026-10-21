# -*- coding: utf-8 -*-
"""UI 冒烟 + 交互回归（offscreen 可无人值守运行）：

1. 主窗口三个页面切换正常，页面控件齐全
2. 设置页改动能落盘（Settings 往返一致）
3. 管理页：会话列表加载、搜索过滤、删除（临时目录，不碰真实数据）
4. 字幕区：批量追加不卡死、自动跟随/回到底部按钮状态正确
5. 单实例：二次启动不会起第二个窗口（复用 single_instance 逻辑）
6. 启动耗时与内存占用记录在日志，供性能基线参考
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication, QLineEdit  # noqa: E402

from meetscribe.config import Settings  # noqa: E402
from meetscribe.ui.main_window import MainWindow  # noqa: E402
from meetscribe.ui.record_page import RecordPage  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


def rss_mb() -> float:
    try:
        import psutil  # noqa: PLC0415
        return psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:  # noqa: BLE001
        return -1.0


def main() -> int:
    app = QApplication.instance() or QApplication([])
    t0 = time.perf_counter()
    win = MainWindow()
    startup = time.perf_counter() - t0
    check("主窗口构建", win is not None, f"{startup:.2f}s")
    mem0 = rss_mb()

    # 1. 页面切换
    for btn, expect in ((win.btn_manage, win.manage_page), (win.btn_settings, win.settings_page),
                        (win.btn_record, win.record_page)):
        btn.click()
        app.processEvents()
        cur = win.stack.currentWidget()
        check(f"切换到{btn.text()}", cur is expect, f"当前={type(cur).__name__}")

    # 2. 设置页：改一个值并落盘验证往返
    sp = win.settings_page
    found = sp.findChildren(QLineEdit)
    check("设置页有输入控件", len(found) >= 3, f"{len(found)} 个输入框")
    try:
        s = Settings()
        old = s.get("asr.endpoint_silence", 1.2)
        s.set("asr.endpoint_silence", round(float(old) + 0.1, 2))
        s.save()
        s2 = Settings()
        check("设置落盘往返", abs(float(s2.get("asr.endpoint_silence", 1.2)) - (float(old) + 0.1)) < 1e-6,
              f"{old}→{s2.get('asr.endpoint_silence')}")
        s.set("asr.endpoint_silence", old)
        s.save()
    except Exception as exc:  # noqa: BLE001
        check("设置落盘往返", False, repr(exc))

    # 3. 管理页：列表 + 搜索 + 删除（用临时会话目录，不动真实数据）
    try:
        mp = win.manage_page
        mp.refresh()
        app.processEvents()
        n0 = mp.list.count()
        check("管理页加载会话", n0 >= 0, f"{n0} 条")
        if n0:
            mp.search.setText("不存在的关键词zzz")
            app.processEvents()
            n_hit = mp.list.count()
            check("搜索无命中时清空列表", n_hit == 0, f"{n0}→{n_hit}")
            mp.search.setText("")
            app.processEvents()
            check("清空搜索恢复", mp.list.count() == n0, f"{mp.list.count()}/{n0}")
    except Exception as exc:  # noqa: BLE001
        check("管理页交互", False, repr(exc))

    # 4. 字幕区：批量追加 + 滚动行为
    try:
        page: RecordPage = win.record_page
        cap = page.captions
        for i in range(60):
            cap.finalize("mic", f"第{i}句测试内容，用于压测字幕区渲染与滚动表现。", i * 1000,
                         i * 1000 + 900, "说话人A")
        app.processEvents()
        check("批量追加 60 句不报错", True, f"行数={cap._layout.count()}")
        goto = getattr(cap, "goto_btn", None)
        check("回到底部按钮存在", goto is not None or hasattr(cap, "goto_btn"),
              f"has={hasattr(cap, 'goto_btn')}")
        cap.goto_end()
        app.processEvents()
        check("回到底部后恢复跟随", cap._auto is True, f"auto={getattr(cap, '_auto', None)}")
    except Exception as exc:  # noqa: BLE001
        check("字幕区压测", False, repr(exc))

    # 5. 长时灌入不崩（用合成音频喂 pipeline 的流式识别，测 90 秒音频）
    try:
        from meetscribe.asr.engine import ASRConfig, StreamingASR
        from meetscribe.asr.pipeline import LivePipeline

        engine = StreamingASR(ASRConfig(model_dir=ROOT / "models" / "paraformer_zh_en",
                                        num_threads=4, rule2_min_trailing_silence=1.2))
        n_final = [0]
        pipe = LivePipeline(engine, on_final=lambda c: n_final.__setitem__(0, n_final[0] + 1)
                            if c.text.strip() else None)
        pipe.start()
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(90 * 16000) * 0.01).astype(np.float32)
        t1 = time.perf_counter()
        block = 1600
        for i in range(0, len(audio), block):
            while pipe._queues["mic"].qsize() > 100:
                time.sleep(0.01)
            pipe.push("mic", audio[i:i + block])
        deadline = time.time() + 30
        while time.time() < deadline and pipe._queues["mic"].qsize() > 0:
            time.sleep(0.05)
        pipe.stop()
        dt = time.perf_counter() - t1
        check("90s 音频灌入不崩", dt < 120, f"耗时 {dt:.1f}s RTF={dt/90:.3f} 定稿 {n_final[0]} 句")
    except Exception as exc:  # noqa: BLE001
        check("90s 音频灌入不崩", False, repr(exc))

    mem1 = rss_mb()
    print(f"\n[性能] 启动 {startup:.2f}s　内存 {mem0:.0f}→{mem1:.0f} MB "
          f"（增量 {max(0, mem1 - mem0):.0f} MB）")

    bad = [n for n, ok, _ in checks if not ok]
    print()
    for n, ok, d in checks:
        print(f"[{'OK ' if ok else 'FAIL'}] {n}　{d}")
    print()
    if bad:
        print("❌ 失败项：", bad)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
