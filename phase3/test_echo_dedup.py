# -*- coding: utf-8 -*-
"""回声去重 + 说话人标签上限回归：

1. 麦克风句子与回采句子时间重叠且文本相似 ≥0.6 → 判定为回声，丢弃
2. 文本不相似（同时间不同内容）→ 不误杀
3. 回采轨自身的句子不参与回声判定
4. speaker_max 设为 10 时，连立 10 个说话人不崩（标签 A..Z）
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from meetscribe.config import Settings  # noqa: E402
from meetscribe.ui.worker import CaptureWorker  # noqa: E402
from meetscribe.asr.diarize import EmbeddingSpeakerRegistry  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


def main() -> int:
    app = QApplication.instance() or QApplication([])
    tmp = Path(tempfile.mkdtemp(prefix="ms_echo_"))
    worker = CaptureWorker(ROOT / "models" / "paraformer_zh_en", tmp / "data", ROOT / "config")
    worker._records = [
        {"track": "loop", "speaker": "对方", "start_ms": 1000, "end_ms": 5000,
         "text": "塞嘴布要填下资产领用单哦", "refined": False, "raw": "", "mixed": 0.0},
    ]

    # 1. 时间重叠 + 文本高度相似 → 回声
    ok1 = worker._is_echo_of_loop("mic", 1200, 5200, "塞嘴布要填下资产领用单哦")
    check("同文同时段判为回声", ok1 is True, f"got={ok1}")

    # 2. 带转写噪声的回声副本（部分字错）仍应命中
    ok2 = worker._is_echo_of_loop("mic", 1500, 5500, "塞嘴布要填下子产领用单哦啊")
    check("带噪声副本判为回声", ok2 is True, f"got={ok2}")

    # 3. 时间重叠但内容完全不同 → 不是回声
    ok3 = worker._is_echo_of_loop("mic", 1200, 5200, "大家看一下下季度的预算方案")
    check("同段不同内容不误杀", ok3 is False, f"got={ok3}")

    # 4. 回采轨自己的句子不参与判定
    ok4 = worker._is_echo_of_loop("loop", 1200, 5200, "塞嘴布要填下资产领用单哦")
    check("回采轨不判回声", ok4 is False, f"got={ok4}")

    # 5. 时间不重叠的相同文本 → 不是回声（相隔久可能真是重复发言）
    worker._records.append(
        {"track": "loop", "speaker": "对方", "start_ms": 600000, "end_ms": 604000,
         "text": "塞嘴布要填下资产领用单哦", "refined": False, "raw": "", "mixed": 0.0})
    ok5 = worker._is_echo_of_loop("mic", 300000, 304000, "塞嘴布要填下资产领用单哦")
    check("时段不重叠不判回声", ok5 is False, f"got={ok5}")

    # 6. speaker_max=10：连立 10 个互不相似的说话人不崩（LABELS 需支持到 J）
    model = ROOT / "models" / "campplus_zh" / "campplus_zh.onnx"
    if model.exists():
        class FakeExt:
            """确定性伪嵌入：按段首幅值生成互不相似的单位向量（cos≈0）。"""

            def create_stream(self) -> "FakeExt":
                return self

            def accept_waveform(self, sr: int, seg) -> None:
                self._seg = seg

            def input_finished(self) -> None:
                pass

            def compute(self, stream):
                idx = int(abs(float(self._seg[0])) * 100000) % 192
                v = np.zeros(192, dtype=np.float32)
                v[idx] = 1.0
                return v

        reg = EmbeddingSpeakerRegistry(model, max_speakers=10)
        reg._ext = FakeExt()
        labs = []
        for i in range(10):
            seg = np.full(16000, 0.02 * (i + 1), dtype=np.float32)  # 第 i 个"说话人"（幅度过 RMS 门限）
            lab, _ = reg.assign_sentence(seg)
            labs.append(lab)
        check("10 个独立说话人可建立", len(set(labs)) == 10, f"唯一标签数={len(set(labs))}")
        check("第10个标签是说话人J", labs[-1] == "说话人J", f"lab={labs[-1]}")
    else:
        print("[SKIP] 无声纹模型，跳过说话人上限测试")

    print()
    bad = [n for n, ok, _ in checks if not ok]
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
