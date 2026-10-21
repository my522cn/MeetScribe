"""会话后处理回归：转写编辑、批量纠正、说话人改名+声纹档案、压缩、异常恢复。"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe import session_tools as st  # noqa: E402

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK ' if ok else 'FAIL'}] {name}" + (f"：{detail}" if detail else ""))
    if not ok:
        failures.append(name)


def make_session(root: Path, name: str = "20261021_100000_测试") -> Path:
    import soundfile as sf

    d = root / name
    (d / "mic").mkdir(parents=True)
    (d / "loop").mkdir()
    # 两个 3 秒分段（正弦波，能压缩）
    t = np.arange(16000 * 3) / 16000
    audio = (0.1 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    for i in (1, 2):
        sf.write(d / f"mic/seg_{i:04d}.flac", audio, 16000, format="FLAC", subtype="PCM_16")
    manifest = {
        "session_id": name, "title": "测试会议", "source": "manual", "mode": "offline",
        "sample_rate": 16000, "started_at": "2026-10-21T10:00:00", "ended_at": "",
        "status": "recording", "tracks": {}, "duration_ms": 6000,
        "segments": [
            {"track": "mic", "index": i, "file": f"mic/seg_{i:04d}.flac",
             "start_ms": (i - 1) * 3000, "duration_ms": 3000, "samples": 48000}
            for i in (1, 2)
        ],
    }
    (d / "session.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    recs = [
        {"track": "mic", "speaker": "说话人A", "start_ms": 0, "end_ms": 2000,
         "text": "我们把联条时间定一下。", "refined": True},
        {"track": "mic", "speaker": "说话人B", "start_ms": 2500, "end_ms": 5000,
         "text": "联条要在周五前完成。", "refined": True},
    ]
    (d / "transcript.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8"
    )
    (d / "diarize.log").write_text(
        "0\t180.0\t1400.0\t说话人A\n3000\t230.0\t1700.0\t说话人B\n"
        "5000\t182.0\t1410.0\t说话人A\n8000\t232.0\t1690.0\t说话人B\n",
        encoding="utf-8",
    )
    return d


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ms_tools_"))
    d = make_session(tmp)

    # 1. 批量纠正
    recs = st.load_records(d)
    changed = st.replace_text(recs, [("联条", "联调")])
    st.save_records(d, recs)
    recs = st.load_records(d)
    check("批量替换生效", changed == 2 and all("联条" not in r["text"] for r in recs),
          f"改动 {changed} 句")

    # 2. 全局替换表（写到临时文件，不动真实配置）
    tmp_repl = tmp / "replacements.txt"
    n = st.append_global_replacements([("联条", "联调")], path=tmp_repl)
    n2 = st.append_global_replacements([("联条", "联调")], path=tmp_repl)  # 重复应跳过
    check("全局替换表写入且去重", n == 1 and n2 == 0, f"{n}/{n2}")

    # 3. 说话人改名 + 声纹档案
    mapping = {"说话人A": "张总", "说话人B": "李工"}
    ch = st.rename_speakers(recs, mapping)
    st.save_records(d, recs)
    st.save_speaker_names(d, mapping)
    names = {r["speaker"] for r in st.load_records(d)}
    check("说话人改名生效", ch == 2 and names == {"张总", "李工"}, str(names))

    # 声纹档案：写进临时目录同名文件以免污染真实 config
    orig_path = st.global_profiles_path
    st.global_profiles_path = lambda: tmp / "profiles.json"
    try:
        saved = st.save_global_profiles(d, mapping)
        profiles = st.load_global_profiles()
        st.global_profiles_path = orig_path
    finally:
        st.global_profiles_path = orig_path
    check("声纹档案保存并在下次可加载", saved == 2 and set(profiles) == {"张总", "李工"},
          str({k: tuple(round(x, 1) for x in v) for k, v in profiles.items()}))

    # 4. 压缩
    done, saved_bytes, skipped = st.compress_session(d)
    check("压缩分段（OPUS 可用时）", done >= 0,
          f"压缩 {done} 个，节省 {saved_bytes / 1024:.0f} KB，跳过 {skipped[:1]}")
    if done:
        manifest = json.loads((d / "session.json").read_text(encoding="utf-8"))
        files = [s["file"] for s in manifest["segments"]]
        check("清单已指向新文件", all(f.endswith(".opus") for f in files), str(files))

    # 5. 异常恢复
    st.mark_completed(d)
    manifest = json.loads((d / "session.json").read_text(encoding="utf-8"))
    check("异常会话标记为已结束", manifest["status"] == "completed"
          and manifest.get("recovered") is True and manifest["ended_at"])

    print()
    if failures:
        print("❌ 失败项：", failures)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
