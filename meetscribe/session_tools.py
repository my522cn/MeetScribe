"""会话后处理：转写编辑、批量同音字纠正、说话人改名与声纹归档、录音压缩、异常退出恢复。

被会议管理页调用。所有函数都只读写该会话目录内的文件：
- transcript.jsonl  逐句转写（说话人/时间/文本）
- session.json      清单
- diarize.log       每句声纹特征（start_ms / f0 / centroid / label）
- speakers.json     本会话的说话人命名
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 转写读写
def load_records(session_dir: Path) -> list[dict]:
    path = Path(session_dir) / "transcript.jsonl"
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def save_records(session_dir: Path, records: list[dict]) -> None:
    """原子写：先写临时文件再替换，避免中途崩溃把转写写坏。"""
    path = Path(session_dir) / "transcript.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for rec in records:
            keep = {k: v for k, v in rec.items()
                    if k not in ("refine_from", "refine_rejected", "llm_from", "llm")}
            fh.write(json.dumps(keep, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------- 批量纠正
def replace_text(records: list[dict], pairs: list[tuple[str, str]]) -> int:
    """对全部句子做批量替换，返回被改动的句子数。pairs=[(错, 对), ...]"""
    changed = 0
    pairs = [(a, b) for a, b in pairs if a]
    for rec in records:
        old = rec.get("text", "")
        new = old
        for wrong, right in pairs:
            new = new.replace(wrong, right)
        if new != old:
            rec["text"] = new
            changed += 1
    return changed


def append_global_replacements(pairs: list[tuple[str, str]],
                              path: Path | None = None) -> int:
    """把纠正写入全局替换表（下次录音自动生效），返回新增条数。"""
    target = Path(path) if path else PROJECT_ROOT / "config" / "replacements.txt"
    existing = set()
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            if "=>" in line and not line.startswith("#"):
                existing.add(line.split("=>")[0].strip())
    added = 0
    lines = []
    for wrong, right in pairs:
        if wrong and wrong not in existing:
            lines.append(f"{wrong}=>{right}")
            added += 1
    if lines:
        with target.open("a", encoding="utf-8") as fh:
            fh.write("\n" + "\n".join(lines) + "\n")
    return added


# ---------------------------------------------------------------- 说话人
def speakers_in(records: list[dict]) -> list[str]:
    seen: list[str] = []
    for rec in records:
        sp = rec.get("speaker") or ""
        if sp and sp not in seen:
            seen.append(sp)
    return seen


def rename_speakers(records: list[dict], mapping: dict[str, str]) -> int:
    """把"说话人A"改成真实姓名，返回改动句数。"""
    changed = 0
    for rec in records:
        old = rec.get("speaker")
        if old in mapping and mapping[old] and mapping[old] != old:
            rec["speaker"] = mapping[old]
            changed += 1
    return changed


def load_diarize_log(session_dir: Path) -> list[tuple[int, float, float, str]]:
    path = Path(session_dir) / "diarize.log"
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            out.append((int(parts[0]), float(parts[1]), float(parts[2]), parts[3].strip()))
        except ValueError:
            continue
    return out


def speaker_features(session_dir: Path) -> dict[str, tuple[float, float, int]]:
    """按标签聚合声纹特征 → {标签: (f0均值, 质心均值, 有效句数)}（只统计特征有效的句子）。"""
    acc: dict[str, list[float]] = {}
    cnt: dict[str, int] = {}
    for _start, f0, cent, label in load_diarize_log(session_dir):
        if f0 <= 0 or cent <= 0:
            continue
        a = acc.setdefault(label, [0.0, 0.0])
        a[0] += f0
        a[1] += cent
        cnt[label] = cnt.get(label, 0) + 1
    return {
        label: (acc[label][0] / cnt[label], acc[label][1] / cnt[label], cnt[label])
        for label in acc
    }


def save_speaker_names(session_dir: Path, mapping: dict[str, str]) -> None:
    path = Path(session_dir) / "speakers.json"
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")


def load_speaker_names(session_dir: Path) -> dict[str, str]:
    path = Path(session_dir) / "speakers.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


# ---- 全局声纹档案（跨会话长期学习）----
def global_profiles_path() -> Path:
    return PROJECT_ROOT / "config" / "speakers.json"


def save_global_profiles(session_dir: Path, mapping: dict[str, str]) -> int:
    """把本次会话"已命名说话人"的声纹特征存进全局档案，供下次会议自动认人。返回保存人数。"""
    feats = speaker_features(session_dir)
    path = global_profiles_path()
    data: dict[str, dict] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {}
    saved = 0
    for label, name in mapping.items():
        name = (name or "").strip()
        if not name or label not in feats:
            continue
        f0, cent, n = feats[label]
        if n < 2:  # 样本太少不可靠
            continue
        prev = data.get(name) or {}
        pn = int(prev.get("samples", 0))
        # 滑动平均：老档案和新观测按样本量加权
        data[name] = {
            "f0": round((prev.get("f0", f0) * pn + f0 * n) / (pn + n), 2),
            "cent": round((prev.get("cent", cent) * pn + cent * n) / (pn + n), 1),
            "samples": pn + n,
        }
        saved += 1
    if saved:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return saved


def load_global_profiles() -> dict[str, tuple[float, float]]:
    path = global_profiles_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for name, v in data.items():
        try:
            out[name] = (float(v["f0"]), float(v["cent"]))
        except Exception:  # noqa: BLE001
            continue
    return out


# ---------------------------------------------------------------- 压缩
def compress_session(session_dir: Path, on_progress=None) -> tuple[int, int, list[str]]:
    """把 FLAC 分段转成 OPUS（体积约 1/2），更新清单并删除原文件。

    返回 (压缩文件数, 节省字节数, 跳过原因列表)。libsndfile 不支持 OPUS 时全部跳过。
    """
    import soundfile as sf

    session = Path(session_dir)
    manifest_path = session / "session.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    segs = data.get("segments", [])
    try:
        opus_ok = "OPUS" in sf.available_subtypes("OGG")
    except Exception:  # noqa: BLE001
        opus_ok = False
    if not opus_ok:
        return 0, 0, ["当前音频库不支持 OPUS 编码，无法压缩（可保留 FLAC）"]

    done, saved, skipped = 0, 0, []
    for i, seg in enumerate(segs, 1):
        rel = seg.get("file", "")
        if not rel.lower().endswith(".flac"):
            continue
        src = session / rel
        if not src.exists():
            continue
        if on_progress:
            on_progress(f"正在压缩 {i}/{len(segs)}…")
        try:
            audio, sr = sf.read(src, dtype="float32", always_2d=True)
            dst_rel = rel[:-5] + ".opus"
            dst = session / dst_rel
            sf.write(dst, audio, sr, format="OGG", subtype="OPUS")
            old_size, new_size = src.stat().st_size, dst.stat().st_size
            if new_size >= old_size:  # 没省空间就不换
                dst.unlink()
                continue
            saved += old_size - new_size
            src.unlink()
            seg["file"] = dst_rel
            done += 1
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{rel}: {exc}")
    if done:
        manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return done, saved, skipped


# ---------------------------------------------------------------- 异常恢复
def mark_completed(session_dir: Path) -> None:
    """把异常退出的会话标记为已结束（音频与转写保留，可在管理页正常查看）。"""
    from datetime import datetime

    path = Path(session_dir) / "session.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("ended_at"):
        data["ended_at"] = datetime.now().isoformat(timespec="seconds")
    data["status"] = "completed"
    data["recovered"] = True
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
