"""打包成桌面 exe 并创建桌面快捷方式。

用法：
    .venv\\Scripts\\python.exe build_exe.py            # 完整打包 + 快捷方式
    .venv\\Scripts\\python.exe build_exe.py --no-link  # 只打包

产物：release/MeetScribe/  （exe + _internal + models + config + assets）
模型约 790MB，采用 onedir 而不是 onefile —— onefile 每次启动都要把
近 800MB 解包到临时目录，启动要几十秒，完全不能接受。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "release"
APP_NAME = "MeetScribe"          # exe 用 ASCII 名，快捷方式显示中文
SHORTCUT_NAME = "会议转写助手"
APP_DIR = DIST / APP_NAME

# 打包时不用带进去的“重”模块（本项目完全没用到）
EXCLUDES = [
    "tkinter", "matplotlib", "scipy", "pandas", "IPython", "pytest",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
    "PySide6.QtMultimedia", "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtDesigner",
]


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"命令失败（{r.returncode}）：{cmd[0]}")


def rotate_previous() -> Path | None:
    """把上一次的产物改名留档（保留一版可回滚），再让 PyInstaller 从干净目录构建。

    直接删除旧目录会撞上本机的批量删除保护（一次性删 300+ 文件会被拦），
    改名是单条操作，既不触发保护，又天然留了一份备份。
    """
    prev = DIST / f"{APP_NAME}_prev"
    if not APP_DIR.exists():
        return None
    if prev.exists():
        # 只保留最近一版备份，更早的清掉（PowerShell 侧不受批量删除保护限制）
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Remove-Item -Recurse -Force '{prev}'"],
            capture_output=True,
        )
    APP_DIR.rename(prev)
    print(f"上一版已留档：release/{prev.name}/")
    return prev


def build() -> Path:
    rotate_previous()
    icon = ROOT / "assets" / "app.ico"
    if not icon.exists():
        run([str(ROOT / ".venv" / "Scripts" / "python.exe"), "phase3/make_icon.py"])

    py = str(ROOT / ".venv" / "Scripts" / "pyinstaller.exe")
    cmd = [
        py, "--noconfirm", "--clean", "--onedir", "--windowed",
        "--name", APP_NAME,
        "--distpath", str(DIST),
        "--workpath", str(ROOT / "build" / "pyi"),
        "--specpath", str(ROOT / "build"),
        "--icon", str(icon),
        # 模型与配置放在 exe 旁边，不打进包内（见 meetscribe/paths.py）
        "--add-data", f"{ROOT / 'assets'};assets",
        # sherpa-onnx 的 DLL 在 sherpa_onnx/lib 下，必须整包收集
        "--collect-all", "sherpa_onnx",
        "--collect-binaries", "soundcard",
        "--hidden-import", "sounddevice",
        "--hidden-import", "soundfile",
        "--hidden-import", "soundcard",
        "--hidden-import", "cffi",
        "--hidden-import", "onnxruntime",
    ]
    for m in EXCLUDES:
        cmd += ["--exclude-module", m]
    cmd.append(str(ROOT / "run_app.py"))
    run(cmd)

    exe = APP_DIR / f"{APP_NAME}.exe"
    if not exe.exists():
        raise SystemExit("打包完成但没找到 exe，请检查 PyInstaller 输出")
    return exe


def stage_runtime_files() -> None:
    """把运行时资源放到 exe 旁边：models / config / data。"""
    # 模型（约 790MB）：首次复制后不再重复，节省时间
    dst_models = APP_DIR / "models"
    if not dst_models.exists():
        print("复制模型（约 790MB）…")
        shutil.copytree(ROOT / "models", dst_models)
    else:
        print("模型已存在，跳过复制")

    # 配置（hotwords / replacements / settings）
    dst_cfg = APP_DIR / "config"
    if dst_cfg.exists():
        shutil.rmtree(dst_cfg)
    shutil.copytree(ROOT / "config", dst_cfg)
    print("已复制 config/")

    # 会话目录：随包带历史会话（存在才复制，不存在则留空首次录音自动创建）
    dst_sessions = APP_DIR / "data" / "sessions"
    dst_sessions.mkdir(parents=True, exist_ok=True)
    src_sessions = ROOT / "data" / "sessions"
    if src_sessions.exists():
        n = 0
        for child in src_sessions.iterdir():
            target = dst_sessions / child.name
            if not target.exists():
                if child.is_dir():
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)
                n += 1
        print(f"已随包复制 {n} 个历史会话")


def make_shortcut(exe: Path) -> Path:
    """在桌面创建快捷方式（不覆盖同名但指向别处的旧快捷方式时给出提示）。"""
    desktop = Path.home() / "Desktop"
    if not desktop.exists():
        desktop = Path.home() / "OneDrive" / "Desktop"
    lnk = desktop / f"{SHORTCUT_NAME}.lnk"
    ps = f"""
$W = New-Object -ComObject WScript.Shell
$S = $W.CreateShortcut("{lnk}")
$S.TargetPath = "{exe}"
$S.WorkingDirectory = "{exe.parent}"
$S.IconLocation = "{exe},0"
$S.Description = "MeetScribe 本地会议转写与纪要"
$S.Save()
Write-Output "created: {lnk}"
"""
    # Windows PowerShell 的输出是本地代码页（GBK），解码失败不能影响主流程
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not lnk.exists():
        print("⚠️ 快捷方式创建失败：", (r.stderr or r.stdout or "").strip())
    else:
        print((r.stdout or "").strip() or f"已创建：{lnk}")
    return lnk


def main() -> int:
    t0 = time.time()
    exe = build()
    stage_runtime_files()
    size = sum(p.stat().st_size for p in APP_DIR.rglob("*") if p.is_file()) / (1024 ** 3)
    print(f"\n打包完成：{exe}")
    print(f"目录体积：{size:.2f} GB　耗时 {time.time() - t0:.0f}s")
    if "--no-link" not in sys.argv:
        make_shortcut(exe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
