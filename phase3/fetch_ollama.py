"""多连接分段下载（单连接只有 170KB/s，16 路并发能到 2-3MB/s）。

用法：python phase3/fetch_ollama.py <url> <output> [connections]
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://github.com/ollama/ollama/releases/download/v0.40.6/OllamaSetup.exe"
OUT = Path("C:/Users/my522/Downloads/OllamaSetup.exe")
PARTS = OUT.parent / "ollama_parts"
N = 16


def content_length(url: str) -> int:
    r = subprocess.run(
        ["curl", "-sIL", "--max-time", "25", url],
        capture_output=True, text=True,
    )
    length = 0
    for line in r.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            length = int(line.split(":", 1)[1].strip())
    return length


def fetch(idx: int, start: int, end: int, url: str) -> tuple[int, int]:
    """按"缺多少补多少"的方式续下：每次都从已有长度处再要剩余区间，追加写入。

    注意不能同时用 -r 和 -C -（curl 会同时按两者算偏移，结果错位）。
    """
    PARTS.mkdir(parents=True, exist_ok=True)
    part = PARTS / f"p{idx:02d}"
    tmp = PARTS / f"p{idx:02d}.part"
    want = end - start + 1
    for _ in range(8):
        have = part.stat().st_size if part.exists() else 0
        if have >= want:
            break
        subprocess.run(
            ["curl", "-sL", "--fail", "--max-time", "900",
             "-r", f"{start + have}-{end}", "-o", str(tmp), url],
            capture_output=True,
        )
        if tmp.exists() and tmp.stat().st_size:
            with open(part, "ab") as f:
                f.write(tmp.read_bytes())
            tmp.unlink()
        time.sleep(1)
    return idx, part.stat().st_size if part.exists() else 0


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else URL
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else OUT
    n = int(sys.argv[3]) if len(sys.argv) > 3 else N
    total = content_length(url)
    print(f"总大小 {total} bytes，{n} 路下载")
    if total <= 0:
        raise SystemExit("拿不到文件大小")

    chunk = total // n
    ranges = []
    for i in range(n):
        s = i * chunk
        e = total - 1 if i == n - 1 else (i + 1) * chunk - 1
        ranges.append((i, s, e))

    t0 = time.time()
    done = {"n": 0}

    def run(item):
        r = fetch(*item, url)
        done["n"] += 1
        got = sum(p.stat().st_size for p in PARTS.glob("p*"))
        print(f"  完成 {done['n']}/{n}  累计 {got/1048576:.1f}MB / {total/1048576:.1f}MB"
              f"  {got/max(1,time.time()-t0)/1048576:.2f}MB/s", flush=True)
        return r

    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(run, ranges))

    bad = [i for i, size in results if size != ranges[i][2] - ranges[i][1] + 1]
    print("分片校验：", "全部完整" if not bad else f"不完整 {bad}")
    if bad:
        raise SystemExit("有分片未下完，请重跑")

    out.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(out, "wb") as f:
        for i in range(n):
            data = (PARTS / f"p{i:02d}").read_bytes()
            f.write(data)
            h.update(data)
    print(f"合并完成 {out.stat().st_size} bytes  用时 {time.time()-t0:.0f}s")
    print("sha256", h.hexdigest())


if __name__ == "__main__":
    main()
