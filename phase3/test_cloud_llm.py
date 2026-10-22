# -*- coding: utf-8 -*-
"""云端模型（OpenAI 兼容）链路回归 —— 用本地假服务跑通，不需要真 API Key。

验证：
1. CloudClient 能按 OpenAI 协议请求、解析 choices[0].message.content
2. 推理模型返回的 <think>…</think> 会被剥掉，正文不被污染
3. 401/500/非 JSON 等异常能给出可读报错，不会崩
4. 无 Key 时 alive() 为 False、聊天直接报错（不会悄悄把内容发出去）
5. 预设切换能把官方端点/模型名填进设置，设置能存能读
6. 开了云端开关后，纪要走云端而不是本地（用假服务验证调用发生在云端）
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from meetscribe.llm import CloudClient  # noqa: E402
from meetscribe.llm.cloud import PRESETS, strip_think  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


REPLY = {
    "id": "x", "model": "mock-model",
    "choices": [{"message": {"role": "assistant",
                             "content": "<think>先想一下</think>## 一句话结论\n会议定了下周一上线。"}}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 30},
}


class Handler(BaseHTTPRequestHandler):
    mode = "ok"

    def log_message(self, *a) -> None:  # 静音
        pass

    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if Handler.mode == "http500":
            self._send(500, b"boom")
            return
        self._send(200, json.dumps({"data": [{"id": "mock-model"}]}).encode())

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        Handler.last_body = json.loads(raw.decode("utf-8")) if raw else {}
        Handler.last_auth = self.headers.get("Authorization", "")
        if Handler.mode == "http500":
            self._send(500, b"boom")
        elif Handler.mode == "notjson":
            self._send(200, b"<html>not json</html>")
        elif Handler.mode == "http401":
            self._send(401, b'{"error":{"message":"invalid api key"}}')
        else:
            self._send(200, json.dumps(REPLY).encode("utf-8"))


def main() -> int:
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}/v1"

    # 1. 正常调用
    c = CloudClient(base, "sk-test", "mock-model")
    r = c.chat("mock-model", "你是纪要助手", "会议内容：……")
    check("按 OpenAI 协议解析正文", "一句话结论" in r.text, r.text[:40])
    check("剥掉思考链", "<think>" not in r.text and "先想一下" not in r.text, r.text[:40])
    check("统计与耗时回填", r.prompt_tokens == 120 and r.eval_tokens == 30 and r.seconds > 0,
          f"{r.prompt_tokens}/{r.eval_tokens} {r.seconds:.2f}s")
    check("带上 Bearer 鉴权", Handler.last_auth == "Bearer sk-test", Handler.last_auth)
    check("请求体带 system/user", len(Handler.last_body.get("messages", [])) == 2, "")

    # 2. 异常分支
    Handler.mode = "http401"
    try:
        c.chat("mock-model", "s", "u")
        check("401 报错可读", False, "没抛异常")
    except RuntimeError as exc:
        check("401 报错可读", "401" in str(exc) and "invalid api key" in str(exc), str(exc)[:60])
    Handler.mode = "notjson"
    try:
        c.chat("mock-model", "s", "u")
        check("非 JSON 报错可读", False, "没抛异常")
    except RuntimeError as exc:
        check("非 JSON 报错可读", "不是 JSON" in str(exc), str(exc)[:50])
    Handler.mode = "http500"
    check("500 时 alive=False", not c.alive(), "")
    Handler.mode = "ok"
    check("恢复后 alive=True", c.alive(), "")

    # 3. 没 Key 不发声
    nokey = CloudClient(base, "", "mock-model")
    check("无 Key 时 alive=False", not nokey.alive(), "")
    try:
        nokey.chat("mock-model", "s", "u")
        check("无 Key 时拒绝发送", False, "竟然发出去了")
    except RuntimeError as exc:
        check("无 Key 时拒绝发送", "API Key" in str(exc), str(exc)[:40])

    # 4. 三家预设都在，端点格式正确
    need = {"deepseek": "api.deepseek.com", "kimi": "api.moonshot.cn",
            "hunyuan": "api.hunyuan.cloud.tencent.com"}
    ok = all(need[k] in PRESETS[k][0] for k in need) and all(PRESETS[k][1] for k in need)
    check("三家官方端点与默认模型齐全", ok, f"{list(PRESETS)}")

    # 5. 设置页：切预设自动填端点/模型、能存回配置
    from PySide6.QtWidgets import QApplication

    from meetscribe.config import Settings
    from meetscribe.ui.settings_page import SettingsPage

    app = QApplication.instance() or QApplication([])
    s = Settings()
    page = SettingsPage(s)
    page.cmb_cloud.setCurrentIndex(page.cmb_cloud.findData("kimi"))
    check("切到 Kimi 自动填端点",
          "moonshot" in page.edit_cloud_url.text() and page.edit_cloud_model.text().startswith("kimi"),
          f"{page.edit_cloud_url.text()} / {page.edit_cloud_model.text()}")
    page.cmb_cloud.setCurrentIndex(page.cmb_cloud.findData("custom"))
    keep = page.edit_cloud_model.text()
    check("自定义不覆盖已填内容", page.edit_cloud_model.text() == keep, keep)
    page.cmb_cloud.setCurrentIndex(page.cmb_cloud.findData("deepseek"))
    page.edit_cloud_key.setText("sk-abc")
    page.chk_cloud.setChecked(True)
    page._save()
    s2 = Settings()
    check("云端配置存回 settings.json",
          s2.get("llm.cloud_enabled", False) and s2.get("llm.cloud_api_key", "") == "sk-abc"
          and "deepseek" in s2.get("llm.cloud_base_url", ""),
          f"{s2.get('llm.cloud_base_url')} enabled={s2.get('llm.cloud_enabled')}")
    # 还原，别把测试用的 Key 留在配置里
    s2.set("llm.cloud_enabled", False)
    s2.set("llm.cloud_api_key", "")
    s2.save()

    srv.shutdown()
    print()
    for n, ok_, d in checks:
        print(f"[{'OK ' if ok_ else 'FAIL'}] {n}　{d}")
    print()
    bad = [n for n, ok_, _ in checks if not ok_]
    if bad:
        print("❌ 失败项：", bad)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
