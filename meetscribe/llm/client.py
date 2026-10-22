"""大模型接口：极简 Ollama HTTP 客户端，只依赖标准库。"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int = 0
    eval_tokens: int = 0
    seconds: float = 0.0
    thinking: str = ""


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ---------- 基础 ----------
    def _get(self, path: str) -> dict:
        req = urllib.request.Request(self.base_url + path, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=8) as r:  # noqa: S310
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(f"连不上 Ollama（{self.base_url}）：{exc.reason}") from exc
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(str(exc)) from exc

    def models(self) -> list[str]:
        data = self._get("/api/tags")
        return [m.get("name", "") for m in data.get("models", [])]

    @staticmethod
    def _size_b(name: str) -> float:
        """从模型名里解析参数规模（qwen2.5:7b-instruct → 7.0），解析不出返回 0。"""
        import re

        m = re.search(r":(\d+(?:\.\d+)?)b", name.lower())
        return float(m.group(1)) if m else 0.0

    def pick_model(self, wanted: str, fallback_hint: str = "qwen2.5",
                   prefer: str = "small") -> str:
        """配置里的模型可能已被卸载/改名：本机没有就回退到一个已装的同族模型。

        prefer="small"：回退到同族里最小的（纠错要快，1.5b 好过 14b）；
        prefer="large"：回退到同族里最大的（纪要要质量，14b 好过 1.5b）。
        返回空串表示本机一个可用模型都没有（调用方应关掉 LLM 功能）。
        """
        try:
            installed = [m for m in self.models() if m]
        except OllamaError:
            return ""
        if not installed:
            return ""
        if wanted in installed:
            return wanted
        same_family = [m for m in installed if fallback_hint in m]
        pool = same_family or installed
        key = self._size_b
        return min(pool, key=key) if prefer == "small" else max(pool, key=key)

    def alive(self) -> bool:
        try:
            self.models()
            return True
        except OllamaError:
            return False

    @staticmethod
    def locate_exe() -> str | None:
        import os
        import shutil

        found = shutil.which("ollama")
        if found:
            return found
        cand = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
        return cand if os.path.exists(cand) else None

    def ensure_server(self, wait: float = 12.0) -> bool:
        """服务没起来就自己拉一个（Ollama 桌面版开机自启，但被关掉时会连不上）。"""
        import subprocess
        import time

        if self.alive():
            return True
        exe = self.locate_exe()
        if not exe:
            return False
        try:
            flags = 0
            if hasattr(subprocess, "CREATE_NO_WINDOW"):
                flags = subprocess.CREATE_NO_WINDOW
            subprocess.Popen(
                [exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=flags,
            )
        except Exception:  # noqa: BLE001
            return False
        deadline = time.time() + wait
        while time.time() < deadline:
            if self.alive():
                return True
            time.sleep(0.6)
        return False

    def chat(
        self, model: str, system: str, user: str,
        temperature: float = 0.0, num_predict: int = 512, timeout: float | None = None,
        think: bool = False,
    ) -> ChatResult:
        import time

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # 思考链默认关闭；部分 qwen3 版本不认这个开关，此时思考会进 thinking 字段或混进正文，
            # 由调用方（Corrector/摘要）用 _clean 兜底。
            "think": think,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/api/chat", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:  # noqa: S310
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaError(f"请求失败：{exc.reason}") from exc
        except Exception as exc:  # noqa: BLE001
            raise OllamaError(str(exc)) from exc
        msg = data.get("message") or {}
        return ChatResult(
            text=str(msg.get("content", "")).strip(),
            model=str(data.get("model", model)),
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            eval_tokens=int(data.get("eval_count", 0)),
            seconds=time.perf_counter() - t0,
            thinking=str(msg.get("thinking", "") or ""),
        )

    def pull(self, model: str) -> None:
        """阻塞拉取模型（仅供命令行工具用）。"""
        payload = json.dumps({"model": model, "stream": True}).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/api/pull", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=3600) as r:  # noqa: S310
            for line in r:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line.decode("utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                if evt.get("error"):
                    raise OllamaError(str(evt["error"]))
                if evt.get("status") == "success":
                    return
