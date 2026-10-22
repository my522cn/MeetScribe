# -*- coding: utf-8 -*-
"""云端大模型（OpenAI 兼容接口）：DeepSeek / Kimi / 混元 / 自定义端点。

为什么要有这一层：
- 本地 CPU 跑 14b 纪要要 2 分钟、长会议还得截断；云端 GPU 十几秒出结果，
  上下文还能到 128K~1M，长会议能整场丢进去，质量完全不同量级。
- 代价是会议内容离开本机，所以默认关闭，必须用户自己填 Key 并勾选启用。

chat() 的签名和 OllamaClient 一致（返回 ChatResult），所以纪要/纠错
两条链路不用改，直接把 client 换掉就行。
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from .client import ChatResult

# 三家官方端点都是 OpenAI 兼容，实测均可达（未带 Key 时返回 401）
PRESETS: dict[str, tuple[str, str, str]] = {
    # 名称: (base_url, 默认模型, 一句话说明)
    "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat",
                 "最便宜，约 0.02 元/场会议；长上下文"),
    "kimi": ("https://api.moonshot.cn/v1", "kimi-k2.6",
             "上下文最长（K3 到 1M），长会议最稳，价格最高"),
    "hunyuan": ("https://api.hunyuan.cloud.tencent.com/v1", "hunyuan-turbos",
                "腾讯混元，256K 上下文，价格居中"),
    "custom": ("", "", "自建/代理端点（OpenAI 兼容）"),
}

THINK_RE = re.compile(r"<think>.*?(?:</think>|<\|think\|>|$)", re.S)


def strip_think(text: str) -> str:
    """云端推理模型（R1、Kimi 思考模式）会把思考过程一起返回，正文前要剥掉。"""
    return THINK_RE.sub("", text or "").strip()


class CloudClient:
    """OpenAI 兼容 chat 调用。不引入 openai 依赖，直接用 urllib。"""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 120.0) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = (api_key or "").strip()
        self.model = (model or "").strip()
        self.timeout = float(timeout)

    # ---------- 探测 ----------
    def alive(self) -> bool:
        if not self.base_url or not self.api_key:
            return False
        try:
            self._get("/models")
            return True
        except Exception:  # noqa: BLE001
            return False

    def models(self) -> list[str]:
        data = self._get("/models")
        return [m.get("id", "") for m in data.get("data", []) if m.get("id")]

    def last_error(self) -> str:
        return self._last_error

    _last_error = ""

    # ---------- 核心 ----------
    def chat(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.2,
        num_predict: int = 2000,
        timeout: float | None = None,
        think: bool = False,
    ) -> ChatResult:
        """签名与 OllamaClient.chat 保持一致。think 参数云端用不上，保留占位。"""
        t0 = time.time()
        if not self.api_key:
            raise RuntimeError("未填写 API Key")
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max(64, int(num_predict)),
            "stream": False,
        }
        data = self._post("/chat/completions", payload, timeout=timeout or self.timeout)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"云端返回无法解析：{str(data)[:200]}") from exc
        usage = data.get("usage") or {}
        return ChatResult(
            text=strip_think(text),
            model=data.get("model", model or self.model),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            eval_tokens=int(usage.get("completion_tokens", 0) or 0),
            seconds=time.time() - t0,
        )

    # ---------- 内部 ----------
    def _request(self, path: str, payload: dict | None, timeout: float) -> dict:
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            self._last_error = f"HTTP {exc.code} {body}"
            raise RuntimeError(f"云端返回 {exc.code}：{body}") from exc
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            raise
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"云端返回不是 JSON：{raw[:200]}") from exc

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        return self._request(path, payload, timeout)

    def _get(self, path: str, timeout: float = 8.0) -> dict:
        return self._request(path, None, timeout)
