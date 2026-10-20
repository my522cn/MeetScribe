"""上下文纠错：用本地小模型把 ASR 的同音错别字改对。

为什么不直接让模型"重写"整句：会议转写最怕模型自由发挥。所以有三道闸：
1. 提示词严格限定"只改明显听错/同音错字"，禁止改写、增删、润色；
2. 输出必须逐行编号对齐，解析失败的句子原样保留；
3. 改动幅度超过阈值（字符级相似度 < 0.75 或长度变化 > 30%）就丢弃，保留原文。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .client import OllamaClient, OllamaError

SYSTEM = """你是中文会议转写的校稿员。输入是语音识别（ASR）刚转出的句子，可能有同音错别字、专名听错、漏字。
规则：
1. 只修正"明显是因为听错而写错的字/词"，例如"联条"→"联调"、"研调"→"联调"、"time mine"→"timeline"。
2. 禁止润色、改写、调整语序、增删标点以外的内容；句子意思不许变。
3. 不确定的一律不动。宁可漏改，也不要改错。
4. 术语表里出现过的写法必须照抄术语表。
5. 输出格式：每行一条，`编号|改好的句子`，条数与编号必须和输入完全一致，不要解释。"""


@dataclass
class FixRequest:
    key: tuple[str, int]     # (track, start_ms) 用于回填
    text: str


def _norm(text: str) -> str:
    drop = set("，。、？！：；“”‘’（）()[]{},.?!:;\"' \n\t-—…")
    return "".join(ch.lower() for ch in text if ch not in drop)


def _similar(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _accept(old: str, new: str) -> bool:
    a, b = _norm(old), _norm(new)
    if not b:
        return False
    if len(b) < 0.7 * len(a) or len(b) > 1.3 * len(a):
        return False
    return _similar(old, new) >= 0.75


class Corrector:
    def __init__(self, client: OllamaClient, model: str, hotwords: list[str] | None = None) -> None:
        self.client = client
        self.model = model
        self.hotwords = [w for w in (hotwords or []) if w.strip()]
        self.last_stats: dict[str, float] = {}

    def _glossary(self) -> str:
        if not self.hotwords:
            return ""
        return "\n术语表（正确写法）：" + "、".join(self.hotwords[:80])

    def fix(self, items: list[FixRequest]) -> dict[tuple[str, int], str]:
        """批量纠错，返回 {key: 新文本}，只包含真正被修正的条目。"""
        if not items:
            return {}
        lines = [f"{i+1}|{it.text}" for i, it in enumerate(items)]
        user = self._glossary() + "\n\n待校稿句子：\n" + "\n".join(lines)
        try:
            res = self.client.chat(
                self.model, SYSTEM, user,
                temperature=0.0, num_predict=64 * len(items) + 128,
            )
        except OllamaError:
            return {}
        out: dict[tuple[str, int], str] = {}
        for raw in res.text.splitlines():
            raw = raw.strip()
            if not raw or "|" not in raw:
                continue
            head, _, body = raw.partition("|")
            idx = re.sub(r"\D", "", head)
            if not idx:
                continue
            i = int(idx) - 1
            if not 0 <= i < len(items):
                continue
            body = body.strip().strip("`").strip()
            old = items[i].text
            if body and body != old and _accept(old, body):
                out[items[i].key] = body
        self.last_stats = {
            "sent": len(items),
            "fixed": len(out),
            "seconds": res.seconds,
            "prompt_tokens": res.prompt_tokens,
            "eval_tokens": res.eval_tokens,
        }
        return out
