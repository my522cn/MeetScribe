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
5. 输出格式：每行一条，`编号|改好的句子`，条数与编号必须和输入完全一致。
6. 严禁输出分析、思路、解释、前言后语；第一行必须就是 `1|...`，不要复述规则，不要把多条句子改成一样的内容。"""

# 部分 qwen3 版本不听 think:false，把思维过程写进正文；用 <arg_key:6124c78e> 指令硬压一次
NO_THINK = "<arg_key:6124c78e> 不要输出思考过程。直接按格式输出结果。\n"


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


# 接受守卫：只放行"几乎一样、只动了极少字"的修正。
# 实测（26 句真实样本）：
#   qwen2.5:7b 相似度≥0.75 → 采纳 12 条、改好 4 改坏 5，净收益为负；
#   qwen2.5:7b 相似度≥0.97 → 采纳 5 条、改好 4 改坏 0，零伤害（只修正长句里的一两字）；
#   qwen2.5:1.5b 相似度≥0.97 → 改好 2 改坏 0，且 0.8s/条（7b 需 5.8s/条）；
#   按"改了几个字"判定 → 会放进更多坏修改，反而更差。
# 所以默认取最保守档 + 1.5b（唯一零伤害且最快），用户可在设置里调低"纠错严格度"换取更多修正。
FIX_MIN_SIMILAR = 0.97


def _accept(old: str, new: str, min_similar: float = FIX_MIN_SIMILAR) -> bool:
    a, b = _norm(old), _norm(new)
    if not b:
        return False
    if len(b) < 0.7 * len(a) or len(b) > 1.3 * len(a):
        return False
    return _similar(old, new) >= min_similar


class Corrector:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        hotwords: list[str] | None = None,
        batch: int = 3,
        think_off: bool = True,
        min_similar: float = FIX_MIN_SIMILAR,
    ) -> None:
        self.client = client
        self.model = model
        self.hotwords = [w for w in (hotwords or []) if w.strip()]
        self.batch = max(1, min(8, int(batch)))
        self.think_off = think_off
        self.min_similar = min_similar
        self.last_stats: dict[str, float] = {}
        self.rejected = 0   # 被"退化输出"守卫拦下的批次数，便于诊断

    def _glossary(self) -> str:
        if not self.hotwords:
            return ""
        return "\n术语表（正确写法）：" + "、".join(self.hotwords[:80])

    def _parse(self, text: str, items: list[FixRequest]) -> dict[int, str]:
        """解析模型输出：容忍思考前言、markdown 代码块；识别"退化"输出（全批相同）。"""
        got: dict[int, str] = {}
        for raw in text.splitlines():
            raw = raw.strip().strip("`").strip()
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
            if body:
                got.setdefault(i, body)

        # 退化守卫：多条输入却只给出一种结果（小模型常见崩塌）→ 整批作废
        if len(items) > 1 and len(got) > 1 and len({v for v in got.values()}) == 1:
            self.rejected += 1
            return {}
        return got

    def fix(self, items: list[FixRequest]) -> dict[tuple[str, int], str]:
        """批量纠错，返回 {key: 新文本}，只包含真正被修正的条目。

        实测：一次喂 8 句会让小模型崩塌（把整批改成一个样、甚至改坏），
        所以内部按 self.batch（默认 3）切片，逐批请求。
        """
        if not items:
            return {}
        out: dict[tuple[str, int], str] = {}
        seconds = prompt_tok = eval_tok = 0.0
        for start in range(0, len(items), self.batch):
            chunk = items[start:start + self.batch]
            lines = [f"{i+1}|{it.text}" for i, it in enumerate(chunk)]
            user = self._glossary() + "\n\n待校稿句子：\n" + "\n".join(lines)
            if self.think_off:
                user = NO_THINK + user
            try:
                res = self.client.chat(
                    self.model, SYSTEM, user,
                    temperature=0.0, num_predict=64 * len(chunk) + 128,
                )
            except OllamaError:
                continue
            seconds += res.seconds
            prompt_tok += res.prompt_tokens
            eval_tok += res.eval_tokens
            for i, body in self._parse(res.text, chunk).items():
                old = chunk[i].text
                if body != old and _accept(old, body, self.min_similar):
                    out[chunk[i].key] = body
        self.last_stats = {
            "sent": len(items),
            "fixed": len(out),
            "seconds": seconds,
            "prompt_tokens": prompt_tok,
            "eval_tokens": eval_tok,
        }
        return out
