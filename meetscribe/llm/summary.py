"""会议纪要生成：把转写文本交给本地模型，输出固定结构的 Markdown 纪要。

结构固定成 6 段，避免模型自由发挥：
  一句话结论 / 讨论主题 / 各方观点 / 争议与未决 / 结论 / 待办任务
长会议按"分块摘要 → 合并"两段式，避免超出上下文窗口。
"""
from __future__ import annotations

from .client import OllamaClient, OllamaError

SYSTEM = """你是会议纪要撰写员。根据给定的会议转写稿，输出 Markdown 纪要。
转写稿格式：每行 `说话人 时间 内容`，时间是会议内的偏移。
要求：
1. 只写转写稿里确实出现过的内容，不要补充、不要推测。信息不足的条目标注"（未提及）"。
2. 待办任务必须包含：事项、负责人、时间。负责人取该行开头的说话人；转写稿里没说时间就写"待定"。
3. 观点按说话人分组，一句话概括，不要逐字复述。
4. 输出结构固定为以下 6 个二级标题，不要加别的标题：
## 一句话结论
## 讨论主题
## 各方观点
## 争议与未决
## 结论
## 待办任务"""

CHUNK_CHARS = 4000


def _chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    lines = text.splitlines()
    out, buf, n = [], [], 0
    for line in lines:
        buf.append(line)
        n += len(line)
        if n >= size:
            out.append("\n".join(buf))
            buf, n = [], 0
    if buf:
        out.append("\n".join(buf))
    return out


def _partial(client: OllamaClient, model: str, idx: int, total: int, chunk: str, title: str) -> str:
    user = (
        f"会议《{title}》，这是第 {idx}/{total} 段转写稿。请按结构输出这一段的纪要要点，"
        f"不要写开头结尾套话。\n\n转写稿：\n{chunk}"
    )
    return client.chat(model, SYSTEM, user, temperature=0.2, num_predict=1500, timeout=600).text


def summarize(client: OllamaClient, model: str, title: str, transcript: str) -> str:
    parts = _chunks(transcript)
    if len(parts) <= 1:
        user = f"会议《{title}》完整转写稿：\n\n{transcript}"
        return client.chat(model, SYSTEM, user, temperature=0.2, num_predict=2000, timeout=600).text

    notes = []
    for i, chunk in enumerate(parts, 1):
        try:
            notes.append(_partial(client, model, i, len(parts), chunk, title))
        except OllamaError as exc:
            notes.append(f"（第 {i} 段摘要失败：{exc}）")
    merged = "\n\n".join(f"### 第 {i} 段要点\n{n}" for i, n in enumerate(notes, 1))
    user = (
        f"会议《{title}》较长，已分段摘要。请把下面的分段要点合并成一份完整纪要，"
        f"去重、按主题归并，保持 6 个固定标题结构。\n\n{merged}"
    )
    return client.chat(model, SYSTEM, user, temperature=0.2, num_predict=2500, timeout=900).text
