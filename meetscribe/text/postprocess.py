"""文本后处理：同音纠错 + 标点恢复 + 中英文间距。

三层处理，都不依赖大模型：
1. 精确替换：词表里 "错词=>正确词"（含英文短语，如 "time mine=>timeline"）
2. 同音替换：对热词表里的词，在全文里找拼音完全相同的片段并替换（联条→联调）
3. 标点恢复：CT-Transformer 中英标点模型，缺模型时退化为句末补句号
"""
from __future__ import annotations

import re
from pathlib import Path

CJK = r"\u4e00-\u9fff"
_CJK_RE = re.compile(f"[{CJK}]")
_LATIN_CJK = re.compile(f"([{CJK}])([A-Za-z0-9])")
_CJK_LATIN = re.compile(f"([A-Za-z0-9])([{CJK}])")
_SENT_END = "。！？!?；;"


def _load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def apply_spacing(text: str) -> str:
    """中英文之间补一个空格，读起来才有间隔感。"""
    text = _LATIN_CJK.sub(r"\1 \2", text)
    text = _CJK_LATIN.sub(r"\1 \2", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def has_punctuation(text: str) -> bool:
    return any(ch in text for ch in "，。！？；、,.")


class Punctuator:
    def __init__(self, model_path: Path, num_threads: int = 1) -> None:
        self.available = False
        self._punct = None
        if not model_path.exists():
            return
        try:
            import sherpa_onnx

            cfg = sherpa_onnx.OfflinePunctuationConfig(
                model=sherpa_onnx.OfflinePunctuationModelConfig(
                    ct_transformer=str(model_path),
                    num_threads=num_threads,
                )
            )
            self._punct = sherpa_onnx.OfflinePunctuation(cfg)
            self.available = True
        except Exception:  # noqa: BLE001
            self.available = False

    def add(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text
        if self._punct is not None:
            try:
                out = self._punct.add_punctuation(text).strip()
                if out:
                    return out
            except Exception:  # noqa: BLE001
                pass
        if text[-1] not in _SENT_END:
            text += "。"
        return text


class TextPostProcessor:
    def __init__(self, hotwords_file: Path, replacements_file: Path, punct_model: Path) -> None:
        self.hotwords = [w for w in _load_lines(hotwords_file) if len(w) >= 2 and _CJK_RE.match(w)]
        self.replacements: dict[str, str] = {}
        for line in _load_lines(replacements_file):
            if "=>" in line:
                src, dst = line.split("=>", 1)
                self.replacements[src.strip().lower()] = dst.strip()
        self.punctuator = Punctuator(punct_model)
        self._py_cache: dict[str, tuple[frozenset[str], ...]] = {}
        try:
            from pypinyin import pinyin, Style

            def _sets(s: str) -> tuple[frozenset[str], ...]:
                # 多音字取全部读音，比较时按字位取交集（"调"含 diao/tiao，能匹配"条"）
                rows = pinyin(s, style=Style.NORMAL, heteronym=True, errors="ignore")
                return tuple(frozenset(r) for r in rows)

            self._pinyin = _sets
        except Exception:  # noqa: BLE001
            self._pinyin = None

    # ---------- 对外 ----------
    def process(self, text: str, punctuate: bool = True) -> str:
        if not text.strip():
            return ""
        out = self._exact(text)
        out = self._homophone(out)
        if punctuate:
            out = self.punctuator.add(out)
        elif out[-1] not in _SENT_END:
            out += "。"
        out = apply_spacing(out)
        return out

    # ---------- 内部 ----------
    def _exact(self, text: str) -> str:
        out = text
        for src, dst in self.replacements.items():
            if src not in out.lower():
                continue
            # 纯英文按整词匹配，避免 dem→demo 误伤 demand 之类
            if src.isascii():
                out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
            else:
                out = out.replace(src, dst)
        return out

    def _pinyin_of(self, s: str) -> tuple[frozenset[str], ...]:
        if s not in self._py_cache:
            self._py_cache[s] = self._pinyin(s) if self._pinyin else ()
        return self._py_cache[s]

    @staticmethod
    def _same_sound(a: tuple[frozenset[str], ...], b: tuple[frozenset[str], ...]) -> bool:
        if not a or not b or len(a) != len(b):
            return False
        return all(x & y for x, y in zip(a, b))

    def _homophone(self, text: str) -> str:
        """按拼音把热词的正确写法替换回文本中（处理同音错别字）。"""
        if not self.hotwords or self._pinyin is None:
            return text
        out = text
        for word in self.hotwords:
            if word in out:
                continue
            target = self._pinyin_of(word)
            if not target:
                continue
            n = len(word)
            i = 0
            chars = list(out)
            while i <= len(chars) - n:
                seg = "".join(chars[i:i + n])
                if all(_CJK_RE.match(c) for c in seg) and self._same_sound(self._pinyin_of(seg), target):
                    chars[i:i + n] = list(word)
                    i += n
                    continue
                i += 1
            out = "".join(chars)
        return out
