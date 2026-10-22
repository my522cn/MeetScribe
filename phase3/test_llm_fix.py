# -*- coding: utf-8 -*-
"""LLM 纠错改动回归（不依赖 Ollama 是否可用，用假 client 验证逻辑）：

1. 批量切片：一次 7 句按 batch=3 切成 3 批请求
2. 退化守卫：模型把整批输出成同一句 → 整批作废，不回填
3. 严格守卫：改动幅度大的（相似度 < 下限）不采纳，只放行一两字修正
4. <arg_key:6124c78e> 前缀生效（think_off=True）
5. 解析健壮性：跳过思考前言、只认 `编号|句子`，不会把前言当结果
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from meetscribe.llm.correct import FIX_MIN_SIMILAR, Corrector, FixRequest, _accept  # noqa: E402

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


class FakeClient:
    """按脚本返回响应，记录收到的请求。"""

    def __init__(self, responder) -> None:
        self.responder = responder
        self.calls: list[str] = []

    def chat(self, model, system, user, temperature=0.0, num_predict=512, timeout=None, think=False):
        self.calls.append(user)

        class R:
            def __init__(self, text):
                self.text = text
                self.seconds = 0.1
                self.prompt_tokens = self.eval_tokens = 1
                self.thinking = ""

        return R(self.responder(user, len(self.calls)))


def main() -> int:
    # 1. 批量切片
    def resp_lines(user, n):
        lines = [l for l in user.splitlines() if "|" in l]
        return "\n".join(f"{i+1}|{l.split('|',1)[1]}" for i, l in enumerate(lines))

    items = [FixRequest(("mic", i * 1000), f"第{i}句原始转写内容") for i in range(7)]
    client = FakeClient(resp_lines)
    cor = Corrector(client, "fake", batch=3)
    cor.fix(items)
    check("按 batch=3 切片", len(client.calls) == 3, f"{len(client.calls)} 次请求")

    # 2. 退化守卫：整批输出同一个句子
    client2 = FakeClient(lambda user, n: "1|全都一样\n2|全都一样\n3|全都一样")
    cor2 = Corrector(client2, "fake", batch=3)
    out2 = cor2.fix(items[:3])
    check("退化输出整批作废", out2 == {} and cor2.rejected >= 1,
          f"回填 {len(out2)} 条，拦截 {cor2.rejected} 批")

    # 3. 严格守卫放行一两字修正、拦截大改
    # 长句里改一个同音字应放行；整句重写必须拦截
    long_old = "我们下一步要把这个模块的联条工作安排到下周进行并且要提前通知相关的同事们准备好材料"
    long_new = "我们下一步要把这个模块的联调工作安排到下周进行并且要提前通知相关的同事们准备好材料"
    check("放行长句同音小改", _accept(long_old, long_new) is True, f"下限 {FIX_MIN_SIMILAR}")
    check("拦截整句重写", _accept("明天下午开会", "会议改到下周三上午十点") is False, "")
    # 短句宁可漏改（实测放宽会放进更多坏修改）
    check("中短句保守不改（宁可漏改）", _accept("我们下一步要把这个模块的联条工作安排", 
                                      "我们下一步要把这个模块的联调工作安排") is False, "")

    # 4. <arg_key:6124c78e> 前缀
    client4 = FakeClient(resp_lines)
    Corrector(client4, "fake", batch=3, think_off=True).fix(items[:2])
    has = "<arg_key:6124c78e>" in client4.calls[0]
    client5 = FakeClient(resp_lines)
    Corrector(client5, "fake", batch=3, think_off=False).fix(items[:2])
    hasnt = "<arg_key:6124c78e>" not in client5.calls[0]
    check("think_off 加 <arg_key:6124c78e>", has and hasnt, f"开={has} 关={hasnt}")

    # 5. 解析健壮性：前面堆一段思考，后面才是结果
    def resp_think(user, n):
        lines = [l for l in user.splitlines() if "|" in l]
        body = "\n".join(f"{i+1}|{l.split('|',1)[1]}改" for i, l in enumerate(lines))
        return "首先，用户给了几句转写，我需要先检查有没有错别字。\n" + body

    client6 = FakeClient(resp_think)
    cor6 = Corrector(client6, "fake", batch=2, min_similar=0.75)
    out6 = cor6.fix([FixRequest(("mic", 0), "这是一段比较长的测试句子内容用于验证解析逻辑是否正确"),
                     FixRequest(("mic", 1), "这是另一段比较长的测试句子内容用于验证解析逻辑是否正确")])
    check("跳过思考前言只解析编号行", len(out6) == 2, f"回填 {len(out6)} 条")
    check("前言不会污染结果", all("首先" not in v for v in out6.values()), "")

    # 6. 配置模型已卸载 → 回退到本机已装的同类模型
    try:
        from meetscribe.llm.client import OllamaClient

        class FakeTags(OllamaClient):
            def models(self):
                return ["qwen2.5:7b-instruct", "qwen2.5:3b-instruct"]

        c = FakeTags()
        check("已卸载模型回退到同类(小)", c.pick_model("qwen3:4b") == "qwen2.5:3b-instruct",
              f"-> {c.pick_model('qwen3:4b')}")
        check("已卸载模型回退到同类(大)",
              c.pick_model("qwen3:4b", prefer="large") == "qwen2.5:7b-instruct",
              f"-> {c.pick_model('qwen3:4b', prefer='large')}")
        check("已装模型原样返回", c.pick_model("qwen2.5:3b-instruct") == "qwen2.5:3b-instruct", "")
        check("无同族回退到任意已装(小)",
              c.pick_model("llama3:8b", fallback_hint="llama") in
              ("qwen2.5:3b-instruct", "qwen2.5:7b-instruct"), "")
    except Exception as exc:  # noqa: BLE001
        check("模型回退", False, repr(exc))

    bad = [n for n, ok, _ in checks if not ok]
    print()
    for n, ok, d in checks:
        print(f"[{'OK ' if ok else 'FAIL'}] {n}　{d}")
    print()
    if bad:
        print("❌ 失败项：", bad)
        return 1
    print("全部通过 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
