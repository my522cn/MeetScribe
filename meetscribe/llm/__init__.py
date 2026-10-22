"""大模型相关能力：本地 Ollama + 可选云端（OpenAI 兼容）。"""
from .client import ChatResult, OllamaClient, OllamaError
from .cloud import PRESETS, CloudClient
from .correct import Corrector, FixRequest

__all__ = [
    "OllamaClient", "OllamaError", "ChatResult", "Corrector", "FixRequest",
    "CloudClient", "PRESETS",
]
