"""LLM payload rendering and held-out transfer sweep (M10).

Eval-only: attack payloads are rendered by a local LLM (``qwen2.5:3b`` via
Ollama) instead of templates, to demonstrate the trained defender transfers
to LLM-authored attacks. Training stays scripted/templated (fast).
"""

from arena.llm.client import LLMClient, available
from arena.llm.payloads import render_payload, validate_payload
from arena.llm.sweep import run_sweep

__all__ = [
    "LLMClient",
    "available",
    "render_payload",
    "validate_payload",
    "run_sweep",
]
