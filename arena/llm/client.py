"""Cached Ollama client for LLM payload rendering.

Disk cache keyed by ``(model, prompt, options_hash)`` so a sweep is
re-runnable offline. Falls back to templates if Ollama is absent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from arena.config import LLMConfig

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def available(cfg: LLMConfig | None = None) -> bool:
    """Is the Ollama daemon up and the configured model pulled?"""
    cfg = cfg or LLMConfig()
    try:
        import ollama as _ollama

        models = _ollama.list()
        names = [m.model for m in models.models]
        return any(cfg.model in n for n in names)
    except Exception:
        return False


def _cache_path(cfg: LLMConfig, key: str) -> Path:
    d = REPO_ROOT / cfg.cache_dir
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _hash_key(model: str, prompt: str, options: dict | None) -> str:
    raw = json.dumps({"model": model, "prompt": prompt, "options": options or {}}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class LLMClient:
    """Thin wrapper over the ``ollama`` python client with disk caching.

    >>> client = LLMClient(cfg)
    >>> client.available()          # False if daemon is down
    >>> text = client.generate("hello")  # cached on disk
    """

    def __init__(self, cfg: LLMConfig | None = None) -> None:
        self.cfg = cfg or LLMConfig()
        self._client = None
        self._available: bool | None = None  # lazy

    def _ensure_client(self):
        if self._client is None:
            try:
                import ollama as _ollama

                self._client = _ollama
            except ImportError:
                self._client = False
        return self._client

    def is_available(self) -> bool:
        """Check once and cache the result."""
        if self._available is None:
            client = self._ensure_client()
            if client is False or client is None:
                self._available = False
            else:
                self._available = available(self.cfg)
        return self._available

    def generate(self, prompt: str, *, options: dict | None = None) -> str | None:
        """Generate text. Returns cached response if available, or ``None``
        if Ollama is absent / model is not pulled / request fails."""
        key = _hash_key(self.cfg.model, prompt, options)
        cp = _cache_path(self.cfg, key)

        # Cache hit
        if cp.exists():
            try:
                data = json.loads(cp.read_text())
                if data.get("response"):
                    return data["response"]
            except (json.JSONDecodeError, KeyError):
                pass  # corrupt entry -> treat as miss

        # Cache miss
        client = self._ensure_client()
        if client is False or client is None:
            return None

        try:
            opts = {"temperature": self.cfg.temperature}
            if options:
                opts.update(options)
            resp = client.generate(
                model=self.cfg.model,
                prompt=prompt,
                options=opts,
            )
            text = resp.get("response", "")
            if text:
                cp.write_text(json.dumps({"prompt": prompt, "response": text}))
            return text or None
        except Exception:
            return None
