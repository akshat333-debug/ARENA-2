"""Slow integration tests for arena/llm/ (M10).

Requires a running Ollama daemon with qwen2.5:3b pulled.
Skipped automatically if Ollama is not available.

Run: python -m pytest tests/test_m10_integration.py -q -m slow
"""

from __future__ import annotations

import pytest

from arena.config import ArenaConfig, LLMConfig
from arena.llm.client import LLMClient, available
from arena.llm.payloads import render_payload, validate_payload
from arena.llm.sweep import run_sweep
from arena.scenarios import AttackFamily, ObjectiveKind


# Mark all tests in this file as slow
pytestmark = pytest.mark.slow


def _ollama_available() -> bool:
    """Check if Ollama is running with the required model."""
    return available(LLMConfig())


@pytest.fixture(scope="module")
def client():
    """Shared LLM client, skips module if Ollama unavailable."""
    if not _ollama_available():
        pytest.skip("Ollama daemon not running or qwen2.5:3b not pulled")
    return LLMClient()


# ---------------------------------------------------------------------------
# Client tests
# ---------------------------------------------------------------------------

class TestLLMClientIntegration:
    def test_available(self, client):
        assert client.is_available() is True

    def test_generate_returns_text(self, client):
        result = client.generate("Say hello in exactly three words.")
        assert result is not None
        assert len(result.strip()) > 0

    def test_generate_caches_on_disk(self, client, tmp_path):
        cfg = LLMConfig(cache_dir=str(tmp_path / "cache"))
        c = LLMClient(cfg)

        r1 = c.generate("What is 2 + 2? Answer with just the number.")
        assert r1 is not None

        r2 = c.generate("What is 2 + 2? Answer with just the number.")
        assert r1 == r2  # cache hit returns same result


# ---------------------------------------------------------------------------
# Payload rendering
# ---------------------------------------------------------------------------

class TestPayloadRendering:
    @pytest.mark.parametrize("family", list(AttackFamily))
    def test_render_all_families(self, client, family):
        rp = render_payload(family, client)
        assert rp.family == family
        assert len(rp.payload.strip()) > 0
        # Either LLM succeeded or we got the fallback
        if rp.via_llm:
            assert validate_payload(rp.payload, family)
        else:
            from arena.llm.payloads import _FALLBACK_PAYLOADS
            assert rp.payload == _FALLBACK_PAYLOADS[family]

    def test_injected_control(self, client):
        rp = render_payload(
            AttackFamily.INDIRECT_PROMPT_INJECTION,
            client,
            objective_kind=ObjectiveKind.INJECTED_CONTROL,
        )
        assert len(rp.payload.strip()) > 0


# ---------------------------------------------------------------------------
# Sweep (abbreviated — just checks it runs without error)
# ---------------------------------------------------------------------------

class TestSweepIntegration:
    def test_sweep_runs(self, client):
        from arena.baselines import StaticAllowList

        cfg = ArenaConfig()
        defenders = {"static_allowlist": StaticAllowList()}
        result = run_sweep(defenders, cfg, seed=0)
        assert result.label == "small"
        # Sweep should have run the templated path at minimum
        assert len(result.templated_rows) > 0
