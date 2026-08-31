"""Slow integration tests for arena/llm/ (M10).

Requires a running Ollama daemon with qwen2.5:3b pulled.
Skipped automatically if Ollama is not available.

Run: python -m pytest tests/test_m10_integration.py -q -m slow
"""

from __future__ import annotations

import pytest

from arena.config import LLMConfig
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
        from arena.config import load_config

        # load_config("small.yaml"), not ArenaConfig() -- the bare default is
        # named "unnamed". Asserting "small" against it is how this test shipped
        # red: it only ran when Ollama was present, and it never was.
        cfg = load_config("small.yaml")
        defenders = {"static_allowlist": StaticAllowList()}
        result = run_sweep(
            defenders, cfg,
            n_decision_adv=8, n_decision_benign=8, br_steps=512, n_eval=20, seed=0,
        )
        assert result.label == "small"
        assert len(result.templated_rows) > 0
        assert result.llm_available is True
        assert result.llm_rows and len(result.llm_rows) == len(result.templated_rows)

    def test_sweep_arms_actually_differ(self, client):
        """The whole point of M10. If the LLM arm cannot move a single one of
        Blue's decisions, the sweep is a formality and the two tables are the
        same measurement reported twice."""
        import numpy as np

        from arena.baselines.collect import collect_decisions
        from arena.config import load_config
        from arena.llm.sweep import _collect_llm_decisions

        cfg = load_config("small.yaml")
        scripted = collect_decisions(15, 15, config=cfg, seed=7)
        llm_rows, stats = _collect_llm_decisions(15, 15, config=cfg, llm=client, seed=7)

        assert stats.n_llm_planned > 0, "LLM planned nothing; sweep is vacuous"
        assert stats.n_differing_from_scripted > 0

        differing = sum(
            1 for a, b in zip(scripted, llm_rows)
            if not all(
                np.array_equal(np.asarray(a.blue_obs[k]), np.asarray(b.blue_obs[k]))
                for k in a.blue_obs
            )
        )
        assert differing > 0, (
            "LLM-planned episodes produced byte-identical Blue observations -- "
            "the sweep measures nothing"
        )
