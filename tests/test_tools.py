"""M1 unit tests: tool specs and registry sampling."""

from __future__ import annotations

import numpy as np
import pytest

from arena.tools import (
    COMMON_TOOLS,
    DOMAIN_TOOLS,
    Domain,
    SideEffect,
    ToolRegistry,
    ToolSpec,
    domain_pool,
    sample_registry,
)


# --- valid input ---------------------------------------------------------


def test_every_domain_pool_can_host_a_full_chain():
    """Without a source, a propagator and a sink, exfiltration is unsatisfiable."""
    for domain in Domain:
        pool = domain_pool(domain)
        assert any(t.is_source for t in pool), f"{domain} has no taint source"
        assert any(t.propagates for t in pool), f"{domain} has no propagator"
        assert any(t.is_sink for t in pool), f"{domain} has no sink"


def test_tool_names_are_unique_within_a_pool():
    for domain in Domain:
        names = [t.name for t in domain_pool(domain)]
        assert len(names) == len(set(names)), f"duplicate tool names in {domain}"


@pytest.mark.parametrize("domain", list(Domain))
def test_sample_registry_respects_size_and_chain(domain: Domain):
    rng = np.random.default_rng(0)
    reg = sample_registry(rng, domain, 6)
    assert len(reg) == 6
    assert len(set(reg.names)) == 6, "registry contains duplicate tools"
    assert reg.sources and reg.propagators and reg.sinks


def test_sample_registry_is_deterministic_under_seed():
    a = sample_registry(np.random.default_rng(42), Domain.FINANCE, 8)
    b = sample_registry(np.random.default_rng(42), Domain.FINANCE, 8)
    assert a.names == b.names


def test_sample_registry_varies_across_seeds():
    seen = {sample_registry(np.random.default_rng(s), Domain.LEGAL, 6).names for s in range(20)}
    assert len(seen) > 1, "sampler ignores its rng"


def test_registry_index_round_trips():
    reg = sample_registry(np.random.default_rng(1), Domain.NEWS, 7)
    for i, tool in enumerate(reg):
        assert reg.index(tool.name) == i


def test_registry_ordering_is_canonical():
    """Action indices must depend on the tool set, not on draw order."""
    reg = sample_registry(np.random.default_rng(3), Domain.HEALTHCARE, 9)
    assert list(reg.names) == sorted(reg.names)


def test_side_effect_predicates_are_mutually_consistent():
    for tool in COMMON_TOOLS + tuple(t for ts in DOMAIN_TOOLS.values() for t in ts):
        assert sum([tool.is_source, tool.is_sink, tool.propagates]) <= 1


# --- invalid input -------------------------------------------------------


def test_sample_registry_rejects_oversized_request():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="cannot sample"):
        sample_registry(rng, Domain.NEWS, len(domain_pool(Domain.NEWS)) + 1)


def test_sample_registry_rejects_size_below_chain_requirement():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_tools >="):
        sample_registry(rng, Domain.NEWS, 2, require_chain=True)


def test_sample_registry_can_guarantee_an_untrusted_read():
    rng = np.random.default_rng(0)
    for _ in range(20):
        reg = sample_registry(rng, Domain.FINANCE, 6, require_untrusted_read=True)
        assert reg.untrusted_reads


def test_sample_registry_untrusted_read_needs_room():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_tools >="):
        sample_registry(rng, Domain.NEWS, 3, require_chain=True, require_untrusted_read=True)


def test_registry_index_raises_on_unknown_tool():
    reg = sample_registry(np.random.default_rng(0), Domain.NEWS, 6)
    with pytest.raises(KeyError):
        reg.index("no_such_tool")


def test_sensitivity_is_bounded():
    with pytest.raises(ValueError):
        ToolSpec(
            name="bad", category="x", side_effect=SideEffect.READ_BENIGN,
            sensitivity=5, description="out of range",
        )


def test_tool_spec_is_frozen():
    with pytest.raises(Exception):
        COMMON_TOOLS[0].name = "renamed"  # type: ignore[misc]


# --- edge cases ----------------------------------------------------------


def test_minimum_viable_registry_is_exactly_the_chain():
    reg = sample_registry(np.random.default_rng(0), Domain.FINANCE, 3)
    assert len(reg) == 3
    assert len(reg.sources) >= 1 and len(reg.propagators) >= 1 and len(reg.sinks) >= 1


def test_full_pool_registry_contains_everything():
    pool = domain_pool(Domain.LEGAL)
    reg = sample_registry(np.random.default_rng(0), Domain.LEGAL, len(pool))
    assert set(reg.names) == {t.name for t in pool}


def test_by_side_effect_filters():
    reg = ToolRegistry(tools=COMMON_TOOLS)
    sinks = reg.by_side_effect(SideEffect.EXTERNAL_SINK)
    assert sinks == reg.sinks
    assert all(t.is_sink for t in sinks)
    assert reg.by_side_effect() == ()
