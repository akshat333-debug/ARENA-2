"""M1 integration: config -> generator -> registry -> objective, end to end.

Unit tests check each piece; these check that the pieces agree with each other
over a realistic sample, and that the invariants the rest of ARENA will rely on
actually hold in bulk.
"""

from __future__ import annotations

from collections import Counter

from arena.config import load_config
from arena.scenarios import ALL_FAMILIES, EpisodeType, ScenarioGenerator
from arena.tools import Domain, SideEffect, domain_pool

N = 400


def _gen(name: str = "small.yaml") -> ScenarioGenerator:
    return ScenarioGenerator.from_config(load_config(name).scenario)


def test_bulk_generation_holds_every_structural_invariant():
    for sc in _gen().sample_batch(N):
        reg = sc.registry
        assert len(set(reg.names)) == len(reg), f"{sc.scenario_id}: duplicate tools"
        assert list(reg.names) == sorted(reg.names), "action indices not canonical"
        assert reg.sources and reg.propagators and reg.sinks
        assert sc.max_steps >= 1
        if sc.is_adversarial:
            obj = sc.objective
            assert obj is not None
            # Every objective must be satisfiable inside its own registry, and
            # must require more than one call to complete.
            assert set(obj.source_tools) <= set(reg.names)
            assert set(obj.sink_tools) <= set(reg.names)
            assert obj.min_chain_len > 1
            assert obj.min_chain_len <= sc.max_steps, "objective longer than the episode"
            for name in obj.source_tools:
                assert reg[reg.index(name)].is_source
            for name in obj.sink_tools:
                assert reg[reg.index(name)].is_sink
        else:
            assert sc.objective is None


def test_bulk_generation_covers_the_whole_space():
    batch = _gen().sample_batch(N)
    assert {s.domain for s in batch} == set(Domain), "not all domains sampled"
    families = {s.family for s in batch if s.is_adversarial}
    assert families == set(ALL_FAMILIES), f"missing families: {set(ALL_FAMILIES) - families}"


def test_adversarial_ratio_is_respected_in_aggregate():
    cfg = load_config("small.yaml").scenario
    counts = Counter(s.episode_type for s in _gen().sample_batch(N))
    observed = counts[EpisodeType.ADVERSARIAL] / N
    assert abs(observed - cfg.adversarial_ratio) < 0.08, f"ratio drift: {observed}"


def test_pipeline_is_reproducible_from_config():
    a = [s.model_dump() for s in _gen().sample_batch(50)]
    b = [s.model_dump() for s in _gen().sample_batch(50)]
    assert a == b


def test_both_shipped_configs_generate():
    for name in ("small.yaml", "paper.yaml"):
        batch = _gen(name).sample_batch(40)
        assert len(batch) == 40
        assert all(len(s.registry) >= 3 for s in batch)


def test_every_registry_can_host_a_chained_exfiltration():
    """The invariant the project actually depends on: sensitive source -> some
    propagator -> external sink. Taint propagates through TRANSFORM *and* EXEC, so
    a `run_script` chain is as valid an exfiltration path as a `summarize` one.
    If any registry lacked this shape, that episode would be unwinnable for Red by
    construction and would silently pollute the exploitability metric.
    """
    for sc in _gen().sample_batch(N):
        reg = sc.registry
        assert any(t.side_effect is SideEffect.READ_SENSITIVE for t in reg)
        assert reg.propagators, f"{sc.scenario_id} has no taint propagator"
        assert any(t.side_effect is SideEffect.EXTERNAL_SINK for t in reg)


def test_canonical_transform_chain_is_the_common_case():
    """The proposal's worked example (project.md S5) is specifically
    read sensitive -> summarize/transform -> external sink. It is not required in
    every registry (EXEC chains are equally valid), but it must be the dominant
    shape, or the headline scenario would be rare in training.
    """
    hosts = sum(
        any(t.side_effect is SideEffect.TRANSFORM for t in sc.registry)
        for sc in _gen().sample_batch(N)
    )
    assert hosts / N > 0.75, f"canonical transform chain too rare: {hosts}/{N}"


def test_canonical_chain_is_reachable_in_every_domain():
    """Every domain must be able to produce the literal worked example."""
    for domain in Domain:
        gen = ScenarioGenerator(seed=0, domains=(domain,), n_tools_max=len(domain_pool(domain)))
        assert any(
            any(t.side_effect is SideEffect.READ_SENSITIVE for t in sc.registry)
            and any(t.side_effect is SideEffect.TRANSFORM for t in sc.registry)
            and any(t.side_effect is SideEffect.EXTERNAL_SINK for t in sc.registry)
            for sc in gen.sample_batch(20)
        ), f"{domain.value} never produced the canonical chain"


def test_every_family_generates_across_every_domain():
    """Eval sweeps slice by (family, domain); no cell may be empty."""
    for domain in Domain:
        gen = ScenarioGenerator(seed=0, domains=(domain,))
        for family in ALL_FAMILIES:
            sc = gen.sample(force_family=family)
            assert sc.domain is domain and sc.family is family
            assert sc.objective is not None
