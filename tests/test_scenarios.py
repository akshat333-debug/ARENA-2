"""M1 unit tests: scenario generation and attack objectives."""

from __future__ import annotations

import pytest

from arena.scenarios import (
    ALL_FAMILIES,
    FAMILY_SPECS,
    TASK_TEMPLATES,
    AttackFamily,
    EpisodeType,
    ScenarioGenerator,
)
from arena.tools import Domain


# --- valid input ---------------------------------------------------------


def test_every_family_is_generable_and_well_formed():
    """All six TAMAS families must produce a satisfiable objective."""
    gen = ScenarioGenerator(seed=0)
    for family in ALL_FAMILIES:
        sc = gen.sample(force_family=family)
        assert sc.is_adversarial
        assert sc.objective is not None
        assert sc.objective.family is family
        assert sc.objective.source_tools and sc.objective.sink_tools
        # The objective must be satisfiable inside its own registry.
        assert set(sc.objective.source_tools) <= set(sc.registry.names)
        assert set(sc.objective.sink_tools) <= set(sc.registry.names)


def test_all_families_are_chained_variants():
    """The novelty is sequential attacks: no family may complete in one call."""
    for family, spec in FAMILY_SPECS.items():
        assert spec["min_chain_len"] > 1, f"{family} is single-shot, not a chained variant"


def test_family_specs_cover_every_family():
    assert set(FAMILY_SPECS) == set(ALL_FAMILIES)
    assert len(ALL_FAMILIES) == 6


def test_benign_scenarios_carry_no_objective():
    gen = ScenarioGenerator(seed=1)
    for sc in gen.sample_batch(30, force_type=EpisodeType.BENIGN):
        assert sc.episode_type is EpisodeType.BENIGN
        assert sc.objective is None
        assert sc.family is None


def test_adversarial_scenarios_always_carry_an_objective():
    gen = ScenarioGenerator(seed=2)
    for sc in gen.sample_batch(30, force_type=EpisodeType.ADVERSARIAL):
        assert sc.objective is not None


def test_generation_is_reproducible_under_seed():
    a = ScenarioGenerator(seed=7).sample_batch(15)
    b = ScenarioGenerator(seed=7).sample_batch(15)
    assert [s.model_dump() for s in a] == [s.model_dump() for s in b]


def test_reset_rewinds_the_stream():
    gen = ScenarioGenerator(seed=5)
    first = gen.sample_batch(5)
    gen.reset()
    assert [s.model_dump() for s in gen.sample_batch(5)] == [s.model_dump() for s in first]


def test_different_seeds_diverge():
    a = ScenarioGenerator(seed=1).sample_batch(10)
    b = ScenarioGenerator(seed=2).sample_batch(10)
    assert [s.scenario_id for s in a] != [s.scenario_id for s in b] or \
        [s.task for s in a] != [s.task for s in b]


def test_registry_always_supports_a_chain():
    gen = ScenarioGenerator(seed=3)
    for sc in gen.sample_batch(40):
        assert sc.registry.sources and sc.registry.propagators and sc.registry.sinks


def test_scenario_ids_are_unique_within_a_run():
    gen = ScenarioGenerator(seed=4)
    ids = [s.scenario_id for s in gen.sample_batch(100)]
    assert len(set(ids)) == len(ids)


def test_tasks_come_from_the_scenario_domain():
    gen = ScenarioGenerator(seed=6)
    for sc in gen.sample_batch(40):
        assert sc.task in TASK_TEMPLATES[sc.domain]


def test_indirect_injection_requires_an_untrusted_read():
    gen = ScenarioGenerator(seed=8)
    sc = gen.sample(force_family=AttackFamily.INDIRECT_PROMPT_INJECTION)
    assert sc.objective is not None
    assert sc.objective.requires_untrusted_read


# --- invalid input -------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"domains": ()}, "domains"),
        ({"families": ()}, "families"),
        ({"adversarial_ratio": 1.5}, "adversarial_ratio"),
        ({"adversarial_ratio": -0.1}, "adversarial_ratio"),
        ({"n_tools_min": 9, "n_tools_max": 4}, "n_tools_min"),
        ({"n_tools_min": 2}, "n_tools_min"),
    ],
)
def test_generator_rejects_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ScenarioGenerator(seed=0, **kwargs)


def test_scenario_is_frozen():
    sc = ScenarioGenerator(seed=0).sample()
    with pytest.raises(Exception):
        sc.task = "mutated"  # type: ignore[misc]


# --- edge cases ----------------------------------------------------------


def test_ratio_zero_yields_only_benign():
    gen = ScenarioGenerator(seed=0, adversarial_ratio=0.0)
    assert all(not s.is_adversarial for s in gen.sample_batch(50))


def test_ratio_one_yields_only_adversarial():
    gen = ScenarioGenerator(seed=0, adversarial_ratio=1.0)
    assert all(s.is_adversarial for s in gen.sample_batch(50))


def test_force_family_overrides_a_benign_ratio():
    """An eval sweep must be able to demand a family even at ratio 0."""
    gen = ScenarioGenerator(seed=0, adversarial_ratio=0.0)
    sc = gen.sample(force_family=AttackFamily.COLLUDING_AGENTS)
    assert sc.is_adversarial and sc.family is AttackFamily.COLLUDING_AGENTS


def test_single_domain_generator():
    gen = ScenarioGenerator(seed=0, domains=(Domain.HEALTHCARE,))
    assert {s.domain for s in gen.sample_batch(20)} == {Domain.HEALTHCARE}


def test_minimum_registry_size_still_generates_all_families():
    gen = ScenarioGenerator(seed=0, n_tools_min=3, n_tools_max=3)
    for family in ALL_FAMILIES:
        sc = gen.sample(force_family=family)
        assert len(sc.registry) == 3
        assert sc.objective is not None
