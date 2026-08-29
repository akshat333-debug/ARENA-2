"""M1 unit tests: config loading and validation."""

from __future__ import annotations

import pytest

from arena.config import CONFIG_DIR, ArenaConfig, ScenarioConfig, load_config
from arena.scenarios import ALL_FAMILIES, ScenarioGenerator
from arena.tools import Domain, domain_pool


# --- valid input ---------------------------------------------------------


@pytest.mark.parametrize("name", ["small.yaml", "paper.yaml"])
def test_shipped_configs_load(name: str):
    cfg = load_config(name)
    assert cfg.name == name.removesuffix(".yaml")
    assert cfg.scenario.n_tools_min <= cfg.scenario.n_tools_max


def test_load_config_accepts_a_bare_filename():
    assert load_config("small.yaml").name == load_config(CONFIG_DIR / "small.yaml").name


def test_shipped_configs_are_actually_satisfiable():
    """n_tools_max must not exceed the smallest domain pool, or sampling explodes."""
    for name in ("small.yaml", "paper.yaml"):
        cfg = load_config(name)
        smallest = min(len(domain_pool(d)) for d in cfg.scenario.domains)
        assert cfg.scenario.n_tools_max <= smallest, f"{name}: n_tools_max exceeds pool"


def test_generator_round_trips_through_config():
    cfg = load_config("small.yaml")
    gen = ScenarioGenerator.from_config(cfg.scenario)
    assert gen.max_steps == cfg.scenario.max_steps
    assert gen.domains == tuple(cfg.scenario.domains)
    assert len(gen.sample_batch(5)) == 5


def test_small_config_covers_all_six_families():
    assert set(load_config("small.yaml").scenario.families) == set(ALL_FAMILIES)


def test_top_level_seed_propagates_to_sections(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: t\nseed: 99\n")
    assert load_config(p).scenario.seed == 99


def test_explicit_section_seed_wins_over_top_level(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: t\nseed: 99\nscenario:\n  seed: 5\n")
    assert load_config(p).scenario.seed == 5


def test_defaults_are_usable_with_no_yaml():
    gen = ScenarioGenerator.from_config(ArenaConfig().scenario)
    assert len(gen.sample_batch(3)) == 3


# --- invalid input -------------------------------------------------------


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")


def test_unknown_key_is_rejected(tmp_path):
    """Silently ignoring a typo'd key is how a run is quietly misconfigured."""
    p = tmp_path / "c.yaml"
    p.write_text("name: t\nscenrio:\n  max_steps: 3\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_non_mapping_yaml_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_tools_min": 9, "n_tools_max": 4},
        {"adversarial_ratio": 2.0},
        {"max_steps": 0},
        {"n_tools_min": 1},
        {"domains": ()},
        {"families": ()},
    ],
)
def test_scenario_config_validates(kwargs):
    with pytest.raises(ValueError):
        ScenarioConfig(**kwargs)


# --- edge cases ----------------------------------------------------------


def test_empty_yaml_falls_back_to_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("")
    assert load_config(p).scenario.max_steps == ScenarioConfig().max_steps


def test_config_is_frozen():
    cfg = load_config("small.yaml")
    with pytest.raises(Exception):
        cfg.seed = 3  # type: ignore[misc]


def test_single_domain_config_is_valid(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("scenario:\n  domains: [finance]\n")
    assert tuple(load_config(p).scenario.domains) == (Domain.FINANCE,)
