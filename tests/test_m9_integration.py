"""M9 integration: a real fetch from the public sources, end to end.

Marked ``slow`` and skipped when the network / dataset is unreachable — the unit
suite (``test_data.py``) covers the logic with monkeypatched I/O. This test only
runs when someone actually wants to exercise the live endpoints.
"""

from __future__ import annotations

import urllib.error

import pytest

from arena.config import ArenaConfig, ScenarioConfig
from arena.data import fetch as fetchmod
from arena.data.toucan import BenignProfile, category_of_spec, load_profile
from arena.scenarios import ScenarioGenerator
from arena.scripted import BenignRoller

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def fetched(tmp_path_factory):
    dest = tmp_path_factory.mktemp("data")
    try:
        man = fetchmod.fetch_all(dest, n_toucan=200, force=True)
    except (urllib.error.URLError, RuntimeError, TimeoutError) as e:
        pytest.skip(f"dataset endpoints unreachable: {e}")
    return dest, man


def test_manifest_verifies_and_records_provenance(fetched):
    dest, man = fetched
    assert fetchmod.verify(dest)
    arts = man["artifacts"]
    assert set(arts) == {"tamas/source.tar.gz", "toucan/rows.jsonl", "toucan/profile.json"}
    tou = arts["toucan/rows.jsonl"]
    assert tou["rows"] > 0
    assert tou["licence"] == "Apache-2.0"
    assert len(tou["dataset_sha"]) >= 8  # HF commit sha pinned
    assert "generated_utc" in man


def test_real_profile_is_well_formed_and_drives_benign_traffic(fetched):
    dest, _ = fetched
    prof = load_profile(dest / "toucan" / "profile.json")
    assert isinstance(prof, BenignProfile)
    assert prof.n_trajectories > 0
    assert abs(sum(prof.category_weights.values()) - 1.0) < 1e-6
    assert sum(prof.length_hist.values()) == pytest.approx(1.0)
    # the keyword classifier recognises most real tools; reads are the plurality
    assert prof.category_weights["other"] < 0.6
    assert prof.category_weights["read"] > 0.25

    gen = ScenarioGenerator.from_config(ArenaConfig().scenario, seed=0)
    sc = next(s for s in (gen.sample() for _ in range(200)) if not s.is_adversarial)
    roller = BenignRoller(sc, seed=0, profile=prof)
    cats = [category_of_spec(sc.registry[roller({})]) for _ in range(2000)]
    # the fetched profile actually moved the mix toward reads vs uniform choice
    uniform = [category_of_spec(sc.registry[BenignRoller(sc, seed=0)({})]) for _ in range(2000)]
    assert cats.count("read") / len(cats) > uniform.count("read") / len(uniform)


def test_toucan_source_flag_end_to_end(fetched, monkeypatch):
    dest, _ = fetched
    prof = load_profile(dest / "toucan" / "profile.json")
    monkeypatch.setattr("arena.data.toucan.load_profile", lambda *a, **k: prof)

    from arena.baselines.collect import collect_decisions

    cfg = ArenaConfig(scenario=ScenarioConfig(benign_source="toucan"))
    rows = collect_decisions(20, 20, config=cfg, seed=2)
    assert any(not r.adversarial for r in rows)
    assert any(r.adversarial for r in rows)
