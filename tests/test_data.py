"""M9 unit tests: dataset fetch (idempotency + manifest) and the Toucan-derived
benign-traffic profile. No network — the fetch helpers are monkeypatched."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from arena.config import ArenaConfig, ScenarioConfig
from arena.data import fetch as fetchmod
from arena.data.toucan import (
    CATEGORIES,
    BenignProfile,
    build_profile,
    category_of_spec,
    classify_name,
    load_profile,
)
from arena.scenarios import ScenarioGenerator
from arena.scripted import BenignRoller
from arena.tools import SideEffect, ToolSpec

CFG = ArenaConfig()


# --------------------------------------------------------------------------
# classify_name / category_of_spec
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("get_user", "read"),
        ("list-issues", "read"),
        ("lyrical-mcp-count_syllables", "read"),
        ("find_rhymes", "read"),
        ("create_issue", "act"),
        ("send_email", "act"),
        ("github-mcp-delete_branch", "act"),
        ("run_script", "act"),
        ("summarize_thread", "other"),
        ("weather", "other"),
        ("", "other"),
    ],
)
def test_classify_name(name, expected):
    assert classify_name(name) == expected


def test_classify_name_uses_description_when_name_is_opaque():
    assert classify_name("tool_x7", "Search the knowledge base for matching docs") == "read"


@pytest.mark.parametrize(
    "eff,expected",
    [
        (SideEffect.READ_BENIGN, "read"),
        (SideEffect.READ_SENSITIVE, "read"),
        (SideEffect.TRANSFORM, "other"),
        (SideEffect.WRITE_LOCAL, "act"),
        (SideEffect.EXTERNAL_SINK, "act"),
        (SideEffect.EXEC, "act"),
    ],
)
def test_category_of_spec_covers_every_side_effect(eff, expected):
    t = ToolSpec(name="t", category="c", side_effect=eff, sensitivity=0, description="d")
    assert category_of_spec(t) == expected
    assert set(CATEGORIES) == {"read", "act", "other"}


# --------------------------------------------------------------------------
# BenignProfile
# --------------------------------------------------------------------------

def _msgs(*tool_names):
    ms = [{"role": "user", "content": "hi"}]
    for n in tool_names:
        ms.append({"role": "tool_call", "content": "{'name': '%s', 'arguments': '{}'}" % n})
    return json.dumps(ms)


def test_build_profile_from_in_memory_rows():
    rows = [
        {"messages": _msgs("get_a", "get_b", "create_c"), "tools": []},
        {"messages": _msgs("list_x", "delete_y"), "tools": []},
        {"messages": _msgs(), "tools": []},  # no calls -> skipped
    ]
    p = build_profile(rows)
    assert p.n_trajectories == 2
    assert abs(sum(p.category_weights.values()) - 1.0) < 1e-9
    # 5 calls total: get,get,list = read (3/5); create,delete = act (2/5)
    assert p.category_weights["read"] == pytest.approx(0.6)
    assert p.category_weights["act"] == pytest.approx(0.4)
    assert abs(sum(p.length_hist.values()) - 1.0) < 1e-9
    assert p.length_hist == {3: pytest.approx(0.5), 2: pytest.approx(0.5)}


def test_build_profile_rejects_empty():
    with pytest.raises(ValueError):
        build_profile([{"messages": _msgs(), "tools": []}])


def test_profile_validation():
    with pytest.raises(ValueError):
        BenignProfile(category_weights={"read": 0.9, "act": 0.05}, length_hist={}, n_trajectories=1)
    with pytest.raises(ValueError):
        BenignProfile(category_weights={"bogus": 1.0}, length_hist={}, n_trajectories=1)
    with pytest.raises(ValueError):
        BenignProfile(category_weights={"read": 1.0}, length_hist={}, n_trajectories=0)


def test_profile_roundtrip(tmp_path):
    p = BenignProfile(
        category_weights={"read": 0.5, "act": 0.3, "other": 0.2},
        length_hist={1: 0.25, 4: 0.75},
        n_trajectories=123,
    )
    f = tmp_path / "sub" / "profile.json"
    p.save(f)
    q = BenignProfile.load(f)
    assert q == p
    assert load_profile(f) == p
    assert load_profile(tmp_path / "missing.json") is None


def test_profile_sampling_follows_weights():
    rng = np.random.default_rng(0)
    p = BenignProfile(
        category_weights={"read": 0.8, "act": 0.15, "other": 0.05},
        length_hist={2: 1.0}, n_trajectories=10,
    )
    draws = [p.sample_category(rng) for _ in range(4000)]
    assert 0.75 < draws.count("read") / len(draws) < 0.85
    assert all(p.sample_length(rng) == 2 for _ in range(50))


# --------------------------------------------------------------------------
# the swap-in: a profile drives BenignRoller's category mix
# --------------------------------------------------------------------------

def _benign_scenario(seed=0):
    gen = ScenarioGenerator.from_config(CFG.scenario, seed=seed)
    for _ in range(200):
        sc = gen.sample()
        if not sc.is_adversarial:
            return sc
    raise AssertionError("no benign scenario")


def _category_mix(profile, *, steps=4000, seed=1):
    sc = _benign_scenario()
    roller = BenignRoller(sc, seed=seed, profile=profile)
    counts = {c: 0 for c in CATEGORIES}
    for _ in range(steps):
        i = roller({})
        counts[category_of_spec(sc.registry[i])] += 1
        # invariant preserved: never a sink after touching a source
        assert not (roller._touched_source and sc.registry[i].is_sink)
    tot = sum(counts.values())
    return {c: counts[c] / tot for c in CATEGORIES}


def test_toucan_profile_shifts_benign_distribution():
    read_heavy = BenignProfile(
        category_weights={"read": 0.9, "act": 0.08, "other": 0.02},
        length_hist={4: 1.0}, n_trajectories=100,
    )
    act_heavy = BenignProfile(
        category_weights={"read": 0.1, "act": 0.88, "other": 0.02},
        length_hist={4: 1.0}, n_trajectories=100,
    )
    mix_read = _category_mix(read_heavy)
    mix_act = _category_mix(act_heavy)

    # the profile actually moves the traffic, and in the right direction
    assert mix_read["read"] > 0.7
    assert mix_act["act"] > mix_read["act"] + 0.3
    # sanity: uniform roller is nowhere near the read-heavy skew
    uniform = _category_mix(None)
    assert mix_read["read"] > uniform["read"] + 0.15


def test_benign_roller_without_profile_is_unchanged():
    sc = _benign_scenario()
    a = BenignRoller(sc, seed=7)
    b = BenignRoller(sc, seed=7)
    assert [a({}) for _ in range(50)] == [b({}) for _ in range(50)]


# --------------------------------------------------------------------------
# fetch: manifest + idempotency (monkeypatched network)
# --------------------------------------------------------------------------

def _fake_get(url):
    return b"fake-tamas-tarball-" + url.encode()


def _fake_get_json(url):
    if "/api/datasets/" in url:
        return {"sha": "deadbeef0000"}
    if "/rows?" in url:
        off = int(url.split("offset=")[1].split("&")[0])
        if off > 0:
            return {"rows": []}  # one page, then stop
        rows = []
        for k in range(20):
            names = ["get_x", "get_y", "create_z"] if k % 2 else ["list_a", "delete_b"]
            rows.append({"row": {"messages": _msgs(*names), "tools": []}})
        return {"rows": rows}
    raise AssertionError(f"unexpected url {url}")


@pytest.fixture
def fake_net(monkeypatch):
    monkeypatch.setattr(fetchmod, "_get", _fake_get)
    monkeypatch.setattr(fetchmod, "_get_json", _fake_get_json)


def test_fetch_all_writes_a_complete_manifest(tmp_path, fake_net):
    man = fetchmod.fetch_all(tmp_path, n_toucan=100, force=True)
    arts = man["artifacts"]
    assert set(arts) == {"tamas/source.tar.gz", "toucan/rows.jsonl", "toucan/profile.json"}
    for meta in arts.values():
        assert set(meta) >= {"sha256", "bytes", "source", "licence", "retrieved_utc"}
        assert len(meta["sha256"]) == 64
        assert meta["bytes"] > 0
        assert meta["retrieved_utc"].startswith("20")
    assert "Apache-2.0" in arts["toucan/rows.jsonl"]["licence"]
    assert "CDLA-Permissive-2.0" in arts["tamas/source.tar.gz"]["licence"]
    assert arts["toucan/rows.jsonl"]["dataset_sha"] == "deadbeef0000"

    # profile really got built from the fetched rows
    prof = BenignProfile.load(tmp_path / "toucan" / "profile.json")
    assert prof.n_trajectories == 20
    assert abs(sum(prof.category_weights.values()) - 1.0) < 1e-9


def test_fetch_all_is_idempotent(tmp_path, fake_net, monkeypatch):
    first = fetchmod.fetch_all(tmp_path, n_toucan=100, force=True)
    assert fetchmod.verify(tmp_path)

    # second run must not touch the network at all
    def boom(*_a, **_k):
        raise AssertionError("network hit on an idempotent re-fetch")

    monkeypatch.setattr(fetchmod, "_get", boom)
    monkeypatch.setattr(fetchmod, "_get_json", boom)
    second = fetchmod.fetch_all(tmp_path, n_toucan=100)
    assert second == first


def test_verify_fails_when_an_artifact_is_tampered(tmp_path, fake_net):
    fetchmod.fetch_all(tmp_path, n_toucan=100, force=True)
    assert fetchmod.verify(tmp_path)
    rows = tmp_path / "toucan" / "rows.jsonl"
    rows.write_text(rows.read_text() + '\n{"messages":"[]"}\n')
    assert not fetchmod.verify(tmp_path)


def test_verify_false_without_manifest(tmp_path):
    assert fetchmod.verify(tmp_path) is False
    assert fetchmod.load_manifest(tmp_path) is None


# --------------------------------------------------------------------------
# config wiring + THIRD_PARTY.md
# --------------------------------------------------------------------------

def test_benign_source_config():
    assert ArenaConfig().scenario.benign_source == "synthetic"
    cfg = ArenaConfig(scenario=ScenarioConfig(benign_source="toucan"))
    assert cfg.scenario.benign_source == "toucan"
    with pytest.raises(Exception):
        ScenarioConfig(benign_source="parquet")


def test_collect_decisions_honours_toucan_source(monkeypatch):
    from arena.baselines import collect as collectmod

    prof = BenignProfile(
        category_weights={"read": 0.9, "act": 0.08, "other": 0.02},
        length_hist={3: 1.0}, n_trajectories=50,
    )
    monkeypatch.setattr("arena.data.toucan.load_profile", lambda *a, **k: prof)
    cfg = ArenaConfig(scenario=ScenarioConfig(benign_source="toucan"))
    rows = collectmod.collect_decisions(3, 3, config=cfg, seed=1)
    assert rows and any(not r.adversarial for r in rows)
    # synthetic path still returns None
    assert collectmod._benign_profile(ArenaConfig()) is None


def test_third_party_md_records_both_datasets():
    txt = (Path(__file__).resolve().parent.parent / "THIRD_PARTY.md").read_text()
    for token in [
        "Toucan-1.5M", "Apache-2.0", "TAMAS", "MIT", "CDLA-Permissive-2.0",
        "github.com/microsoft/TAMAS", "huggingface.co/datasets/Agent-Ark/Toucan-1.5M",
    ]:
        assert token in txt
