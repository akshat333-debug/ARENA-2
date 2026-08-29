"""M4 unit tests: the three baseline defenders."""

from __future__ import annotations

import numpy as np
import pytest

from arena.baselines import CausalMonitor, SingleShotDetector, StaticAllowList
from arena.baselines.collect import Decision, collect_decisions, split
from arena.config import load_config
from arena.env import BLUE, QUARANTINE, RED, ARENAEnv
from arena.features import VERDICTS
from arena.scenarios import AttackFamily, ScenarioGenerator
from arena.scripted import ScriptedAttacker
from tests.test_env import run_aec

CFG = load_config("small.yaml")


@pytest.fixture(scope="module")
def decisions():
    return collect_decisions(60, 60, config=CFG, seed=3)


# --- every baseline is a usable Blue policy ---------------------


@pytest.mark.parametrize("make", [StaticAllowList, SingleShotDetector, CausalMonitor])
def test_baseline_is_a_callable_blue_policy_returning_valid_verdicts(make, decisions):
    b = make()
    if hasattr(b, "fit"):
        b.fit(decisions)
    env = ARENAEnv(CFG)
    env.reset(seed=0)
    seen = set()
    for agent in env.agent_iter():
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        if agent == RED:
            env.step(0)
        else:
            v = b(env.observe(BLUE))
            assert v in VERDICTS
            seen.add(v)
            env.step(v)
    assert seen  # it actually made decisions


# --- StaticAllowList ------------------------------------------


def test_allowlist_misses_the_canonical_split_chain():
    sc = ScenarioGenerator(seed=0).sample(force_family=AttackFamily.DIRECT_PROMPT_INJECTION)
    env = ARENAEnv(scenario=sc)
    run_aec(env, ScriptedAttacker(sc), StaticAllowList())
    assert env.last_outcome.objective_completed  # every call individually in policy


def test_allowlist_blocks_a_lone_high_sensitivity_sink():
    al = StaticAllowList(block_sensitivity=2)
    sc = ScenarioGenerator(seed=1).sample(force_family=AttackFamily.DIRECT_PROMPT_INJECTION)
    # find a sensitivity-2 sink in some registry
    gen = ScenarioGenerator(seed=2)
    for _ in range(40):
        sc = gen.sample(force_family=AttackFamily.DIRECT_PROMPT_INJECTION)
        sinks = [t for t in sc.registry if t.is_sink and t.sensitivity >= 2]
        if sinks:
            break
    else:
        pytest.skip("no sensitivity-2 sink sampled")
    env = ARENAEnv(scenario=sc)
    env.reset(seed=0)
    env.step(sc.registry.index(sinks[0].name))
    verdict = al(env.observe(BLUE))
    assert verdict == QUARANTINE


def test_allowlist_needs_no_training():
    assert StaticAllowList().fit([]) is not None


# --- trained detectors --------------------------------------


def test_single_shot_fits_and_predicts_in_range(decisions):
    ss = SingleShotDetector().fit(decisions)
    for d in decisions[:20]:
        s = ss.score(d.blue_obs)
        assert 0.0 <= s <= 1.0


def test_causal_monitor_fits_and_predicts_in_range(decisions):
    cm = CausalMonitor().fit(decisions)
    for d in decisions[:20]:
        assert 0.0 <= cm.score(d.blue_obs) <= 1.0


def test_causal_monitor_scores_the_sink_of_a_real_chain_higher_than_a_lone_read(decisions):
    cm = CausalMonitor().fit(decisions)
    chain = [d for d in decisions if d.on_attack_path and d.pending_tool.is_sink]
    lone = [d for d in decisions if (not d.adversarial) and d.step_index == 0]
    assert chain and lone
    hi = np.mean([cm.score(d.blue_obs) for d in chain])
    lo = np.mean([cm.score(d.blue_obs) for d in lone])
    assert hi > lo


def test_single_shot_cannot_use_context_two_identical_calls_score_the_same(decisions):
    """A read_env_file call scores the same for single-shot whether it is step 0
    of an attack or step 0 of a benign session — that is the limitation."""
    ss = SingleShotDetector().fit(decisions)
    reads = [d for d in decisions if d.pending_tool.name == "read_env_file"]
    if len(reads) < 2:
        pytest.skip("not enough read_env_file decisions")
    scores = {round(ss.score(d.blue_obs), 6) for d in reads}
    assert len(scores) == 1  # context-free: all identical


# --- collect / split ----------------------------------------


def test_collect_decisions_labels_are_consistent(decisions):
    assert all(isinstance(d, Decision) for d in decisions)
    # on-path decisions only occur in adversarial episodes
    assert all(d.adversarial for d in decisions if d.on_attack_path)
    # benign episodes never complete an objective
    assert not any(d.episode_completed for d in decisions if not d.adversarial)


def test_split_is_disjoint_and_covers(decisions):
    tr, va = split(decisions, frac=0.75, seed=0)
    assert len(tr) + len(va) == len(decisions)
    assert len(tr) > len(va) > 0
