"""M3 integration: generator -> env -> taint -> rewards, and the Gym wrapper.

Unit tests exercise the env mechanics in isolation. These run whole episodes for
every attack family through the real stack and check the game-level properties
the later modules depend on:

* every family is winnable by Red using *only* tool-index actions (the env's
  auto data-flow default), so M5's Red does not need a structured action space;
* a sequence-aware Blue policy beats the canonical split-chain attack that a
  per-call policy cannot (the project's headline claim, demonstrated);
* the Gymnasium single-agent view produces a coherent best-response signal, which
  is what the M8 exploitability sweep is built on.
"""

from __future__ import annotations

import numpy as np

from arena.config import load_config
from arena.env import ALLOW, QUARANTINE, RED, ARENAEnv, SingleAgentARENA
from arena.scenarios import ALL_FAMILIES, EpisodeType, ScenarioGenerator

import pytest

from tests.test_env import ScriptedRed, allow_blue, run_aec, winning_chain_indices, rr


@pytest.mark.parametrize("family", list(ALL_FAMILIES))
def test_every_family_is_winnable_with_tool_index_actions_only(family):
    gen = ScenarioGenerator(seed=0, max_steps=16)
    wins = 0
    for _ in range(15):
        sc = gen.sample(force_family=family)
        env = ARENAEnv(scenario=sc)
        run_aec(env, ScriptedRed(winning_chain_indices(sc)), allow_blue)
        wins += env.last_outcome.objective_completed
    assert wins == 15, f"{family.value}: only {wins}/15 scripted attacks completed"


@pytest.mark.parametrize("family", list(ALL_FAMILIES))
def test_reward_signs_match_the_game_outcome(family):
    sc = ScenarioGenerator(seed=1, max_steps=16).sample(force_family=family)
    env = ARENAEnv(scenario=sc)
    run_aec(env, ScriptedRed(winning_chain_indices(sc)), allow_blue)
    r_red, r_blue = rr(env)
    assert r_red > 0 and r_blue < 0, f"{family.value}: red {r_red}, blue {r_blue}"


def test_sequence_aware_blue_beats_the_split_chain_a_per_call_policy_misses():
    """A per-call Blue that judges each call in isolation allows the canonical
    read -> summarize -> exfiltrate chain (every call is individually in policy).
    A Blue that watches the trajectory can stop it. Same episodes, both policies.
    """
    gen = ScenarioGenerator(seed=5, max_steps=14)
    scenarios = [gen.sample(force_family=ALL_FAMILIES[1]) for _ in range(20)]  # direct prompt injection

    def per_call_blue(obs):
        # Looks only at the pending call: allow anything that isn't, by itself,
        # an obviously forbidden action. A lone external sink call is common in
        # benign work, so it allows those too -> misses the chain.
        return ALLOW

    def sequence_blue(obs):
        # Watches the sequence: if a sensitive read has already happened and the
        # pending call is an external sink, quarantine.
        calls = obs["calls"]
        length = int(obs["length"][0])
        se = calls[:length, : 6]  # side-effect one-hot block
        seen_sensitive = bool(se[:, 1].any())  # index 1 == read_sensitive
        pending_is_sink = bool(se[-1, 4] == 1.0)  # index 4 == external_sink
        return QUARANTINE if (seen_sensitive and pending_is_sink) else ALLOW

    per_call_success = 0
    sequence_success = 0
    for sc in scenarios:
        e1 = ARENAEnv(scenario=sc)
        run_aec(e1, ScriptedRed(winning_chain_indices(sc)), per_call_blue)
        per_call_success += e1.last_outcome.objective_completed

        e2 = ARENAEnv(scenario=sc)
        run_aec(e2, ScriptedRed(winning_chain_indices(sc)), sequence_blue)
        sequence_success += e2.last_outcome.objective_completed

    assert per_call_success == len(scenarios), "per-call Blue should miss every split chain"
    assert sequence_success == 0, "sequence-aware Blue should catch every split chain"


def test_benign_episodes_do_not_punish_a_calm_blue_but_do_punish_a_paranoid_one():
    gen = ScenarioGenerator(seed=8, adversarial_ratio=0.0, max_steps=12)
    calm_total = 0.0
    paranoid_total = 0.0
    for _ in range(20):
        sc = gen.sample()
        assert sc.episode_type is EpisodeType.BENIGN

        e_calm = ARENAEnv(scenario=sc)
        run_aec(e_calm, lambda o: 0, allow_blue)
        calm_total += rr(e_calm)[1]

        e_par = ARENAEnv(scenario=sc)
        run_aec(e_par, lambda o: 0, lambda o: QUARANTINE)
        paranoid_total += rr(e_par)[1]

    assert calm_total == 0.0
    assert paranoid_total < 0.0


def test_exploitability_style_sweep_against_a_frozen_blue():
    """The M8 primitive: freeze Blue, measure a best-response Red's success rate.
    Here 'best response' is the scripted winning chain; a passive Blue should be
    almost fully exploitable, a hard Blue much less so."""

    def passive_blue(obs):
        return ALLOW

    def hardline_blue(obs):
        # quarantine the moment an external sink is proposed after any read
        calls = obs["calls"]
        length = int(obs["length"][0])
        se = calls[:length, :6]
        if length >= 1 and se[-1, 4] == 1.0 and se[:, :2].any():
            return QUARANTINE
        return ALLOW

    def exploitability(blue_policy, n=40):
        succ = 0
        for i in range(n):
            gen = ScenarioGenerator(seed=1000 + i, max_steps=16)
            sc = gen.sample(force_family=ALL_FAMILIES[i % len(ALL_FAMILIES)])
            genv = SingleAgentARENA(RED, blue_policy, scenario=sc)
            genv.reset(seed=0)
            term = trunc = False
            for a in winning_chain_indices(sc):
                _, _, term, trunc, info = genv.step(a)
                if term or trunc:
                    break
            succ += int(info.get("objective_completed", False))
        return succ / n

    passive = exploitability(passive_blue)
    hard = exploitability(hardline_blue)
    assert passive == 1.0, f"passive Blue should be fully exploitable, got {passive}"
    assert hard < passive, f"hardline Blue should reduce exploitability ({hard} vs {passive})"


def test_full_stack_seeded_run_is_reproducible():
    def go():
        env = ARENAEnv(load_config("small.yaml"))
        env.reset(seed=2024)
        rng = np.random.default_rng(0)
        out = []
        for agent in env.agent_iter():
            _, _, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                continue
            env.step(int(rng.integers(env.action_space(agent).n)))
            out.append((agent, env.last_outcome is not None))
        b = env.last_breakdown
        return out, (round(b.r_red, 6), round(b.r_blue, 6))

    assert go() == go()
