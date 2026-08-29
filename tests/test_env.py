"""M3 unit tests: the PettingZoo AEC env and its Gymnasium wrapper."""

from __future__ import annotations

import numpy as np
import pytest
from pettingzoo.test import api_test

from arena.config import load_config
from arena.env import (
    ALLOW,
    BLUE,
    FLAG,
    QUARANTINE,
    RED,
    ARENAEnv,
    SingleAgentARENA,
)
from arena.scenarios import AttackFamily, EpisodeType, ScenarioGenerator


# --- helpers ---------------------------------------------------------


def adversarial_scenario(family=AttackFamily.DIRECT_PROMPT_INJECTION, seed=0):
    return ScenarioGenerator(seed=seed, max_steps=12).sample(force_family=family)


def winning_chain_indices(sc):
    """Registry indices for a minimal source -> propagator... -> sink chain."""
    reg = sc.registry
    obj = sc.objective
    src = obj.source_tools[0]
    sink = obj.sink_tools[0] if obj.kind.value != "injected_control" else reg.sinks[0].name
    prop = reg.propagators[0].name
    n_prop = max(1, obj.min_chain_len - 2)
    names = [src] + [prop] * n_prop + [sink]
    return [reg.index(n) for n in names]


class ScriptedRed:
    def __init__(self, indices):
        self._it = list(indices)
        self._i = 0

    def __call__(self, obs):
        a = self._it[self._i] if self._i < len(self._it) else 0
        self._i += 1
        return a


def allow_blue(obs):
    return ALLOW


def quarantine_blue(obs):
    return QUARANTINE


def run_aec(env, red_policy, blue_policy, seed=0):
    env.reset(seed=seed)
    for agent in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        if term or trunc:
            env.step(None)
            continue
        act = red_policy(obs) if agent == RED else blue_policy(obs)
        env.step(act)
    return env


def rr(env):
    """Final (r_red, r_blue) — env.rewards is pruned after the agents drain."""
    b = env.last_breakdown
    return b.r_red, b.r_blue


# --- API conformance -----------------------------------------------


def test_pettingzoo_api_conformance():
    api_test(ARENAEnv(load_config("small.yaml")), num_cycles=400)


def test_reset_is_reproducible_under_seed():
    a = ARENAEnv(load_config("small.yaml"))
    b = ARENAEnv(load_config("small.yaml"))
    a.reset(seed=123)
    b.reset(seed=123)
    assert a.scenario.model_dump() == b.scenario.model_dump()
    # ... and a different seed gives a different episode
    b.reset(seed=124)
    assert a.scenario.scenario_id != b.scenario.scenario_id or \
        a.scenario.model_dump() != b.scenario.model_dump()


def test_full_rollout_reproducible():
    def go(seed):
        env = ARENAEnv(load_config("small.yaml"))
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        trace = []
        for agent in env.agent_iter():
            _, r, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                continue
            act = int(rng.integers(env.action_space(agent).n))
            env.step(act)
            trace.append((agent, act, env.rewards[RED], env.rewards[BLUE]))
        return trace

    assert go(7) == go(7)


# --- episode termination ------------------------------------------


def test_quarantine_ends_episode_immediately_and_call_does_not_execute():
    sc = adversarial_scenario()
    env = ARENAEnv(scenario=sc)
    env.reset(seed=0)
    env.step(winning_chain_indices(sc)[0])  # red proposes the sensitive source
    env.step(QUARANTINE)                     # blue quarantines it
    assert all(env.terminations.values())
    assert env.last_outcome.n_steps == 0     # nothing executed
    assert env.last_outcome.quarantined


def test_objective_completion_ends_episode():
    sc = adversarial_scenario()
    env = ARENAEnv(scenario=sc)
    run_aec(env, ScriptedRed(winning_chain_indices(sc)), allow_blue)
    assert env.last_outcome.objective_completed
    assert all(env.terminations.values())
    assert not any(env.truncations.values())  # completion is a termination, not truncation


def test_step_cap_truncates():
    sc = adversarial_scenario(seed=3)
    env = ARENAEnv(scenario=sc)
    # Red keeps reading a benign file; nothing ever completes.
    benign_idx = sc.registry.index("read_file") if "read_file" in sc.registry.names else 0
    run_aec(env, lambda o: benign_idx, allow_blue)
    assert env.last_outcome.n_steps == sc.max_steps
    assert all(env.truncations.values())
    assert all(env.terminations.values())


# --- reward wiring ----------------------------------------------


def test_red_reward_positive_when_attack_completes_undetected():
    sc = adversarial_scenario()
    env = ARENAEnv(scenario=sc)
    run_aec(env, ScriptedRed(winning_chain_indices(sc)), allow_blue)
    r_red, r_blue = rr(env)
    assert r_red > 0
    assert r_blue < 0  # missed it


def test_blue_reward_positive_when_it_quarantines_the_attack_in_time():
    sc = adversarial_scenario()
    idxs = winning_chain_indices(sc)
    env = ARENAEnv(scenario=sc)
    env.reset(seed=0)
    # allow the first calls, quarantine on the sink
    for k, red_act in enumerate(idxs):
        env.step(red_act)
        env.step(QUARANTINE if k == len(idxs) - 1 else ALLOW)
        if all(env.terminations.values()):
            break
    r_red, r_blue = rr(env)
    assert not env.last_outcome.objective_completed
    assert r_blue > r_red


def test_flag_executes_the_call_but_is_recorded():
    sc = adversarial_scenario()
    idxs = winning_chain_indices(sc)
    env = ARENAEnv(scenario=sc)
    env.reset(seed=0)
    for red_act in idxs:
        env.step(red_act)
        env.step(FLAG)
        if all(env.terminations.values()):
            break
    out = env.last_outcome
    assert out.objective_completed          # flag does not block execution
    assert len(out.flagged_steps) >= 1
    assert set(out.flagged_steps) & set(out.malicious_steps)  # some flags were on-chain


# --- ANTI-LEAKAGE INVARIANT (M3 gate) ---------------------------


def _obs_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(np.array_equal(a[k], b[k]) for k in a)


def test_blue_observation_is_invariant_to_taint_state_perturbation():
    sc = adversarial_scenario()
    env = ARENAEnv(scenario=sc)
    env.reset(seed=0)
    idxs = winning_chain_indices(sc)
    # play two full MCP steps
    for red_act in idxs[:2]:
        env.step(red_act)
        env.step(ALLOW)

    before = env.observe(BLUE)

    # Reach into the env's ground-truth taint graph and corrupt it wholesale.
    from arena.taint import Artifact, FlowEvent, EventKind, TaintColor, TaintTag
    from arena.tools import SideEffect

    fake_tag = TaintTag(TaintColor.SENSITIVE, "read_credentials", 0, 2)
    env._tracker._artifacts.append(
        Artifact(999, "read_credentials", 0, SideEffect.READ_SENSITIVE, frozenset({fake_tag}), (0,))
    )
    env._tracker._events.append(
        FlowEvent(EventKind.EXFIL, 1, "post_webhook", frozenset({fake_tag}), (0, 1))
    )
    env._tracker._last_id = 999

    after = env.observe(BLUE)
    assert _obs_equal(before, after), "Blue's observation changed when taint state was perturbed"


def test_blue_observation_identical_for_same_calls_regardless_of_objective():
    """Same registry, same tool sequence, same verdicts — one adversarial, one
    benign. Blue must see exactly the same thing: the objective/taint is not in
    its observation."""
    sc_adv = adversarial_scenario(seed=11)
    sc_ben = sc_adv.model_copy(update={"episode_type": EpisodeType.BENIGN, "objective": None})

    env_a = ARENAEnv(scenario=sc_adv)
    env_b = ARENAEnv(scenario=sc_ben)
    env_a.reset(seed=0)
    env_b.reset(seed=0)

    idxs = winning_chain_indices(sc_adv)
    for red_act in idxs:
        env_a.step(red_act)
        env_b.step(red_act)
        # compare Blue's view of the pending proposal
        if not all(env_a.terminations.values()):
            assert _obs_equal(env_a.observe(BLUE), env_b.observe(BLUE))
        env_a.step(ALLOW)
        if not all(env_b.terminations.values()):
            env_b.step(ALLOW)
        if all(env_a.terminations.values()):
            break


def test_blue_obs_has_no_field_that_could_carry_taint():
    env = ARENAEnv(load_config("small.yaml"))
    env.reset(seed=0)
    obs = env.observe(BLUE)
    assert set(obs) == {"domain", "calls", "length"}


# --- invalid input --------------------------------------------


def test_blue_rejects_unknown_verdict():
    env = ARENAEnv(scenario=adversarial_scenario())
    env.reset(seed=0)
    env.step(0)
    with pytest.raises(ValueError):
        env.step(99)


def test_out_of_range_red_action_wraps_into_the_registry():
    sc = adversarial_scenario()
    env = ARENAEnv(scenario=sc)
    env.reset(seed=0)
    env.step(len(sc.registry) + 3)  # wraps to index 3
    env.step(ALLOW)
    assert env.last_outcome is None or env._history  # a call was recorded, no crash
    assert env._history[0].tool is sc.registry[3]


# --- Gymnasium wrapper --------------------------------------


def test_single_agent_wrapper_red_vs_frozen_blue():
    sc = adversarial_scenario()
    genv = SingleAgentARENA(RED, allow_blue, scenario=sc)
    obs, info = genv.reset(seed=0)
    assert genv.observation_space.contains(obs)

    idxs = winning_chain_indices(sc)
    total = 0.0
    term = trunc = False
    for a in idxs:
        obs, r, term, trunc, info = genv.step(a)
        total += r
        if term or trunc:
            break
    assert term and total > 0  # Red completed the attack against a passive Blue


def test_single_agent_wrapper_blue_vs_frozen_red():
    sc = adversarial_scenario()
    idxs = winning_chain_indices(sc)
    red = ScriptedRed(idxs)
    genv = SingleAgentARENA(BLUE, red, scenario=sc)
    obs, info = genv.reset(seed=0)

    r = 0.0
    term = trunc = False
    # Blue quarantines on its second decision.
    for k in range(len(idxs)):
        obs, r, term, trunc, info = genv.step(QUARANTINE if k == 1 else ALLOW)
        if term or trunc:
            break
    assert term and r > 0  # Blue stopped it and was rewarded


def test_single_agent_wrapper_reward_not_double_counted():
    sc = adversarial_scenario()
    genv = SingleAgentARENA(RED, allow_blue, scenario=sc)
    genv.reset(seed=0)
    idxs = winning_chain_indices(sc)
    rewards = []
    for a in idxs:
        _, r, term, trunc, _ = genv.step(a)
        rewards.append(r)
        if term or trunc:
            break
    # exactly one non-zero reward, on the terminal transition
    assert sum(1 for x in rewards if x != 0.0) == 1
    assert rewards[-1] != 0.0


def test_wrapper_step_after_done_raises():
    genv = SingleAgentARENA(RED, quarantine_blue, scenario=adversarial_scenario())
    genv.reset(seed=0)
    genv.step(0)  # blue quarantines immediately -> episode over
    with pytest.raises(RuntimeError):
        genv.step(0)
