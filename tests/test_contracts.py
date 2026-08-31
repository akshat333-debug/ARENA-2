"""Cross-module contract tests (added by the M1–M9 audit).

Every other test file checks one module against hand-built inputs. Nothing
checked the *hand-offs*, and that is exactly where the audit's three bugs lived:
the env told PPO an episode had ended in a way PPO's boundary logic could not
read, self-play measured one quantity and labelled it another, and the harness
scored a learned policy under different rules than the baselines it was ranked
against.

These run randomised episodes and assert the invariants that span modules. They
are deliberately blunt: cheap to run, and they fail loudly when two modules
start disagreeing about a contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from arena.config import load_config
from arena.env import ALLOW, BLUE, FLAG, QUARANTINE, RED, ARENAEnv, SingleAgentARENA
from arena.rewards import compute_rewards
from arena.scenarios import EpisodeType, ScenarioGenerator
from arena.taint import TaintTracker

CFG = load_config("small.yaml")
N_EPISODES = 250


def _random_episode(env, rng):
    """Play one episode with random-but-legal actions from both sides."""
    for agent in env.agent_iter():
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        if agent == RED:
            # Includes indices past the registry, to exercise the modulo wrap.
            env.step(int(rng.integers(0, CFG.scenario.n_tools_max)))
        else:
            env.step(int(rng.choice([ALLOW, FLAG, QUARANTINE], p=[0.7, 0.15, 0.15])))


@pytest.fixture(scope="module")
def episodes():
    """(scenario, outcome, breakdown, executed_calls) for many random episodes."""
    rng = np.random.default_rng(0)
    gen = ScenarioGenerator.from_config(CFG.scenario, seed=11)
    out = []
    for ep in range(N_EPISODES):
        sc = gen.sample()
        env = ARENAEnv(CFG, scenario=sc)
        env.reset(seed=ep)
        _random_episode(env, rng)
        out.append((sc, env.last_outcome, env.last_breakdown,
                    [(rc.tool, rc.step_index) for rc in env._history]))
    return out


def test_the_fixture_actually_exercises_all_three_endings(episodes):
    """Guard against a vacuous suite: if every episode ended the same way, the
    invariants below would be far weaker than they look."""
    quarantined = sum(o.quarantined for _, o, _, _ in episodes)
    completed = sum(o.objective_completed for _, o, _, _ in episodes)
    capped = sum(not (o.quarantined or o.objective_completed) for _, o, _, _ in episodes)
    assert quarantined > 10 and completed > 5 and capped > 10, (
        f"ending mix too skewed: quarantine={quarantined} complete={completed} cap={capped}"
    )


def test_env_rewards_are_exactly_the_reward_engines(episodes):
    """env.py must not compute reward itself — it delegates to rewards.py."""
    for sc, outcome, bd, _ in episodes:
        want = compute_rewards(outcome, CFG.reward)
        assert want.r_red == pytest.approx(bd.r_red, abs=1e-12)
        assert want.r_blue == pytest.approx(bd.r_blue, abs=1e-12)


def test_outcome_fields_are_internally_consistent(episodes):
    for sc, o, _, history in episodes:
        assert o.n_steps == len(history)
        assert o.n_steps <= sc.max_steps
        if o.quarantined:
            # The quarantined call never ran, so the step count is the quarantine step.
            assert o.quarantine_step == o.n_steps
        if o.objective_completed:
            assert 0 <= o.completion_step < sc.max_steps
        assert all(0 <= s < sc.max_steps for s in o.flagged_steps)
        assert all(0 <= s < sc.max_steps for s in o.malicious_steps)
        if not sc.is_adversarial:
            assert not o.malicious_steps and not o.objective_completed
            assert not o.caught_in_flight


def test_a_completed_objective_is_reproducible_by_an_independent_replay(episodes):
    """The env's verdict on "did the attack land" must be a pure function of the
    executed call sequence — reproducible from the history alone."""
    checked = 0
    for sc, o, _, history in episodes:
        if not o.objective_completed:
            continue
        tr = TaintTracker()
        for tool, step_index in history:
            tr.step(tool, step_index)
        assert tr.objective_completed(sc.objective), (
            f"{sc.scenario_id}: env says completed, independent replay disagrees"
        )
        checked += 1
    assert checked > 0, "no completed episodes to check — test would be vacuous"


# --- the anti-leakage invariant, at the observation boundary --------------


def test_blue_observation_is_invariant_to_the_ground_truth(episodes):
    """Same registry, same tool sequence, different objective (and a benign twin
    with no objective at all) => byte-identical Blue observations.

    Constructed explicitly rather than by searching for naturally-occurring
    pairs: a search that finds none leaves the check silently vacuous.
    """
    gen = ScenarioGenerator.from_config(CFG.scenario, seed=3)
    adv = [s for s in (gen.sample() for _ in range(400)) if s.is_adversarial]
    assert adv, "no adversarial scenarios generated"

    compared = 0
    for base in adv[:10]:
        obj = base.objective
        variants = [
            base.model_copy(update={"objective": obj.model_copy(update={
                "source_tools": tuple(reversed(obj.source_tools)),
                "sink_tools": tuple(reversed(obj.sink_tools)),
                "min_chain_len": max(1, obj.min_chain_len - 1),
            })}),
            base.model_copy(update={"objective": None, "episode_type": EpisodeType.BENIGN}),
        ]
        for variant in variants:
            e1, e2 = ARENAEnv(CFG, scenario=base), ARENAEnv(CFG, scenario=variant)
            e1.reset(seed=0)
            e2.reset(seed=0)
            for k in range(min(4, len(base.registry))):
                e1.step(k)
                e2.step(k)
                o1, o2 = e1.observe(BLUE), e2.observe(BLUE)
                assert np.array_equal(o1["calls"], o2["calls"])
                assert np.array_equal(o1["length"], o2["length"])
                assert np.array_equal(o1["domain"], o2["domain"])
                e1.step(ALLOW)
                e2.step(ALLOW)
            compared += 1
    assert compared >= 10


# --- observation-space conformance ---------------------------------------


def test_observations_are_finite_and_inside_the_declared_spaces():
    env = ARENAEnv(CFG)
    env.reset(seed=0)
    for _ in range(300):
        for agent in (RED, BLUE):
            obs = env.observe(agent)
            for key, value in obs.items():
                arr = np.asarray(value)
                assert np.all(np.isfinite(arr)), f"non-finite in {agent}/{key}"
            assert env.observation_space(agent).contains(obs), f"{agent} obs outside its space"
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.reset()
            continue
        env.step(env.action_space(env.agent_selection).sample())


# --- the env <-> Gym-wrapper contract PPO depends on ----------------------


@pytest.mark.parametrize("learner", [RED, BLUE])
def test_gym_episode_return_equals_the_reward_breakdown(learner):
    """`SingleAgentARENA` accumulates PettingZoo's per-agent rewards. The sum over
    an episode must be exactly that agent's final reward, counted once — this is
    what PPO optimises, so a mis-attribution here is invisible and fatal."""
    opponent = (lambda o: ALLOW) if learner == RED else (lambda o: 0)
    env = SingleAgentARENA(learner, opponent, config=CFG)
    for ep in range(60):
        env.reset(seed=0 if ep == 0 else None)
        total, done = 0.0, False
        while not done:
            _, reward, term, trunc, _ = env.step(env.action_space.sample())
            total += reward
            done = term or trunc
        bd = env.aec.last_breakdown
        expected = bd.r_red if learner == RED else bd.r_blue
        assert total == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("learner", [RED, BLUE])
def test_every_episode_end_sets_exactly_one_of_terminated_truncated(learner):
    """Gymnasium's contract, and the specific thing that broke PPO's GAE: an env
    raising both flags makes `terminated and not truncated` false, so the
    boundary vanishes. See docs/audit-m1-m9.md."""
    opponent = (lambda o: ALLOW) if learner == RED else (lambda o: 0)
    env = SingleAgentARENA(learner, opponent, config=CFG)
    ends = 0
    env.reset(seed=0)
    for _ in range(600):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            assert term != trunc, "exactly one of terminated/truncated must fire"
            assert "hit_step_cap" in info
            ends += 1
            env.reset()
    assert ends > 20, "too few episode ends to be a meaningful check"


def test_causal_features_do_not_open_a_leakage_channel(episodes):
    """`blue_causal_features` (M11) hands Blue the same hand-engineered summary
    the causal_monitor baseline uses. It is derived from Blue's own observable
    call rows, so it must inherit the invariance — assert that rather than
    assume it, since this is the one place a new Blue input was added since M3.
    """
    import torch

    from arena.features import sequence_features_torch
    from arena.policies import to_batch

    gen = ScenarioGenerator.from_config(CFG.scenario, seed=11)
    adv = [s for s in (gen.sample() for _ in range(400)) if s.is_adversarial]
    assert adv, "no adversarial scenarios generated"

    compared = 0
    for base in adv[:10]:
        twin = base.model_copy(update={"objective": None, "episode_type": EpisodeType.BENIGN})
        e1, e2 = ARENAEnv(CFG, scenario=base), ARENAEnv(CFG, scenario=twin)
        e1.reset(seed=0)
        e2.reset(seed=0)
        for _ in range(4):
            for e in (e1, e2):
                if not (e.terminations[RED] or e.truncations[RED]):
                    e.step(0)
            if e1.terminations[BLUE] or e1.truncations[BLUE]:
                break
            o1, o2 = e1.observe(BLUE), e2.observe(BLUE)
            f1 = sequence_features_torch(*[to_batch(o1)[k] for k in ("calls", "length")])
            f2 = sequence_features_torch(*[to_batch(o2)[k] for k in ("calls", "length")])
            torch.testing.assert_close(f1, f2)
            compared += 1
            for e in (e1, e2):
                if not (e.terminations[BLUE] or e.truncations[BLUE]):
                    e.step(ALLOW)
    assert compared > 0, "no observations compared — test would be vacuous"
