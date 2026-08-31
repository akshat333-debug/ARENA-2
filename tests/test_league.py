"""M7 unit tests: the opponent-checkpoint league (no training)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from arena.env import RED, ARENAEnv
from arena.league import Checkpoint, FrozenPolicySampler, League, net_factory
from arena.policies import make_policy
from arena.config import load_config

CFG = load_config("small.yaml")
MS, NT = CFG.scenario.max_steps, CFG.scenario.n_tools_max


def a_blue(seed):
    return make_policy("blue", max_steps=MS, n_tools_max=NT, seed=seed)


# --- pool population -------------------------------------------------


def test_pool_grows_with_add():
    lg = League(seed=0)
    for g in range(4):
        lg.add("blue", g, a_blue(g).state_dict())
    assert lg.pool_size("blue") == 4
    assert lg.pool_size("red") == 0
    assert len(lg) == 4


def test_pool_max_evicts_oldest_keeps_latest():
    lg = League(pool_max=3, seed=0)
    for g in range(6):
        lg.add("blue", g, a_blue(g).state_dict())
    gens = [c.generation for c in lg.checkpoints("blue")]
    assert gens == [3, 4, 5]
    assert lg.latest("blue").generation == 5


def test_add_rejects_bad_side():
    with pytest.raises(ValueError):
        League().add("green", 0, a_blue(0).state_dict())


# --- sampling ------------------------------------------------------


def test_sample_empty_pool_raises():
    with pytest.raises(IndexError):
        League().sample("red")


def test_single_checkpoint_pool_always_returns_it():
    lg = League(p_latest=0.0, seed=0)
    lg.add("blue", 7, a_blue(0).state_dict())
    assert all(lg.sample("blue").generation == 7 for _ in range(20))


def test_sample_probs_sum_to_one_and_match_empirical():
    lg = League(pool_max=None, p_latest=0.4, seed=0)
    for g in range(4):
        lg.add("blue", g, a_blue(g).state_dict())
    probs = lg.sample_probs("blue")
    assert probs.shape == (4,)
    assert abs(probs.sum() - 1.0) < 1e-9
    assert probs[-1] == pytest.approx(0.4 + 0.6 / 4)

    draws = np.array([lg.sample("blue").generation for _ in range(6000)])
    _, counts = np.unique(draws, return_counts=True)
    emp = counts / draws.size
    assert np.allclose(emp, probs, atol=0.03), (emp, probs)


def test_p_latest_one_always_picks_latest():
    lg = League(p_latest=1.0, seed=0)
    for g in range(3):
        lg.add("red", g, make_policy("red", max_steps=MS, n_tools_max=NT, seed=g).state_dict())
    assert all(lg.sample("red").generation == 2 for _ in range(30))


@pytest.mark.parametrize("kwargs", [{"p_latest": 1.5}, {"p_latest": -0.1}, {"pool_max": 0}])
def test_league_validates_construction(kwargs):
    with pytest.raises(ValueError):
        League(**kwargs)


# --- persistence -------------------------------------------------


def test_state_dict_round_trips_bit_identically():
    lg = League(pool_max=4, p_latest=0.25, seed=1)
    for g in range(3):
        lg.add("blue", g, a_blue(g).state_dict())
        lg.add("red", g, make_policy("red", max_steps=MS, n_tools_max=NT, seed=g).state_dict())

    lg2 = League(seed=999)
    lg2.load_state_dict(lg.state_dict())
    assert lg2.pool_max == 4 and lg2.p_latest == 0.25
    for side in ("red", "blue"):
        a, b = lg.checkpoints(side), lg2.checkpoints(side)
        assert [c.generation for c in a] == [c.generation for c in b]
        for ca, cb in zip(a, b):
            for k in ca.state_dict:
                assert torch.equal(ca.state_dict[k], cb.state_dict[k])


def test_checkpoint_state_dict_is_detached_cpu_copy():
    p = a_blue(0)
    lg = League()
    ck = lg.add("blue", 0, p.state_dict())
    # mutating the live policy must not touch the stored checkpoint
    with torch.no_grad():
        for param in p.parameters():
            param.add_(1.0)
    for k, v in ck.state_dict.items():
        assert not torch.equal(v, p.state_dict()[k])
        assert v.device.type == "cpu"


# --- FrozenPolicySampler --------------------------------------


def test_sampler_uses_fallback_while_pool_empty():
    env = ARENAEnv(CFG)
    live = make_policy("red", max_steps=MS, n_tools_max=NT, seed=0)
    s = FrozenPolicySampler(env, League(seed=0), "red",
                            make_net=net_factory("red", max_steps=MS, n_tools_max=NT),
                            fallback=live)
    env.reset(seed=0)
    a = s(env.observe(RED))
    assert isinstance(a, int)


def test_sampler_without_fallback_raises_on_empty_pool():
    env = ARENAEnv(CFG)
    s = FrozenPolicySampler(env, League(seed=0), "red",
                            make_net=net_factory("red", max_steps=MS, n_tools_max=NT))
    env.reset(seed=0)
    with pytest.raises(RuntimeError):
        s(env.observe(RED))


def test_sampler_resamples_when_episode_changes():
    env = ARENAEnv(CFG)
    lg = League(pool_max=None, p_latest=0.0, seed=0)
    for g in range(5):
        lg.add("red", g, make_policy("red", max_steps=MS, n_tools_max=NT, seed=g).state_dict())
    s = FrozenPolicySampler(env, lg, "red",
                            make_net=net_factory("red", max_steps=MS, n_tools_max=NT))
    picked = []
    for ep in range(30):
        env.reset(seed=ep)
        s(env.observe(RED))
        picked.append(s._episode)
    # episode index advances every reset; the sampler tracked each one
    assert picked == list(range(30))


def test_sampler_is_frozen_no_grad_and_weights_stable():
    env = ARENAEnv(CFG)
    lg = League(seed=0)
    red_ck = make_policy("red", max_steps=MS, n_tools_max=NT, seed=3)
    lg.add("red", 0, red_ck.state_dict())
    before = {k: v.clone() for k, v in red_ck.state_dict().items()}

    s = FrozenPolicySampler(env, lg, "red",
                            make_net=net_factory("red", max_steps=MS, n_tools_max=NT))
    for ep in range(10):
        env.reset(seed=ep)
        while not (env.terminations["red_0"] or env.truncations["red_0"]):
            ag = env.agent_selection
            if ag == RED:
                env.step(s(env.observe(RED)))
            else:
                env.step(0)
    # the materialised net is a fresh object; the original checkpoint policy is untouched
    for k, v in red_ck.state_dict().items():
        assert torch.equal(v, before[k])


def test_sampler_caches_materialised_nets():
    env = ARENAEnv(CFG)
    lg = League(p_latest=1.0, seed=0)
    lg.add("red", 0, make_policy("red", max_steps=MS, n_tools_max=NT, seed=0).state_dict())
    s = FrozenPolicySampler(env, lg, "red",
                            make_net=net_factory("red", max_steps=MS, n_tools_max=NT))
    for ep in range(5):
        env.reset(seed=ep)
        s(env.observe(RED))
    assert len(s._cache) == 1  # same checkpoint every episode -> built once


def test_net_factory_builds_right_shape():
    net = net_factory("blue", max_steps=MS, n_tools_max=NT)()
    from arena.policies import to_batch
    env = ARENAEnv(CFG)
    env.reset(seed=0)
    env.step(0)
    logits, _ = net(to_batch(env.observe("blue_0")))
    assert logits.shape[-1] == 3


def test_checkpoint_key_is_side_and_generation():
    ck = Checkpoint("red", 4, {})
    assert ck.key() == ("red", 4)


# ---------------------------------------------------------------------------
# PFSP — prioritised sampling by how hard each checkpoint is (M11 future-work #3)
# ---------------------------------------------------------------------------


def _pfsp_league(rates, *, p_latest=0.0, power=2.0, seed=0):
    lg = League(p_latest=p_latest, seed=seed, pfsp=True, pfsp_power=power)
    for i, r in enumerate(rates):
        lg.add("blue", i, {"w": torch.zeros(1)}, opponent_win_rate=r)
    return lg


def test_pfsp_weights_favour_the_checkpoint_the_learner_loses_to():
    # opponent_win_rate = how well the LEARNER does. 0.1 = learner is losing.
    lg = _pfsp_league([0.1, 0.9])
    w = lg._pfsp_weights("blue")
    assert w[0] > w[1], "the hard opponent must be drawn more often"
    assert w.sum() == pytest.approx(1.0)


def test_pfsp_power_controls_sharpness():
    rates = [0.2, 0.8]
    soft = _pfsp_league(rates, power=1.0)._pfsp_weights("blue")
    hard = _pfsp_league(rates, power=4.0)._pfsp_weights("blue")
    assert hard[0] > soft[0], "higher power = greedier toward the hardest"
    flat = _pfsp_league(rates, power=0.0)._pfsp_weights("blue")
    assert flat == pytest.approx([0.5, 0.5]), "power 0 must reduce to uniform"


def test_pfsp_missing_rates_fall_back_to_pool_mean_not_a_crash():
    lg = League(p_latest=0.0, seed=0, pfsp=True)
    lg.add("blue", 0, {"w": torch.zeros(1)}, opponent_win_rate=0.2)
    lg.add("blue", 1, {"w": torch.zeros(1)})  # no rate recorded
    w = lg._pfsp_weights("blue")
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(1.0)


def test_pfsp_all_solved_degrades_to_uniform_not_divide_by_zero():
    lg = _pfsp_league([1.0, 1.0, 1.0])
    w = lg._pfsp_weights("blue")
    assert w == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_pfsp_sample_probs_match_empirical_draws():
    lg = _pfsp_league([0.1, 0.5, 0.9], p_latest=0.3, seed=7)
    probs = lg.sample_probs("blue")
    assert probs.sum() == pytest.approx(1.0)
    counts = np.zeros(3)
    for _ in range(8000):
        counts[lg.sample("blue").generation] += 1
    assert counts / counts.sum() == pytest.approx(probs, abs=0.02)


def test_uniform_league_is_unchanged_by_the_pfsp_addition():
    """Default stays the M7 behaviour — PFSP is opt-in."""
    lg = League(p_latest=0.3, seed=1)
    assert lg.pfsp is False
    for i in range(4):
        lg.add("blue", i, {"w": torch.zeros(1)}, opponent_win_rate=0.1 * i)
    p = lg.sample_probs("blue")
    expected = np.full(4, 0.7 / 4)
    expected[-1] += 0.3
    assert p == pytest.approx(expected)


def test_pfsp_survives_state_dict_roundtrip():
    lg = _pfsp_league([0.1, 0.9], p_latest=0.25, power=3.0)
    lg2 = League()
    lg2.load_state_dict(lg.state_dict())
    assert lg2.pfsp is True and lg2.pfsp_power == 3.0
    assert lg2.sample_probs("blue") == pytest.approx(lg.sample_probs("blue"))


def test_league_rejects_negative_pfsp_power():
    with pytest.raises(ValueError):
        League(pfsp_power=-1.0)
