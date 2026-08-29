"""M5 unit tests: PPO internals — GAE, truncation bootstrap, the update step."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from arena.config import PPOConfig, load_config
from arena.env import RED, SingleAgentARENA
from arena.policies import make_policy
from arena.ppo import PPOTrainer, RolloutBuffer, compute_gae, resolve_device
from arena.scenarios import AttackFamily, ScenarioGenerator
from arena.scripted import passive_blue

CFG = load_config("small.yaml")


def a_scenario(seed=0, max_steps=10):
    return ScenarioGenerator(seed=seed, max_steps=max_steps).sample(
        force_family=AttackFamily.DIRECT_PROMPT_INJECTION
    )


def a_trainer(total=512, **ppo_kw):
    sc = a_scenario()
    env = SingleAgentARENA(RED, passive_blue, scenario=sc)
    pol = make_policy("red", max_steps=sc.max_steps, n_tools_max=len(sc.registry))
    cfg = PPOConfig(n_steps=256, minibatch_size=64, total_steps=total, **ppo_kw)
    return PPOTrainer(env, pol, cfg, seed=0), sc


# --- GAE --------------------------------------------------------------


def test_gae_terminal_has_no_bootstrap():
    r = np.array([0.0, 1.0])
    v = np.array([0.5, 0.5])
    term = np.array([False, True])
    adv, ret = compute_gae(r, v, term, [None, None], 0.0, gamma=1.0, lam=1.0)
    # last step terminates: delta = 1.0 + 0 - 0.5
    assert adv[-1] == pytest.approx(0.5)
    assert ret[-1] == pytest.approx(1.0)


def test_gae_truncation_bootstraps_from_the_cut_value():
    """A step-cap truncation is NOT a terminal state. Without bootstrapping, the
    value function learns that running out of time is as bad as being caught."""
    r = np.array([0.0])
    v = np.array([0.0])
    term = np.array([False])
    trunc_only, _ = compute_gae(r, v, term, [None], 0.0, gamma=0.99, lam=0.95)
    bootstrapped, _ = compute_gae(r, v, term, [5.0], 0.0, gamma=0.99, lam=0.95)
    assert bootstrapped[0] > trunc_only[0]
    assert bootstrapped[0] == pytest.approx(0.99 * 5.0)


def test_gae_trace_resets_at_episode_boundaries():
    """Advantage must not leak across episodes."""
    r = np.array([1.0, 1.0, 1.0])
    v = np.zeros(3)
    term = np.array([False, True, False])
    adv, _ = compute_gae(r, v, term, [None, None, None], 0.0, gamma=1.0, lam=1.0)
    # step 1 terminates -> its advantage is just its own delta
    assert adv[1] == pytest.approx(1.0)
    # step 0 therefore cannot see step 2's reward
    assert adv[0] == pytest.approx(2.0)


def test_gae_matches_hand_computation_no_boundaries():
    r = np.array([1.0, 2.0])
    v = np.array([0.0, 0.0])
    term = np.array([False, False])
    adv, ret = compute_gae(r, v, term, [None, None], 3.0, gamma=0.5, lam=1.0)
    # t=1: delta = 2 + 0.5*3 - 0 = 3.5
    # t=0: delta = 1 + 0.5*0 - 0 = 1.0 ; adv = 1.0 + 0.5*1*3.5 = 2.75
    assert adv[1] == pytest.approx(3.5)
    assert adv[0] == pytest.approx(2.75)
    assert ret == pytest.approx(adv + v)


# --- buffer -----------------------------------------------------------


def test_buffer_add_and_clear():
    b = RolloutBuffer(4)
    b.add({"x": 1}, 0, -0.5, 1.0, 0.2, terminal=False, truncated_value=None)
    assert len(b) == 1
    b.clear()
    assert len(b) == 0


# --- device -----------------------------------------------------------


def test_resolve_device_auto_is_cpu():
    assert resolve_device("auto").type == "cpu"


def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_ppo_config_rejects_bad_device():
    with pytest.raises(ValueError):
        PPOConfig(device="cuda")


def test_ppo_config_rejects_minibatch_larger_than_rollout():
    with pytest.raises(ValueError, match="minibatch_size"):
        PPOConfig(n_steps=64, minibatch_size=128)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_runs_on_mps():
    trainer, _ = a_trainer(total=256)
    trainer.device = torch.device("mps")
    trainer.policy = trainer.policy.to(trainer.device)
    stats = trainer.train(256)
    assert stats.steps == 256


# --- training mechanics ----------------------------------------------


def test_train_runs_and_updates_parameters():
    trainer, _ = a_trainer(total=512)
    before = [p.detach().clone() for p in trainer.policy.parameters()]
    stats = trainer.train(512)
    after = list(trainer.policy.parameters())
    assert stats.steps == 512 and stats.updates >= 1
    assert any(not torch.equal(a, b) for a, b in zip(before, after)), "no parameter moved"


def test_train_records_episode_returns():
    trainer, _ = a_trainer(total=512)
    stats = trainer.train(512)
    assert stats.episode_returns
    assert np.isfinite(stats.mean_return)


def test_losses_are_finite():
    trainer, _ = a_trainer(total=512)
    stats = trainer.train(512)
    for v in (stats.policy_loss, stats.value_loss, stats.entropy):
        assert np.isfinite(v)


def test_training_is_reproducible_under_seed():
    """Two runs with the same seed must land on identical weights — without any
    help from the caller seeding globals."""

    def run():
        sc = a_scenario()
        env = SingleAgentARENA(RED, passive_blue, scenario=sc)
        pol = make_policy("red", max_steps=sc.max_steps, n_tools_max=len(sc.registry), seed=0)
        PPOTrainer(env, pol, PPOConfig(n_steps=256, minibatch_size=64), seed=0).train(512)
        return torch.cat([p.detach().flatten() for p in pol.parameters()])

    assert torch.allclose(run(), run())


def test_training_does_not_depend_on_ambient_global_rng():
    """A run must not change because something else touched np.random/torch first.
    This is what makes a reported experiment reproducible."""

    def run():
        sc = a_scenario()
        env = SingleAgentARENA(RED, passive_blue, scenario=sc)
        pol = make_policy("red", max_steps=sc.max_steps, n_tools_max=len(sc.registry), seed=0)
        PPOTrainer(env, pol, PPOConfig(n_steps=256, minibatch_size=64), seed=0).train(512)
        return torch.cat([p.detach().flatten() for p in pol.parameters()])

    np.random.seed(1234)
    torch.manual_seed(4321)
    a = run()
    np.random.seed(9)
    torch.manual_seed(99)
    b = run()
    assert torch.allclose(a, b)


def test_callback_is_invoked_each_update():
    trainer, _ = a_trainer(total=512)
    seen = []
    trainer.train(512, callback=lambda s: seen.append(s.updates))
    assert seen == list(range(1, len(seen) + 1))
