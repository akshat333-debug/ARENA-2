"""M5 unit tests: policy networks — shapes, masking, gradients, determinism."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from arena.config import PolicyConfig, load_config
from arena.env import BLUE, RED, ARENAEnv, SingleAgentARENA
from arena.features import ALLOW, VERDICTS
from arena.policies import (
    TorchPolicyAdapter,
    make_policy,
    stack_obs,
    to_batch,
)
from arena.scenarios import AttackFamily, ScenarioGenerator

CFG = load_config("small.yaml")


def a_scenario(seed=0):
    return ScenarioGenerator(seed=seed, max_steps=10).sample(
        force_family=AttackFamily.DIRECT_PROMPT_INJECTION
    )


def red_obs_batch(n=4, seed=0):
    sc = a_scenario(seed)
    env = ARENAEnv(scenario=sc)
    env.reset(seed=seed)
    return sc, [env.observe(RED) for _ in range(n)]


def blue_obs_batch(n=4, seed=0):
    sc = a_scenario(seed)
    env = ARENAEnv(scenario=sc)
    env.reset(seed=seed)
    env.step(0)
    return sc, [env.observe(BLUE) for _ in range(n)]


# --- shapes -----------------------------------------------------------


def test_blue_forward_shapes():
    sc, obs = blue_obs_batch(5)
    pol = make_policy("blue", max_steps=sc.max_steps, n_tools_max=len(sc.registry))
    logits, value = pol(stack_obs(obs))
    assert logits.shape == (5, len(VERDICTS))
    assert value.shape == (5,)


def test_red_forward_shapes():
    sc, obs = red_obs_batch(5)
    n = len(sc.registry)
    pol = make_policy("red", max_steps=sc.max_steps, n_tools_max=n)
    logits, value = pol(stack_obs(obs))
    assert logits.shape == (5, n)
    assert value.shape == (5,)


def test_act_returns_valid_actions():
    sc, obs = blue_obs_batch(3)
    pol = make_policy("blue", max_steps=sc.max_steps, n_tools_max=len(sc.registry))
    action, logprob, value = pol.act(stack_obs(obs))
    assert action.shape == (3,)
    assert all(int(a) in VERDICTS for a in action)


# --- masking ----------------------------------------------------------


def test_red_never_samples_a_masked_action():
    """Red's Discrete space is n_tools_max; a smaller registry masks the tail.
    Sampling outside the mask would index a tool that does not exist."""
    sc = a_scenario()
    n_real = len(sc.registry)
    n_max = n_real + 5
    env = ARENAEnv(scenario=sc)
    env.reset(seed=0)
    obs = env.observe(RED)
    # widen the observation to a larger action space, mask still marks n_real
    padded = {
        "task": obs["task"],
        "registry": np.pad(obs["registry"], ((0, 5), (0, 0))),
        "registry_mask": np.pad(obs["registry_mask"], (0, 5)),
        "objective": np.zeros(4 + 2 * n_max, dtype=np.float32),
        "history": obs["history"],
        "length": obs["length"],
    }
    pol = make_policy("red", max_steps=sc.max_steps, n_tools_max=n_max)
    batch = stack_obs([padded] * 200)
    action, _, _ = pol.act(batch)
    assert int(action.max()) < n_real, "sampled a masked-out tool index"


def test_red_masked_logits_are_minus_inf():
    sc = a_scenario()
    n_max = len(sc.registry) + 3
    pol = make_policy("red", max_steps=sc.max_steps, n_tools_max=n_max)
    env = ARENAEnv(scenario=sc)
    env.reset(seed=0)
    obs = env.observe(RED)
    padded = {
        "task": obs["task"],
        "registry": np.pad(obs["registry"], ((0, 3), (0, 0))),
        "registry_mask": np.pad(obs["registry_mask"], (0, 3)),
        "objective": np.zeros(4 + 2 * n_max, dtype=np.float32),
        "history": obs["history"],
        "length": obs["length"],
    }
    logits, _ = pol(to_batch(padded))
    assert torch.isneginf(logits[0, len(sc.registry):]).all() or \
        (logits[0, len(sc.registry):] <= torch.finfo(logits.dtype).min).all()


# --- gradients --------------------------------------------------------


@pytest.mark.parametrize("side", ["red", "blue"])
def test_gradients_flow_to_every_parameter(side):
    sc = a_scenario()
    obs = (red_obs_batch if side == "red" else blue_obs_batch)(4)[1]
    pol = make_policy(side, max_steps=sc.max_steps, n_tools_max=len(sc.registry))
    batch = stack_obs(obs)
    logits, value = pol(batch)
    loss = logits.sum() + value.sum()
    loss.backward()
    dead = [n for n, p in pol.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not dead, f"no/!finite gradient for: {dead}"


@pytest.mark.parametrize("side", ["red", "blue"])
def test_evaluate_matches_act_logprobs(side):
    sc = a_scenario()
    obs = (red_obs_batch if side == "red" else blue_obs_batch)(6)[1]
    pol = make_policy(side, max_steps=sc.max_steps, n_tools_max=len(sc.registry))
    batch = stack_obs(obs)
    action, logprob, _ = pol.act(batch)
    lp2, entropy, _ = pol.evaluate(batch, action)
    assert torch.allclose(logprob, lp2, atol=1e-5)
    assert (entropy >= 0).all()


# --- determinism ------------------------------------------------------


def test_deterministic_act_is_argmax():
    sc, obs = blue_obs_batch(4)
    pol = make_policy("blue", max_steps=sc.max_steps, n_tools_max=len(sc.registry))
    batch = stack_obs(obs)
    logits, _ = pol(batch)
    action, _, _ = pol.act(batch, deterministic=True)
    assert torch.equal(action, logits.argmax(-1))


def test_same_seed_same_init():
    sc = a_scenario()
    torch.manual_seed(0)
    a = make_policy("blue", max_steps=sc.max_steps, n_tools_max=len(sc.registry))
    torch.manual_seed(0)
    b = make_policy("blue", max_steps=sc.max_steps, n_tools_max=len(sc.registry))
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


# --- adapter ----------------------------------------------------------


def test_adapter_is_a_drop_in_policy():
    sc = a_scenario()
    pol = make_policy("blue", max_steps=sc.max_steps, n_tools_max=len(sc.registry))
    blue = TorchPolicyAdapter(pol)
    env = SingleAgentARENA(RED, blue, scenario=sc)
    obs, _ = env.reset(seed=0)
    obs, r, term, trunc, info = env.step(0)
    assert isinstance(r, float)


def test_adapter_returns_plain_int():
    sc = a_scenario()
    pol = make_policy("blue", max_steps=sc.max_steps, n_tools_max=len(sc.registry))
    _, obs = blue_obs_batch(1)
    v = TorchPolicyAdapter(pol)(obs[0])
    assert isinstance(v, int) and v in VERDICTS


# --- config / invalid input ------------------------------------------


def test_policy_sizes_come_from_config():
    sc = a_scenario()
    small = make_policy("blue", max_steps=sc.max_steps, n_tools_max=6, cfg=PolicyConfig(hidden=16, gru_hidden=16))
    big = make_policy("blue", max_steps=sc.max_steps, n_tools_max=6, cfg=PolicyConfig(hidden=128, gru_hidden=128))
    assert sum(p.numel() for p in small.parameters()) < sum(p.numel() for p in big.parameters())


def test_make_policy_rejects_unknown_side():
    with pytest.raises(ValueError, match="red.*blue|side"):
        make_policy("green", max_steps=8, n_tools_max=6)


def test_zero_length_observation_is_handled():
    """At episode start Blue has seen nothing; the net must still produce a
    finite verdict rather than indexing off the front of the sequence."""
    sc = a_scenario()
    env = ARENAEnv(scenario=sc)
    env.reset(seed=0)
    obs = env.observe(BLUE)
    assert int(obs["length"][0]) == 0
    pol = make_policy("blue", max_steps=sc.max_steps, n_tools_max=len(sc.registry))
    logits, value = pol(to_batch(obs))
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


# ---------------------------------------------------------------------------
# blue_causal_features (M11 future-work #5)
# ---------------------------------------------------------------------------


def _blue_obs(seed=0, steps=3):
    from arena.config import load_config
    from arena.env import BLUE, ARENAEnv

    cfg = load_config("small.yaml")
    env = ARENAEnv(cfg)
    env.reset(seed=seed)
    for _ in range(steps):
        env.step(0)          # red proposes
        env.step(ALLOW)      # blue allows
    env.step(0)
    return cfg, env.observe(BLUE)


def test_causal_features_widen_the_trunk_by_exactly_sequence_feats():
    from arena.features import SEQUENCE_FEATS

    cfg, _ = _blue_obs()
    off = make_policy("blue", max_steps=cfg.scenario.max_steps,
                      n_tools_max=cfg.scenario.n_tools_max,
                      cfg=PolicyConfig(blue_causal_features=False), seed=0)
    on = make_policy("blue", max_steps=cfg.scenario.max_steps,
                     n_tools_max=cfg.scenario.n_tools_max,
                     cfg=PolicyConfig(blue_causal_features=True), seed=0)
    assert on.net.trunk[0].in_features - off.net.trunk[0].in_features == SEQUENCE_FEATS


def test_causal_blue_produces_valid_output_and_gradients():
    cfg, obs = _blue_obs()
    pol = make_policy("blue", max_steps=cfg.scenario.max_steps,
                      n_tools_max=cfg.scenario.n_tools_max,
                      cfg=PolicyConfig(blue_causal_features=True), seed=0)
    logits, value = pol(to_batch(obs))
    assert logits.shape == (1, len(VERDICTS)) and value.shape == (1,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    (logits.sum() + value.sum()).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in pol.parameters())


def test_causal_blue_is_off_by_default():
    assert PolicyConfig().blue_causal_features is False
    cfg, obs = _blue_obs()
    pol = make_policy("blue", max_steps=cfg.scenario.max_steps,
                      n_tools_max=cfg.scenario.n_tools_max, seed=0)
    assert pol.net.causal is False


def test_causal_features_actually_reach_the_trunk():
    """Guard against the flag widening the layer while feeding it zeros — the
    silent no-op version of this feature."""
    from arena.features import sequence_features_torch

    cfg, obs = _blue_obs(steps=4)
    batch = to_batch(obs)
    feats = sequence_features_torch(batch["calls"], batch["length"])
    assert torch.count_nonzero(feats) > 0, "fixture produced an empty summary"

    pol = make_policy("blue", max_steps=cfg.scenario.max_steps,
                      n_tools_max=cfg.scenario.n_tools_max,
                      cfg=PolicyConfig(blue_causal_features=True), seed=0)
    base, _ = pol(batch)
    # zero only the causal slice of the trunk's weights; output must move
    with torch.no_grad():
        pol.net.trunk[0].weight[:, -feats.shape[1]:] = 0.0
    muted, _ = pol(batch)
    assert not torch.allclose(base, muted), "causal features had no effect on the output"
