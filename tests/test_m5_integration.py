"""M5 integration: PPO actually learns, on both sides of the real game.

These are the M5 gate. They are slow-ish by test standards (tens of seconds) but
they are the only thing that distinguishes "the code runs" from "the trainer
works", and every later module (self-play, league, exploitability) is built on
top of a trainer that must genuinely optimise.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from arena.config import PPOConfig, load_config
from arena.env import ARENAEnv, BLUE, RED, SingleAgentARENA
from arena.policies import TorchPolicyAdapter, make_policy
from arena.ppo import PPOTrainer, evaluate_policy
from arena.scenarios import AttackFamily, ScenarioGenerator
from arena.scripted import AdaptiveScriptedRed, paranoid_blue, passive_blue

CFG = load_config("small.yaml")

pytestmark = pytest.mark.slow


def _blue_env(cfg):
    env = SingleAgentARENA(BLUE, lambda o: 0, config=cfg)
    env._opponent = AdaptiveScriptedRed(env, seed=0)
    return env


def test_red_learns_to_complete_attacks_against_a_passive_blue():
    sc = ScenarioGenerator(seed=0, max_steps=10).sample(
        force_family=AttackFamily.DIRECT_PROMPT_INJECTION
    )
    env = SingleAgentARENA(RED, passive_blue, scenario=sc)
    pol = make_policy("red", max_steps=sc.max_steps, n_tools_max=len(sc.registry), cfg=CFG.policy, seed=0)

    before = evaluate_policy(env, pol, 60, deterministic=False)["attack_success_rate"]
    PPOTrainer(env, pol, CFG.ppo, seed=0).train(15_000)
    after = evaluate_policy(env, pol, 60, deterministic=False)["attack_success_rate"]

    assert after > before + 0.25, f"Red did not learn: {before:.3f} -> {after:.3f}"
    assert after >= 0.4


def test_blue_learns_to_discriminate_rather_than_quarantine_everything():
    """The M5 headline. Blue must beat *both* degenerate policies, and its
    quarantine rate must track the adversarial rate rather than saturating at 1.
    """
    env = _blue_env(CFG)
    pol = make_policy(
        "blue", max_steps=CFG.scenario.max_steps,
        n_tools_max=CFG.scenario.n_tools_max, cfg=CFG.policy, seed=1,
    )
    PPOTrainer(env, pol, CFG.ppo, seed=1).train(60_000)
    r = evaluate_policy(env, pol, 300, deterministic=False)

    # Reference returns on this config, measured: passive ~-0.49, paranoid ~-0.54.
    assert r["mean_return"] > 0.2, f"Blue no better than the trivial policies: {r}"
    assert r["attack_success_rate"] < 0.15, f"Blue is missing attacks: {r}"
    assert 0.3 < r["quarantine_rate"] < 0.75, (
        f"Blue looks degenerate (quarantine rate {r['quarantine_rate']:.3f}); "
        "~0.5 is discrimination, ~1.0 is paranoia, ~0.0 is passivity"
    )


def test_trained_blue_beats_the_scripted_reference_defenders():
    """Same episodes, three defenders, compared on total reward."""
    env = _blue_env(CFG)
    pol = make_policy(
        "blue", max_steps=CFG.scenario.max_steps,
        n_tools_max=CFG.scenario.n_tools_max, cfg=CFG.policy, seed=1,
    )
    PPOTrainer(env, pol, CFG.ppo, seed=1).train(60_000)
    trained = TorchPolicyAdapter(pol, deterministic=True)

    def mean_r_blue(blue, n=200, seed=11):
        e = ARENAEnv(CFG)
        e.reset(seed=seed)
        red = AdaptiveScriptedRed(e, seed=0)
        out = []
        for ep in range(n):
            if ep:
                e.reset()
            for agent in e.agent_iter():
                _, _, t, tc, _ = e.last()
                if t or tc:
                    e.step(None)
                    continue
                e.step(red(e.observe(RED)) if agent == RED else blue(e.observe(BLUE)))
            out.append(e.last_breakdown.r_blue)
        return float(np.mean(out))

    r_trained = mean_r_blue(trained)
    r_passive = mean_r_blue(passive_blue)
    r_paranoid = mean_r_blue(paranoid_blue)
    assert r_trained > r_passive, f"trained {r_trained:.3f} <= passive {r_passive:.3f}"
    assert r_trained > r_paranoid, f"trained {r_trained:.3f} <= paranoid {r_paranoid:.3f}"


def test_trained_policies_plug_in_as_opponents():
    """A trained policy must be usable exactly where a scripted one is — this is
    the interface self-play (M6) and the league (M7) depend on."""
    sc = ScenarioGenerator(seed=1, max_steps=10).sample(
        force_family=AttackFamily.COLLUDING_AGENTS
    )
    blue_net = make_policy("blue", max_steps=sc.max_steps, n_tools_max=len(sc.registry), cfg=CFG.policy, seed=0)
    blue = TorchPolicyAdapter(blue_net)

    # Red trains against the (untrained but real) Blue network as its opponent.
    env = SingleAgentARENA(RED, blue, scenario=sc)
    pol = make_policy("red", max_steps=sc.max_steps, n_tools_max=len(sc.registry), cfg=CFG.policy, seed=0)
    stats = PPOTrainer(env, pol, PPOConfig(n_steps=512, minibatch_size=128), seed=0).train(2048)
    assert stats.steps == 2048 and stats.episode_returns


def test_frozen_opponent_does_not_change_during_training():
    """Self-play correctness depends on 'frozen' actually meaning frozen."""
    sc = ScenarioGenerator(seed=2, max_steps=10).sample(
        force_family=AttackFamily.DIRECT_PROMPT_INJECTION
    )
    blue_net = make_policy("blue", max_steps=sc.max_steps, n_tools_max=len(sc.registry), cfg=CFG.policy, seed=0)
    before = [p.detach().clone() for p in blue_net.parameters()]

    env = SingleAgentARENA(RED, TorchPolicyAdapter(blue_net), scenario=sc)
    red = make_policy("red", max_steps=sc.max_steps, n_tools_max=len(sc.registry), cfg=CFG.policy, seed=0)
    PPOTrainer(env, red, PPOConfig(n_steps=512, minibatch_size=128), seed=0).train(2048)

    for a, b in zip(before, blue_net.parameters()):
        assert torch.equal(a, b), "frozen opponent's weights moved during training"
