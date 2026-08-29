"""M6 unit tests: self-play wiring (no training)."""

from __future__ import annotations

import pytest
import torch

from arena.config import SelfPlayConfig, load_config
from arena.env import RED, ARENAEnv
from arena.features import VERDICTS
from arena.scenarios import AttackFamily, EpisodeType, ScenarioGenerator
from arena.scripted import AdaptiveRed, AdaptiveScriptedRed, BenignRoller
from arena.selfplay import GenerationStats, SelfPlayTrainer

CFG = load_config("small.yaml")


def test_trainer_builds_both_policies_with_config_sizes():
    sp = SelfPlayTrainer(CFG, seed=0)
    nt = CFG.scenario.n_tools_max
    # red action logits must span n_tools_max, blue must span the verdicts
    env_r = ARENAEnv(CFG)
    env_r.reset(seed=0)
    from arena.policies import to_batch

    rl, _ = sp.red(to_batch(env_r.observe(RED)))
    assert rl.shape[-1] == nt
    env_r.step(0)
    bl, _ = sp.blue(to_batch(env_r.observe("blue_0")))
    assert bl.shape[-1] == len(VERDICTS)


def test_red_and_blue_start_from_different_seeds():
    sp = SelfPlayTrainer(CFG, seed=0)
    # they are different networks; a couple of parameters should differ
    rp = torch.cat([p.flatten() for p in sp.red.parameters()])
    bp = torch.cat([p.flatten() for p in sp.blue.parameters()])
    assert rp.shape != bp.shape or not torch.equal(rp[: bp.numel()], bp)


def test_state_dict_round_trips():
    a = SelfPlayTrainer(CFG, seed=0)
    b = SelfPlayTrainer(CFG, seed=99)
    b.load_state_dict(a.state_dict())
    for k, v in a.red.state_dict().items():
        assert torch.equal(v, b.red.state_dict()[k])
    for k, v in a.blue.state_dict().items():
        assert torch.equal(v, b.blue.state_dict()[k])


def test_generation_stats_summary_is_a_string():
    from arena.ppo import TrainStats

    gs = GenerationStats(0, TrainStats(), TrainStats(), 0.4, -0.1, 0.5, 0.1)
    assert isinstance(gs.summary(), str) and "gen 0" in gs.summary()


# --- AdaptiveRed with a fixed attacker (the self-play opponent) ------


class _FixedAttacker:
    """Always returns action 3 — a stand-in for a frozen learned Red."""

    def __call__(self, obs: dict) -> int:
        return 3


def test_adaptive_red_uses_the_fixed_attacker_on_adversarial_episodes():
    gen = ScenarioGenerator.from_config(CFG.scenario, seed=1)
    sc = gen.sample(force_family=AttackFamily.DIRECT_PROMPT_INJECTION)
    env = ARENAEnv(CFG, scenario=sc)
    opp = AdaptiveRed(env, seed=0, attacker=_FixedAttacker())
    env.reset(seed=0)
    assert opp(env.observe(RED)) == 3
    assert opp._inner is not None and not isinstance(opp._inner, BenignRoller)


def test_adaptive_red_uses_benign_roller_on_benign_episodes():
    gen = ScenarioGenerator.from_config(CFG.scenario, seed=2)
    # find a benign scenario
    sc = None
    for _ in range(50):
        cand = gen.sample()
        if cand.episode_type is EpisodeType.BENIGN:
            sc = cand
            break
    assert sc is not None
    env = ARENAEnv(CFG, scenario=sc)
    opp = AdaptiveRed(env, seed=0, attacker=_FixedAttacker())
    env.reset(seed=0)
    opp(env.observe(RED))  # benign branch -> BenignRoller, not the fixed attacker
    assert isinstance(opp._inner, BenignRoller)


def test_adaptive_scripted_red_alias_still_works():
    assert AdaptiveScriptedRed is AdaptiveRed
    env = ARENAEnv(CFG)
    env.reset(seed=0)
    opp = AdaptiveScriptedRed(env, seed=0)  # no attacker -> scripted
    a = opp(env.observe(RED))
    assert isinstance(a, int)


# --- config --------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {"n_generations": 0},
    {"steps_per_side": 0},
    {"eval_episodes": 0},
])
def test_selfplay_config_validates(kwargs):
    with pytest.raises(ValueError):
        SelfPlayConfig(**kwargs)


def test_selfplay_config_defaults_load():
    assert load_config("small.yaml").selfplay.n_generations >= 1
    assert load_config("paper.yaml").selfplay.steps_per_side >= 1
