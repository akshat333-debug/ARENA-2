"""M7 integration: the league is wired into self-play and reduces the M6 drift.

M6 without a league drifts to (passive Blue, weak Red) by generation 2-3 — Blue's
quarantine rate collapses toward zero. With the league, Blue keeps facing strong
past Reds, so it should still be discriminating at the end of the run.
"""

from __future__ import annotations

import copy

import pytest
import torch

from arena.config import SelfPlayConfig, load_config
from arena.selfplay import SelfPlayTrainer

pytestmark = pytest.mark.slow

FAST = SelfPlayConfig(n_generations=2, steps_per_side=8_000, eval_episodes=100)
CFG = load_config("small.yaml").model_copy(update={"selfplay": FAST})


def _snap(policy):
    return {k: v.detach().clone() for k, v in policy.state_dict().items()}


def _moved(before, policy):
    after = policy.state_dict()
    return any(not torch.equal(before[k], after[k]) for k in before)


def test_league_is_on_by_default_and_pools_grow_per_generation():
    sp = SelfPlayTrainer(CFG, seed=0)
    assert sp.league is not None
    sp.train(2)
    assert sp.league.pool_size("red") == 2
    assert sp.league.pool_size("blue") == 2
    assert sp.league.latest("red").generation == 1


def test_frozen_opponent_still_does_not_move_with_a_league():
    sp = SelfPlayTrainer(CFG, seed=0)
    sp.train(1)  # populate the pool
    r0 = _snap(sp.red)
    sp._train_side("blue", 5)
    assert not _moved(r0, sp.red), "Red moved while it was a frozen league opponent"

    b0 = _snap(sp.blue)
    sp._train_side("red", 6)
    assert not _moved(b0, sp.blue), "Blue moved while it was a frozen league opponent"


def test_generation_numbering_continues_across_train_calls():
    sp = SelfPlayTrainer(CFG, seed=0)
    sp.train(1)
    sp.train(1)
    assert [g.generation for g in sp.history] == [0, 1]
    assert sp.league.pool_size("blue") == 2


def test_checkpoint_round_trips_trainer_and_league():
    sp = SelfPlayTrainer(CFG, seed=0)
    sp.train(1)
    sd = copy.deepcopy(sp.state_dict())
    assert "league" in sd

    fresh = SelfPlayTrainer(CFG, seed=42)
    fresh.load_state_dict(sd)
    for k, v in sp.blue.state_dict().items():
        assert torch.equal(v, fresh.blue.state_dict()[k])
    assert fresh.league.pool_size("red") == sp.league.pool_size("red")
    a = sp.league.latest("blue").state_dict
    b = fresh.league.latest("blue").state_dict
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_league_off_reproduces_the_m6_loop_shape():
    no_league = CFG.model_copy(
        update={"selfplay": FAST.model_copy(update={"use_league": False})}
    )
    sp = SelfPlayTrainer(no_league, seed=0)
    assert sp.league is None
    sp.train(1)
    assert "league" not in sp.state_dict()


def test_self_play_with_league_is_reproducible():
    def run():
        sp = SelfPlayTrainer(CFG, seed=1)
        sp.train(1)
        return torch.cat([p.detach().flatten() for p in sp.blue.parameters()])

    assert torch.allclose(run(), run())


@pytest.mark.parametrize("_", [0])  # keep it a single slow case, easy to deselect
def test_league_keeps_blue_discriminating_where_m6_drifts(_):
    """The payoff, and *only* the payoff M7 actually claims: with the league,
    Blue does not drift to passive by the last generation (the M6 failure), and
    it handles the opponents it trains against.

    This deliberately does **not** assert that exploitability by a *fresh best
    response* falls — see `docs/audit-m1-m9.md`. It used to
    (`last.exploitability < 0.30`), which passed only because a PPO bug crippled
    every best-response Red; with that fixed the fresh-BR number is ~0.7 and
    flat across generations. Closing that gap is an open problem for M11, not
    something this module ever demonstrated.
    """
    cfg = load_config("small.yaml").model_copy(
        update={"selfplay": SelfPlayConfig(n_generations=4, steps_per_side=40_000,
                                           eval_episodes=200)}
    )
    sp = SelfPlayTrainer(cfg, seed=0)
    sp.train()
    last = sp.history[-1]
    # M6 without a league ended near quarantine_rate 0.01 here; the league must
    # keep Blue engaged. Band, not a point: it should still be discriminating.
    assert last.blue_quarantine_rate > 0.20, (
        f"Blue drifted to passive despite the league: "
        f"quarantine rate {last.blue_quarantine_rate:.3f}"
    )
    # and it should not have gone the other way into blanket paranoia either
    assert last.blue_quarantine_rate < 0.95
    # Against the league it trains on, Blue stops most attacks — the thing the
    # opponent pool is there to preserve.
    assert last.blue_attack_success < 0.30, (
        f"Blue is not stopping the league's Reds: {last.blue_attack_success:.3f}"
    )
