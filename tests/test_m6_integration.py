"""M6 integration: the alternating self-play loop actually runs and alternates.

The M6 gate: one full generation completes on small.yaml in minutes, and the
frozen side's weights provably do not move while the other side trains.
"""

from __future__ import annotations

import copy

import pytest
import torch

from arena.config import SelfPlayConfig, load_config
from arena.selfplay import SelfPlayTrainer

pytestmark = pytest.mark.slow

# Small enough to run in a test, big enough that both sides genuinely update.
FAST = SelfPlayConfig(n_generations=2, steps_per_side=8_000, eval_episodes=100)
CFG = load_config("small.yaml").model_copy(update={"selfplay": FAST})


def _snapshot(policy):
    return {k: v.detach().clone() for k, v in policy.state_dict().items()}


def _moved(before, policy):
    after = policy.state_dict()
    return any(not torch.equal(before[k], after[k]) for k in before)


def test_one_generation_completes_and_records_stats():
    sp = SelfPlayTrainer(CFG, seed=0)
    hist = sp.train(1)
    assert len(hist) == 1
    g = hist[0]
    assert g.generation == 0
    assert 0.0 <= g.exploitability <= 1.0
    assert 0.0 <= g.blue_quarantine_rate <= 1.0
    assert torch.isfinite(torch.tensor(g.blue_return_vs_red))
    assert g.red_train.steps == FAST.steps_per_side
    assert g.blue_train.steps == FAST.steps_per_side


def test_both_policies_change_over_a_generation():
    sp = SelfPlayTrainer(CFG, seed=0)
    r0, b0 = _snapshot(sp.red), _snapshot(sp.blue)
    sp.train(1)
    assert _moved(r0, sp.red), "Red did not update"
    assert _moved(b0, sp.blue), "Blue did not update"


def test_frozen_blue_does_not_move_while_red_trains():
    sp = SelfPlayTrainer(CFG, seed=0)
    b0 = _snapshot(sp.blue)
    sp._train_side("red", 0)
    assert not _moved(b0, sp.blue), "Blue's weights moved while it was the frozen opponent"


def test_frozen_red_does_not_move_while_blue_trains():
    sp = SelfPlayTrainer(CFG, seed=0)
    r0 = _snapshot(sp.red)
    sp._train_side("blue", 0)
    assert not _moved(r0, sp.red), "Red's weights moved while it was the frozen opponent"


def test_blue_does_not_collapse_to_blanket_quarantine_in_self_play():
    """The M5 failure mode must not resurface once the opponent is a learned Red."""
    sp = SelfPlayTrainer(CFG, seed=0)
    hist = sp.train(2)
    last = hist[-1]
    assert last.blue_quarantine_rate < 0.95, (
        f"Blue looks degenerate: quarantine rate {last.blue_quarantine_rate:.3f}"
    )


def test_self_play_run_is_reproducible():
    def run():
        sp = SelfPlayTrainer(CFG, seed=1)
        sp.train(1)
        return torch.cat([p.detach().flatten() for p in sp.blue.parameters()])

    a, b = run(), run()
    assert torch.allclose(a, b)


def test_checkpoint_round_trips_a_trained_pair():
    sp = SelfPlayTrainer(CFG, seed=0)
    sp.train(1)
    sd = copy.deepcopy(sp.state_dict())
    fresh = SelfPlayTrainer(CFG, seed=42)
    fresh.load_state_dict(sd)
    for k, v in sp.blue.state_dict().items():
        assert torch.equal(v, fresh.blue.state_dict()[k])
