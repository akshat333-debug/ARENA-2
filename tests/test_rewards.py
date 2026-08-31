"""M2 unit tests: asymmetric reward engine."""

from __future__ import annotations

import pytest

from arena.config import RewardConfig
from arena.rewards import EpisodeOutcome, compute_rewards, heuristic_plausibility
from arena.tools import COMMON_TOOLS

W = RewardConfig()
BY_NAME = {t.name: t for t in COMMON_TOOLS}


# --- valid input: the three canonical outcomes -------------------------


def test_red_wins_undetected_exfiltration():
    o = EpisodeOutcome(
        is_adversarial=True, objective_completed=True, completion_step=2,
        n_steps=3, plausibility=1.0, malicious_steps=(0, 1, 2),
    )
    rb = compute_rewards(o, W)
    assert rb.red_evade == W.w_evade
    assert rb.blue_miss == W.w_miss
    assert rb.blue_detect == 0.0
    assert rb.r_red > 0 > rb.r_blue


def test_blue_wins_quarantine_before_completion():
    o = EpisodeOutcome(
        is_adversarial=True, objective_completed=False, n_steps=3, plausibility=1.0,
        quarantined=True, quarantine_step=2, flagged_steps=(1,), malicious_steps=(0, 1, 2),
        caught_in_flight=True,
    )
    rb = compute_rewards(o, W)
    assert rb.blue_detect == W.w_detect
    assert rb.blue_flag == W.w_flag       # step 1 was on the chain
    assert rb.red_evade == 0.0            # attack denied: no evasion payout
    assert rb.r_blue > rb.r_red           # Red keeps only its plausibility credit


def test_false_positive_on_benign_session():
    o = EpisodeOutcome(
        is_adversarial=False, objective_completed=False, n_steps=4, plausibility=1.0,
        quarantined=True, quarantine_step=1,
    )
    rb = compute_rewards(o, W)
    assert rb.blue_false_quarantine == W.w_false_quarantine
    assert rb.r_blue == -W.w_false_quarantine


def test_partial_credit_flag_without_quarantine():
    o = EpisodeOutcome(
        is_adversarial=True, objective_completed=True, completion_step=3,
        n_steps=4, plausibility=1.0, flagged_steps=(2,), malicious_steps=(0, 2, 3),
    )
    rb = compute_rewards(o, W)
    assert rb.blue_flag == W.w_flag
    assert rb.blue_miss == W.w_miss
    # flag credit does not cover the miss
    assert rb.r_blue == W.w_flag - W.w_miss


def test_wrong_flags_are_penalised_even_on_adversarial_episodes():
    o = EpisodeOutcome(
        is_adversarial=True, objective_completed=False, n_steps=5, plausibility=1.0,
        quarantined=True, quarantine_step=4, flagged_steps=(1, 2), malicious_steps=(3, 4),
    )
    rb = compute_rewards(o, W)
    assert rb.blue_false_flag == 2 * W.w_false_flag
    assert rb.blue_flag == 0.0  # neither flag was on the chain


def test_step_cost_scales_with_length():
    short = compute_rewards(EpisodeOutcome(True, True, 2, 1.0, completion_step=1, malicious_steps=(0, 1)), W)
    long = compute_rewards(EpisodeOutcome(True, True, 10, 1.0, completion_step=1, malicious_steps=(0, 1)), W)
    assert long.red_step_cost > short.red_step_cost
    assert long.r_red < short.r_red


def test_plausibility_feeds_red_reward_linearly():
    lo = compute_rewards(EpisodeOutcome(True, False, 3, 0.0), W)
    hi = compute_rewards(EpisodeOutcome(True, False, 3, 1.0), W)
    assert hi.red_plausibility - lo.red_plausibility == pytest.approx(W.w_plausibility)


def test_zero_weights_zero_reward():
    z = RewardConfig(w_evade=0, w_plausibility=0, w_step=0, w_detect=0, w_flag=0,
                     w_miss=0, w_false_quarantine=0, w_false_flag=0)
    o = EpisodeOutcome(True, True, 3, 1.0, completion_step=2, quarantined=True,
                       quarantine_step=2, flagged_steps=(0,), malicious_steps=(1, 2))
    rb = compute_rewards(o, z)
    assert rb.r_red == 0.0 and rb.r_blue == 0.0


# --- invalid input ----------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(is_adversarial=True, objective_completed=False, n_steps=-1, plausibility=1.0),
        dict(is_adversarial=True, objective_completed=False, n_steps=1, plausibility=1.5),
        dict(is_adversarial=True, objective_completed=False, n_steps=1, plausibility=-0.1),
        dict(is_adversarial=True, objective_completed=True, completion_step=None, n_steps=1, plausibility=1.0),
        dict(is_adversarial=False, objective_completed=True, completion_step=1, n_steps=1, plausibility=1.0),
        dict(is_adversarial=True, objective_completed=False, n_steps=1, plausibility=1.0,
             quarantined=True, quarantine_step=None),
        dict(is_adversarial=False, objective_completed=False, n_steps=1, plausibility=1.0,
             malicious_steps=(0,)),
    ],
)
def test_episode_outcome_validates(kwargs):
    with pytest.raises(ValueError):
        EpisodeOutcome(**kwargs)


def test_negative_reward_weight_rejected():
    with pytest.raises(ValueError):
        RewardConfig(w_evade=-1.0)


# --- edge cases -----------------------------------------------------


def test_benign_session_left_alone_costs_blue_nothing():
    o = EpisodeOutcome(is_adversarial=False, objective_completed=False, n_steps=5, plausibility=1.0)
    assert compute_rewards(o, W).r_blue == 0.0


def test_adversarial_completed_but_also_quarantined_after_is_still_a_miss():
    # env recorded completion at step 2; a late quarantine at step 3 cannot un-leak it
    o = EpisodeOutcome(
        is_adversarial=True, objective_completed=True, completion_step=2, n_steps=4,
        plausibility=1.0, quarantined=True, quarantine_step=3, malicious_steps=(0, 1, 2),
    )
    rb = compute_rewards(o, W)
    assert rb.blue_detect == 0.0
    assert rb.blue_miss == W.w_miss


def test_heuristic_plausibility_is_monotone_in_risky_fraction():
    reads = [BY_NAME["read_file"]] * 4
    one_sink = reads + [BY_NAME["post_webhook"]]
    all_risky = [BY_NAME["read_env_file"], BY_NAME["post_webhook"]]
    assert heuristic_plausibility(reads) == 1.0
    assert heuristic_plausibility(one_sink) >= heuristic_plausibility(all_risky)
    assert 0.0 <= heuristic_plausibility(all_risky) < 1.0


def test_heuristic_plausibility_empty_sequence():
    assert heuristic_plausibility([]) == 1.0


# ---------------------------------------------------------------------------
# dense_flag_credit (M11 future-work #4) — coverage-proportional flag reward
# ---------------------------------------------------------------------------


def _flagged(flags, malicious=(0, 1, 2)):
    return EpisodeOutcome(
        is_adversarial=True, objective_completed=False, n_steps=4, plausibility=1.0,
        quarantined=True, quarantine_step=3, flagged_steps=flags,
        malicious_steps=malicious, caught_in_flight=True,
    )


def test_flat_flag_credit_is_the_default_and_is_binary():
    w = RewardConfig()
    assert w.dense_flag_credit is False
    one = compute_rewards(_flagged((0,)), w).blue_flag
    all3 = compute_rewards(_flagged((0, 1, 2)), w).blue_flag
    assert one == all3 == w.w_flag


def test_dense_flag_credit_is_monotone_in_chain_coverage():
    """The reason this exists: with flat credit the marginal value of a second
    correct flag is zero while a wrong flag still costs, so Blue's cheapest
    policy is to flag once and stop looking."""
    w = RewardConfig(dense_flag_credit=True)
    vals = [compute_rewards(_flagged(f), w).blue_flag for f in [(0,), (0, 1), (0, 1, 2)]]
    assert vals == sorted(vals) and vals[0] < vals[-1]
    assert vals == pytest.approx([w.w_flag / 3, 2 * w.w_flag / 3, w.w_flag])


def test_dense_flag_credit_is_capped_at_w_flag():
    w = RewardConfig(dense_flag_credit=True)
    # every malicious step flagged, plus the cap cannot be exceeded
    assert compute_rewards(_flagged((0, 1, 2)), w).blue_flag == pytest.approx(w.w_flag)


def test_dense_flag_credit_still_charges_for_off_chain_flags():
    w = RewardConfig(dense_flag_credit=True)
    rb = compute_rewards(_flagged((0, 7, 8)), w)
    assert rb.blue_flag == pytest.approx(w.w_flag / 3)
    assert rb.blue_false_flag == pytest.approx(2 * w.w_false_flag)


def test_dense_flag_credit_preserves_the_degenerate_defender_tie():
    """The invariant the whole reward design rests on: quarantine-always and
    allow-always must score exactly equally badly, or the trivial strategy wins.
    Flags do not enter it (neither degenerate policy flags), but assert it under
    the new mode rather than assuming."""
    for w in (RewardConfig(), RewardConfig(dense_flag_credit=True)):
        p = 0.5
        quarantine_always = p * w.w_detect - (1 - p) * w.w_false_quarantine
        allow_always = -p * w.w_miss
        assert quarantine_always == pytest.approx(allow_always)


def test_dense_flag_credit_with_no_malicious_steps_does_not_divide_by_zero():
    w = RewardConfig(dense_flag_credit=True)
    o = EpisodeOutcome(
        is_adversarial=True, objective_completed=False, n_steps=2, plausibility=1.0,
        quarantined=True, quarantine_step=1, flagged_steps=(0,), malicious_steps=(),
    )
    rb = compute_rewards(o, w)
    assert rb.blue_flag == 0.0
    assert rb.blue_false_flag == pytest.approx(w.w_false_flag)
