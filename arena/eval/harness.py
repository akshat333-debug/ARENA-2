"""Leaderboard harness — every defender on the same scale (project.md S7).

For each defender: a per-decision detection sweep (AUROC, TPR@5%FPR against the
taint-graph ground truth) and an episode-level exploitability run (best-response
attack success). The defenders are the three M4 baselines and any ARENA-trained
Blue, so the table directly answers "does co-evolution beat the static
baselines".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from arena.baselines.collect import Decision, collect_decisions
from arena.config import ArenaConfig
from arena.eval.exploitability import exploitability
from arena.eval.metrics import roc_auc, tpr_at_fpr
from arena.features import QUARANTINE, VERDICTS
from arena.policies import ActorCritic, to_batch

BluePolicy = Callable[[dict], int]
ScoreFn = Callable[[dict], float]


@dataclass(frozen=True)
class LeaderboardRow:
    name: str
    auroc: float
    tpr_at_5pct_fpr: float
    #: Best-response attack success against this defender at its calibrated
    #: operating point (trainable defenders calibrated to `calibration_fpr` first).
    exploitability: float
    fresh_red_start: float
    #: The false-positive rate the defender actually sits at on the decision set
    #: after calibration — a sanity column, should be ~`calibration_fpr`.
    operating_fpr: float
    n_decisions: int

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "auroc": round(self.auroc, 4),
            "tpr_at_5pct_fpr": round(self.tpr_at_5pct_fpr, 4),
            "exploitability": round(self.exploitability, 4),
            "fresh_red_start": round(self.fresh_red_start, 4),
            "operating_fpr": round(self.operating_fpr, 4),
            "n_decisions": self.n_decisions,
        }


def policy_score_fn(policy: ActorCritic, *, device: str = "cpu") -> ScoreFn:
    """A trained Blue's per-decision detection score: P(intervene) = softmax
    probability the actor assigns to FLAG or QUARANTINE. Lets a learned policy be
    ranked on the same ROC as the rule/statistical baselines."""
    import torch

    q_ix = VERDICTS.index(QUARANTINE)

    def _score(blue_obs: dict) -> float:
        with torch.no_grad():
            logits, _ = policy(to_batch(blue_obs, device))
            p = torch.softmax(logits, dim=-1)[0]
        # intervene = flag + quarantine; quarantine weighted since it is the
        # decisive action, flag is a softer signal.
        return float(p[q_ix] + 0.5 * p[VERDICTS.index(1)])

    return _score


def _score_fn_for(defender) -> ScoreFn:
    if hasattr(defender, "score"):
        return defender.score
    if isinstance(defender, ActorCritic):
        return policy_score_fn(defender)
    if callable(defender):
        # A plain Blue policy exposes no soft score; its "score" is its own hard
        # verdict, so a reference like passive/paranoid gets an honest AUROC of
        # 0.5 (it does not discriminate) rather than an error.
        def _verdict_score(blue_obs: dict) -> float:
            v = defender(blue_obs)
            return 1.0 if v == QUARANTINE else (0.5 if v == VERDICTS.index(1) else 0.0)

        return _verdict_score
    raise TypeError(
        f"don't know how to get a detection score from {type(defender).__name__}; "
        "pass a BlueBaseline, an ActorCritic, or a Blue policy callable"
    )


def _blue_policy_for(defender) -> BluePolicy:
    if isinstance(defender, ActorCritic):
        from arena.policies import TorchPolicyAdapter

        return TorchPolicyAdapter(defender, deterministic=False)
    if callable(defender):
        return defender  # a BlueBaseline is callable
    raise TypeError(f"cannot turn {type(defender).__name__} into a Blue policy")


def _calibrate(defender, decisions: list[Decision], target_fpr: float) -> float:
    """If the defender exposes a tunable ``threshold``, set it to the
    ``1 - target_fpr`` quantile of its scores on the *benign* decisions. Return
    the false-positive rate the defender actually sits at on those decisions —
    from its threshold if it has one, otherwise by running its own verdict."""
    benign = [d for d in decisions if not d.on_attack_path]
    if not benign:
        return float("nan")

    if hasattr(defender, "threshold"):
        score_fn = _score_fn_for(defender)
        neg = np.array([score_fn(d.blue_obs) for d in benign])
        thr = np.nextafter(float(np.quantile(neg, 1.0 - target_fpr)), np.inf)
        defender.threshold = thr
        return float(np.mean(neg >= thr))

    # No tunable threshold (rule-based or a policy): measure the FPR it produces.
    policy = _blue_policy_for(defender)
    fp = sum(1 for d in benign if policy(d.blue_obs) == QUARANTINE)
    return fp / len(benign)


def evaluate_defenders(
    defenders: dict[str, object],
    config: ArenaConfig | None = None,
    *,
    decisions: list[Decision] | None = None,
    n_decision_adv: int | None = None,
    n_decision_benign: int | None = None,
    br_steps: int | None = None,
    n_eval: int | None = None,
    seed: int = 0,
) -> list[LeaderboardRow]:
    cfg = config or ArenaConfig()
    if decisions is None:
        na = n_decision_adv if n_decision_adv is not None else cfg.eval.n_decision_adv
        nb = n_decision_benign if n_decision_benign is not None else cfg.eval.n_decision_benign
        decisions = collect_decisions(na, nb, config=cfg, seed=seed + 7)

    y = [int(d.on_attack_path) for d in decisions]
    rows: list[LeaderboardRow] = []
    for name, defender in defenders.items():
        score_fn = _score_fn_for(defender)
        scores = [score_fn(d.blue_obs) for d in decisions]
        # AUROC / TPR@5%FPR are threshold-free, so compute them first, then
        # calibrate the operating point for the exploitability run.
        auroc = roc_auc(scores, y)
        tpr5 = tpr_at_fpr(scores, y, 0.05)
        op_fpr = _calibrate(defender, decisions, cfg.eval.calibration_fpr)
        expl = exploitability(
            _blue_policy_for(defender), cfg, br_steps=br_steps, n_eval=n_eval, seed=seed,
        )
        rows.append(LeaderboardRow(
            name=name,
            auroc=auroc,
            tpr_at_5pct_fpr=tpr5,
            exploitability=expl.exploitability,
            fresh_red_start=expl.fresh_red_start,
            operating_fpr=op_fpr,
            n_decisions=len(decisions),
        ))
    # Rank by exploitability, then by discrimination (AUROC) as the tie-break.
    rows.sort(key=lambda r: (r.exploitability, -r.auroc))
    return rows


def format_leaderboard(rows: list[LeaderboardRow]) -> str:
    head = (f"{'defender':<20}{'AUROC':>9}{'TPR@5%FPR':>12}"
            f"{'exploitability':>16}{'op.FPR':>9}{'(fresh Red)':>13}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r.name:<20}{r.auroc:>9.3f}{r.tpr_at_5pct_fpr:>12.3f}"
            f"{r.exploitability:>16.3f}{r.operating_fpr:>9.3f}{r.fresh_red_start:>13.3f}"
        )
    return "\n".join(lines)
