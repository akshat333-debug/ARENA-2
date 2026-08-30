"""Light scoring for baselines — enough to demonstrate the M4 claim.

The full metric harness (exploitability curves, AUROC / TPR@5%FPR leaderboard vs
ARENA-trained defenders) is M8. Here: attack-success rate, false-quarantine rate,
and a plain AUROC over recorded decisions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from arena.baselines.collect import Decision
from arena.config import ArenaConfig
from arena.env import RED, ARENAEnv
from arena.eval.metrics import roc_auc
from arena.eval.metrics import tpr_at_fpr as _tpr_at_fpr
from arena.scenarios import ScenarioGenerator
from arena.scripted import BenignRoller, ScriptedAttacker

BluePolicy = Callable[[dict], int]


@dataclass(frozen=True)
class BaselineScore:
    name: str
    attack_success_rate: float  # attacks that completed despite the defender (== exploitability proxy)
    false_quarantine_rate: float  # benign SESSIONS the defender quarantined (episode-level)
    auroc: float  # per-decision, scores vs ground-truth on_attack_path
    tpr_at_5pct_fpr: float  # per-decision TPR at a 5% per-decision FPR — the M8 headline metric
    n_adversarial: int
    n_benign: int


def _play(env: ARENAEnv, red: BluePolicy, blue: BluePolicy, seed: int) -> None:
    env.reset(seed=seed)
    for agent in env.agent_iter():
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        obs = env.observe(agent)
        env.step(red(obs) if agent == RED else blue(obs))


def attack_success_rate(blue: BluePolicy, n: int, *, config: ArenaConfig, seed: int = 0) -> float:
    cfg = config
    gen = ScenarioGenerator.from_config(cfg.scenario, seed=seed)
    done = 0
    completed = 0
    tries = 0
    while done < n and tries < n * 20:
        tries += 1
        sc = gen.sample()
        if not sc.is_adversarial:
            continue
        env = ARENAEnv(cfg, scenario=sc)
        _play(env, ScriptedAttacker(sc), blue, seed=seed + done)
        completed += int(env.last_outcome.objective_completed)
        done += 1
    return completed / max(1, done)


def false_quarantine_rate(blue: BluePolicy, n: int, *, config: ArenaConfig, seed: int = 0) -> float:
    cfg = config
    gen = ScenarioGenerator.from_config(cfg.scenario, seed=seed + 10_000)
    from arena.baselines.collect import _benign_profile

    profile = _benign_profile(cfg)
    done = 0
    quarantined = 0
    tries = 0
    while done < n and tries < n * 20:
        tries += 1
        sc = gen.sample()
        if sc.is_adversarial:
            continue
        env = ARENAEnv(cfg, scenario=sc)
        _play(env, BenignRoller(sc, seed=seed + done, profile=profile), blue, seed=seed + done)
        quarantined += int(env.last_outcome.quarantined)
        done += 1
    return quarantined / max(1, done)


def calibrate_threshold(baseline, benign_decisions: list[Decision], target_fpr: float = 0.05) -> float:
    """Set the baseline's decision threshold to the ``1 - target_fpr`` quantile of
    its scores on benign traffic, so every detector is compared at the same
    false-positive budget (the M8 methodology, in miniature)."""
    s = np.array([baseline.score(d.blue_obs) for d in benign_decisions if not d.adversarial])
    if len(s) == 0:
        return 0.5
    thr = float(np.quantile(s, 1.0 - target_fpr))
    # nudge above ties so exactly-at-quantile benign scores don't fire
    baseline.threshold = np.nextafter(thr, np.inf)
    return baseline.threshold


# These two used to carry their own rank-AUROC and quantile-threshold
# implementations, written before M8 existed. They agreed with
# `arena.eval.metrics` exactly (checked over 910 decisions in the M1-M9 audit),
# but two copies of a metric is one copy too many — and the local AUROC computed
# its ranks twice, the first result dead. They now delegate to the M8 versions,
# which are the ones verified against `sklearn.metrics`.


def decision_auroc(score_fn: Callable[[dict], float], decisions: list[Decision]) -> float:
    """Per-decision AUROC of ``score_fn`` against the taint-graph ground truth."""
    y = [int(d.on_attack_path) for d in decisions]
    if len(set(y)) < 2:
        return float("nan")
    return roc_auc([score_fn(d.blue_obs) for d in decisions], y)


def tpr_at_fpr(score_fn: Callable[[dict], float], decisions: list[Decision], fpr: float = 0.05) -> float:
    """Per-decision true-positive rate at a fixed per-decision false-positive
    rate. The threshold is the ``1 - fpr`` quantile of scores on the off-path
    (benign) decisions; TPR is measured on the on-path decisions."""
    return _tpr_at_fpr(
        [score_fn(d.blue_obs) for d in decisions],
        [int(d.on_attack_path) for d in decisions],
        fpr,
    )


def score_baseline(
    baseline,
    *,
    config: ArenaConfig,
    n_adversarial: int = 40,
    n_benign: int = 40,
    decisions: list[Decision] | None = None,
    seed: int = 0,
) -> BaselineScore:
    return BaselineScore(
        name=getattr(baseline, "name", type(baseline).__name__),
        attack_success_rate=attack_success_rate(baseline, n_adversarial, config=config, seed=seed),
        false_quarantine_rate=false_quarantine_rate(baseline, n_benign, config=config, seed=seed),
        auroc=decision_auroc(baseline.score, decisions) if decisions else float("nan"),
        tpr_at_5pct_fpr=tpr_at_fpr(baseline.score, decisions) if decisions else float("nan"),
        n_adversarial=n_adversarial,
        n_benign=n_benign,
    )
