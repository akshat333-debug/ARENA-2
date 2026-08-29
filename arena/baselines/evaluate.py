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
    done = 0
    quarantined = 0
    tries = 0
    while done < n and tries < n * 20:
        tries += 1
        sc = gen.sample()
        if sc.is_adversarial:
            continue
        env = ARENAEnv(cfg, scenario=sc)
        _play(env, BenignRoller(sc, seed=seed + done), blue, seed=seed + done)
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


def decision_auroc(score_fn: Callable[[dict], float], decisions: list[Decision]) -> float:
    y = np.array([int(d.on_attack_path) for d in decisions])
    if len(np.unique(y)) < 2:
        return float("nan")
    s = np.array([score_fn(d.blue_obs) for d in decisions])
    # rank-based AUROC, no sklearn dependency needed here
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def tpr_at_fpr(score_fn: Callable[[dict], float], decisions: list[Decision], fpr: float = 0.05) -> float:
    """Per-decision true-positive rate at a fixed per-decision false-positive
    rate. The threshold is the ``1 - fpr`` quantile of scores on the off-path
    (benign) decisions; TPR is measured on the on-path decisions."""
    neg = np.array([score_fn(d.blue_obs) for d in decisions if not d.on_attack_path])
    pos = np.array([score_fn(d.blue_obs) for d in decisions if d.on_attack_path])
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thr = np.nextafter(float(np.quantile(neg, 1.0 - fpr)), np.inf)
    return float(np.mean(pos >= thr))


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
