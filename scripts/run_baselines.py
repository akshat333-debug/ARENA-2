"""Train the three baseline defenders and print the comparison table.

    python3 scripts/run_baselines.py [config.yaml] [--adv N] [--benign N]

The full leaderboard (vs ARENA-trained Blue, with exploitability curves) is M8.
This is the static-baseline half of it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.baselines import CausalMonitor, SingleShotDetector, StaticAllowList  # noqa: E402
from arena.baselines.collect import collect_decisions, split
from arena.baselines.evaluate import score_baseline
from arena.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="small.yaml")
    ap.add_argument("--adv", type=int, default=200)
    ap.add_argument("--benign", type=int, default=200)
    ap.add_argument("--eval-adv", type=int, default=100)
    ap.add_argument("--eval-benign", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"config={cfg.name}  collecting {args.adv} adversarial + {args.benign} benign episodes...")
    rows = collect_decisions(args.adv, args.benign, config=cfg, seed=args.seed + 1)
    train, val = split(rows, frac=0.75, seed=args.seed)
    print(f"  {len(rows)} decisions  ({sum(d.on_attack_path for d in rows)} on attack chains)")

    baselines = [
        StaticAllowList(),
        SingleShotDetector().fit(train),
        CausalMonitor().fit(train),
    ]

    print()
    print(f"{'baseline':<18}  {'AUROC':>7}  {'TPR@5%FPR':>10}  {'attack_success':>14}  {'episode_FP':>11}")
    print("-" * 70)
    for b in baselines:
        s = score_baseline(
            b, config=cfg,
            n_adversarial=args.eval_adv, n_benign=args.eval_benign,
            decisions=val, seed=args.seed + 99,
        )
        print(
            f"{s.name:<18}  {s.auroc:>7.3f}  {s.tpr_at_5pct_fpr:>10.3f}  "
            f"{s.attack_success_rate:>14.3f}  {s.false_quarantine_rate:>11.3f}"
        )
    print()
    print("attack_success is the exploitability proxy: fraction of scripted attacks that")
    print("complete despite the defender. TAMAS reports ~0.80-0.82 for static single-shot.")


if __name__ == "__main__":
    main()
