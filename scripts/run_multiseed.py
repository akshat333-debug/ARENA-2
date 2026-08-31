"""Multi-seed leaderboard — is the gap a result or is it noise?

    python3 scripts/run_multiseed.py small.yaml --seeds 0 1 2 3 4
    python3 scripts/run_multiseed.py small.yaml --seeds 0 1 2 --no-arena-blue

Trains a fresh self-play Blue per seed (so `arena_blue` is not one fixed net
scored five times), runs the full leaderboard per seed, and reports mean +/-
spread plus the paired per-seed difference between the two defenders that
matter.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.baselines import CausalMonitor, SingleShotDetector, StaticAllowList  # noqa: E402
from arena.baselines.collect import collect_decisions, split  # noqa: E402
from arena.config import load_config  # noqa: E402
from arena.eval.multiseed import (  # noqa: E402
    format_aggregate,
    multiseed_leaderboard,
    paired_delta,
)
from arena.selfplay import SelfPlayTrainer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="small.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="PPO steps per side per generation")
    ap.add_argument("--decisions", type=int, default=None)
    ap.add_argument("--br-steps", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--no-arena-blue", action="store_true",
                    help="baselines only (skips per-seed self-play training)")
    ap.add_argument("--out", type=str, default=None, help="write results JSON here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    t0 = time.time()

    def build(seed: int) -> dict:
        tr, _ = split(collect_decisions(200, 200, config=cfg, seed=seed + 1), seed=seed)
        d = {
            "static_allowlist": StaticAllowList(),
            "single_shot": SingleShotDetector().fit(tr),
            "causal_monitor": CausalMonitor().fit(tr),
        }
        if not args.no_arena_blue:
            print(f"  [seed {seed}] training self-play Blue...", flush=True)
            sp = SelfPlayTrainer(cfg, seed=seed)
            if args.steps is not None:
                sp.cfg = cfg.model_copy(update={
                    "selfplay": cfg.selfplay.model_copy(update={"steps_per_side": args.steps})
                })
            sp.train(args.generations)
            d["arena_blue"] = sp.blue
        return d

    def progress(seed, rows):
        print(f"  [seed {seed}] done ({time.time() - t0:.0f}s)", flush=True)

    print(f"=== multi-seed leaderboard: {cfg.name}, seeds={args.seeds} ===", flush=True)
    agg, per_seed = multiseed_leaderboard(
        build, cfg, seeds=args.seeds,
        n_decision_adv=args.decisions, n_decision_benign=args.decisions,
        br_steps=args.br_steps, n_eval=args.n_eval, progress=progress,
    )

    print()
    print(format_aggregate(agg))
    print()

    names = {r.name for r in agg}
    print("--- paired per-seed differences (the question that matters) ---")
    pairs = []
    if {"arena_blue", "causal_monitor"} <= names:
        pairs.append(("arena_blue", "causal_monitor"))
    if {"causal_monitor", "static_allowlist"} <= names:
        pairs.append(("causal_monitor", "static_allowlist"))
    for a, b in pairs:
        for metric in ("exploitability", "auroc"):
            d = paired_delta(per_seed, a, b, metric)
            print("  " + d.summary())
            print(f"      per-seed: {[round(x, 3) for x in d.deltas]}")

    print(f"\n=== done in {time.time() - t0:.0f}s ===")

    if args.out:
        out = {
            "config": cfg.name,
            "seeds": args.seeds,
            "aggregate": [r.as_dict() for r in agg],
            "per_seed": {
                str(s): [r.as_dict() for r in rows] for s, rows in per_seed.items()
            },
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
