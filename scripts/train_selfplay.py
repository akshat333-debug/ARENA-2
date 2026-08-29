"""Run alternating self-play (M6).

    python3 scripts/train_selfplay.py small.yaml --generations 4 --steps 40000
    python3 scripts/train_selfplay.py small.yaml --out runs/sp.pt

Each generation: freeze Blue -> train Red; freeze Red -> train Blue; evaluate.
Exploitability (best-response attack success vs the current Blue) is printed per
generation. The M7 league adds an opponent-checkpoint pool to stop the
non-transitive drift this loop shows on its own; the proper exploitability curve
is M8.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.config import SelfPlayConfig, load_config  # noqa: E402
from arena.selfplay import SelfPlayTrainer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="small.yaml")
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None, help="PPO steps per side per generation")
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="save {red, blue} state_dicts here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sp_cfg = cfg.selfplay
    overrides = {}
    if args.generations is not None:
        overrides["n_generations"] = args.generations
    if args.steps is not None:
        overrides["steps_per_side"] = args.steps
    if args.eval_episodes is not None:
        overrides["eval_episodes"] = args.eval_episodes
    if overrides:
        sp_cfg = SelfPlayConfig(**{**sp_cfg.model_dump(), **overrides})
        cfg = cfg.model_copy(update={"selfplay": sp_cfg})

    print(f"config={cfg.name} seed={args.seed} "
          f"generations={sp_cfg.n_generations} steps/side={sp_cfg.steps_per_side} "
          f"league={'on' if sp_cfg.use_league else 'off'}")

    sp = SelfPlayTrainer(cfg, seed=args.seed)
    sp.train(callback=lambda s: print("  " + s.summary()))

    if sp.league is not None:
        print(f"  league pools: red={sp.league.pool_size('red')} blue={sp.league.pool_size('blue')}")

    if args.out:
        import torch

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(sp.state_dict(), args.out)
        print(f"  saved -> {args.out}")

    xs = [g.exploitability for g in sp.history]
    print(f"\nexploitability by generation: {['%.3f' % x for x in xs]}")
    if len(xs) >= 2:
        print(f"  first {xs[0]:.3f} -> last {xs[-1]:.3f}  (delta {xs[-1] - xs[0]:+.3f})")


if __name__ == "__main__":
    main()
