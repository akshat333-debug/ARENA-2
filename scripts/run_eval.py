"""M8 evaluation harness — leaderboard + exploitability curve.

    python3 scripts/run_eval.py small.yaml                 # baselines only
    python3 scripts/run_eval.py small.yaml --blue runs/sp.pt  # + ARENA-trained Blue
    python3 scripts/run_eval.py small.yaml --curve runs/sp.pt # exploitability curve over its league

`--blue` / `--curve` take a checkpoint saved by scripts/train_selfplay.py
(a dict with "blue" and optionally "league").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from arena.baselines import CausalMonitor, SingleShotDetector, StaticAllowList  # noqa: E402
from arena.baselines.collect import collect_decisions, split  # noqa: E402
from arena.config import load_config  # noqa: E402
from arena.eval import (  # noqa: E402
    evaluate_defenders,
    exploitability_curve,
    format_leaderboard,
)
from arena.eval.exploitability import blue_from_state_dict  # noqa: E402
from arena.policies import make_policy  # noqa: E402


def _load_blue(path: str, cfg):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    blue_sd = sd["blue"] if "blue" in sd else sd
    net = make_policy("blue", max_steps=cfg.scenario.max_steps,
                      n_tools_max=cfg.scenario.n_tools_max, cfg=cfg.policy)
    net.load_state_dict(blue_sd)
    net.eval()
    return net, sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="small.yaml")
    ap.add_argument("--blue", type=str, default=None, help="checkpoint to add as 'arena_blue'")
    ap.add_argument("--curve", type=str, default=None,
                    help="checkpoint whose league Blue pool to plot an exploitability curve over")
    ap.add_argument("--br-steps", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--decisions", type=int, default=120, help="adversarial (and equal benign) episodes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.curve:
        from arena.selfplay import SelfPlayTrainer

        sp = SelfPlayTrainer(cfg, seed=args.seed)
        sp.load_state_dict(torch.load(args.curve, map_location="cpu", weights_only=True))
        if sp.league is None:
            print("checkpoint has no league; cannot build a curve")
            return
        blues = [
            (f"gen{ck.generation}", blue_from_state_dict(ck.state_dict, cfg))
            for ck in sp.league.checkpoints("blue")
        ]
        pts = exploitability_curve(blues, cfg, br_steps=args.br_steps, n_eval=args.n_eval, seed=args.seed)
        print("exploitability curve (fresh best-response Red per point):")
        for p in pts:
            print(f"  {p.label:<8} exploitability={p.result.exploitability:.3f} "
                  f"(fresh Red start {p.result.fresh_red_start:.3f})")
        xs = [p.result.exploitability for p in pts]
        if len(xs) >= 2:
            print(f"  first {xs[0]:.3f} -> last {xs[-1]:.3f}  (delta {xs[-1] - xs[0]:+.3f})")
        return

    tr, _ = split(collect_decisions(200, 200, config=cfg, seed=args.seed + 1), seed=args.seed)
    defenders = {
        "static_allowlist": StaticAllowList(),
        "single_shot": SingleShotDetector().fit(tr),
        "causal_monitor": CausalMonitor().fit(tr),
    }
    if args.blue:
        net, _ = _load_blue(args.blue, cfg)
        defenders["arena_blue"] = net

    rows = evaluate_defenders(
        defenders, cfg,
        n_decision_adv=args.decisions, n_decision_benign=args.decisions,
        br_steps=args.br_steps, n_eval=args.n_eval, seed=args.seed,
    )
    print(format_leaderboard(rows))
    print("\nexploitability = best-response attack success at a calibrated ~5% FPR.")
    print("(fresh Red) = same Red's success before best-response training — near chance,")
    print("             the check that no warm start leaked in.")


if __name__ == "__main__":
    main()
