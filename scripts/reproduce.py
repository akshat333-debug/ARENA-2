"""M11 — reproduce all figures and tables from a single command.

    python3 scripts/reproduce.py small.yaml
    python3 scripts/reproduce.py small.yaml --blue runs/sp.pt
    python3 scripts/reproduce.py small.yaml --out report/

Regenerates:
  1. Baseline leaderboard table
  2. ARENA-trained Blue row (if --blue provided)
  3. Exploitability curve (if --blue has a league)
  4. Plot files (PNG) in the output directory
  5. Transfer sweep table (M10, with --sweep)
  6. results.json with all raw numbers

Every number in the report MUST come from this script.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from arena.baselines import CausalMonitor, SingleShotDetector, StaticAllowList  # noqa: E402
from arena.baselines.collect import collect_decisions, split  # noqa: E402
from arena.config import load_config  # noqa: E402
from arena.eval import evaluate_defenders, exploitability_curve, format_leaderboard  # noqa: E402
from arena.eval.exploitability import blue_from_state_dict  # noqa: E402
from arena.eval.report import (  # noqa: E402
    format_leaderboard_table,
    format_exploitability_curve,
    plot_exploitability_curve,
    plot_leaderboard_comparison,
    save_results,
)
from arena.llm.sweep import run_sweep  # noqa: E402
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
    ap = argparse.ArgumentParser(description="Reproduce all M11 figures and tables")
    ap.add_argument("config", nargs="?", default="small.yaml")
    ap.add_argument("--blue", type=str, default=None, help="checkpoint to add as 'arena_blue'")
    ap.add_argument("--out", type=str, default="report", help="output directory")
    ap.add_argument("--decisions", type=int, default=150, help="adversarial (and equal benign) episodes")
    ap.add_argument("--br-steps", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="also run the M10 LLM transfer sweep (needs Ollama; slow)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== ARENA M11 Reproduce ===")
    print(f"config: {cfg.name}  seed: {args.seed}")
    print(f"output: {out_dir}/")
    print()

    results = {"config": cfg.name, "seed": args.seed, "sections": {}}

    # --- 1. Baseline leaderboard ---
    print("[1/5] Collecting decisions for baselines...")
    tr, _ = split(collect_decisions(200, 200, config=cfg, seed=args.seed + 1), seed=args.seed)
    defenders = {
        "static_allowlist": StaticAllowList(),
        "single_shot": SingleShotDetector().fit(tr),
        "causal_monitor": CausalMonitor().fit(tr),
    }

    # --- 2. ARENA-trained Blue ---
    if args.blue:
        print("[2/5] Loading ARENA-trained Blue...")
        net, sd = _load_blue(args.blue, cfg)
        defenders["arena_blue"] = net
    else:
        print("[2/5] No --blue provided, skipping ARENA Blue.")

    # --- 3. Evaluate ---
    print("[3/5] Running evaluation (this may take a few minutes)...")
    rows = evaluate_defenders(
        defenders, cfg,
        n_decision_adv=args.decisions, n_decision_benign=args.decisions,
        br_steps=args.br_steps, n_eval=args.n_eval, seed=args.seed,
    )
    print(format_leaderboard(rows))

    leaderboard_rows = [r.as_dict() for r in rows]
    results["sections"]["leaderboard"] = leaderboard_rows

    # Save leaderboard table
    lb_md = format_leaderboard_table(leaderboard_rows, title=f"Baseline Leaderboard ({cfg.name})")
    (out_dir / "leaderboard.md").write_text(lb_md)
    print(f"  -> {out_dir}/leaderboard.md")

    # Save leaderboard plot
    plot_leaderboard_comparison(leaderboard_rows, out_dir / "leaderboard.png",
                                title=f"Defender Comparison ({cfg.name})")
    print(f"  -> {out_dir}/leaderboard.png")

    # --- 4. Exploitability curve ---
    if args.blue and "blue" in sd and "league" in sd:
        print("[4/5] Building exploitability curve from league...")
        from arena.selfplay import SelfPlayTrainer

        sp = SelfPlayTrainer(cfg, seed=args.seed)
        sp.load_state_dict(sd)
        if sp.league is not None:
            blues = [
                (f"gen{ck.generation}", blue_from_state_dict(ck.state_dict, cfg))
                for ck in sp.league.checkpoints("blue")
            ]
            pts = exploitability_curve(blues, cfg, br_steps=args.br_steps,
                                       n_eval=args.n_eval, seed=args.seed)
            curve_data = [
                {"label": p.label, "exploitability": p.result.exploitability,
                 "fresh_red_start": p.result.fresh_red_start}
                for p in pts
            ]
            results["sections"]["exploitability_curve"] = curve_data

            print(format_exploitability_curve(curve_data, title="Exploitability Curve"))
            curve_md = format_exploitability_curve(curve_data, title=f"Exploitability Curve ({cfg.name})")
            (out_dir / "exploitability_curve.md").write_text(curve_md)
            print(f"  -> {out_dir}/exploitability_curve.md")

            plot_exploitability_curve(curve_data, out_dir / "exploitability_curve.png")
            print(f"  -> {out_dir}/exploitability_curve.png")
        else:
            print("  (checkpoint has no league; skipping curve)")
    else:
        print("[4/5] No league checkpoint; skipping exploitability curve.")

    # --- 5. M10 transfer sweep ---
    if args.sweep:
        print("[5/5] Running the M10 transfer sweep (LLM-planned attacks)...")
        sweep = run_sweep(
            defenders, cfg,
            n_decision_adv=args.decisions, n_decision_benign=args.decisions,
            br_steps=args.br_steps, n_eval=args.n_eval, seed=args.seed,
        )
        print(sweep.format())
        results["sections"]["transfer_sweep"] = {
            "llm_available": sweep.llm_available,
            "plan_stats": (
                {
                    "n_adversarial": sweep.plan_stats.n_adversarial,
                    "n_llm_planned": sweep.plan_stats.n_llm_planned,
                    "n_differing_from_scripted": sweep.plan_stats.n_differing_from_scripted,
                    "reasons": sweep.plan_stats.reasons,
                }
                if sweep.plan_stats else None
            ),
            "scripted_rows": [r.as_dict() for r in sweep.templated_rows],
            "llm_rows": [r.as_dict() for r in sweep.llm_rows] if sweep.llm_rows else None,
        }
        (out_dir / "transfer_sweep.md").write_text(sweep.format())
        print(f"  -> {out_dir}/transfer_sweep.md")
    else:
        print("[5/5] Transfer sweep skipped (pass --sweep to run it).")

    # --- Save raw results ---
    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)
    save_results(results, out_dir / "results.json")
    print(f"\n  -> {out_dir}/results.json")

    print(f"\n=== Done in {elapsed:.1f}s ===")
    print(f"All figures and tables in {out_dir}/")
    print("Every number in the report must come from this script.")


if __name__ == "__main__":
    main()
