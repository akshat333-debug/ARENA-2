"""M10 LLM transfer sweep — one command.

    python3 scripts/run_sweep.py small.yaml                      # baselines only
    python3 scripts/run_sweep.py small.yaml --blue runs/sp.pt    # + ARENA-trained Blue

Runs ``evaluate_defenders`` twice: once with templated payloads (standard)
and once with LLM-rendered payloads (if Ollama is available).  Reports both
leaderboards side by side.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from arena.baselines import CausalMonitor, SingleShotDetector, StaticAllowList  # noqa: E402
from arena.baselines.collect import split, collect_decisions  # noqa: E402
from arena.config import load_config  # noqa: E402
from arena.llm.sweep import run_sweep  # noqa: E402
from arena.policies import make_policy  # noqa: E402


def _load_blue(path: str, cfg):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    blue_sd = sd["blue"] if "blue" in sd else sd
    net = make_policy("blue", max_steps=cfg.scenario.max_steps,
                      n_tools_max=cfg.scenario.n_tools_max, cfg=cfg.policy)
    net.load_state_dict(blue_sd)
    net.eval()
    return net


def main() -> None:
    ap = argparse.ArgumentParser(description="M10 LLM transfer sweep")
    ap.add_argument("config", nargs="?", default="small.yaml")
    ap.add_argument("--blue", type=str, default=None, help="checkpoint to add as 'arena_blue'")
    ap.add_argument("--br-steps", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--decisions", type=int, default=120, help="adversarial (and equal benign) episodes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr, _ = split(collect_decisions(200, 200, config=cfg, seed=args.seed + 1), seed=args.seed)

    defenders = {
        "static_allowlist": StaticAllowList(),
        "single_shot": SingleShotDetector().fit(tr),
        "causal_monitor": CausalMonitor().fit(tr),
    }
    if args.blue:
        net = _load_blue(args.blue, cfg)
        defenders["arena_blue"] = net

    result = run_sweep(
        defenders, cfg,
        n_decision_adv=args.decisions, n_decision_benign=args.decisions,
        br_steps=args.br_steps, n_eval=args.n_eval, seed=args.seed,
    )
    print(result.format())


if __name__ == "__main__":
    main()
