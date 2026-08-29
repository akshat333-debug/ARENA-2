"""Train one side with PPO against a fixed scripted opponent (M5).

    python3 scripts/train_ppo.py blue --steps 60000
    python3 scripts/train_ppo.py red  --steps 15000 --fixed-scenario

Alternating self-play across generations is M6; this trains a single
best response, which is also the primitive the M8 exploitability sweep uses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.config import load_config  # noqa: E402
from arena.env import BLUE, RED, SingleAgentARENA  # noqa: E402
from arena.policies import make_policy  # noqa: E402
from arena.ppo import PPOTrainer, evaluate_policy  # noqa: E402
from arena.scenarios import AttackFamily, ScenarioGenerator  # noqa: E402
from arena.scripted import AdaptiveScriptedRed, passive_blue  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("side", choices=["red", "blue"])
    ap.add_argument("config", nargs="?", default="small.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=300)
    ap.add_argument("--fixed-scenario", action="store_true",
                    help="red only: train on one scenario instead of the distribution")
    ap.add_argument("--out", type=str, default=None, help="save the trained policy here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    steps = args.steps if args.steps is not None else cfg.ppo.total_steps

    if args.side == "red":
        if args.fixed_scenario:
            sc = ScenarioGenerator(seed=args.seed, max_steps=cfg.scenario.max_steps).sample(
                force_family=AttackFamily.DIRECT_PROMPT_INJECTION
            )
            env = SingleAgentARENA(RED, passive_blue, scenario=sc)
            max_steps, n_tools = sc.max_steps, len(sc.registry)
        else:
            env = SingleAgentARENA(RED, passive_blue, config=cfg)
            max_steps, n_tools = cfg.scenario.max_steps, cfg.scenario.n_tools_max
    else:
        env = SingleAgentARENA(BLUE, lambda o: 0, config=cfg)
        env._opponent = AdaptiveScriptedRed(env, seed=args.seed)
        max_steps, n_tools = cfg.scenario.max_steps, cfg.scenario.n_tools_max

    policy = make_policy(args.side, max_steps=max_steps, n_tools_max=n_tools,
                         cfg=cfg.policy, seed=args.seed)

    before = evaluate_policy(env, policy, args.eval_episodes, deterministic=False)
    print(f"config={cfg.name} side={args.side} steps={steps} seed={args.seed}")
    print(f"  before: return={before['mean_return']:+.3f} "
          f"attack_success={before['attack_success_rate']:.3f} "
          f"quarantine={before['quarantine_rate']:.3f}")

    def log(stats):
        if stats.updates % 10 == 0:
            print(f"  [{stats.steps:>7}] return={stats.last_mean_return:+.3f} "
                  f"pi={stats.policy_loss:+.4f} v={stats.value_loss:.4f} H={stats.entropy:.3f}")

    trainer = PPOTrainer(env, policy, cfg.ppo, seed=args.seed)
    trainer.train(steps, callback=log)

    after = evaluate_policy(env, policy, args.eval_episodes, deterministic=False)
    print(f"  after : return={after['mean_return']:+.3f} "
          f"attack_success={after['attack_success_rate']:.3f} "
          f"quarantine={after['quarantine_rate']:.3f}")

    if args.out:
        import torch

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy.state_dict(), args.out)
        print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()
