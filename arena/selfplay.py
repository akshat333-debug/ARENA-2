"""Alternating self-play (architecture.md S7).

One generation:

1. Freeze Blue, train Red to best-respond to it (find exploits).
2. Freeze Red, train Blue to best-respond to *that* Red (patch them).
3. Evaluate: how exploitable is the current Blue by the Red just trained
   against it, and how does Blue do against the current Red.

Repeat. Exploitability is expected to fall across generations as Blue closes the
gaps Red keeps finding — that curve is the project's headline result, produced
properly by the M8 harness; here it is tracked per generation as a running check.

The league / opponent-checkpoint pool that guards against cyclic strategies is
M7. This module is only the alternation, against the *current* opponent.

Frozen means frozen: the opponent is queried through
:class:`~arena.policies.TorchPolicyAdapter`, whose ``act`` is under
``torch.no_grad`` and whose parameters are never handed to an optimiser. The M5
suite already proves an opponent's weights do not move during training;
``tests/test_m6_integration`` re-checks it for this loop specifically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from arena.config import ArenaConfig
from arena.env import BLUE, RED, SingleAgentARENA
from arena.policies import ActorCritic, TorchPolicyAdapter, make_policy
from arena.ppo import PPOTrainer, TrainStats, evaluate_policy, resolve_device
from arena.scripted import AdaptiveRed


@dataclass
class GenerationStats:
    generation: int
    red_train: TrainStats
    blue_train: TrainStats
    #: Best-response attack success of the Red trained this generation against the
    #: Blue it was trained on — the running exploitability proxy. Should trend down.
    exploitability: float
    blue_return_vs_red: float
    blue_quarantine_rate: float
    blue_attack_success: float  # attacks Blue let through, vs the current Red

    def summary(self) -> str:
        return (
            f"gen {self.generation}: exploitability={self.exploitability:.3f}  "
            f"blue_return={self.blue_return_vs_red:+.3f}  "
            f"blue_quarantine={self.blue_quarantine_rate:.3f}  "
            f"blue_miss={self.blue_attack_success:.3f}"
        )


class SelfPlayTrainer:
    def __init__(self, config: ArenaConfig | None = None, *, seed: int = 0,
                 device: torch.device | str | None = None) -> None:
        self.cfg = config or ArenaConfig()
        self.seed = seed
        self.device = torch.device(device) if device is not None else resolve_device(self.cfg.ppo.device)

        ms = self.cfg.scenario.max_steps
        nt = self.cfg.scenario.n_tools_max
        self.red: ActorCritic = make_policy("red", max_steps=ms, n_tools_max=nt,
                                            cfg=self.cfg.policy, seed=seed).to(self.device)
        self.blue: ActorCritic = make_policy("blue", max_steps=ms, n_tools_max=nt,
                                             cfg=self.cfg.policy, seed=seed + 1).to(self.device)
        self.history: list[GenerationStats] = []

    # --- frozen-opponent adapters ------------------------------------

    def _frozen(self, policy: ActorCritic) -> TorchPolicyAdapter:
        return TorchPolicyAdapter(
            policy,
            deterministic=not self.cfg.selfplay.stochastic_opponent,
            device=self.device,
        )

    def _red_env(self) -> SingleAgentARENA:
        """Red learns against the frozen Blue across the scenario distribution."""
        return SingleAgentARENA(RED, self._frozen(self.blue), config=self.cfg)

    def _blue_env(self) -> SingleAgentARENA:
        """Blue learns against the frozen Red on adversarial episodes and against
        realistic benign traffic on the rest."""
        env = SingleAgentARENA(BLUE, lambda o: 0, config=self.cfg)
        env._opponent = AdaptiveRed(env, seed=self.seed, attacker=self._frozen(self.red))
        return env

    # --- one generation --------------------------------------------

    def _train_side(self, side: str, gen: int) -> TrainStats:
        env = self._red_env() if side == "red" else self._blue_env()
        policy = self.red if side == "red" else self.blue
        # Distinct, deterministic seed per (generation, side).
        tr = PPOTrainer(env, policy, self.cfg.ppo, seed=self.seed + 1000 * gen + (0 if side == "red" else 1),
                        device=self.device)
        return tr.train(self.cfg.selfplay.steps_per_side)

    def _evaluate(self, gen: int, rs: TrainStats, bs: TrainStats) -> GenerationStats:
        n = self.cfg.selfplay.eval_episodes
        red_vs_blue = evaluate_policy(self._red_env(), self.red, n, deterministic=False, device=self.device)
        blue_vs_red = evaluate_policy(self._blue_env(), self.blue, n, deterministic=False, device=self.device)
        return GenerationStats(
            generation=gen,
            red_train=rs,
            blue_train=bs,
            exploitability=red_vs_blue["attack_success_rate"],
            blue_return_vs_red=blue_vs_red["mean_return"],
            blue_quarantine_rate=blue_vs_red["quarantine_rate"],
            blue_attack_success=blue_vs_red["attack_success_rate"],
        )

    def train(self, n_generations: int | None = None, *,
              callback: Callable[[GenerationStats], None] | None = None) -> list[GenerationStats]:
        gens = n_generations if n_generations is not None else self.cfg.selfplay.n_generations
        for gen in range(gens):
            rs = self._train_side("red", gen)
            bs = self._train_side("blue", gen)
            stats = self._evaluate(gen, rs, bs)
            self.history.append(stats)
            if callback is not None:
                callback(stats)
        return self.history

    # --- persistence ---------------------------------------------

    def state_dict(self) -> dict:
        return {"red": self.red.state_dict(), "blue": self.blue.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.red.load_state_dict(sd["red"])
        self.blue.load_state_dict(sd["blue"])
