"""Single-file PPO for ARENA.

Deliberately not Stable-Baselines3 (architecture.md S2): ARENA's contribution is
the training *loop* — alternating freeze, league opponent sampling, spawning a
fresh best-response learner on demand for exploitability. Two policies with
different observation *and* action spaces, one of them frozen and acting inside
the env, is not the shape SB3 optimises for. This is ~250 readable lines and the
trainer sits behind one interface, so swapping it later touches one file.

Standard PPO: GAE(lambda), clipped surrogate, value loss, entropy bonus,
grad-norm clipping. Two details that matter here:

* **Truncation bootstrapping.** Episodes hit the step cap often, and a step-cap
  truncation is *not* a terminal state — the value function must bootstrap
  through it or Red learns that running out of time is as bad as being caught.
* **Dict observations.** Both sides observe dicts; the buffer stores per-key
  arrays and batches them with :func:`arena.policies.stack_obs`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from arena.config import PPOConfig
from arena.policies import ActorCritic, stack_obs, to_batch


def resolve_device(spec: str) -> torch.device:
    """``auto`` -> cpu. For nets this small the MPS kernel-launch overhead costs
    more than the parallelism buys; ``mps`` stays available as an explicit opt-in
    for the scaled-up configs."""
    if spec == "auto":
        return torch.device("cpu")
    if spec == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but MPS is not available")
    return torch.device(spec)


@dataclass
class TrainStats:
    """What one :meth:`PPOTrainer.train` call did."""

    steps: int = 0
    updates: int = 0
    episode_returns: list[float] = field(default_factory=list)
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.episode_returns)) if self.episode_returns else float("nan")

    @property
    def last_mean_return(self, k: int = 20) -> float:
        tail = self.episode_returns[-k:]
        return float(np.mean(tail)) if tail else float("nan")


class RolloutBuffer:
    """Fixed-size on-policy buffer over dict observations."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.clear()

    def clear(self) -> None:
        self.obs: list[dict] = []
        self.actions: list[int] = []
        self.logprobs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        #: True only on a real terminal state (not a step-cap truncation).
        self.terminals: list[bool] = []
        #: Value of the state the episode was cut at, for truncation bootstrap.
        self.truncated_values: list[float | None] = []

    def __len__(self) -> int:
        return len(self.actions)

    def add(self, obs, action, logprob, reward, value, terminal, truncated_value) -> None:
        self.obs.append(obs)
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.terminals.append(bool(terminal))
        self.truncated_values.append(truncated_value)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminals: np.ndarray,
    truncated_values: list,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GAE(lambda) with correct handling of truncation.

    On a *terminated* step there is no future, so the TD target is just the
    reward. On a *truncated* step the episode continues in principle, so the
    target bootstraps from the value of the cut state — but the advantage trace
    still resets, because the next transition belongs to a different episode.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float64)
    last_gae = 0.0
    for t in reversed(range(n)):
        truncated = truncated_values[t] is not None
        episode_boundary = terminals[t] or truncated
        if terminals[t]:
            next_value = 0.0
        elif truncated:
            next_value = float(truncated_values[t])
        else:
            next_value = values[t + 1] if t + 1 < n else last_value
        delta = rewards[t] + gamma * next_value - values[t]
        if episode_boundary:
            last_gae = delta
        else:
            last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


class PPOTrainer:
    """Trains one :class:`ActorCritic` against a fixed Gymnasium env.

    The env is normally a :class:`arena.env.SingleAgentARENA` with the opponent
    frozen — "train Red vs this Blue" or vice versa.
    """

    def __init__(
        self,
        env,
        policy: ActorCritic,
        cfg: PPOConfig | None = None,
        *,
        seed: int = 0,
        device: torch.device | str | None = None,
    ) -> None:
        self.cfg = cfg or PPOConfig()
        self.device = torch.device(device) if device is not None else resolve_device(self.cfg.device)
        self.env = env
        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.seed = seed
        # The trainer owns *all* its randomness. Using the global numpy RNG for
        # minibatch shuffling makes a run depend on whatever else touched
        # np.random first, so two identical calls can train differently.
        self._rng = np.random.default_rng(seed)
        self._rng_seeded = False
        self._obs: dict | None = None
        self._episode_return = 0.0
        self.buffer = RolloutBuffer(self.cfg.n_steps)

    # --- rollout -------------------------------------------------------

    def _reset_env(self) -> dict:
        if not self._rng_seeded:
            obs, _ = self.env.reset(seed=self.seed)
            self._rng_seeded = True
        else:
            obs, _ = self.env.reset()
        return obs

    def collect(self, n_steps: int, stats: TrainStats) -> float:
        """Fill the buffer with ``n_steps`` transitions. Returns V(next state) for
        the final bootstrap."""
        self.buffer.clear()
        if self._obs is None:
            self._obs = self._reset_env()
            self._episode_return = 0.0

        for _ in range(n_steps):
            batch = to_batch(self._obs, self.device)
            action, logprob, value = self.policy.act(batch)
            a = int(action.item())
            next_obs, reward, terminated, truncated, _ = self.env.step(a)
            self._episode_return += float(reward)

            truncated_value = None
            if truncated and not terminated:
                with torch.no_grad():
                    _, v = self.policy(to_batch(next_obs, self.device))
                truncated_value = float(v.item())

            self.buffer.add(
                self._obs, a, float(logprob.item()), float(reward), float(value.item()),
                terminal=bool(terminated and not truncated),
                truncated_value=truncated_value,
            )

            if terminated or truncated:
                stats.episode_returns.append(self._episode_return)
                self._episode_return = 0.0
                self._obs = self._reset_env()
            else:
                self._obs = next_obs

        with torch.no_grad():
            _, last_v = self.policy(to_batch(self._obs, self.device))
        return float(last_v.item())

    # --- update -------------------------------------------------------

    def update(self, last_value: float, stats: TrainStats) -> None:
        cfg = self.cfg
        buf = self.buffer
        rewards = np.asarray(buf.rewards, dtype=np.float64)
        values = np.asarray(buf.values, dtype=np.float64)
        terminals = np.asarray(buf.terminals, dtype=bool)
        adv_np, ret_np = compute_gae(
            rewards, values, terminals, buf.truncated_values,
            last_value, cfg.gamma, cfg.gae_lambda,
        )

        obs = stack_obs(buf.obs, self.device)
        actions = torch.as_tensor(buf.actions, dtype=torch.long, device=self.device)
        old_logprobs = torch.as_tensor(buf.logprobs, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(adv_np, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(ret_np, dtype=torch.float32, device=self.device)

        n = len(buf)
        idx = np.arange(n)
        p_loss = v_loss = ent = 0.0
        n_batches = 0
        for _ in range(cfg.n_epochs):
            self._rng.shuffle(idx)
            for start in range(0, n, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                if len(mb) < 2:
                    continue
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                mb_obs = {k: v[mb_t] for k, v in obs.items()}

                logprobs, entropy, value = self.policy.evaluate(mb_obs, actions[mb_t])
                ratio = torch.exp(logprobs - old_logprobs[mb_t])

                mb_adv = advantages[mb_t]
                if cfg.normalize_advantage and len(mb) > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                policy_loss = torch.max(pg1, pg2).mean()
                value_loss = 0.5 * ((value - returns[mb_t]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                p_loss += float(policy_loss.item())
                v_loss += float(value_loss.item())
                ent += float(entropy_loss.item())
                n_batches += 1

        if n_batches:
            stats.policy_loss = p_loss / n_batches
            stats.value_loss = v_loss / n_batches
            stats.entropy = ent / n_batches
        stats.updates += 1

    # --- driver -------------------------------------------------------

    def train(self, total_steps: int | None = None, *, callback: Callable | None = None) -> TrainStats:
        total = total_steps if total_steps is not None else self.cfg.total_steps
        torch.manual_seed(self.seed)
        stats = TrainStats()
        while stats.steps < total:
            n = min(self.cfg.n_steps, total - stats.steps)
            last_value = self.collect(n, stats)
            stats.steps += n
            self.update(last_value, stats)
            if callback is not None:
                callback(stats)
        return stats


def evaluate_policy(env, policy: ActorCritic, n_episodes: int = 20, *,
                    deterministic: bool = True, device: torch.device | str = "cpu") -> dict:
    """Run ``n_episodes`` and report mean return plus the env's own info flags."""
    returns: list[float] = []
    completed = 0
    quarantined = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        info: dict = {}
        while True:
            action, _, _ = policy.act(to_batch(obs, device), deterministic=deterministic)
            obs, reward, term, trunc, info = env.step(int(action.item()))
            total += float(reward)
            if term or trunc:
                break
        returns.append(total)
        completed += int(bool(info.get("objective_completed")))
        quarantined += int(bool(info.get("quarantined")))
    return {
        "mean_return": float(np.mean(returns)),
        "attack_success_rate": completed / n_episodes,
        "quarantine_rate": quarantined / n_episodes,
        "n_episodes": n_episodes,
    }
