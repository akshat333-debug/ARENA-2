"""Red and Blue policy networks.

Both are small actor-critic modules over the M4 observation encoders. Sizes come
from :class:`arena.config.PolicyConfig` — nothing here hardcodes scale.

Blue is a **GRU over the call sequence**: trajectory-level by construction, which
is the whole point (a per-call classifier is the M4 single-shot baseline, and it
provably cannot separate the split chain).

Red scores each tool in the episode's registry by dot-product against a context
vector built from the task, the objective, and the history *including Blue's past
verdicts* — that verdict channel is what lets Red adapt to the defender it faces.
Scoring by embedding rather than a fixed output head means one Red generalises
across registries of different sizes and contents; the fixed ``Discrete`` action
space is masked down to the episode's real registry.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from arena.config import PolicyConfig
from arena.features import CALL_FEATS, DOMAINS, RED_EXTRA, VERDICTS

RED_HIST_FEATS = CALL_FEATS + RED_EXTRA


def _init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    """Orthogonal init — the standard PPO choice; markedly more stable than the
    default for small actor-critics."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def stack_obs(obs_list: Sequence[dict], device: torch.device | str = "cpu") -> dict:
    """Batch a list of observation dicts into a dict of tensors."""
    out = {}
    for k in obs_list[0]:
        arr = np.stack([np.asarray(o[k]) for o in obs_list])
        dtype = torch.long if k == "length" else torch.float32
        out[k] = torch.as_tensor(arr, dtype=dtype, device=device)
    return out


def to_batch(obs: dict, device: torch.device | str = "cpu") -> dict:
    """Single observation -> batch of one."""
    return stack_obs([obs], device)


def _last_valid(seq_out: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """Gather the GRU output at the last filled row of each sequence.

    ``length`` is the number of filled rows (0 is possible at episode start, in
    which case row 0 is all zeros and its encoding is a well-defined constant).
    """
    b = seq_out.shape[0]
    idx = torch.clamp(length.reshape(b) - 1, min=0)
    return seq_out[torch.arange(b, device=seq_out.device), idx]


class BluePolicy(nn.Module):
    """GRU over the observed call sequence -> allow / flag / quarantine + value."""

    def __init__(self, max_steps: int, cfg: PolicyConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or PolicyConfig()
        self.max_steps = max_steps
        self.cfg = cfg
        self.call_enc = nn.Sequential(_init(nn.Linear(CALL_FEATS, cfg.hidden)), nn.Tanh())
        self.gru = nn.GRU(cfg.hidden, cfg.gru_hidden, batch_first=True)
        self.trunk = nn.Sequential(
            _init(nn.Linear(cfg.gru_hidden + len(DOMAINS), cfg.hidden)), nn.Tanh()
        )
        self.actor = _init(nn.Linear(cfg.hidden, len(VERDICTS)), std=0.01)
        self.critic = _init(nn.Linear(cfg.hidden, 1), std=1.0)

        # Start permissive: bias the actor toward ALLOW and away from QUARANTINE.
        #
        # Quarantine is an *absorbing* action — it ends the episode. A Blue that
        # starts trigger-happy only ever sees length-1 episodes, so it never
        # collects the long sequences that carry the evidence it needs, and it
        # settles into the "quarantine everything" basin permanently. Starting
        # permissive lets it observe whole chains first and learn when to
        # intervene. Measured: without this, 2 of 5 seeds collapse into that
        # basin; with it, all 5 learn to discriminate.
        with torch.no_grad():
            self.actor.bias.copy_(torch.tensor([1.0, 0.0, -1.0]))

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.call_enc(obs["calls"])
        seq, _ = self.gru(h)
        summary = _last_valid(seq, obs["length"])
        z = self.trunk(torch.cat([summary, obs["domain"]], dim=-1))
        return self.actor(z), self.critic(z).squeeze(-1)


class RedPolicy(nn.Module):
    """Context-vs-tool-embedding scorer -> masked distribution over the registry."""

    def __init__(self, max_steps: int, n_tools_max: int, cfg: PolicyConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or PolicyConfig()
        self.max_steps = max_steps
        self.n_tools_max = n_tools_max
        self.cfg = cfg
        obj_dim = 4 + 2 * n_tools_max
        task_dim = len(DOMAINS) + 1

        self.tool_enc = nn.Sequential(
            _init(nn.Linear(CALL_FEATS, cfg.hidden)), nn.Tanh(),
            _init(nn.Linear(cfg.hidden, cfg.tool_embed)),
        )
        self.hist_enc = nn.Sequential(_init(nn.Linear(RED_HIST_FEATS, cfg.hidden)), nn.Tanh())
        self.gru = nn.GRU(cfg.hidden, cfg.gru_hidden, batch_first=True)
        self.ctx = nn.Sequential(
            _init(nn.Linear(cfg.gru_hidden + task_dim + obj_dim, cfg.hidden)), nn.Tanh(),
            _init(nn.Linear(cfg.hidden, cfg.hidden)), nn.Tanh(),
        )
        self.ctx_proj = _init(nn.Linear(cfg.hidden, cfg.tool_embed), std=0.01)
        self.critic = _init(nn.Linear(cfg.hidden, 1), std=1.0)

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.hist_enc(obs["history"])
        seq, _ = self.gru(h)
        summary = _last_valid(seq, obs["length"])
        z = self.ctx(torch.cat([summary, obs["task"], obs["objective"]], dim=-1))

        tool_emb = self.tool_enc(obs["registry"])            # (B, n_tools, d)
        q = self.ctx_proj(z).unsqueeze(1)                     # (B, 1, d)
        logits = (tool_emb * q).sum(-1)                       # (B, n_tools)

        mask = obs["registry_mask"] > 0.5
        # A registry is never empty, so at least one action always survives.
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits, self.critic(z).squeeze(-1)


class ActorCritic(nn.Module):
    """Thin wrapper giving both policies one sampling/evaluation interface."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, obs: dict):
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs: dict, *, deterministic: bool = False):
        logits, value = self.net(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs: dict, actions: torch.Tensor):
        logits, value = self.net(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


def make_policy(side: str, *, max_steps: int, n_tools_max: int,
                cfg: PolicyConfig | None = None, seed: int | None = None) -> ActorCritic:
    """``side`` is ``"red"`` or ``"blue"`` (bare names, not the env's agent ids).

    Pass ``seed`` to make initialisation independent of whatever ambient global
    torch RNG state the process happens to be in. Without it, two identical runs
    can start from different weights — and since the training outcome depends on
    the initial policy, a whole experiment becomes irreproducible. Always seed for
    anything whose numbers get reported.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if side == "red":
        return ActorCritic(RedPolicy(max_steps, n_tools_max, cfg))
    if side == "blue":
        return ActorCritic(BluePolicy(max_steps, cfg))
    raise ValueError(f"side must be 'red' or 'blue', got {side!r}")


class TorchPolicyAdapter:
    """Wrap a trained :class:`ActorCritic` as the plain ``obs -> int`` callable the
    env, the baselines and the league all speak.

    ``deterministic=True`` gives the greedy policy used for evaluation; frozen
    opponents during self-play stay stochastic so the learner faces the policy's
    actual distribution.
    """

    def __init__(self, policy: ActorCritic, *, deterministic: bool = False,
                 device: torch.device | str = "cpu") -> None:
        self.policy = policy
        self.deterministic = deterministic
        self.device = device

    def __call__(self, obs: dict) -> int:
        batch = to_batch(obs, self.device)
        action, _, _ = self.policy.act(batch, deterministic=self.deterministic)
        return int(action.item())
