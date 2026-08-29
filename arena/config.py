"""Typed configuration, loaded from YAML.

Every size, step count and generation count in ARENA lives here — nothing in
``arena/`` hardcodes scale. Scaling up is a YAML edit (architecture.md S10).
Sections are added as modules land; this file currently covers M1–M2.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from arena.scenarios import ALL_FAMILIES, AttackFamily
from arena.tools import Domain

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


class ScenarioConfig(BaseModel):
    """Controls the scenario generator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = 0
    domains: tuple[Domain, ...] = tuple(Domain)
    families: tuple[AttackFamily, ...] = ALL_FAMILIES
    n_tools_min: int = Field(default=6, ge=4)
    n_tools_max: int = Field(default=12, ge=4)
    adversarial_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    max_steps: int = Field(default=12, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "ScenarioConfig":
        if self.n_tools_min > self.n_tools_max:
            raise ValueError(f"n_tools_min ({self.n_tools_min}) > n_tools_max ({self.n_tools_max})")
        if not self.domains:
            raise ValueError("domains must be non-empty")
        if not self.families:
            raise ValueError("families must be non-empty")
        return self


class RewardConfig(BaseModel):
    """Weights for the asymmetric reward engine (arena/rewards.py).

    All entries are non-negative magnitudes; the signs live in the reward
    formula, not here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    w_evade: float = Field(default=1.0, ge=0.0)
    w_plausibility: float = Field(default=0.3, ge=0.0)
    w_step: float = Field(default=0.02, ge=0.0)
    w_detect: float = Field(default=1.0, ge=0.0)
    w_flag: float = Field(default=0.3, ge=0.0)
    w_miss: float = Field(default=1.0, ge=0.0)
    #: Default is w_detect + w_miss, which makes the two degenerate defenders —
    #: quarantine-everything and allow-everything — score *exactly equally badly*
    #: at a 50/50 adversarial ratio (both -0.5 per episode, vs +0.5 for a
    #: discriminating Blue). Any lower and blanket paranoia is the better trivial
    #: strategy, which is what the proposal's "precision, not paranoia" rules out.
    w_false_quarantine: float = Field(default=2.0, ge=0.0)
    w_false_flag: float = Field(default=0.25, ge=0.0)


class PolicyConfig(BaseModel):
    """Network sizes for both policies (arena/policies.py).

    Small by default: these must train on an M3 in minutes. Scaling up is a YAML
    edit, never a code change.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hidden: int = Field(default=64, ge=8)
    #: GRU hidden width for the sequence encoders.
    gru_hidden: int = Field(default=64, ge=8)
    #: Embedding width used to score tools against the context (Red).
    tool_embed: int = Field(default=32, ge=4)


class PPOConfig(BaseModel):
    """PPO hyperparameters (arena/ppo.py)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: 1e-3, not the usual 3e-4. Blue's reward landscape has a wide, flat basin
    #: around the degenerate "quarantine everything" policy, and quarantine is an
    #: absorbing action — it ends the episode, so a paranoid policy stops
    #: collecting the late-sequence states it needs to learn discrimination. At
    #: 3e-4 Blue reliably converges into that basin; at 1e-3 it escapes and learns
    #: to discriminate. Measured, not guessed — see docs/m5-policies-and-ppo.md.
    lr: float = Field(default=1e-3, gt=0.0)
    #: Transitions collected per update.
    n_steps: int = Field(default=1024, ge=8)
    n_epochs: int = Field(default=4, ge=1)
    minibatch_size: int = Field(default=256, ge=1)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    clip_coef: float = Field(default=0.2, gt=0.0)
    ent_coef: float = Field(default=0.01, ge=0.0)
    vf_coef: float = Field(default=0.5, ge=0.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    #: Total environment transitions for one training call.
    total_steps: int = Field(default=20_000, ge=1)
    #: "cpu", "mps", or "auto". Auto picks cpu — for nets this small the MPS
    #: kernel-launch overhead outweighs the parallelism.
    device: str = "auto"
    normalize_advantage: bool = True

    @model_validator(mode="after")
    def _check(self) -> "PPOConfig":
        if self.minibatch_size > self.n_steps:
            raise ValueError(
                f"minibatch_size ({self.minibatch_size}) > n_steps ({self.n_steps})"
            )
        if self.device not in ("cpu", "mps", "auto"):
            raise ValueError(f"device must be cpu/mps/auto, got {self.device!r}")
        return self


class ArenaConfig(BaseModel):
    """Top-level config. One YAML file maps to one of these."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "unnamed"
    #: Global seed. Sub-configs may override for their own component.
    seed: int = 0
    scenario: ScenarioConfig = ScenarioConfig()
    reward: RewardConfig = RewardConfig()
    policy: PolicyConfig = PolicyConfig()
    ppo: PPOConfig = PPOConfig()

    @model_validator(mode="after")
    def _propagate_seed(self) -> "ArenaConfig":
        # A single top-level seed should drive everything unless a section opts out
        # by setting its own non-default value. Keeps `seed: 7` reproducible runs
        # from needing the seed repeated in every section.
        if self.scenario.seed == 0 and self.seed != 0:
            object.__setattr__(self, "scenario", self.scenario.model_copy(update={"seed": self.seed}))
        return self


def load_config(path: str | Path) -> ArenaConfig:
    """Load and validate a YAML config file."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        candidate = CONFIG_DIR / p
        if candidate.exists():
            p = candidate
    if not p.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {p} must contain a YAML mapping, got {type(data).__name__}")
    return ArenaConfig(**data)
