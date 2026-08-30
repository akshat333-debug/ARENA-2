"""Typed configuration, loaded from YAML.

Every size, step count and generation count in ARENA lives here — nothing in
``arena/`` hardcodes scale. Scaling up is a YAML edit (architecture.md S10).
Sections are added as modules land; this file currently covers M1–M2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
    #: Where benign traffic comes from. "synthetic" = the uniform BenignRoller
    #: (M1). "toucan" = draw tool categories from the Toucan-1.5M profile fetched
    #: by ``arena.data.fetch`` (M9); falls back to synthetic if ``data/`` absent.
    benign_source: Literal["synthetic", "toucan"] = "synthetic"

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


class SelfPlayConfig(BaseModel):
    """Alternating self-play loop (arena/selfplay.py) and its league (arena/league.py)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_generations: int = Field(default=4, ge=1)
    #: PPO transitions per side, per generation. Blue needs enough per generation
    #: to escape the paranoid basin from a cold start (~40k on small.yaml); Red
    #: warms up faster. This is the single biggest lever on wall-clock.
    steps_per_side: int = Field(default=40_000, ge=1)
    #: Episodes used for the per-generation eval sweep.
    eval_episodes: int = Field(default=200, ge=1)
    #: Freeze opponents as stochastic (sample from their distribution) rather than
    #: greedy — the learner should face the policy it will actually meet.
    stochastic_opponent: bool = True
    #: Train each side against a *sample* of the opposing side's past checkpoints,
    #: not only the current one (architecture.md S7). Off reproduces the M6 loop,
    #: which drifts to (passive Blue, weak Red) after a couple of generations.
    use_league: bool = True
    #: Max checkpoints kept per side (oldest evicted, latest always retained).
    #: None = unbounded.
    league_pool_max: int | None = Field(default=8, ge=1)
    #: Probability the opponent sampler returns the most recent checkpoint rather
    #: than drawing uniformly from the whole pool.
    league_p_latest: float = Field(default=0.35, ge=0.0, le=1.0)


class EvalConfig(BaseModel):
    """Evaluation harness (arena/eval/)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Best-response training budget for one exploitability measurement. Larger =
    #: a tighter (higher, more honest) exploitability number, at linear cost.
    br_steps: int = Field(default=20_000, ge=1)
    #: Episodes for the post-best-response attack-success estimate.
    n_eval_episodes: int = Field(default=300, ge=1)
    #: Episodes whose Blue decisions feed the AUROC / TPR sweep.
    n_decision_adv: int = Field(default=150, ge=1)
    n_decision_benign: int = Field(default=150, ge=1)
    #: Every trainable defender is calibrated to this per-decision false-positive
    #: rate before its exploitability is measured, so "low exploitability" means
    #: "discriminates well" and not "quarantines everything" (project.md S3.3).
    calibration_fpr: float = Field(default=0.05, gt=0.0, lt=1.0)


class LLMConfig(BaseModel):
    """LLM eval sweep (arena/llm/, M10)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Ollama model name for payload rendering.
    model: str = "qwen2.5:3b"
    #: Sampling temperature.
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    #: Request timeout in seconds.
    timeout: int = Field(default=60, ge=1)
    #: Max retries on transient failures.
    max_retries: int = Field(default=2, ge=0)
    #: Directory for the disk cache (relative to repo root, git-ignored).
    cache_dir: str = ".cache/llm"


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
    selfplay: SelfPlayConfig = SelfPlayConfig()
    eval: EvalConfig = EvalConfig()
    llm: LLMConfig = LLMConfig()

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
