"""Typed configuration, loaded from YAML.

Every size, step count and generation count in ARENA lives here — nothing in
``arena/`` hardcodes scale. Scaling up is a YAML edit (architecture.md S10).
Sections are added as modules land; this file currently covers M1.
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
    n_tools_min: int = Field(default=6, ge=3)
    n_tools_max: int = Field(default=12, ge=3)
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


class ArenaConfig(BaseModel):
    """Top-level config. One YAML file maps to one of these."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "unnamed"
    #: Global seed. Sub-configs may override for their own component.
    seed: int = 0
    scenario: ScenarioConfig = ScenarioConfig()

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
