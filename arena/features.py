"""Observation encoders and baseline feature views — OBSERVABLE METADATA ONLY.

Everything here is built from what a defender can legitimately see about a tool
call: its declared side-effect class, category, sensitivity, position in the
session, and past verdicts. Nothing reads :class:`arena.taint.TaintTracker`.
Extracted from ``arena/env.py`` (M3) so the M4 baseline detectors and the M5
policy networks share exactly one definition of "the observable state".

Two feature views for the baselines:

* :func:`single_call_features` — the pending call in isolation (no history). What
  a TAMAS-style single-shot per-call detector gets. A lone ``read_env_file`` is
  byte-identical whether the episode is a benign changelog task or an
  exfiltration, so this view *cannot* separate the split chain — which is the
  whole point of the comparison.
* :func:`sequence_features` — a fixed-width trajectory summary with observable
  causal proxies (``sensitive_read_before_sink`` etc.). What a CASPIAN-style
  causal monitor gets, and what ARENA's trained Blue must beat.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from gymnasium import spaces

from arena.tools import ToolSpec

# Verdict encoding, shared with the env.
ALLOW, FLAG, QUARANTINE = 0, 1, 2
VERDICTS = (ALLOW, FLAG, QUARANTINE)

# Fixed one-hot orderings: an observation means the same thing in every episode
# regardless of which tools this registry happens to hold.
SIDE_EFFECTS = ("read_benign", "read_sensitive", "transform", "write_local", "external_sink", "exec")
CATEGORIES = ("filesystem", "secrets", "network", "compute", "communication", "content", "records", "transaction")
DOMAINS = ("news", "education", "finance", "healthcare", "legal")

_SE_IX = {s: i for i, s in enumerate(SIDE_EFFECTS)}
_CAT_IX = {c: i for i, c in enumerate(CATEGORIES)}
_DOM_IX = {d: i for i, d in enumerate(DOMAINS)}

SE_SENSITIVE = _SE_IX["read_sensitive"]
SE_SINK = _SE_IX["external_sink"]

# Per-call feature row: side-effect one-hot (6) + category one-hot (8)
#   + sensitivity/2 + is_untrusted_read + is_privileged + position + resolved? (5)
#   + verdict one-hot (allow, flag) (2)
CALL_FEATS = len(SIDE_EFFECTS) + len(CATEGORIES) + 7
# Red only: Blue's verdict incl. quarantine appended, one-hot (3).
RED_EXTRA = 3

# Width of single_call_features: the two one-hot blocks + 3 scalars.
SINGLE_CALL_FEATS = len(SIDE_EFFECTS) + len(CATEGORIES) + 3


def encode_call(
    tool: ToolSpec,
    step_index: int,
    max_steps: int,
    *,
    resolved: bool,
    verdict: int | None,
    for_red: bool,
) -> np.ndarray:
    """One call as a feature row. ``verdict`` is ``None`` for the pending
    proposal. ``for_red`` appends Blue's full verdict (Red's feedback channel);
    Blue's own view stops at allow/flag."""
    width = CALL_FEATS + (RED_EXTRA if for_red else 0)
    v = np.zeros(width, dtype=np.float32)
    o = 0
    v[o + _SE_IX[tool.side_effect.value]] = 1.0
    o += len(SIDE_EFFECTS)
    v[o + _CAT_IX[tool.category]] = 1.0
    o += len(CATEGORIES)
    v[o] = tool.sensitivity / 2.0
    v[o + 1] = float(tool.is_untrusted_read)
    v[o + 2] = float(tool.is_privileged)
    v[o + 3] = step_index / max(1, max_steps)
    v[o + 4] = float(resolved)
    o += 5
    if resolved and verdict in (ALLOW, FLAG):
        v[o + verdict] = 1.0
    o += 2
    if for_red and resolved and verdict in VERDICTS:
        v[o + verdict] = 1.0
    return v


def history_matrix(
    history: Sequence[tuple[ToolSpec, int, int]],
    pending: tuple[ToolSpec, int] | None,
    max_steps: int,
    *,
    for_red: bool,
) -> tuple[np.ndarray, int]:
    """``history`` is ``(tool, step_index, verdict)`` per resolved call; ``pending``
    is ``(tool, step_index)`` for the unresolved proposal or ``None``."""
    width = CALL_FEATS + (RED_EXTRA if for_red else 0)
    mat = np.zeros((max_steps, width), dtype=np.float32)
    for i, (tool, step_index, verdict) in enumerate(history):
        mat[i] = encode_call(
            tool, step_index, max_steps, resolved=True, verdict=verdict, for_red=for_red
        )
    length = len(history)
    if pending is not None and length < max_steps:
        tool, step_index = pending
        mat[length] = encode_call(
            tool, step_index, max_steps, resolved=False, verdict=None, for_red=for_red
        )
        length += 1
    return mat, length


def domain_onehot(domain_value: str) -> np.ndarray:
    v = np.zeros(len(DOMAINS), dtype=np.float32)
    v[_DOM_IX[domain_value]] = 1.0
    return v


def blue_observation_space(max_steps: int) -> spaces.Dict:
    return spaces.Dict(
        {
            "domain": spaces.Box(0.0, 1.0, (len(DOMAINS),), np.float32),
            "calls": spaces.Box(0.0, 1.0, (max_steps, CALL_FEATS), np.float32),
            "length": spaces.Box(0, max_steps, (1,), np.int32),
        }
    )


def red_observation_space(max_steps: int, n_tools_max: int) -> spaces.Dict:
    return spaces.Dict(
        {
            "task": spaces.Box(0.0, 1.0, (len(DOMAINS) + 1,), np.float32),
            "registry": spaces.Box(0.0, 1.0, (n_tools_max, CALL_FEATS), np.float32),
            "registry_mask": spaces.MultiBinary(n_tools_max),
            "objective": spaces.Box(0.0, 1.0, (4 + 2 * n_tools_max,), np.float32),
            "history": spaces.Box(0.0, 1.0, (max_steps, CALL_FEATS + RED_EXTRA), np.float32),
            "length": spaces.Box(0, max_steps, (1,), np.int32),
        }
    )


# --- baseline feature views ------------------------------------------


def single_call_features(tool: ToolSpec) -> np.ndarray:
    """The pending call in isolation — no history, no position. This is
    deliberately blind to the trajectory."""
    v = np.zeros(SINGLE_CALL_FEATS, dtype=np.float32)
    v[_SE_IX[tool.side_effect.value]] = 1.0
    o = len(SIDE_EFFECTS)
    v[o + _CAT_IX[tool.category]] = 1.0
    o += len(CATEGORIES)
    v[o] = tool.sensitivity / 2.0
    v[o + 1] = float(tool.is_untrusted_read)
    v[o + 2] = float(tool.is_privileged)
    return v


SEQUENCE_FEATURE_NAMES = (
    *(f"frac_{s}" for s in SIDE_EFFECTS),
    "n_calls_norm",
    "frac_risky",
    "has_sensitive_read",
    "has_untrusted_read",
    "has_external_sink",
    "sensitive_read_before_sink",
    "untrusted_read_before_privileged",
    "launder_depth_norm",
    "max_sensitivity",
)
SEQUENCE_FEATS = len(SEQUENCE_FEATURE_NAMES)
_SEQ_NORM = 24.0  # a generous session length for normalisation


# A call reduced to the observable fields the sequence features need.
#   (side_effect_value, sensitivity, is_untrusted_read, is_privileged)
_CallView = tuple[str, int, bool, bool]


def _views_from_tools(tools: Sequence[ToolSpec]) -> list[_CallView]:
    return [(t.side_effect.value, t.sensitivity, t.is_untrusted_read, t.is_privileged) for t in tools]


def _views_from_rows(rows: np.ndarray) -> list[_CallView]:
    """``rows`` is an ``(n, >=17)`` slice of encoded call rows (:func:`encode_call`
    output). Only the observable prefix is read."""
    out: list[_CallView] = []
    for row in rows:
        se = SIDE_EFFECTS[int(np.argmax(row[: len(SIDE_EFFECTS)]))]
        sensitivity = int(round(float(row[len(SIDE_EFFECTS) + len(CATEGORIES)]) * 2))
        untrusted = bool(row[len(SIDE_EFFECTS) + len(CATEGORIES) + 1] > 0.5)
        privileged = bool(row[len(SIDE_EFFECTS) + len(CATEGORIES) + 2] > 0.5)
        out.append((se, sensitivity, untrusted, privileged))
    return out


def _sequence_features_core(views: Sequence[_CallView]) -> np.ndarray:
    v = np.zeros(SEQUENCE_FEATS, dtype=np.float32)
    n = len(views)
    if n == 0:
        return v

    counts = np.zeros(len(SIDE_EFFECTS), dtype=np.float32)
    first_sensitive = first_untrusted = None
    last_sink = last_privileged = None
    n_transforms_after_sensitive = 0
    max_sens = 0
    for i, (se, sensitivity, untrusted, privileged) in enumerate(views):
        counts[_SE_IX[se]] += 1
        max_sens = max(max_sens, sensitivity)
        is_sink = se == "external_sink"
        propagates = se in ("transform", "exec")
        if se == "read_sensitive" and first_sensitive is None:
            first_sensitive = i
        if untrusted and first_untrusted is None:
            first_untrusted = i
        if is_sink:
            last_sink = i
        if privileged:
            last_privileged = i
        if propagates and first_sensitive is not None and i > first_sensitive:
            n_transforms_after_sensitive += 1

    v[: len(SIDE_EFFECTS)] = counts / n
    o = len(SIDE_EFFECTS)
    v[o] = min(1.0, n / _SEQ_NORM)
    risky = counts[SE_SENSITIVE] + counts[SE_SINK]
    v[o + 1] = risky / n
    v[o + 2] = float(first_sensitive is not None)
    v[o + 3] = float(first_untrusted is not None)
    v[o + 4] = float(last_sink is not None)
    v[o + 5] = float(
        first_sensitive is not None and last_sink is not None and last_sink > first_sensitive
    )
    v[o + 6] = float(
        first_untrusted is not None
        and last_privileged is not None
        and last_privileged > first_untrusted
    )
    v[o + 7] = min(1.0, n_transforms_after_sensitive / 6.0)
    v[o + 8] = max_sens / 2.0
    return v


def sequence_features(tools: Sequence[ToolSpec]) -> np.ndarray:
    """Fixed-width observable summary of a call sequence (including the pending
    call), from :class:`ToolSpec` objects. Carries the causal proxies a
    sequence-aware detector needs."""
    return _sequence_features_core(_views_from_tools(tools))


def sequence_features_from_rows(rows: np.ndarray) -> np.ndarray:
    """Same summary, computed from encoded call rows — what a defender has at
    inference time from its observation alone."""
    return _sequence_features_core(_views_from_rows(rows))
