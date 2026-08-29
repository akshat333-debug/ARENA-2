"""Toucan-1.5M -> a benign-traffic profile that drives :class:`BenignRoller`.

The env generates its own scenarios, so no *training* dataset is needed. But
Blue's false-positive penalty is only meaningful against a realistic benign
distribution. Toucan-1.5M (Apache-2.0, 1.5M real MCP tool-call trajectories)
supplies that: we summarise a subsample into a small :class:`BenignProfile`
(one JSON), and the roller draws tool categories from it instead of uniformly.

Nothing from Toucan is vendored. `arena.data.fetch` pulls a row subsample into
the git-ignored ``data/`` dir; :func:`build_profile` turns it into the profile;
the hot path only ever loads the resulting JSON.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from arena.tools import SideEffect

#: Coarse tool categories. Three is enough to shape benign traffic without
#: pretending a keyword heuristic can recover fine-grained side effects.
CATEGORIES = ("read", "act", "other")

_READ_KW = frozenset((
    "get", "list", "search", "read", "find", "fetch", "query", "lookup", "view",
    "show", "describe", "check", "count", "retrieve", "browse", "scan", "inspect",
    "download", "load", "compare", "compara",
))
_ACT_KW = frozenset((
    "create", "update", "delete", "send", "post", "write", "set", "add", "remove",
    "run", "execute", "submit", "upload", "publish", "insert", "modify", "edit",
    "move", "cancel", "book", "make", "put", "patch", "trigger", "deploy", "start",
    "generate", "draw", "fill", "render", "build", "format", "save", "convert",
))

_SPEC_CATEGORY = {
    SideEffect.READ_BENIGN: "read",
    SideEffect.READ_SENSITIVE: "read",
    SideEffect.EXTERNAL_SINK: "act",
    SideEffect.EXEC: "act",
    SideEffect.WRITE_LOCAL: "act",
    SideEffect.TRANSFORM: "other",
}

DEFAULT_PROFILE_PATH = Path("data") / "toucan" / "profile.json"


def _tokens(text: str) -> list[str]:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)  # split camelCase
    return [t for t in re.split(r"[\s_.\-/]+", text.lower()) if t]


def classify_name(name: str, description: str = "") -> str:
    """Bucket a real MCP tool into read / act / other by its verb.

    ponytail: keyword heuristic. Real Toucan names carry a long server prefix
    (``office-word-document-server-create_document``), so we scan every token —
    camelCase-split — and take the first read/act keyword hit, falling back to
    the description. Calibrate against a hand-labelled sample only if the benign
    distribution turns out to matter to a headline number.
    """
    for t in _tokens(name):
        if t in _READ_KW:
            return "read"
        if t in _ACT_KW:
            return "act"
    for t in _tokens(description)[:12]:
        if t in _READ_KW:
            return "read"
        if t in _ACT_KW:
            return "act"
    return "other"


def category_of_spec(spec) -> str:
    """Category of one of *our* :class:`~arena.tools.ToolSpec` tools."""
    return _SPEC_CATEGORY[spec.side_effect]


@dataclass(frozen=True)
class BenignProfile:
    """Empirical shape of benign Toucan traffic. All histograms are normalised."""

    category_weights: dict[str, float]
    length_hist: dict[int, float]
    n_trajectories: int
    source: str = "Agent-Ark/Toucan-1.5M"

    def __post_init__(self) -> None:
        if not self.category_weights or abs(sum(self.category_weights.values()) - 1.0) > 1e-6:
            raise ValueError("category_weights must be a normalised distribution")
        for c in self.category_weights:
            if c not in CATEGORIES:
                raise ValueError(f"unknown category {c!r}")
        if self.n_trajectories <= 0:
            raise ValueError("n_trajectories must be positive")

    def weight(self, category: str) -> float:
        return self.category_weights.get(category, 0.0)

    def sample_category(self, rng) -> str:
        cats = list(self.category_weights)
        return str(rng.choice(cats, p=[self.category_weights[c] for c in cats]))

    def sample_length(self, rng) -> int:
        if not self.length_hist:
            return 0
        ks = sorted(self.length_hist)
        return int(rng.choice(ks, p=[self.length_hist[k] for k in ks]))

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "n_trajectories": self.n_trajectories,
            "category_weights": self.category_weights,
            "length_hist": {str(k): v for k, v in self.length_hist.items()},
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, d: dict) -> "BenignProfile":
        return cls(
            category_weights={k: float(v) for k, v in d["category_weights"].items()},
            length_hist={int(k): float(v) for k, v in d.get("length_hist", {}).items()},
            n_trajectories=int(d["n_trajectories"]),
            source=d.get("source", "Agent-Ark/Toucan-1.5M"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BenignProfile":
        return cls.from_dict(json.loads(Path(path).read_text()))


def load_profile(path: str | Path = DEFAULT_PROFILE_PATH) -> BenignProfile | None:
    """The fetched profile if it exists, else ``None`` so callers fall back to
    the synthetic roller. Keeps ``data/`` optional for everyone who just runs
    the tests."""
    path = Path(path)
    return BenignProfile.load(path) if path.is_file() else None


# --------------------------------------------------------------------------
# building the profile from a fetched Toucan row subsample
# --------------------------------------------------------------------------

def _tool_calls(messages: list[dict]) -> list[str]:
    """Names of the tools actually invoked in one trajectory's message list."""
    names: list[str] = []
    for m in messages:
        if m.get("role") != "tool_call":
            continue
        raw = m.get("content", "")
        name = ""
        try:  # content is a py-repr-ish dict string: {'name': '...', 'arguments': '...'}
            name = str(json.loads(raw.replace("'", '"')).get("name", ""))
        except (ValueError, AttributeError):
            if "'name':" in raw:
                name = raw.split("'name':", 1)[1].split(",", 1)[0].strip().strip("'\" ")
        if name:
            names.append(name)
    return names


def _iter_rows(source: str | Path):
    """Yield row dicts from a JSONL subsample or a list already in memory."""
    if isinstance(source, (str, Path)) and Path(source).is_file():
        for line in Path(source).read_text().splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)
    else:  # already an iterable of dicts
        yield from source


def build_profile(source, *, max_rows: int | None = None) -> BenignProfile:
    """Summarise fetched Toucan rows into a :class:`BenignProfile`.

    ``source`` is a path to the ``rows.jsonl`` written by ``arena.data.fetch``,
    or an in-memory iterable of row dicts (each with a JSON ``messages`` and a
    ``tools`` list of function specs).
    """
    cat_counts: Counter[str] = Counter()
    len_counts: Counter[int] = Counter()
    n = 0
    for row in _iter_rows(source):
        if max_rows is not None and n >= max_rows:
            break
        msgs = row.get("messages")
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        calls = _tool_calls(msgs or [])
        if not calls:
            continue
        descs = {}
        raw_tools = row.get("tools")
        if isinstance(raw_tools, str):
            raw_tools = json.loads(raw_tools)
        for t in raw_tools or []:
            fn = t.get("function", t)
            descs[fn.get("name", "")] = fn.get("description", "")
        for name in calls:
            cat_counts[classify_name(name, descs.get(name, ""))] += 1
        len_counts[len(calls)] += 1
        n += 1

    if n == 0:
        raise ValueError("no usable trajectories in the Toucan subsample")

    tot_c = sum(cat_counts.values())
    tot_l = sum(len_counts.values())
    return BenignProfile(
        category_weights={c: cat_counts.get(c, 0) / tot_c for c in CATEGORIES},
        length_hist={k: v / tot_l for k, v in sorted(len_counts.items())},
        n_trajectories=n,
    )
