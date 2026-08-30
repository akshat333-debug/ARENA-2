"""Report generation helpers (M11).

Plotting and table-formatting utilities consumed by ``scripts/reproduce.py``
and the paper draft.  Keeps the report decoupled from the eval harness —
every function takes data, not config.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def format_leaderboard_table(rows: list[dict], *, title: str = "Leaderboard") -> str:
    """Render a markdown table from a list of row dicts."""
    if not rows:
        return f"**{title}**\n\nNo data."
    cols = list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "|---" * len(cols) + "|"
    lines = [f"**{title}**", "", header, sep]
    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def format_exploitability_curve(points: list[dict], *, title: str = "Exploitability Curve") -> str:
    """Render a markdown table from curve points."""
    if not points:
        return f"**{title}**\n\nNo data."
    lines = [f"**{title}**", "", "| Generation | Exploitability | Fresh Red Start | Delta |",
             "|---|---|---|---|"]
    prev = None
    for p in points:
        delta = f"{p['exploitability'] - prev:+.3f}" if prev is not None else "—"
        lines.append(
            f"| {p['label']} | {p['exploitability']:.3f} | {p['fresh_red_start']:.3f} | {delta} |"
        )
        prev = p["exploitability"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting (lazy matplotlib import so the module loads without it)
# ---------------------------------------------------------------------------

def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    return plt


def plot_exploitability_curve(
    points: list[dict],
    out: str | Path,
    *,
    title: str = "Exploitability Across Self-Play Generations",
    xlabel: str = "Generation",
    ylabel: str = "Exploitability (attack success)",
) -> Path:
    """Save an exploitability curve plot. Each point has 'label', 'exploitability',
    'fresh_red_start'. Returns the output path."""
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [p["label"] for p in points]
    expl = [p["exploitability"] for p in points]
    fresh = [p["fresh_red_start"] for p in points]
    x = list(range(len(points)))

    ax.plot(x, expl, "o-", color="#d32f2f", linewidth=2, markersize=8, label="Exploitability")
    ax.plot(x, fresh, "s--", color="#90a4ae", linewidth=1.5, markersize=6, label="Fresh Red start")

    # Reference line for TAMAS static baseline failure (~0.80)
    ax.axhline(y=0.80, color="#ffb74d", linestyle=":", linewidth=1.5, alpha=0.7,
               label="TAMAS static baseline (~0.80)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_leaderboard_comparison(
    rows: list[dict],
    out: str | Path,
    *,
    title: str = "Defender Comparison",
) -> Path:
    """Save a grouped bar chart comparing defenders on AUROC, TPR@5%FPR, exploitability."""
    plt = _import_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    names = [r["name"] for r in rows]
    metrics = ["auroc", "tpr_at_5pct_fpr", "exploitability"]
    metric_labels = ["AUROC", "TPR@5%FPR", "Exploitability"]
    colors = ["#1976d2", "#388e3c", "#d32f2f"]

    for ax, metric, label, color in zip(axes, metrics, metric_labels, colors):
        vals = [r[metric] for r in rows]
        bars = ax.bar(names, vals, color=color, alpha=0.8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", rotation=30)
        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def save_results(data: dict, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, default=str))
    return out


def load_results(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
