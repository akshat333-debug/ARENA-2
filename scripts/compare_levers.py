"""Compare multi-seed lever runs against the baseline (M11).

    python3 scripts/compare_levers.py report/multiseed_baseline.json report/lev_*.json

Each input is a `run_multiseed.py --out` JSON. Judged on BOTH the mean and the
spread: the M11 multi-seed result showed the open problem is arena_blue's seed
variance (+/-0.274 exploitability), not a mean gap, so a lever that moves the
mean while leaving the spread untouched has not addressed it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFENDER = "arena_blue"
METRICS = ("exploitability", "auroc", "tpr_at_5pct_fpr")


def load(path: Path) -> tuple[str, dict, dict]:
    d = json.loads(path.read_text())
    label = d.get("label") or path.stem
    row = next((r for r in d["aggregate"] if r["name"] == DEFENDER), None)
    if row is None:
        raise SystemExit(f"{path}: no {DEFENDER} row (was it run with --no-arena-blue?)")
    per_seed = {
        s: next(r for r in rows if r["name"] == DEFENDER)
        for s, rows in d["per_seed"].items()
    }
    return label, row, per_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("levers", type=Path, nargs="*")
    args = ap.parse_args()

    base_label, base, base_seeds = load(args.baseline)
    rows = [(base_label, base, base_seeds)]
    for p in args.levers:
        if p.resolve() != args.baseline.resolve():
            rows.append(load(p))

    head = f"{'lever':<12}" + "".join(f"{m:>26}" for m in METRICS)
    print(f"=== {DEFENDER} across levers (mean +/- spread, n seeds) ===")
    print(head)
    print("-" * len(head))
    for label, r, _ in rows:
        cells = "".join(
            f"{r[m + '_mean']:.3f}+/-{r[m + '_std']:.3f}".rjust(26) for m in METRICS
        )
        print(f"{label:<12}{cells}")

    print()
    print("--- vs baseline: does the lever move the mean, and does it shrink the spread? ---")
    for label, r, _ in rows[1:]:
        for m in METRICS:
            dmean = r[f"{m}_mean"] - base[f"{m}_mean"]
            dstd = r[f"{m}_std"] - base[f"{m}_std"]
            better = "better" if (dmean < 0 if m == "exploitability" else dmean > 0) else "worse"
            spread = "tighter" if dstd < 0 else "wider"
            print(f"  {label:<10} {m:<18} mean {dmean:+.3f} ({better})   spread {dstd:+.3f} ({spread})")
        print()

    print("Reminder: a mean shift smaller than the baseline spread is not a result "
          "(docs/m11-multiseed.md). Read the spread column first.")


if __name__ == "__main__":
    main()
