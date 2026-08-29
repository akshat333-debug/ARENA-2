"""Fetch the public datasets into the git-ignored ``data/`` dir (M9).

    python3 scripts/fetch_data.py [--dest data] [--n-toucan 8000] [--force]

Idempotent — a second run does nothing unless a file changed or --force is set.
Writes data/manifest.json (sha256 + licence + retrieval time per artifact) and
data/toucan/profile.json (the benign-traffic profile). See THIRD_PARTY.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena.data.fetch import DATA_DIR, fetch_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(DATA_DIR))
    ap.add_argument("--n-toucan", type=int, default=8000, help="benign trajectories to subsample")
    ap.add_argument("--force", action="store_true", help="re-fetch even if the manifest verifies")
    a = ap.parse_args()

    man = fetch_all(Path(a.dest), n_toucan=a.n_toucan, force=a.force)
    for rel, meta in man["artifacts"].items():
        print(f"{rel:<26} {meta['bytes']:>10} B  {meta['licence']}  {meta['sha256'][:12]}…")
    print(f"\nmanifest: {Path(a.dest) / 'manifest.json'}")
    print(json.dumps({"artifacts": list(man["artifacts"])}, indent=2))


if __name__ == "__main__":
    main()
