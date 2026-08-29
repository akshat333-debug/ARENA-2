"""Idempotent fetch of the two public datasets (project.md S13).

Nothing is vendored. This script pulls into a git-ignored ``data/`` dir and
writes ``data/manifest.json`` recording, per artifact: sha256, byte size,
source URL, licence, and the UTC retrieval time. A second run is a no-op unless
a file is missing / changed or ``--force`` is given.

Datasets:
  * **TAMAS**   https://github.com/microsoft/TAMAS  — MIT (code) / CDLA-Permissive-2.0 (data)
  * **Toucan-1.5M** https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M — Apache-2.0

Only stdlib is used (urllib / json / hashlib / tarfile) — no dataset library is
a runtime dependency of ARENA.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from arena.data.toucan import build_profile

DATA_DIR = Path("data")
MANIFEST = "manifest.json"

TAMAS_TARBALL = "https://github.com/microsoft/TAMAS/archive/refs/heads/main.tar.gz"
TAMAS_LICENCE = "MIT (code) / CDLA-Permissive-2.0 (data)"

TOUCAN_REPO = "Agent-Ark/Toucan-1.5M"
TOUCAN_CONFIG = "SFT"
TOUCAN_SPLIT = "train"
TOUCAN_LICENCE = "Apache-2.0"
TOUCAN_PAGE = f"https://huggingface.co/datasets/{TOUCAN_REPO}"
_ROWS_API = "https://datasets-server.huggingface.co/rows"
_DS_API = f"https://huggingface.co/api/datasets/{TOUCAN_REPO}"
_PAGE = 100  # datasets-server hard cap per request

_UA = {"User-Agent": "arena-dataset-fetch/1.0 (+https://github.com/akshat333-debug/ARENA-2)"}


# --------------------------------------------------------------------------
# tiny stdlib network helpers (injectable so tests never hit the network)
# --------------------------------------------------------------------------

def _get(url: str, *, retries: int = 4, timeout: int = 60) -> bytes:
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (https only, fixed hosts)
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:  # pragma: no cover - network flake
            last = e
            time.sleep(2 ** i)
    raise RuntimeError(f"GET failed after {retries} tries: {url}") from last


def _get_json(url: str) -> dict:
    return json.loads(_get(url))


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(dest: Path = DATA_DIR) -> Path:
    return Path(dest) / MANIFEST


def load_manifest(dest: Path = DATA_DIR) -> dict | None:
    p = manifest_path(dest)
    return json.loads(p.read_text()) if p.is_file() else None


def verify(dest: Path = DATA_DIR) -> bool:
    """True iff a manifest exists and every artifact it lists is present on disk
    with a matching sha256."""
    man = load_manifest(dest)
    if not man:
        return False
    for rel, meta in man.get("artifacts", {}).items():
        f = Path(dest) / rel
        if not f.is_file() or _sha256(f) != meta["sha256"]:
            return False
    return True


def _record(dest: Path, rel: str, *, source: str, licence: str, **extra) -> dict:
    f = Path(dest) / rel
    return {
        "sha256": _sha256(f),
        "bytes": f.stat().st_size,
        "source": source,
        "licence": licence,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }


# --------------------------------------------------------------------------
# TAMAS
# --------------------------------------------------------------------------

def fetch_tamas(dest: Path = DATA_DIR) -> dict:
    """Download the TAMAS repo tarball. ponytail: kept whole, not extracted —
    nothing in ARENA consumes TAMAS programmatically (its taxonomy is already
    encoded in ``scenarios.py`` and its 80% number is a citation). An audit that
    wants to diff taxonomies can untar it then."""
    out = Path(dest) / "tamas" / "source.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_get(TAMAS_TARBALL))
    return {"tamas/source.tar.gz": _record(
        dest, "tamas/source.tar.gz", source=TAMAS_TARBALL, licence=TAMAS_LICENCE,
    )}


# --------------------------------------------------------------------------
# Toucan
# --------------------------------------------------------------------------

def fetch_toucan(
    dest: Path = DATA_DIR,
    *,
    n_rows: int = 8000,
    config: str = TOUCAN_CONFIG,
) -> dict:
    """Subsample ``n_rows`` benign trajectories via the HF datasets-server rows
    API (JSON, no parquet reader needed), write ``data/toucan/rows.jsonl``, and
    build the ``BenignProfile`` from them."""
    outdir = Path(dest) / "toucan"
    outdir.mkdir(parents=True, exist_ok=True)
    rows_file = outdir / "rows.jsonl"

    try:
        ds_sha = _get_json(_DS_API).get("sha", "")
    except Exception:  # pragma: no cover - provenance is best-effort
        ds_sha = ""

    written = 0
    with rows_file.open("w") as fh:
        for offset in range(0, n_rows, _PAGE):
            length = min(_PAGE, n_rows - offset)
            url = (f"{_ROWS_API}?dataset={TOUCAN_REPO}&config={config}"
                   f"&split={TOUCAN_SPLIT}&offset={offset}&length={length}")
            batch = _get_json(url).get("rows", [])
            if not batch:
                break
            for item in batch:
                fh.write(json.dumps(item.get("row", item)) + "\n")
                written += 1

    if written == 0:
        raise RuntimeError("Toucan rows API returned nothing")

    profile = build_profile(rows_file)
    profile.save(outdir / "profile.json")

    src = f"{TOUCAN_PAGE} (config={config}, split={TOUCAN_SPLIT})"
    return {
        "toucan/rows.jsonl": _record(
            dest, "toucan/rows.jsonl", source=src, licence=TOUCAN_LICENCE,
            rows=written, dataset_sha=ds_sha,
        ),
        "toucan/profile.json": _record(
            dest, "toucan/profile.json", source=src, licence=TOUCAN_LICENCE,
            derived_from="toucan/rows.jsonl", n_trajectories=profile.n_trajectories,
        ),
    }


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------

def fetch_all(dest: Path = DATA_DIR, *, n_toucan: int = 8000, force: bool = False) -> dict:
    """Fetch both datasets into ``dest``. Idempotent: if the manifest verifies
    and ``force`` is false, returns the existing manifest untouched."""
    dest = Path(dest)
    if not force and verify(dest):
        return load_manifest(dest)

    dest.mkdir(parents=True, exist_ok=True)
    artifacts: dict = {}
    artifacts.update(fetch_tamas(dest))
    artifacts.update(fetch_toucan(dest, n_rows=n_toucan))

    man = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifacts": artifacts,
    }
    manifest_path(dest).write_text(json.dumps(man, indent=2, sort_keys=True))
    return man


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default=str(DATA_DIR))
    ap.add_argument("--n-toucan", type=int, default=8000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    m = fetch_all(Path(a.dest), n_toucan=a.n_toucan, force=a.force)
    print(json.dumps(m, indent=2, sort_keys=True))
