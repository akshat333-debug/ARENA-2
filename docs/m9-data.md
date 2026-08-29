# M9 — Public datasets

Module: [`arena/data/`](../arena/data/) — `fetch.py`, `toucan.py`.
Script: [`scripts/fetch_data.py`](../scripts/fetch_data.py). Provenance: [`THIRD_PARTY.md`](../THIRD_PARTY.md).

ARENA generates its own scenarios, so no dataset is needed to train or evaluate.
Two public datasets are pulled in for grounding only, and **never vendored** —
`data/` is git-ignored, a manifest records every file's hash.

## Fetch — `fetch.py`

```bash
python3 scripts/fetch_data.py               # ~8k Toucan rows + the TAMAS tarball
python3 scripts/fetch_data.py --force       # re-fetch and re-hash
python3 scripts/fetch_data.py --n-toucan 20000
```

- **TAMAS** — the repo tarball (`https://github.com/microsoft/TAMAS/.../main.tar.gz`).
  ponytail: kept whole, not extracted. Nothing in `arena/` parses TAMAS — the
  six-family taxonomy is already in `scenarios.py` and the ~80% figure is a
  citation. An audit that wants to diff taxonomies untars it then.
- **Toucan-1.5M** — an ~8k-row subsample of the `SFT` config via the HF
  datasets-server **rows API** (plain JSON), written to `data/toucan/rows.jsonl`.
  No parquet reader, no `datasets` library — **M9 adds zero runtime dependencies**
  (stdlib `urllib` / `json` / `hashlib`).

**Idempotent.** `fetch_all` writes `data/manifest.json` with, per artifact:
sha256, byte size, source URL, licence, UTC retrieval time. A re-run calls
`verify()` (manifest present *and* every listed file still hashes to its
recorded value); if it passes and `--force` is absent, the network is never
touched.

```
data/
  manifest.json
  tamas/source.tar.gz
  toucan/rows.jsonl          # the subsample (git-ignored)
  toucan/profile.json        # the derived benign profile
```

## Benign profile — `toucan.py`

The raw rows are not used at runtime. `build_profile` summarises them into a
small `BenignProfile`:

- `category_weights` — fraction of real tool calls that are **read** / **act** /
  **other**, from a keyword classifier over the (camelCase-split, prefix-heavy)
  tool names: `office-word-document-server-create_document` → `act`,
  `time-mcp-server-get_timestamp` → `read`. On the fetched sample: roughly
  read 0.41 / act 0.31 / other 0.28.
- `length_hist` — normalised distribution of tool calls per trajectory.

ponytail: three coarse categories, one keyword heuristic with a calibration
note in the source. Fine-grained side-effect recovery from a tool name is not
something a word list can do, and the benign distribution does not (yet) feed a
headline number — it shapes Blue's false-positive pressure.

## The swap-in

`ScenarioConfig.benign_source`:

| value | benign traffic |
|---|---|
| `synthetic` (default) | `BenignRoller` picks tools uniformly (M1 behaviour, unchanged) |
| `toucan` | `BenignRoller` draws a **category** from the Toucan profile each step, then a uniform tool of that category — subject to the same "no sink after a source" invariant |

`toucan` falls back to `synthetic` automatically if `data/toucan/profile.json`
is absent, so the test suite never needs the datasets. Wired through
`collect_decisions` and `false_quarantine_rate` (the two places benign realism
changes a number); `AdaptiveRed` takes a `benign_profile=` for self-play.

## Tests

- `tests/test_data.py` (32, fast, network monkeypatched) — `classify_name` /
  `category_of_spec` tables; `build_profile` from in-memory rows (hand-computed
  weights); `BenignProfile` validation + JSON round-trip; **a skewed profile
  measurably shifts `BenignRoller`'s category mix** and in the right direction
  vs the uniform roller, with the sink invariant preserved; `fetch_all` writes a
  complete manifest, is idempotent (asserts zero network calls on re-run), and
  `verify()` fails on a tampered file; the `benign_source` Literal is enforced;
  `THIRD_PARTY.md` records both datasets with licences and URLs.
- `tests/test_m9_integration.py` (3, `slow`, real endpoints, skips if
  unreachable) — a live ~200-row fetch verifies, pins the HF commit sha, and the
  real profile is well-formed and moves benign traffic toward reads vs uniform.
