# Third-party datasets

ARENA generates its own scenarios — no external dataset is required to train or
evaluate. The datasets below are pulled in **only** for baseline grounding and a
realistic benign-traffic distribution. They are **not vendored**: a fetch script
downloads them into a git-ignored `data/` directory and records a manifest
(`data/manifest.json`) with a SHA-256, byte size, source URL, licence, and UTC
retrieval time for every file.

Fetch:

```bash
python3 scripts/fetch_data.py            # ~8k Toucan rows + TAMAS tarball
python3 scripts/fetch_data.py --force    # re-fetch and re-hash
```

Licences below were read from the source, not assumed. Both permit
redistribution; we redistribute nothing regardless.

| Dataset | Source | Licence | Version / date | What ARENA uses it for |
|---|---|---|---|---|
| **TAMAS** | https://github.com/microsoft/TAMAS | **MIT** (code) · **CDLA-Permissive-2.0** (data) | v1, arXiv:2511.05269 (Nov 2025) | Taxonomy grounding for the six attack families and the ~80–82% static-baseline failure reference. Downloaded as the repo tarball and kept whole — nothing in `arena/` parses it; the taxonomy is already encoded in `arena/scenarios.py`. |
| **Toucan-1.5M** | https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M | **Apache-2.0** | 1.5M trajectories, arXiv:2510.01179 (Oct 2025) | Realistic **benign** traffic. A subsample (`SFT` config, ~8k rows via the HF datasets-server rows API) is summarised into `data/toucan/profile.json` — per-category (read / act / other) call frequencies and a trajectory-length histogram. `arena.data.toucan` loads that profile; with `scenario.benign_source: toucan` the `BenignRoller` draws tool categories from it instead of uniformly. The raw rows are not used at runtime. |

**Rejected:** `obaydata/mcp-agent-trajectory-benchmark` (Apache-2.0, but only
~38–49 trajectories — subsumed by Toucan). `MCP-Atlas` (Apache-2.0 harness, but
measures tool-use accuracy, not security).

## Fetch provenance

`data/manifest.json` (written by `arena.data.fetch`) is the authoritative record
for a given checkout. It is git-ignored along with `data/` — regenerate it with
the command above; the hashes make a re-fetch verifiable.
