# ARENA

**An Adversarial Co-Evolving Benchmark for Agentic Tool-Use Security**
Game-Theoretic Multi-Agent RL for MCP Systems · CSI4006 Game Theory · VIT Vellore

A self-play RL arena where an attacker (Red) and a defender (Blue) co-evolve inside a
simulated MCP tool-calling environment. The output is a defender battle-tested against an
*adapting* adversary, and an **exploitability** curve rather than static single-pass accuracy.

Why it matters: every existing multi-agent security benchmark scores defenders against a
frozen attack list. TAMAS (2025) shows current frameworks fail 81–82% of the time against
attacks that *don't even adapt*. ARENA measures what happens when the attacker learns too.

## Documents

| Doc | Contents |
|---|---|
| [project.md](project.md) | Requirements, game model, scope decisions, datasets, risks |
| [architecture.md](architecture.md) | Tech stack, env design, anti-leakage rule, module layout |
| [modular-plan.md](modular-plan.md) | M1–M11 build order and per-module test gates |

## Status

| Module | State |
|---|---|
| **M1** — config, tools, scenarios | ✅ done — 74 tests passing |
| M2 — taint, rewards | next |
| M3 — PettingZoo env | pending |
| M4 — features, baselines | pending |
| M5–M11 | pending |

## Install

```bash
pip install -r requirements.txt
```

Python 3.13 · developed on Apple M3 (MPS).

## Run the tests

```bash
python3 -m pytest -q
```

## What M1 gives you

```python
from arena.config import load_config
from arena.scenarios import ScenarioGenerator

gen = ScenarioGenerator.from_config(load_config("small.yaml").scenario)
sc = gen.sample()

sc.domain          # finance / healthcare / legal / news / education
sc.task            # the legitimate-looking cover task
sc.registry.names  # the tools exposed this episode
sc.objective       # Red's ground-truth win condition (None if benign)
```

- **Tool registry** with observable metadata and side-effect classes
  (`read_sensitive`, `transform`, `external_sink`, …). Every sampled registry is guaranteed
  to host a source → propagator → sink chain, so no episode is unwinnable by construction.
- **Scenario generator** covering all six TAMAS attack families, each extended into a
  *chained* variant (`min_chain_len > 1`) — no attack completes in a single call, which is
  the whole point.
- **Typed config**, seeded and reproducible. `configs/small.yaml` is the default and runs on
  a laptop; `configs/paper.yaml` is the scale-up target. No scale value is hardcoded.

## Configuration

Scale lives entirely in YAML — `small.yaml` (default, M3-friendly) and `paper.yaml`
(scale-up). A single top-level `seed:` propagates to every section, so a whole run is
reproducible from one number. Unknown keys are rejected rather than silently ignored.

## Licence and attribution

Third-party datasets (TAMAS, Toucan-1.5M) are fetched at M9, never vendored, and recorded
with checksum and licence in `THIRD_PARTY.md`. See [project.md §13](project.md).
