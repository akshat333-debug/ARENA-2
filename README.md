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
| **M1** — config, tools, scenarios | ✅ done |
| **M2** — taint tracker, reward engine | ✅ done |
| **M3** — PettingZoo env + Gym wrapper | ✅ done — 173 tests passing |
| M4 — features, baselines | next |
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

## What M2 gives you

```python
from arena.taint import TaintTracker, replay
from arena.rewards import EpisodeOutcome, compute_rewards
from arena.config import RewardConfig

# read .env -> summarize -> POST webhook  ==  exfiltration, detected by data flow
tr = replay([(env_file, None), (summarize, None), (webhook, None)])
tr.objective_completed(scenario.objective)   # True
tr.attack_path(scenario.objective)           # (0, 1, 2) - ground-truth malicious steps
```

- **Taint tracker** (`arena/taint.py`) — the environment's internal ground truth for
  *"did the attack objective actually complete?"*, computed from the executed call
  sequence. Sources mint `SENSITIVE`/`UNTRUSTED` tags, propagators (`TRANSFORM`/`EXEC`)
  carry them, sinks record a `FlowEvent` when tainted data crosses the boundary.
  **Blue never sees any of this** — it is what stops Red from farming reward with
  attacks that never move sensitive data.
- **Reward engine** (`arena/rewards.py`) — pure functions over an `EpisodeOutcome`.
  Asymmetric: Red is paid for undetected completion + task-plausibility − step cost;
  Blue for stopping an attack in time and for flagging real malicious calls, penalised
  for false quarantines and off-chain flags.
- Every family has a verified minimal winning sequence in its own registry — checked in
  `tests/test_m2_integration.py`, so no episode is unwinnable by construction.

## What M3 gives you

```python
from arena.env import ARENAEnv, SingleAgentARENA, RED, ALLOW, QUARANTINE
from arena.config import load_config

env = ARENAEnv(load_config("small.yaml"))   # PettingZoo AEC: agents "red_0", "blue_0"
env.reset(seed=0)
# red_0 picks a tool index; blue_0 returns allow / flag / quarantine on the sequence

# single-agent view for PPO / exploitability: train Red vs a frozen Blue policy
gym_env = SingleAgentARENA(RED, opponent=frozen_blue_policy, config=load_config("small.yaml"))
```

- **PettingZoo AEC env** — one episode = one MCP session. Passes `pettingzoo.test.api_test`.
  Red proposes tool calls (index into the episode registry); Blue adjudicates the whole
  sequence. Ends on quarantine, objective completion, or step cap.
- **Anti-leakage invariant enforced** — Blue's observation is built from observable call
  metadata only and is provably invariant to the internal taint graph
  (`tests/test_env.py`). Two episodes with the same tool sequence but different objectives
  give Blue byte-identical observations.
- **Gymnasium single-agent wrapper** — `SingleAgentARENA` drives one side with a frozen
  policy and exposes the other as a standard `gym.Env`. This is the M8 exploitability
  primitive: freeze Blue, train a fresh best-response Red, measure success rate.
- Headline claim demonstrated in `tests/test_m3_integration.py`: a sequence-aware Blue
  stops the split-chain exfiltration that a per-call Blue allows through.

## Configuration

Scale lives entirely in YAML — `small.yaml` (default, M3-friendly) and `paper.yaml`
(scale-up). A single top-level `seed:` propagates to every section, so a whole run is
reproducible from one number. Unknown keys are rejected rather than silently ignored.

## Licence and attribution

Third-party datasets (TAMAS, Toucan-1.5M) are fetched at M9, never vendored, and recorded
with checksum and licence in `THIRD_PARTY.md`. See [project.md §13](project.md).
