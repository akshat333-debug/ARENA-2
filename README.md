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
| [docs/audit-m1-m9.md](docs/audit-m1-m9.md) | **Full audit — 3 bugs found, and the corrected headline result** |

## Status

| Module | State |
|---|---|
| **M1** — config, tools, scenarios | ✅ done |
| **M2** — taint tracker, reward engine | ✅ done |
| **M3** — PettingZoo env + Gym wrapper | ✅ done |
| **M4** — features + 3 baseline defenders | ✅ done |
| **M5** — Red/Blue policies + PPO | ✅ done |
| **M6** — alternating self-play loop | ✅ done |
| **M7** — league / opponent-checkpoint pool | ✅ done |
| **M8** — evaluation harness (exploitability + AUROC/TPR) | ✅ done |
| **M9** — public-dataset fetch (TAMAS, Toucan) | ✅ done |
| **Audit M1–M9** | ✅ done — 4 bugs fixed; 343 fast + 32 slow passing. [Read it](docs/audit-m1-m9.md) |
| **M10** — cached Ollama client + transfer sweep | ✅ done — 30 fast tests passing. [Read it](docs/m10-llm.md) |
| **M11** — leaderboard + report | ✅ done — reproduce script, paper draft, plots. [Read it](report/paper.md) |

## What Was Added (dev branch)

All M10 and M11 modules were implemented on top of the existing M1–M9 codebase:

| Addition | Files | Tests |
|----------|-------|-------|
| **M10: LLM-planned attacks** | `arena/llm/client.py`, `attacker.py`, `payloads.py`, `sweep.py` | 33 fast + 24 planner + slow |
| **M10: Sweep script** | `scripts/run_sweep.py` | — |
| **M10: Config** | `arena/config.py` (`LLMConfig` section) | — |
| **M11: Report helpers** | `arena/eval/report.py` (tables, plots, JSON) | 8 fast |
| **M11: Reproduce script** | `scripts/reproduce.py` | — |
| **M11: Paper draft** | `report/paper.md` | — |
| **M10: Docs** | `docs/m10-llm.md` | — |
| **Dependencies** | `requirements.txt` (`ollama>=0.4`, `matplotlib>=3.8`) | — |

**M10 note.** The sweep is verified end-to-end against a live `qwen2.5:3b`. Its held-out
variable is the **attack plan**, not payload prose: Blue observes tool-call metadata only,
so an LLM payload appended to the task text is invisible to every defender and measures
nothing. See [docs/m10-llm.md](docs/m10-llm.md). Without Ollama the sweep runs the scripted
arm alone and says so; `ollama pull qwen2.5:3b` enables the held-out arm.

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

## What M4 gives you

```bash
python3 scripts/run_baselines.py small.yaml     # train + score the 3 baselines
```

- **`arena/features.py`** — one shared definition of the observable state
  (`encode_call`, `history_matrix`, and two baseline views: `single_call_features` for a
  per-call detector, `sequence_features` with causal proxies for a trajectory detector).
  Lifted out of `env.py`; the M1–M3 tests pass unchanged.
- **Three baseline defenders** (`arena/baselines/`), each a drop-in Blue policy:
  `StaticAllowList` (per-call, no memory — today's default), `SingleShotDetector`
  (TAMAS-style per-call classifier), `CausalMonitor` (CASPIAN-style trajectory classifier).
- **Result** (`docs/m4-features-and-baselines.md`): allow-list ≈ chance and ~89% attack
  success (matching TAMAS's ~80–82%); single-shot TPR@5%FPR ≈ 0.19 — it *cannot* separate
  the split chain; causal monitor TPR@5%FPR ≈ 0.85. This is the yardstick ARENA's trained
  Blue must beat.

## What M5 gives you

```bash
python3 scripts/train_ppo.py blue --steps 60000    # train the defender
python3 scripts/train_ppo.py red  --steps 15000 --fixed-scenario
```

- **Policy networks** (`arena/policies.py`) — Blue is a GRU over the call sequence
  (trajectory-level by construction); Red scores tools by embedding against a context built
  from the task, objective and Blue's past verdicts, so one Red generalises across
  registries and *adapts to the defender it faces*.
- **Single-file PPO** (`arena/ppo.py`) — own loop, not SB3, with correct truncation
  bootstrapping and dict-observation support. `TorchPolicyAdapter` makes a trained net a
  drop-in for any scripted policy.
- **Result**: trained Blue reaches **+0.52 to +0.61** mean reward across 5 seeds vs
  **−0.09** for the M4 causal monitor, **−0.49** passive and **−0.54** paranoid — with a
  quarantine rate ≈ the adversarial rate, i.e. it discriminates rather than blanket-blocks.
  Red learns too (attack success 0.05 → 0.52 on a fixed scenario in 15k steps / ~6 s).

See [docs/m5-policies-and-ppo.md](docs/m5-policies-and-ppo.md) for the three real bugs this
module surfaced (detection credit needed evidence; the false-positive weight made paranoia
optimal; scenario ids collided across resets) and the two training-stability findings
(learning rate, and permissive initialisation taking Blue from 3/5 to 5/5 seeds learning).

## Configuration

Scale lives entirely in YAML — `small.yaml` (default, M3-friendly) and `paper.yaml`
(scale-up: bigger scenarios *and* bigger nets, rollouts, generations and evaluation
budgets — needs real compute, not a laptop). A single top-level `seed:` propagates to
every section, so a whole run is reproducible from one number. Unknown keys are rejected
rather than silently ignored.

## Licence and attribution

Third-party datasets (TAMAS, Toucan-1.5M) are fetched at M9, never vendored, and recorded
with checksum and licence in `THIRD_PARTY.md`. See [project.md §13](project.md).

## What M6 gives you

```bash
python3 scripts/train_selfplay.py small.yaml --generations 4 --steps 40000
```

- **`SelfPlayTrainer`** (`arena/selfplay.py`) — the alternating loop: freeze Blue → train
  Red, freeze Red → train Blue, evaluate, repeat. Warm-starts each generation. Opponents
  are frozen through `TorchPolicyAdapter` (no grad, params never optimised); tests prove
  neither side's weights move while it is the frozen opponent.
- Blue trains against `AdaptiveRed(attacker=frozen_red)` — the learned Red on attack
  episodes, realistic benign traffic on the rest.
- **What a 4-generation run shows** (`docs/m6-selfplay.md`): gens 0–1 work as intended —
  Blue discriminates (quarantine rate ≈ adversarial rate), exploitability falls, Blue's
  return climbs. Gens 2–3 drift toward `(passive Blue, weak Red)` — the non-transitive
  self-play failure the proposal names. **This is what M7's opponent-checkpoint pool
  exists to fix**; M6 ships the alternation mechanism, not a converged result.

## What M7 gives you

```bash
python3 scripts/train_selfplay.py small.yaml     # league is on by default
```

- **`arena/league.py`** — `League` keeps per-side pools of past checkpoints (bounded,
  oldest evicted, latest always kept). `FrozenPolicySampler` is an `obs -> int` opponent
  that **resamples a checkpoint from the pool every episode**, so a training phase faces a
  distribution of past opponents, not one fixed policy.
- Wired into `SelfPlayTrainer`: after each generation both policies are snapshotted; the
  next generation trains against a *sample* of the opponent's whole history.
- **Why it exists**: M6 without it drifts to `(passive Blue, weak Red)` by generation 2–3
  (Blue's quarantine rate → ~0.01). The league keeps Blue facing strong past Reds, so it
  stays discriminating. `test_league_keeps_blue_discriminating_where_m6_drifts` asserts the
  final quarantine rate stays > 0.20. See `docs/m7-league.md`.

## What M8 gives you

```bash
python3 scripts/run_eval.py small.yaml                    # baseline leaderboard
python3 scripts/run_eval.py small.yaml --blue runs/sp.pt  # + ARENA-trained Blue
python3 scripts/run_eval.py small.yaml --curve runs/sp.pt # exploitability curve over its league
```

- **`arena/eval/exploitability.py`** — the primary metric: freeze the defender, spawn a
  *fresh* random Red, train it as a best response, report attack success. Measured on an
  all-adversarial distribution so it is not capped at the scenario's adversarial ratio.
  `fresh_red_start` (pre-training success ≈ chance) is the built-in no-warm-start check.
- **`arena/eval/metrics.py`** — `roc_auc` / `roc_curve` / `tpr_at_fpr`, rank-based, checked
  exact against `sklearn.metrics` and hand-computed fixtures. `tpr_at_fpr` is on CASPIAN's
  scale (project.md §8).
- **`arena/eval/harness.py`** — `evaluate_defenders` runs every defender (the 3 M4
  baselines + any ARENA Blue) on one scale, calibrating each to a matched ~5% FPR before
  measuring exploitability so "low exploitability" means "discriminates", not "quarantines
  everything".
- **The result** (`docs/m8-evaluation.md`): at `small.yaml` scale the co-evolved Blue
  reaches **exploitability 0.78**, *above* the CASPIAN-style causal monitor's **0.69** —
  it is not yet less exploitable than the static baselines, and the curve across four
  self-play generations is flat. An earlier version of this README claimed the opposite
  (0.12 vs 0.38, "~3× less exploitable"); that was an artifact of a PPO bug found in the
  M1–M9 audit and is corrected in [docs/audit-m1-m9.md](docs/audit-m1-m9.md). The
  allow-list's 0.78 does now match TAMAS's reported ~80% static-baseline failure.

## What M9 gives you

```bash
python3 scripts/fetch_data.py            # ~8k Toucan rows + the TAMAS tarball -> data/ (git-ignored)
```

- **`arena/data/fetch.py`** — idempotent fetch of the two public datasets
  ([`THIRD_PARTY.md`](THIRD_PARTY.md)): **TAMAS** (MIT / CDLA-Permissive-2.0) as a
  tarball for taxonomy provenance, **Toucan-1.5M** (Apache-2.0) as an ~8k-row JSON
  subsample via the HF datasets-server rows API. Writes `data/manifest.json` with a
  sha256 + licence + UTC retrieval time per file; a re-run verifies the hashes and
  touches the network only if something changed. **Zero new runtime dependencies** —
  stdlib `urllib` / `json` / `hashlib`.
- **`arena/data/toucan.py`** — turns the subsample into a small `BenignProfile`
  (read / act / other call-category frequencies + a trajectory-length histogram).
  With `scenario.benign_source: toucan` the `BenignRoller` draws tool categories from
  real benign traffic instead of uniformly; it falls back to synthetic if `data/` is
  absent, so the tests never need the datasets. See [docs/m9-data.md](docs/m9-data.md).

## What M10 gives you

```bash
python3 scripts/run_sweep.py small.yaml                    # templated + LLM sweep
python3 scripts/run_sweep.py small.yaml --blue runs/sp.pt  # + ARENA-trained Blue
```

- **`arena/llm/client.py`** — cached Ollama client. Disk cache keyed by
  `(model, prompt, options_hash)` so a sweep is re-runnable offline. `available()`
  checks if the daemon is up and the model is pulled.
- **`arena/llm/payloads.py`** — per-family prompt templates that ask the LLM to
  produce realistic malicious payloads. A validity check ensures the output
  encodes the attack intent; on failure we fall back to the M1 template payload.
- **`arena/llm/sweep.py`** — runs `evaluate_defenders` twice: once with templated
  payloads (standard) and once with LLM-rendered payloads (if Ollama is available).
  Reports both leaderboards side by side.
- **Degradation**: if Ollama is absent, the sweep runs with templates and labels
  the output accordingly. Zero new runtime dependencies beyond `ollama`.
- See [docs/m10-llm.md](docs/m10-llm.md) for details.
