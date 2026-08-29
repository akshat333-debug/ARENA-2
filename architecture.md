# ARENA — Architecture

Status: **DRAFT — awaiting sign-off (Gate 2)**. No implementation until approved.
Requirements: [project.md](project.md)

---

## 1. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Env API | **Gymnasium + PettingZoo** (`pettingzoo.AECEnv`) | Stated deliverable in the proposal. Non-negotiable. |
| Tensors / NN | **torch 2.10** (installed, MPS available) | Already present, MPS works on M3. |
| RL algorithm | **Own single-file PPO** (CleanRL-style, ~250 lines) — *not* Stable-Baselines3 | See §2. |
| Config | **pydantic + YAML** (both installed) | Typed configs; scale-up = swap YAML, per Gate 1. |
| Baselines | **scikit-learn** (installed) | Static allow-list, TAMAS-style single-shot detector, CASPIAN-style causal monitor. |
| Metrics | **sklearn.metrics + numpy** | AUROC, TPR@5%FPR. |
| LLM eval | **ollama + qwen2.5:3b** (installed, local) | Held-out payload sweep, cached, offline. |
| Data fetch | **huggingface_hub** + pandas parquet | Toucan subset + TAMAS. No `datasets` lib needed. |
| Tests | **pytest** (installed) | Unit + integration per module. |

**New dependencies: exactly three** — `gymnasium`, `pettingzoo`, `huggingface_hub`.

## 2. The one contested decision: own PPO instead of Stable-Baselines3

SB3 is the obvious reach, and it is the wrong tool here:

- ARENA's research contribution *is* the training loop — alternating freeze, league
  checkpoint pool, opponent sampling. SB3 is built around one learning policy in a fixed
  env; injecting a frozen adversary that is itself a policy means fighting `VecEnv`
  wrappers and callbacks.
- Two policies with **different observation and action spaces** (Red: masked categorical
  over tool calls; Blue: sequence model → 3-way verdict) is not the shape SB3 optimizes for.
- Exploitability requires spawning a fresh best-response learner against a frozen
  opponent on demand — trivial with an owned loop, awkward through SB3's API.
- SB3 on Python 3.13 is additional risk for zero gain.

A CleanRL-style PPO is ~250 lines, fully readable, and is what comparable self-play papers
actually ship. **Cost of being wrong is low** — the trainer is one module behind an
interface; swapping it later touches one file.

## 3. Game / environment design

Episode = one simulated MCP session.

**Scenario** (sampled by the generator): a task, a tool registry, and an episode type —
`benign` or `adversarial` (with a Red objective drawn from the 6-family taxonomy).

Each tool carries observable metadata: name, category, declared sensitivity, and a
side-effect class (`read_sensitive`, `read_benign`, `transform`, `write_local`,
`external_sink`, …).

**Turn loop**, at step `t`:
1. Red proposes a tool call (tool + argument template), conditioned on task, registry,
   history, **and Blue's past verdicts** — this feedback channel is what lets Red adapt.
2. Blue observes the whole call sequence including the proposal, emits
   `allow` / `flag` / `quarantine`.
   - `allow` → call executes, appended to history
   - `flag` → executes but is logged: partial credit if adversarial, FP cost if benign
   - `quarantine` → session terminates
3. Env updates state; repeat until quarantine, task completion, or step cap.

**M3 implementation notes.** Agents are `red_0` / `blue_0` (PettingZoo AEC; one MCP step =
two AEC turns). Red's action is a **tool index only** — the env applies M2's
most-recent-artifact default, which covers the canonical chain and every family (verified);
a structured data-flow action is a later extension. Red's `Discrete` space is fixed at
`n_tools_max` and out-of-range indices wrap; `registry_mask` in Red's obs marks the valid
range. `SingleAgentARENA` is the Gym single-agent view used for PPO and the exploitability
sweep. See [docs/m3-environment.md](docs/m3-environment.md).

## 4. Ground truth vs. observation — the anti-leakage rule

The env maintains an internal **taint graph**: which data originated from a sensitive
source, how it propagated through transforms, and whether it reached an external sink.
This is the ground-truth answer to *"did the attack objective actually complete?"*

> **Blue never observes the taint graph.** Blue sees only observable call metadata —
> tool identity, category, declared sensitivity, argument-template features, position,
> step gap. It must *learn* the trajectory-level pattern from those.

Leaking taint into Blue's observation would make the task trivial and the result
worthless. This is a hard invariant, enforced by a test that asserts Blue's observation
tensor is unchanged when the internal taint state is perturbed. It is also a standing
item in the ML audit (Phase 12).

Corollary: attack success is judged by the taint graph, **not** by whether Blue was
fooled — so Red cannot farm reward with degenerate non-attacks.

## 5. Reward engine

```
R_red  = w_evade · [objective_completed ∧ ¬quarantined]
       + w_plaus · plausibility(sequence)          # still looks like legitimate agent work
       − w_step  · steps_used

R_blue = + w_detect · [quarantined an adversarial session before completion]
       + w_flag   · [flagged the malicious call]   # partial credit
       − w_fp     · [quarantined/flagged a benign session]
       − w_miss   · [objective completed undetected]
```

`plausibility` keeps Red from emitting obviously-absurd sequences; `w_fp` pushes Blue
toward precision rather than paranoia (proposal §3.3).

## 6. Policies

| | Red | Blue |
|---|---|---|
| Input | task embedding + registry + history + Blue's past verdicts | sequence of observable call feature vectors |
| Body | MLP over pooled history | **GRU** over the call sequence (trajectory-level by construction) |
| Head | masked categorical over legal tool calls + value | 3-way verdict + value |

Small by default (`configs/small.yaml`): hidden 64–128, few layers. Sizes are config, not code.

## 7. Self-play + league

Alternating PPO: freeze Blue → train Red; freeze Red → train Blue; repeat.
Every `checkpoint_every` generations, the current policy is frozen into an **opponent pool**.
Opponents are sampled from the pool (uniform, or prioritized by win-rate), never only the
newest — the standard guard against cyclic, non-transitive strategies.

## 8. Evaluation

- **Exploitability (primary):** freeze Blue at generation *g*, train a *fresh* best-response
  Red for K steps, report its success rate. Plot across generations; success = monotone decline.
- **Secondary:** AUROC, TPR@5%FPR on a held-out session set — same scale as CASPIAN (2026).
- **Baselines:** static allow-list · TAMAS-style single-shot per-call detector ·
  CASPIAN-style causal/cascade monitor. Each tuned on a validation split so the comparison
  isn't a strawman.

## 9. Module layout

```
arena/
  config.py            # pydantic configs + YAML loading                    [M1-M2]
  tools.py             # tool specs, registry, side-effect classes          [M1]
  scenarios.py         # scenario generator; 6 families -> chained variants  [M1]
  taint.py             # ground-truth data-flow tracker (ENV-INTERNAL ONLY) [M2]
  rewards.py           # asymmetric reward engine                           [M2]
  features.py          # shared observable encoders + baseline feature views [M4]
  env.py               # PettingZoo AEC env + SingleAgentARENA Gym wrapper  [M3]
  scripted.py          # non-learned policies (attacker, benign, refs)      [M4]
  baselines/           # allowlist / single_shot / causal + collect, evaluate [M4]
  policies.py          # Red net, Blue GRU net                              [M5]
  ppo.py               # single-file PPO trainer                            [M5]
  league.py            # checkpoint pool + opponent sampling                [M7]
  eval/                # metrics, exploitability, harness                   [M8]
  llm/                 # ollama client (cached), payload rendering          [M10]
  data/                # fetch (TAMAS + Toucan + manifest)                  [M9]
scripts/               # run_baselines.py [M4]; train_selfplay, run_eval... [M6+]
configs/               # small.yaml (default), paper.yaml (scale-up target)
tests/                 # unit + integration, mirrors arena/
docs/                  # per-module notes
```

## 10. Scale posture

`configs/small.yaml` is the default and must finish on an M3 in minutes. `configs/paper.yaml`
holds the scale-up target and is never the default. No size, step count, or generation count
is hardcoded anywhere in `arena/`.
