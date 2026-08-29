# ARENA — Modular Build Plan

Status: **DRAFT — awaiting sign-off (Gate 2)**
Requirements: [project.md](project.md) · Architecture: [architecture.md](architecture.md)

Every module follows the same cycle, no exceptions:

> **implement → unit test (valid / invalid / edge) → integration test → fix → update docs → commit**

Push only on explicit confirmation, per user instruction. Local commits accumulate freely.

---

## Build order

| # | Module | Ships | Tests that must pass | Depends on |
|---|---|---|---|---|
| **M1** ✅ | `config.py`, `tools.py`, `scenarios.py` | Typed config; tool registry with side-effect classes; scenario generator covering all 6 attack families as chained variants | every family generates; benign scenarios contain no objective; registries are well-formed; seeded runs reproduce exactly | — |
| **M2** ✅ | `taint.py`, `rewards.py`, `RewardConfig` | Ground-truth data-flow tracker; asymmetric reward engine | taint propagates sensitive→(transform\|exec)→sink; canonical `.env → summarize → webhook` detected as completed exfiltration; benign chains are not; every family has a verified minimal winning sequence in its own registry; reward signs behave at every branch. 139 tests. | M1 |
| **M3** | `env.py` | PettingZoo AEC env + Gymnasium wrappers | PettingZoo API conformance test passes; episode terminates on quarantine / completion / step cap; **leakage test: Blue's observation is invariant to internal taint perturbation** | M1, M2 |
| **M4** | `features.py`, `baselines/` | Blue's observable feature extractor; all three baselines (allow-list, TAMAS-style single-shot, CASPIAN-style causal) | each baseline runs end-to-end on the env and produces verdicts; single-shot provably fails the split-chain attack (that's the headline motivation, so it must be demonstrated, not asserted) | M3 |
| **M5** | `policies.py`, `ppo.py` | Red net, Blue GRU net, single-file PPO | PPO passes a sanity task (learns a trivially-learnable env); shapes/masking correct; gradient flows; runs on MPS and CPU | M3 |
| **M6** | self-play loop | Alternating freeze/train in `scripts/train_selfplay.py` | one full generation completes on `small.yaml` in minutes; frozen side's weights provably do not change | M5 |
| **M7** | `league.py` | Checkpoint pool + opponent sampling | pool grows, sampling honors the distribution, checkpoints restore bit-identically | M6 |
| **M8** | `eval/` | Exploitability, AUROC, TPR@5%FPR, harness | metrics match hand-computed values on a fixture; exploitability spawns a genuinely fresh best-response Red | M7, M4 |
| **M9** | `data/` | TAMAS + Toucan fetch, checksum manifest, benign-traffic sampler | fetch is idempotent; manifest records checksum + date; Toucan subset feeds the benign distribution; licenses recorded in `THIRD_PARTY.md` | M3 |
| **M10** | `llm/` | Cached Ollama client; qwen2.5:3b payload rendering; held-out transfer sweep | cache hit/miss correct; sweep degrades gracefully to templates if Ollama is absent or output fails validity check | M8, M9 |
| **M11** | Leaderboard + report | Exploitability curves, baseline comparison table, paper draft, patent claim structure | numbers in the report are regenerable from a single command | M8, M10 |

**Design note on ordering:** baselines (M4) land *before* RL (M5) deliberately — that gives a
working evaluation target and a demonstrated failure case early, so the RL work has something
real to beat from day one. Benign traffic starts synthetic (M1) and swaps to Toucan at M9;
the env takes a pluggable benign source from M3 so that swap costs nothing.

## Gates still ahead

- **Gate 3 — before any push.** Local commits accumulate; the remote
  (`akshat333-debug/ARENA-2`) is not written to without explicit confirmation.
- Audits (Phases 10–12) run before M11 ships: requirements audit, code audit, ML audit
  (leakage, public-dataset provenance, reproducibility).
