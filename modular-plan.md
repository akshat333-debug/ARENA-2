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
| **M3** ✅ | `env.py` | PettingZoo AEC env (`red_0`/`blue_0`) + `SingleAgentARENA` Gym wrapper | `pettingzoo.test.api_test` passes; terminates on quarantine / completion / step cap; **leakage test: Blue's observation is byte-invariant to internal taint perturbation, and identical across same-calls/different-objective episodes**; every family winnable with tool-index-only actions; sequence-aware Blue beats the split chain a per-call Blue misses. 173 tests. | M1, M2 |
| **M4** ✅ | `features.py`, `scripted.py`, `baselines/` (+ `collect`, `evaluate`) | Shared observable feature views; all three baselines (allow-list, TAMAS-style single-shot, CASPIAN-style causal) as drop-in Blue policies | each baseline runs end-to-end and produces verdicts; `test_m4_integration` demonstrates: allow-list ≈ chance / ~89% attack success (matches TAMAS ~80–82%); single-shot TPR@5%FPR ≈ 0.19 — **provably cannot separate the split chain**; causal monitor TPR@5%FPR ≈ 0.85. 203 tests. | M3 |
| **M5** ✅ | `policies.py`, `ppo.py`, `PolicyConfig`/`PPOConfig` | Red embedding-scorer net, Blue GRU net, single-file PPO with truncation bootstrapping, `TorchPolicyAdapter` | shapes/masking/gradients correct; GAE checked against hand-computed values; runs on CPU and MPS; reproducible independent of ambient global RNG. **Red learns** (attack success 0.05→0.52); **Blue learns to discriminate** — +0.52..+0.61 across 5 seeds vs −0.09 causal monitor, −0.49 passive, −0.54 paranoid, quarantine rate ≈ adversarial rate. Frozen opponents provably do not move. 240 tests. | M3 |
| **M6** ✅ | `selfplay.py`, `SelfPlayConfig`, `scripts/train_selfplay.py`; `AdaptiveRed` generalised | `SelfPlayTrainer`: freeze Blue→train Red, freeze Red→train Blue, evaluate, repeat | one generation completes on `small.yaml` in ~40s; frozen side's weights provably do not move (both directions); reproducible; Blue does not collapse to blanket quarantine vs a learned Red. Run shows gens 0–1 discriminating + exploitability falling, gens 2–3 non-transitive drift → **motivates M7 league**. 258 tests. | M5 |
| **M7** ✅ | `league.py` (`League`, `FrozenPolicySampler`, `net_factory`); wired into `SelfPlayTrainer` | Per-side checkpoint pools; opponent resampled from the pool every episode (`p_latest` bias + uniform) | pool grows / evicts oldest / keeps latest; `sample_probs` matches empirical draws; `state_dict` round-trips bit-identically; stored checkpoints are detached CPU copies; sampler stays frozen; **payoff: a 4-gen league run keeps Blue's quarantine rate > 0.20 where M6 collapsed to ~0.01**. `use_league` on by default; off = plain M6 loop. | M6 |
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
