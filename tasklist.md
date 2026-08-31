# ARENA — Task List / Handoff

**Project:** An Adversarial Co-Evolving Benchmark for Agentic Tool-Use Security.
Game-theoretic multi-agent RL for MCP tool-use security. CSI4006 Game Theory, VIT Vellore.

A self-play RL arena where an attacker (**Red**) and a defender (**Blue**) co-evolve
inside a simulated MCP tool-calling environment. Primary metric is **exploitability**
(best-response attack success against a frozen defender), not static single-pass accuracy.

- **Repo:** https://github.com/akshat333-debug/ARENA-2 (branch `main`)
- **Planning docs:** [`project.md`](project.md) (requirements, game model), [`architecture.md`](architecture.md) (tech stack, module layout, anti-leakage rule), [`modular-plan.md`](modular-plan.md) (M1–M11 build order + per-module test gates)
- **Per-module notes:** [`docs/`](docs/) — one file per module
- **Read this first:** [`docs/audit-m1-m9.md`](docs/audit-m1-m9.md) — full audit, 4 bugs, and why the old M8 headline result is withdrawn

---

## 0. ⚠️ Repo state right now

**All 11 modules (M1–M11) are built, audited and merged to `main`.** The `dev`
branch carrying M10+M11 was reviewed, fixed and fast-forwarded in.

The substantive open problem is no longer "close the exploitability gap" — the
5-seed run showed that gap was never established. It is **reduce the co-evolved
defender's seed variance**. Read [`docs/m11-multiseed.md`](docs/m11-multiseed.md)
before touching anything in `arena/selfplay.py` or `arena/eval/`.

---

## 1. Environment setup

### 1.1 Prerequisites

| Thing | Version used | Notes |
|---|---|---|
| Python | **3.13** (3.13.9) | 3.11+ should work; 3.13 is what it's developed on |
| OS | macOS (Apple M3) | Linux fine. `torch` device defaults to **cpu** — MPS/CUDA is opt-in via config, and for these tiny nets CPU is actually faster |
| git | any | interactive rebase/add (`-i`) not needed |
| Ollama | 0.33.2 | **only needed for M10** — see §3.3 |

### 1.2 Clone + install

```bash
git clone https://github.com/akshat333-debug/ARENA-2.git
cd ARENA-2
python3 -m venv .venv && source .venv/bin/activate    # .venv/ is git-ignored
pip install -r requirements.txt
```

`requirements.txt` (grows per module — [`requirements.txt`](requirements.txt)):

```
numpy>=2.0          pydantic>=2.0        PyYAML>=6.0        # M1
gymnasium>=0.29     pettingzoo>=1.24                       # M3
scikit-learn>=1.3                                          # M4 (baselines only; not in the RL hot path)
torch>=2.2                                                 # M5
pytest>=8.0                                                # tests
# M9 added ZERO deps (stdlib urllib/json/hashlib/tarfile + HF datasets-server rows API)
# M10 will add:  ollama            (python client)
# M11 will add:  matplotlib        (exploitability-curve plots)
```

Optional dev tools (not in requirements): `ruff` for lint (`pip install ruff`).

### 1.3 Verify the install

```bash
python3 -m pytest -q -m "not slow"     # ~340 fast tests, ~20s, must be all green
python3 -m pytest -q                   # + 32 slow training-loop tests, ~8 min
python3 -m ruff check arena/ scripts/ tests/
```

Current baseline: **343 fast + 32 slow passing** (375 collected). If a fast test fails
on a clean checkout, stop and fix before doing anything else.

---

## 2. Git-ignored things you must regenerate locally

[`.gitignore`](.gitignore) excludes everything below. None of it is in the repo; you
generate it on your machine.

| Path | What it is | How to (re)create | Needed for |
|---|---|---|---|
| `/data/` | Fetched public datasets (TAMAS tarball + Toucan-1.5M row subsample) + `manifest.json` + derived `toucan/profile.json` | `python3 scripts/fetch_data.py` (idempotent; `--force` to re-fetch) | Optional now — only if you set `scenario.benign_source: toucan`. **Required for M11** realism claims. Downloads ~a few MB. |
| `runs/`, `checkpoints/`, `*.pt` | Trained policy checkpoints from self-play | `python3 scripts/train_selfplay.py small.yaml --out runs/sp.pt` (~3 min on M3) | Any eval against an ARENA-trained Blue (`run_eval.py --blue …`), the exploitability curve, M11 |
| `.cache/` | LLM response cache (M10) | created by the M10 Ollama client on first run | M10 sweeps |
| `.venv/` | virtualenv | `python3 -m venv .venv` | everything |

### 2.1 Dataset fetch details ([`THIRD_PARTY.md`](THIRD_PARTY.md), [`docs/m9-data.md`](docs/m9-data.md))

```bash
python3 scripts/fetch_data.py                 # ~8k Toucan rows + TAMAS tarball -> data/
python3 scripts/fetch_data.py --n-toucan 20000
python3 scripts/fetch_data.py --force         # re-fetch and re-hash
```

- **Idempotent:** a second run verifies `data/manifest.json`'s sha256s against disk and
  does nothing (no network) unless a file changed or `--force`.
- Nothing is vendored. The manifest records sha256 + byte size + source URL + licence +
  UTC retrieval time per artifact.
- Uses **stdlib only** (no `huggingface_hub`, no parquet reader) — pulls Toucan rows as
  JSON via the HF datasets-server rows API.

---

## 3. External resources & links

### 3.1 Datasets (public, licences verified — [`THIRD_PARTY.md`](THIRD_PARTY.md))

| Dataset | URL | Licence | Used for |
|---|---|---|---|
| **TAMAS** | https://github.com/microsoft/TAMAS | MIT (code) / CDLA-Permissive-2.0 (data) | Taxonomy grounding for the 6 attack families; the ~80–82% static-baseline failure reference. Fetched as the repo tarball, not parsed. arXiv: https://arxiv.org/abs/2511.05269 |
| **Toucan-1.5M** | https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M | Apache-2.0 | Realistic benign traffic. Subsampled (`SFT` config, ~8k rows) into a `BenignProfile`. arXiv: https://arxiv.org/abs/2510.01179 |

HF datasets-server API (what `fetch.py` calls, no auth needed):
`https://datasets-server.huggingface.co/rows?dataset=Agent-Ark/Toucan-1.5M&config=SFT&split=train&offset=0&length=100`

### 3.2 Reference papers (for the M11 report)

- TAMAS — Kavathekar et al., *Benchmarking Adversarial Risks in Multi-Agent LLM Systems*, arXiv:2511.05269
- CASPIAN — cross-channel causal monitor (the strong baseline `causal_monitor` is modelled on it); see `project.md §8`
- AlphaStar league / PFSP — Vinyals et al. 2019 (the `league.py` design; PFSP is the M11 refinement)
- Time Limits in RL — Pardo et al. 2018 (why the step cap is a terminal, not a truncation — see audit bug 1)

### 3.3 Ollama + models (M10 only)

- **Install:** https://ollama.com/download  (macOS: `brew install ollama` or the .dmg)
- **Docs / API:** https://github.com/ollama/ollama/blob/main/docs/api.md
- **Python client:** `pip install ollama` — https://github.com/ollama/ollama-python
- Models already pulled on the current machine (`ollama list`):
  - `qwen2.5:3b`   (1.9 GB) — **the eval LLM**
  - `qwen2.5:1.5b` (986 MB) — fallback / faster
  - `tinyllama`    (637 MB) — smoke tests
- Pull with: `ollama pull qwen2.5:3b`
- M10 must **degrade gracefully to templates** when Ollama is absent — so a machine
  without Ollama still runs the whole suite.

---

## 4. What's done (M1–M9 + audit)

All committed and pushed **except `f5faf63`** (see §0). Every module follows the cycle:
implement → unit tests (valid/invalid/edge) → integration test → fix → update docs → commit.

| Module | Ships | State | Notes / doc |
|---|---|---|---|
| **M1** | `config.py`, `tools.py`, `scenarios.py` | ✅ | Typed frozen pydantic config (scale is YAML-only, `extra="forbid"`); tool registry with side-effect classes; scenario generator — 6 TAMAS families each as a **chained** variant (`min_chain_len > 1`). Every sampled registry hosts a source→propagator→sink chain so no episode is unwinnable. |
| **M2** | `taint.py`, `rewards.py` | ✅ | **`TaintTracker`** — env-internal ground truth: sources mint SENSITIVE/UNTRUSTED tags, propagators carry them, sinks record `FlowEvent`s. `_event_satisfies` needs `chain_len >= min_chain_len`. **`compute_rewards`** — asymmetric; detection credit gated on `caught_in_flight` (evidence, not blanket quarantine); `w_false_quarantine = w_detect + w_miss` so the two degenerate defenders tie exactly. [`docs/m2-taint-and-rewards.md`](docs/m2-taint-and-rewards.md) |
| **M3** | `env.py` | ✅ | **`ARENAEnv`** PettingZoo AEC (`red_0` / `blue_0`) + **`SingleAgentARENA`** Gym wrapper. **Anti-leakage invariant:** Blue's observation is built from observable call metadata only; byte-invariant to the taint graph. Passes `pettingzoo.test.api_test`. [`docs/m3-environment.md`](docs/m3-environment.md) |
| **M4** | `features.py`, `scripted.py`, `baselines/` | ✅ | Shared observable encoders + two baseline feature views. Three drop-in Blue baselines: `StaticAllowList` (per-call, ≈ TAMAS ~80% failure), `SingleShotDetector` (TAMAS-style per-call classifier — provably can't separate the split chain), `CausalMonitor` (CASPIAN-style trajectory classifier — the strong baseline). [`docs/m4-features-and-baselines.md`](docs/m4-features-and-baselines.md) |
| **M5** | `policies.py`, `ppo.py` | ✅ | Blue = GRU over the call sequence; Red = tool-embedding scorer that adapts to Blue's past verdicts. **Own single-file PPO** (not SB3) with GAE(λ), truncation bootstrapping, dict observations. `TorchPolicyAdapter` makes a trained net a drop-in policy. [`docs/m5-policies-and-ppo.md`](docs/m5-policies-and-ppo.md) |
| **M6** | `selfplay.py` | ✅ | `SelfPlayTrainer` — freeze Blue→train Red, freeze Red→train Blue, evaluate, repeat. On its own it drifts to `(passive Blue, weak Red)` by gen 2–3 — the non-transitive self-play failure M7 fixes. [`docs/m6-selfplay.md`](docs/m6-selfplay.md) |
| **M7** | `league.py` | ✅ | `League` — per-side checkpoint pools (bounded, latest-kept); `FrozenPolicySampler` resamples a past opponent each episode. Stops the M6 drift: Blue's quarantine rate stays ~0.5+ where the leagueless loop collapses to ~0.01. `Checkpoint.opponent_win_rate` is stored but **unused** — it's there for PFSP (M11). [`docs/m7-league.md`](docs/m7-league.md) |
| **M8** | `eval/metrics.py`, `eval/exploitability.py`, `eval/harness.py` | ✅ | Rank-based `roc_auc` / `roc_curve` / `tpr_at_fpr` (verified exact vs `sklearn.metrics`). `exploitability()` — freeze defender, spawn a **fresh** random Red, train it as a best response, report attack success (all-adversarial, so not capped at the scenario ratio). `evaluate_defenders()` → leaderboard, calibrates each defender to ~5% FPR first. [`docs/m8-evaluation.md`](docs/m8-evaluation.md) |
| **M9** | `data/fetch.py`, `data/toucan.py`, `THIRD_PARTY.md`, `scripts/fetch_data.py` | ✅ | Idempotent dataset fetch + checksum manifest. Toucan subsample → `BenignProfile` (read/act/other tool-category weights) that drives `BenignRoller` when `scenario.benign_source: toucan`; falls back to synthetic if `data/` absent. **Zero new deps.** [`docs/m9-data.md`](docs/m9-data.md) |
| **Audit** | `tests/test_contracts.py`, doc corrections | ✅ (commit `f5faf63`, unpushed) | 4 bugs found & fixed. See §5. |

### 4.1 Config knobs you'll touch ([`arena/config.py`](arena/config.py))

- **`configs/small.yaml`** — the default. Must finish on an M3 in minutes.
- **`configs/paper.yaml`** — the scale-up. Now scales scenario **and** `policy` / `ppo` /
  `selfplay` (12 gens × 250k steps/side) / `eval` (150k-step best response). Needs real
  compute. A single top-level `seed:` propagates to every section.

---

## 5. The audit result — you MUST understand this before M11

Full write-up: [`docs/audit-m1-m9.md`](docs/audit-m1-m9.md). Four bugs, all in the
**measurement path** (the modules were structurally fine):

1. **CRITICAL — GAE boundaries were invisible.** `env.py` raised `terminated` *and*
   `truncated` together at the step cap, so PPO's `terminated and not truncated` was
   `False` and the bootstrap branch was `False` too — ~90% of episode ends recorded **no
   boundary at all**, and GAE bled advantages across episodes. Corrupted every M5–M8
   training run. Fixed: the step cap is a genuine terminal here (budget is observable,
   reward fully settled); flags are now mutually exclusive; `collect()` derives `terminal`
   from `done`.
2. **`selfplay` per-generation `exploitability` measured the wrong thing** — Red vs a
   *sample of past Blues*, capped at `adversarial_ratio`, not the current Blue. New
   `_exploit_env()`.
3. **Learned Blue was scored stochastically** while every baseline is deterministic → now
   greedy at eval everywhere.
4. **`paper.yaml` wasn't a scale-up** — overrode only `scenario`, so a harder problem ran
   on `small.yaml`'s budget. Now scales everything.

**Consequence — the M8 headline is withdrawn.** The old "arena_blue 0.120 vs
causal_monitor 0.376, ~3× less exploitable" was an artifact: `exploitability()` trains its
attacker with the same broken PPO, so *every* defender looked far less exploitable than it
is. Re-measured at `EvalConfig` defaults:

```
defender                AUROC   TPR@5%FPR  exploitability   op.FPR
------------------------------------------------------------------
causal_monitor          0.980       0.861           0.690    0.036
single_shot             0.728       0.034           0.773    0.010
static_allowlist        0.513       0.037           0.777    0.012
arena_blue              0.927       0.461           0.780    0.047   <-- most exploitable
```

Exploitability curve across the league's Blue checkpoints is **flat**
(`0.720 → 0.765 → 0.740 → 0.710`).

**At `small.yaml` scale, co-evolution does NOT beat the static baselines, and four
generations of self-play do not drive exploitability down.** This is the open problem M11
must address — see §7.

What still holds (separately tested): Blue beats every baseline on mean reward vs the
*training* distribution (M5); the league still prevents the M6 drift (M7).

---

## 6. Next task — M10: `arena/llm/`

**Goal:** an eval-only sweep where attack payloads are rendered by a local LLM
(`qwen2.5:3b`) instead of templates, to demonstrate the trained defender transfers to
"real" LLM-authored payloads. Training stays scripted/templated (fast). This is the
"hybrid" fidelity decision in [`project.md §44`](project.md).

**New dep:** `ollama` (add to `requirements.txt`). **Test gate:** cache hit/miss correct;
sweep degrades gracefully to templates if Ollama is absent or output fails a validity
check.

### 6.1 `arena/llm/client.py` — cached Ollama client
- [ ] Thin wrapper over the `ollama` python client (or the HTTP API at `http://localhost:11434`).
- [ ] **Disk cache** keyed by `(model, prompt, options-hash)` → response. Write under
      `.cache/llm/` (already git-ignored). Cache must make a sweep re-runnable **offline**.
- [ ] `available()` → bool (is the Ollama daemon up + is the model pulled). Used by the
      degrade-to-templates path.
- [ ] Config: model name (`qwen2.5:3b`), temperature, timeout, max retries. Add an
      `LLMConfig` section to `arena/config.py` (frozen, `extra="forbid"`, like the others).
- [ ] Tests: cache miss calls the backend once and stores; cache hit returns without
      calling; a corrupt/empty cache entry is treated as a miss; `available()` is False
      when the daemon is down (mock the client).

### 6.2 `arena/llm/payloads.py` — payload rendering
- [ ] For each attack family + objective, a prompt template that asks the LLM to produce a
      realistic malicious payload string (the "content" an untrusted read or injected
      instruction would carry). Keep prompts in one place, versioned.
- [ ] **Validity check** on the LLM output: does it actually encode the attack intent
      (e.g. references the sink / the sensitive data / the injected instruction)? If it
      fails, fall back to the M1 template payload. This keeps the sweep honest — a
      garbage LLM response can't silently make the defender look better or worse.
- [ ] A payload only changes the *content string* on a call, never the tool sequence or
      the taint graph — so the ground truth and the anti-leakage invariant are unchanged.
      Verify this in a test (Blue's observation for an LLM-payload episode must be
      byte-identical to the templated one with the same tool sequence).

### 6.3 `arena/llm/sweep.py` — held-out transfer sweep
- [ ] Run `evaluate_defenders` (from `arena/eval/harness.py`) on a scenario set whose
      payloads come from the LLM instead of templates. Report the same leaderboard
      columns, side by side with the templated numbers.
- [ ] If `client.available()` is False → run with templates and clearly label the output
      "LLM unavailable, templates used" (do not fail).
- [ ] `scripts/run_sweep.py` — one command, `small.yaml` default, `--blue <ckpt>` to add
      an ARENA Blue, writes a table.

### 6.4 M10 wrap-up
- [ ] `tests/test_llm.py` (fast, backend mocked) + `tests/test_m10_integration.py`
      (`slow`, real Ollama, `pytest.skip` if `not available()`).
- [ ] `docs/m10-llm.md` — what it does, the cache format, the degrade path, real numbers.
- [ ] Update `README.md` ("What M10 gives you"), `modular-plan.md` (tick M10),
      `requirements.txt` (`ollama`).
- [ ] Commit. **Push only after explicit sign-off** (see §8).

---

## 7. Then — M11: leaderboard + report (+ close the exploitability gap)

**Test gate:** every number in the report is regenerable from a single command; **AND**
the open problem the audit exposed is addressed head-on — either the exploitability gap
closes, or the report states plainly that co-evolution did not beat the static baselines
at the scale tested. Do not bury this.

### 7.1 Try to close the gap (in rough order of expected payoff)
- [ ] **Scale.** Run `configs/paper.yaml` end to end (self-play + `run_eval` + curve).
      This is the first lever and now actually possible (audit bug 4). Needs real compute
      — a GPU box or a long CPU run. Record wall-clock.
- [ ] **More generations** at `small.yaml` — is the flat curve a budget artifact? Push to
      12–20 generations, plot the curve.
- [ ] **PFSP league sampling.** `Checkpoint.opponent_win_rate` is already stored per
      checkpoint (unused). Implement prioritised sampling in `League.sample()` /
      `sample_probs()` — draw hard past opponents more often. Gate behind a
      `SelfPlayConfig` flag so uniform stays the default until PFSP is shown to help.
- [ ] **Denser Blue reward.** Blue currently gets one sparse episode-level signal. Try a
      shaped per-step signal (e.g. small reward for a correct `flag` on an on-chain call,
      small penalty for an off-chain flag) — the reward engine already tracks
      `flagged_steps` vs `malicious_steps`. Keep the degenerate-defender tie property.
- [ ] **Give Blue the causal features.** The `causal_monitor` wins partly because its
      hand-engineered `sensitive_read_before_sink` feature ~encodes the taint rule. Feed
      `sequence_features` (from `arena/features.py`) into `BluePolicy` alongside the GRU
      and see if it closes the AUROC/TPR gap.
- [ ] Whatever the outcome: multi-seed it. The current leaderboard is **one seed**. Run
      ≥3 seeds and report mean ± spread before claiming anything moved.

### 7.2 The report
- [ ] `scripts/reproduce.py` (or a Makefile target) — one command that regenerates every
      figure and table: baseline leaderboard, `arena_blue` row, exploitability curve
      (`curve_from_league`), the M10 transfer-sweep table.
- [ ] Exploitability-curve plots — add `matplotlib` to `requirements.txt`.
- [ ] `report/` — paper draft: problem, game model, method (self-play + league +
      exploitability metric), results (honest), the audit as a methods-integrity section,
      limitations, future work.
- [ ] Patent claim structure — the contribution is the *training loop* (alternating
      freeze + checkpoint league + on-demand fresh best-response Red for exploitability),
      not any single net. Draft independent + dependent claims around that.

### 7.3 Pre-M11 audits (Phases 10–12, [`modular-plan.md`](modular-plan.md))
- [ ] **Requirements audit** — every claim in `project.md` still supported by code + tests?
- [ ] **Code audit** — dead code, over-engineering, unhandled edges (`ruff`, and a read
      pass over each module).
- [ ] **ML audit** — (a) leakage: re-verify Blue never sees the taint graph, at every
      call site; (b) public-dataset provenance: `THIRD_PARTY.md` matches what
      `fetch.py` actually pulls; (c) reproducibility: fixed seed → identical numbers,
      documented.

---

## 8. Workflow rules (from the project owner — follow exactly)

- **Modular cycle, no exceptions:** small piece → unit tests (valid / invalid / edge) →
  integration test → fix → update docs → commit. One module (or sub-module) per commit.
- **Push ONLY on explicit confirmation, every time.** Local commits accumulate freely.
  The owner says "push" per commit — do not batch-push or assume.
- **Tests are a gate, not a formality.** Non-trivial logic leaves one runnable check
  behind. Fast suite must be green before every commit; slow suite before anything that
  touches training/eval.
- **Public datasets only**, licences verified from source, **never vendored** — fetched
  into git-ignored `data/` with a checksum manifest.
- **Scale is YAML-only.** Nothing in `arena/` hardcodes a size, step count, or generation
  count. If you need a knob, it goes in `arena/config.py`.
- **Anti-leakage is sacred.** Blue never observes the taint graph. Attack success is
  judged by data flow, never by whether Blue was fooled. Any new Blue input goes through
  `arena/features.py` and must pass the invariance tests.
- Commit messages: end with `Co-Authored-By: Claude <noreply@anthropic.com>` if AI-assisted.
- The repo has **caveman** + **ponytail** hooks active (terse prose, laziest-working code)
  — they govern style, not rigour. Input validation, error handling at trust boundaries,
  and tests are never simplified away.

---

## 9. Command cheat-sheet

```bash
# tests
python3 -m pytest -q -m "not slow"                 # fast (~20s)
python3 -m pytest -q                               # + slow training tests (~8 min)
python3 -m pytest tests/test_contracts.py -q       # cross-module invariants (0.4s)
python3 -m ruff check arena/ scripts/ tests/

# datasets (git-ignored -> regenerate)
python3 scripts/fetch_data.py                      # TAMAS + Toucan -> data/

# baselines (M4)
python3 scripts/run_baselines.py small.yaml

# train (M5 single-side / M6-M7 self-play)
python3 scripts/train_ppo.py blue --steps 60000
python3 scripts/train_selfplay.py small.yaml --generations 4 --steps 40000 --out runs/sp.pt

# evaluate (M8)  -- the headline table
python3 scripts/run_eval.py small.yaml                       # baselines only
python3 scripts/run_eval.py small.yaml --blue runs/sp.pt     # + ARENA-trained Blue
python3 scripts/run_eval.py small.yaml --curve runs/sp.pt    # exploitability curve over the league

# reproduce the corrected audit table (~30-40 min: trains then evals)
python3 scripts/train_selfplay.py small.yaml --generations 4 --steps 40000 --out runs/audit.pt
python3 scripts/run_eval.py small.yaml --blue runs/audit.pt
```

---

## 10. Gotchas / things that already bit us

- **Never assert on PettingZoo `terminations` / `truncations` after a full `agent_iter`
  loop** — the loop drains both dicts, so `all()` / `any()` over them is vacuously true.
  This hid a contradiction between two tests for four modules. Snapshot the flags mid-loop
  (see `flags_at_end` in `tests/test_env.py`).
- **`ARENAEnv.episode_index`, not `scenario.scenario_id`**, is the "new episode" signal —
  `reset(seed=)` rebuilds the generator and restarts the id counter, so ids repeat.
- **Always pass `seed=` to `make_policy`** — it otherwise reads ambient global torch RNG
  and a run becomes irreproducible. `PPOTrainer` owns its own numpy Generator for the same
  reason.
- **PPO learning rate is `1e-3`, not the usual `3e-4`** — Blue's reward landscape has a
  wide flat "quarantine everything" basin; `3e-4` falls into it, `1e-3` escapes. Measured.
- **Blue's actor bias starts toward ALLOW** (`[1.0, 0.0, -1.0]`) — quarantine is absorbing,
  so a trigger-happy Blue never sees long sequences. Took seeds-that-learn from 3/5 to 5/5.
- **Learned defenders are evaluated GREEDY** (`deterministic=True`) to match the
  deterministic baselines. This makes `arena_blue` look *worse* (pure strategies are
  easier to best-respond to) — kept anyway, it's the honest comparison.
- Background training runs: `python x &` inside a shell tool detaches but the wrapper
  exits immediately. Poll an output file, and remember Python buffers stdout when
  redirected (results appear only on process exit).
- `data/` ignore is anchored (`/data/`) so it does **not** swallow the `arena/data/`
  source package. If you add a top-level ignore, anchor it.

---

*Handoff prepared after the M1–M9 audit. Questions → read the module's `docs/*.md` first,
then `architecture.md`. The audit doc explains every deviation from what the older docs
claim.*
