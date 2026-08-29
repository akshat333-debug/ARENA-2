# M8 — Evaluation harness

Module: [`arena/eval/`](../arena/eval/) — `metrics.py`, `exploitability.py`, `harness.py`.
Script: [`scripts/run_eval.py`](../scripts/run_eval.py). Config: `EvalConfig`.

## Exploitability — the primary metric

> Freeze the defender. Spawn a **fresh, randomly-initialised** Red. Train it as a best
> response for a fixed budget. Report the fraction of attack episodes it completes.

`exploitability(blue, cfg, br_steps=…, n_eval=…)` → `ExploitabilityResult`. The Red is
built new every call — never warm-started, never one passed in — so the number is *how
beatable is this defender by an attacker starting from scratch*, not *how well did some
existing Red do*. `ExploitabilityResult.fresh_red_start` records that fresh Red's success
**before** best-response training (≈ chance); `test_exploitability` asserts it stays below
0.5, which is the check that no warm start leaked in.

Two design points:

- **All-adversarial measurement.** `small.yaml` has `adversarial_ratio 0.5`; if
  exploitability were measured on that mix, the benign half (no objective, attack success 0
  by definition) would cap it at ~0.5. `exploitability()` overrides the ratio to 1.0 for
  the measurement — every episode is an attack attempt. The frozen Blue's benign behaviour
  is irrelevant to this number (its false-positive cost shows up in the AUROC/FPR columns).
- **`exploitability_curve([(label, blue), …])`** runs one fresh best-response Red per
  point, all from the same seed so the points are comparable. `curve_from_league(trainer)`
  builds it over a finished `SelfPlayTrainer`'s Blue checkpoint pool — the declining curve
  across generations is the headline figure.

## Secondary metrics — `metrics.py`

`roc_auc`, `roc_curve`, `tpr_at_fpr`, `confusion_at_threshold`. Rank-based, no sklearn in
the hot path; `test_metrics.py` checks every one against `sklearn.metrics` on random data
and against hand-computed values on fixtures (`roc_auc([0.1,0.4,0.35,0.8],[0,0,1,1]) ==
0.75` exactly, tie mid-ranks, single-class → 0.5, etc.).

`tpr_at_fpr` is deliberately on CASPIAN's scale (project.md §8): a defender that only stops
attacks by drowning benign traffic scores near zero here.

## Leaderboard — `harness.py`

`evaluate_defenders({name: defender}, cfg)` → one `LeaderboardRow` each:

| column | meaning |
|---|---|
| `auroc`, `tpr_at_5pct_fpr` | threshold-free detection quality, per Blue decision, vs the taint-graph ground truth |
| `exploitability` | best-response attack success **at a calibrated ~5% FPR operating point** |
| `operating_fpr` | the FPR the defender actually sits at after calibration — sanity column |
| `fresh_red_start` | the fresh Red's pre-training success — the no-warm-start check |

**Calibration matters.** A trained per-call detector tuned by Youden's J quarantines
almost everything, which drives its *raw* exploitability to 0 — but that is paranoia, not
skill. So `evaluate_defenders` first calibrates every defender exposing a tunable
`threshold` to `EvalConfig.calibration_fpr` (5%) on the benign decisions, *then* measures
exploitability. `StaticAllowList` has a `threshold` too, so it takes the same path, but
its scores are binary and calibration lands just above 0 — its verdicts are unchanged.
A neural `ActorCritic` has no tunable threshold; its operating FPR is simply measured by
running its own verdict on the benign decisions.

**Learned defenders are scored greedily.** A trained Blue is wrapped
`deterministic=True` for both the exploitability run and the operating-FPR
measurement. Every baseline here is a deterministic threshold rule and a deployed
neural defender takes the argmax, so sampling would hand the attacker free passes that
are not a property of the defender under test. (This makes `arena_blue` look *worse* —
a pure strategy is easier to best-respond to than a mixed one — which is why it is the
conservative choice as well as the like-for-like one.)

A defender is scored by any of: a `BlueBaseline` (`.score`), an `ActorCritic` (softmax
P(flag/quarantine) via `policy_score_fn`), or a plain Blue policy callable (its own hard
verdict → an honest AUROC of 0.5 for a non-discriminating reference like passive/paranoid).

## Result — the co-evolved defender vs the static baselines

> **Corrected after the M1–M9 audit.** An earlier version of this file reported
> `arena_blue` exploitability **0.120** vs the causal monitor's 0.376 — "~3× less
> exploitable". That was an artifact: a PPO bug (`env` raised `terminated` and
> `truncated` together, so ~90% of episode boundaries were invisible to GAE)
> crippled the best-response Red that `exploitability()` trains, which made
> *every* defender look far less exploitable than it is. See
> [audit-m1-m9.md](audit-m1-m9.md). The numbers below are the re-measurement.

`small.yaml` at the `EvalConfig` defaults (20k-step best-response Red, 300 eval episodes,
200+200 decisions), ARENA Blue from a 4-generation league run at 40k steps/side. Fully
regenerable:

```bash
python3 scripts/train_selfplay.py small.yaml --generations 4 --steps 40000 --out runs/audit.pt
python3 scripts/run_eval.py small.yaml --blue runs/audit.pt
```

```
defender                AUROC   TPR@5%FPR  exploitability   op.FPR  (fresh Red)
-------------------------------------------------------------------------------
causal_monitor          0.980       0.861           0.690    0.036        0.090
single_shot             0.728       0.034           0.773    0.010        0.190
static_allowlist        0.513       0.037           0.777    0.012        0.183
arena_blue              0.927       0.461           0.780    0.047        0.123
```

Exploitability curve over the league's Blue checkpoints (15k-step Red) — **flat**:

```
gen0 0.720   gen1 0.765   gen2 0.740   gen3 0.710      (delta -0.010)
```

A weaker 15k-step attacker lowers every row (`causal_monitor` 0.420, `arena_blue` 0.710)
and swaps `single_shot` and `causal_monitor`, whose gap is inside run-to-run variance at
this budget — `test_m8_integration` allows for that with a tolerance band. What is stable
across both budgets is the finding below.

**At this scale the co-evolved defender is *not* less exploitable than the static
baselines — it is the most exploitable row here — and four generations of self-play do
not drive exploitability down.** `arena_blue` is also the weaker detector than the causal
monitor on the threshold-free columns (AUROC 0.927 vs 0.980, TPR@5%FPR 0.461 vs 0.861):
that monitor's hand-engineered `sensitive_read_before_sink` feature essentially encodes
the taint rule, which the GRU has to discover from sparse episode-level reward.

Two things that do still hold, and are separately tested: Blue beats every baseline on
mean reward against the **training** distribution (M5), and the league still prevents the
M6 drift to passive (M7 — quarantine rate 0.585 where the leagueless loop collapsed to
~0.01, stopping >92% of the league's attacks).

So the gap is specifically **robustness to a fresh best response** — the same
static-vs-adversarial evaluation gap the project exists to measure, now pointing at our
own method. Whether it closes with `paper.yaml` scale, more generations, PFSP-weighted
sampling, denser reward, or handing Blue the causal features is open for M11 and is not
claimed anywhere as settled.

Note that `static_allowlist`'s 0.777 now sits squarely in the range of TAMAS's reported
~80% static-baseline failure — the crippled-attacker measurement had understated it at
0.564. The one number the audit made *more* consistent with the literature is the one
ARENA is benchmarked against.

`run_eval.py --blue <checkpoint>` adds an ARENA Blue as `arena_blue`; `--curve
<checkpoint>` plots the exploitability curve over its self-play generations.

## Tests

- `tests/test_metrics.py` (24) — every metric vs sklearn on random data and vs
  hand-computed fixtures; ROC-curve area equals `roc_auc`; input validation.
- `tests/test_exploitability.py` (7 fast + 2 `slow`) — fresh best-response Red each call
  (deterministic, starts near chance, does not mutate a policy passed in); best-response
  training raises exploitability vs a passive Blue; a paranoid Blue is unexploitable;
  measured against an all-adversarial distribution (not capped at the ratio); curve
  ordering; `blue_from_state_dict` round-trips.
- `tests/test_m8_integration.py` (`slow`) — full harness on the three baselines: every row
  well-formed, calibration lands near the target FPR, AUROC ordering matches information
  available, the causal monitor is the least exploitable baseline at a matched FPR,
  single-shot cannot reach its `TPR@5%FPR`, the leaderboard formats, reference policies
  bracket the baselines, curve ordering.
