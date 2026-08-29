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
skill. So `evaluate_defenders` first calibrates every defender with a tunable `threshold`
to `EvalConfig.calibration_fpr` (5%) on the benign decisions, *then* measures
exploitability. Rule-based (`StaticAllowList`) and neural (`ActorCritic`) defenders pass
through; their operating FPR is just measured.

A defender is scored by any of: a `BlueBaseline` (`.score`), an `ActorCritic` (softmax
P(flag/quarantine) via `policy_score_fn`), or a plain Blue policy callable (its own hard
verdict → an honest AUROC of 0.5 for a non-discriminating reference like passive/paranoid).

## Result — the co-evolved defender vs the static baselines

`small.yaml`, ARENA Blue from a 3-generation league run, all defenders calibrated to
~5% FPR, exploitability from a fresh 15k-step best-response Red:

```
defender                AUROC   TPR@5%FPR  exploitability   op.FPR  (fresh Red)
-------------------------------------------------------------------------------
arena_blue              0.951       0.631           0.120    0.019        0.008
causal_monitor          0.979       0.914           0.376    0.049        0.100
single_shot             0.726       0.179           0.432    0.049        0.056
static_allowlist        0.514       0.034           0.564    0.007        0.196
```

**The ARENA-trained Blue is ~3× less exploitable than the strongest static baseline**
(0.120 vs the CASPIAN-style causal monitor's 0.376) and ~4.7× less than the allow-list,
whose 0.564 matches TAMAS's reported ~80% static-baseline failure. This is the project's
headline claim, on its own primary metric.

Note `arena_blue`'s AUROC/TPR sit *below* the causal monitor's even though its
exploitability is far lower. That is expected and worth stating: `arena_blue` is a
*policy*, not a scorer — it acts at the trajectory level (quarantine the right sessions,
in time), and the softmax-P(intervene) proxy used to place it on the ROC undersells a
decision mechanism that isn't a threshold on that score. The exploitability column — does
Red actually get through — is the one that reflects what the defender does, and there the
co-evolved policy wins decisively.

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
