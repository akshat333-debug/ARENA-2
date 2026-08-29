# M4 — Features & baseline defenders

Modules: [`arena/features.py`](../arena/features.py), [`arena/scripted.py`](../arena/scripted.py),
[`arena/baselines/`](../arena/baselines/). New dep: `scikit-learn`.

## `features.py` — one definition of "the observable state"

The per-call and observation encoders were lifted out of `env.py` (M3) so the M4
baselines and the M5 policy networks share exactly one implementation. `env.py`
now imports `encode_call`, `history_matrix`, `blue_observation_space`,
`red_observation_space` from here; the 173 M1–M3 tests pass unchanged, which is
the regression check for the move.

Everything is **observable metadata only** — nothing reads the taint graph.

| Function | View | Consumer |
|---|---|---|
| `encode_call` | one call → 21-float row (or +3 for Red) | env observations, M5 nets |
| `history_matrix` | resolved calls + pending proposal → `(max_steps, F)` | env observations |
| `single_call_features` | pending call alone → 17 floats (the observable prefix of `encode_call`) | single-shot baseline |
| `sequence_features` / `sequence_features_from_rows` | whole sequence → 15 floats with causal proxies | causal baseline |

`sequence_features` carries the proxies that make a sequence-aware defender
possible without seeing the taint graph: `sensitive_read_before_sink`,
`untrusted_read_before_privileged`, laundering depth, per-side-effect fractions,
max declared sensitivity. `_from_rows` recomputes the identical vector from an
encoded observation matrix, so a deployed detector needs nothing but `blue_obs`.

## `scripted.py` — non-learned policies

- `ScriptedAttacker(scenario)` — plays a minimal winning chain (source →
  propagator×k → sink) for the scenario's objective.
- `BenignRoller(scenario)` — plausible benign traffic. It **does** read sensitive
  files and **does** hit external sinks (legitimate dev/clinical tooling), but
  once it has touched a source it stops calling sinks, so no episode it drives is
  accidentally an attack — and a lone sensitive read is not a give-away tell that
  would unfairly flatter the single-shot detector.
- `passive_blue` / `paranoid_blue` — allow-all / quarantine-all references.

## The three baselines

All implement `BlueBaseline`: `score(blue_obs) → P(attack)`, `act` thresholds it
to a verdict, `__call__ = act` so a baseline is directly a Blue policy for
`SingleAgentARENA` and the eval harness.

| Baseline | Feature view | What it can/can't do |
|---|---|---|
| `StaticAllowList` | pending call only, fixed rule | blocks a lone high-sensitivity sink; **no provenance**, so the split chain sails through. Today's default. |
| `SingleShotDetector` | `single_call_features` (call in isolation), logistic regression | a lone `read_env_file` is byte-identical in benign and attack traffic → cannot separate the split chain |
| `CausalMonitor` | `sequence_features` (trajectory), logistic regression | sees `sensitive_read_before_sink` forming → quarantines at the sink, before completion. The strong baseline. |

Trained baselines fit on `Decision` rows from
`arena.baselines.collect.collect_decisions` (scripted attacks + benign rolls
against an allow-all Blue, so the whole chain is observed). Label is
`on_attack_path` (ground truth from the taint graph — training/eval only).
Threshold is tuned by Youden's J on a validation split; the eval harness can
re-calibrate to a fixed false-positive budget.

## Scoring (`baselines/evaluate.py`)

Light — the full leaderboard is M8. Provides:

- `attack_success_rate` — fraction of scripted attacks that complete despite the
  defender (the exploitability proxy).
- `false_quarantine_rate` — benign *sessions* quarantined (episode-level).
- `decision_auroc` — per-decision, no sklearn dependency.
- `tpr_at_fpr` / `calibrate_threshold` — per-decision TPR at a fixed per-decision
  FPR; the M8 headline metric, in miniature.

`python3 scripts/run_baselines.py small.yaml` prints the table. Representative run
(200+200 train episodes, 100+100 eval):

| baseline | AUROC | TPR@5%FPR | attack_success | episode_FP |
|---|--:|--:|--:|--:|
| static_allowlist | 0.51 | 0.03 | **0.89** | 0.12 |
| single_shot | 0.73 | 0.19 | 0.00\* | 1.00\* |
| causal_monitor | 0.98 | 0.85 | 0.00\* | 0.61\* |

\* single_shot / causal `attack_success` and `episode_FP` are at the Youden
threshold, which for these two is paranoid — the honest apples-to-apples is
**TPR@5%FPR**, where single_shot manages only 0.19 vs causal's 0.85. The
allow-list's 0.89 attack-success matches TAMAS's reported ~0.80–0.82 for static
single-shot defences.

## Tests

- `tests/test_features.py` — `encode_call` layout, `single_call_features` is the
  observable prefix of `encode_call`, `sequence_features` causal proxies fire
  correctly and don't false-positive when a sink precedes the read,
  `_from_rows` agrees with `_from_tools`.
- `tests/test_baselines.py` — each baseline is a valid Blue policy; allow-list
  misses the canonical chain and blocks a lone high-sensitivity sink; trained
  detectors fit and score in range; single-shot gives identical scores to the
  same call in different contexts (the limitation, demonstrated); causal scores a
  real chain's sink above a lone read.
- `tests/test_m4_integration.py` — collect → train → score: allow-list at chance
  and ~fully exploitable; causal beats single-shot at a matched 5% FPR by a wide
  margin; AUROC ordering matches the information each detector is given;
  reproducible.
