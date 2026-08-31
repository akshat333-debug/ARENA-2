# Audit — M1 through M9

Full module-wise and data-flow audit run after M9 landed. Three real bugs, all
in the M5–M8 measurement path, all fixed here. **One of them invalidated the M8
headline result**, which is corrected below rather than quietly restated.

## Summary

| # | Bug | Module | Severity |
|---|---|---|---|
| 1 | Env raised `terminated` *and* `truncated` together, so PPO recorded **no GAE boundary** at ~90% of episode ends | `env.py` ↔ `ppo.py` | **critical** — corrupted every training run M5–M8 |
| 2 | `GenerationStats.exploitability` measured Red against a *league sample of past Blues*, not the current Blue, and on a distribution that capped it at `adversarial_ratio` | `selfplay.py` | **high** — the per-generation headline number measured the wrong quantity |
| 3 | Learned Blue evaluated **stochastically** while every baseline is deterministic | `eval/harness.py` | **medium** — apples-to-oranges leaderboard |
| 4 | `paper.yaml` scaled the *scenario* but not the training budget — a harder problem on a laptop-tuned budget | `configs/paper.yaml` | **medium** — the scale-up target didn't scale |

Everything else checked out. A second, randomised pass over 250+ episodes
(`tests/test_contracts.py`) and 20/20 end-to-end data-flow assertions found no
further contract violations: all six attack families generate and are winnable;
benign traffic never exfiltrates; Blue's observation is byte-invariant to the
ground truth; env rewards are exactly `compute_rewards`; a completed objective
is reproducible by independent replay; both baselines' train-time and
inference-time feature paths agree bit-for-bit (0 mismatches over 910
decisions); the reward asymmetry ties the two degenerate defenders exactly; the
M9 profile swap-in shifts the benign mix as intended.

---

## Bug 1 — the GAE boundary was invisible (critical)

`ARENAEnv._end_episode` set `terminations[a] = True` for every ending **and**
`truncations[a] = True` on a step cap. PPO's `collect()` used the standard
idiom:

```python
terminal        = terminated and not truncated   # False when both are set
truncated_value = ... if truncated and not terminated else None   # also None
```

So a step-cap episode was recorded as **neither** terminal nor truncated — not
an episode boundary at all. `compute_gae` then bootstrapped `V(s_{t+1})` from
the first state of the *next* episode and let the advantage trace run straight
through the boundary.

Measured on a 1024-step rollout: **9 of 88 episode ends were marked. 79 were
invisible.** Step-cap endings dominate (29 of 34 episodes in a sampled trace),
so this hit the large majority of transitions in every PPO run from M5 onward.

The GAE unit tests never caught it because they test `compute_gae` in isolation
with hand-built flags — correctly, and they still pass. The gap was the
`env → collect()` hand-off, which nothing tested. Worse, the two env tests that
*looked* like they covered it —

```python
assert all(env.truncations.values())      # test_step_cap_truncates
assert not any(env.truncations.values())  # test_objective_completion_ends_episode
```

— both ran **after** `agent_iter` had drained the agents, so both dicts were
empty and both assertions were vacuously true. Two contradictory expectations
coexisted for four modules.

**Fix.** The step cap is a genuine terminal state here, not a Gymnasium
`TimeLimit` truncation, for two independent reasons:

1. The remaining budget is **observable** — `step_index / max_steps` rides in
   every encoded call row and in Red's task vector. A time-aware MDP owns its
   horizon (Pardo et al. 2018, *Time Limits in RL*).
2. `compute_rewards` **settles the whole episode** at the cap: running out of
   time is already priced. Bootstrapping `γ·V(s)` on top would credit a future
   that cannot happen.

So `_end_episode` now sets `terminations=True, truncations=False` — mutually
exclusive, per Gymnasium's contract — and reports `hit_step_cap` in `infos`.
`collect()` additionally derives `terminal` from `done = terminated or
truncated`, so an env that violates mutual exclusion degrades to a terminal
instead of silently dropping the boundary.

**New tests:** flags are mutually exclusive at all three endings, checked at the
`SingleAgentARENA.step` interface PPO actually consumes; every finished episode
records exactly one GAE boundary; a deliberately-misbehaving both-flags env
still produces boundaries. The vacuous env assertions are replaced with a
`flags_at_end` helper that snapshots the dicts *before* the drain.

## Bug 2 — the per-generation "exploitability" measured the wrong thing

`SelfPlayTrainer._evaluate` computed it from `_red_env()`. With the M7 league,
that env's opponent is a **sample of past Blues**, so the number was Red's
success against the pool average — it rises as Red improves regardless of how
good the current Blue is. It was also measured at `adversarial_ratio = 0.5`,
which caps attack success near 0.5 since benign episodes have no objective.

The tell was in the failing run: the final generation reported
`exploitability=0.48` (i.e. ~96% of the 0.5 ceiling) while the *same* generation
reported `blue_attack_success=0.075` — the current Blue was stopping 92.5% of
attacks. Two numbers, same Blue, opposite stories.

**Fix.** A dedicated `_exploit_env()`: Red vs the **current** Blue, greedy, with
`adversarial_ratio` forced to 1.0 — the same scale as
`arena.eval.exploitability`, so the in-loop proxy and the real metric are
finally comparable.

## Bug 3 — stochastic defender vs deterministic baselines

`harness._blue_policy_for` wrapped a learned Blue in
`TorchPolicyAdapter(deterministic=False)`. Every baseline on the same leaderboard
is a deterministic threshold rule, and a deployed neural defender takes the
argmax. Sampling handed the attacker a free pass whenever the draw fell off the
greedy action.

**Fix.** Greedy at evaluation (`deterministic=True`) in `harness`,
`blue_from_state_dict`, and the self-play proxy. Training opponents stay
stochastic — that is exploration, not deployment.

Note this made `arena_blue` look **worse** (0.655 → 0.710): a pure strategy is
easier to best-respond to than a mixed one, which is a real game-theoretic
effect and worth keeping in view. It is kept anyway because it is the
like-for-like comparison and the conservative choice for our own method.

## Bug 4 — `paper.yaml` was a scale-up in name only

It overrode the `scenario` section and nothing else, so every training and
evaluation budget fell back to defaults tuned for `small.yaml`: 24-step episodes
and up to 19 tools trained for the same 40k steps/side, evaluated with the same
20k-step best response. A *harder* problem on the *same* budget is strictly
worse than `small.yaml`, not a scale-up — and "scale lives entirely in YAML" is
a claim the README makes.

This matters more after Bugs 1–3, because scale is the first lever listed for
closing the exploitability gap. **Fix:** `paper.yaml` now also scales `policy`
(128/128/64), `ppo` (4096-step rollouts), `selfplay` (12 generations ×
250k steps/side) and `eval` (150k-step best response, 500 eval episodes).
Verified to load and train end-to-end.

## New: cross-module contract tests

Every existing test file checked one module against hand-built inputs. Nothing
checked the **hand-offs** — which is where all three measurement bugs lived.
`tests/test_contracts.py` (10 tests, 0.4s) closes that gap with randomised
episodes: env rewards equal `compute_rewards` exactly, outcome fields are
internally consistent, a completed objective replays independently, Blue's
observation is invariant to the ground truth, observations sit inside their
declared spaces, the Gym wrapper's episode return equals the reward breakdown
for both learners, and exactly one of `terminated`/`truncated` fires.

Two of them exist specifically to stop a *vacuous* suite: one asserts the
random-episode fixture actually produced all three ending types, and the
anti-leakage test constructs its scenario pairs explicitly rather than searching
for naturally-occurring ones (a search that finds none passes silently — the
first draft of this very check compared 0 pairs).

Mutation-checked: reintroducing Bug 1 fails three tests immediately. It does
*not* fail PPO's boundary test, because `collect()`'s hardening absorbs a
misbehaving env — the layered defence working as intended.

---

## Consequence: the M8 headline result does not survive

The old leaderboard reported `arena_blue` exploitability **0.120** vs
`causal_monitor` **0.376** — "~3× less exploitable than the strongest static
baseline". That number was produced by a best-response Red crippled by Bug 1.
Since `exploitability()` *trains* its attacker with the same broken PPO, every
defender looked far less exploitable than it is, and the ordering was not
meaningful.

Re-measured with all three fixes, `small.yaml` at the `EvalConfig` defaults
(20k-step fresh best-response Red, 300 eval episodes), ARENA Blue from 4
generations × 40k steps/side, all defenders calibrated to ~5% FPR:

```
defender                AUROC   TPR@5%FPR  exploitability   op.FPR  (fresh Red)
-------------------------------------------------------------------------------
causal_monitor          0.980       0.861           0.690    0.036        0.090
single_shot             0.728       0.034           0.773    0.010        0.190
static_allowlist        0.513       0.037           0.777    0.012        0.183
arena_blue              0.927       0.461           0.780    0.047        0.123
```

`static_allowlist` at 0.777 now matches TAMAS's reported ~80% static-baseline
failure; the crippled-attacker measurement had understated it at 0.564.

And the exploitability curve across the league's Blue checkpoints is **flat**:

```
gen0 0.720   gen1 0.765   gen2 0.740   gen3 0.710      (delta -0.010)
```

**Honest reading.** At `small.yaml` scale the co-evolved defender is *not* less
exploitable than the static baselines — it is the most exploitable row on the
board — and self-play is not driving exploitability down over four generations.
`arena_blue` is also a weaker detector than the causal monitor on the
threshold-free columns (AUROC 0.927 vs 0.980, TPR@5%FPR 0.461 vs 0.861) — the
causal monitor's hand-engineered `sensitive_read_before_sink` feature
essentially encodes the taint rule, which a GRU has to discover from sparse
episode-level reward.

What *does* still hold, and is separately tested:

- Blue beats every baseline on **mean reward against the training
  distribution** (M5, `test_m5_integration`) — trained Blue positive, causal
  monitor / passive / paranoid negative.
- The league still prevents the M6 drift to passive: quarantine rate 0.585 at
  the last generation where the leagueless loop collapsed to ~0.01, and Blue
  stops >92% of the league's attacks (`test_m7_integration`).

So the gap is specifically **robustness to a fresh best response** — which is
exactly the static-vs-adversarial-evaluation gap this project exists to measure,
now pointing at our own method. That is a result, not a failure; it is just not
the result previously written down.

**Open for M11** (not claimed as solved anywhere in the repo): whether the gap
closes with `paper.yaml` scale, more generations, PFSP-weighted league sampling,
reward shaping that densifies Blue's signal, or giving Blue the causal features
the monitor gets. Nothing in the docs now asserts an outcome that has not been
measured.

## Method

- `pytest -q` — 343 fast + 32 slow, all green after the fixes.
- New `tests/test_contracts.py` — randomised cross-module invariants, mutation-checked.
- 20/20 end-to-end data-flow assertions M1→M9 (families, taint ground truth,
  anti-leakage, flag contract, feature shapes, reward tie, M9 profile swap).
- Baseline train-vs-inference feature parity: 0 mismatches over 910 decisions.
- Every metric in `eval/metrics.py` re-verified exact against `sklearn.metrics`.
- Both configs load and train end-to-end.

## Superseded: this audit's own replacement claim was also one seed

The audit withdrew M8's "~3x less exploitable" and replaced it with
`arena_blue` 0.780 vs `causal_monitor` 0.690 — "at small.yaml scale,
co-evolution does NOT beat the static baselines".

That replacement was itself a **single seed**, and a 5-seed repeat does not
support it (`docs/m11-multiseed.md`). Paired per-seed, the difference is
`[+0.090, -0.007, -0.203, +0.123, -0.357]` — mean -0.071, spread 0.204, sign
flipping. Three of five seeds favour the co-evolved defender.

So the correct statement is neither "co-evolution loses" nor "co-evolution
wins": **the comparison is unresolved at this scale**, and `arena_blue`'s seed
variance (+/-0.274) is an order of magnitude larger than any static baseline's.
What survives 5/5 seeds is that the causal monitor beats the allow-list.

Recorded here rather than quietly edited above: the fix for a
single-seed claim is not a better single-seed claim.

## What was *not* re-measured

Stated so nothing here is mistaken for a verified number:

- The M5 reward figures and the M6/M7 per-generation traces predate the fixes.
  Their *orderings* are re-verified by the slow tests; the exact values are not.
- The M7 league-**off** collapse column was not re-run.
- Only one seed was run for the corrected leaderboard. The
  `causal_monitor`-vs-`single_shot` gap in particular moves with budget and seed;
  `arena_blue` being worse than the causal monitor held at both budgets tried.
