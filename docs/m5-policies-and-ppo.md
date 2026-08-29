# M5 — Policy networks & PPO

Modules: [`arena/policies.py`](../arena/policies.py), [`arena/ppo.py`](../arena/ppo.py),
[`scripts/train_ppo.py`](../scripts/train_ppo.py). Config: `PolicyConfig`, `PPOConfig`.
New dep: `torch` (already present).

## Networks

**Blue** — GRU over the observed call sequence → `allow` / `flag` / `quarantine` + value.
Trajectory-level by construction; the per-call alternative is exactly the M4 single-shot
baseline, which provably cannot separate the split chain.

**Red** — scores each tool in the episode's registry by dot product between a tool
embedding and a context vector built from the task, the objective, and the history
*including Blue's past verdicts*. That verdict channel is what lets Red adapt to the
defender it faces. Scoring by embedding (rather than a fixed output head) means one Red
generalises across registries of different size and contents; the fixed `Discrete` action
space is masked down to the episode's real registry, and
`test_red_never_samples_a_masked_action` checks nothing outside the mask is ever sampled.

`TorchPolicyAdapter` wraps a trained net as the plain `obs -> int` callable the env,
baselines and league all speak, so a learned policy is a drop-in for a scripted one.

Sizes come from `PolicyConfig` (default hidden 64 → ~31k params Blue, ~42k Red).

## PPO

Own single-file trainer, not Stable-Baselines3 (architecture.md §2). Standard PPO —
GAE(λ), clipped surrogate, value loss, entropy bonus, grad-norm clipping — plus two
details that matter here:

- **Episode boundaries.** `compute_gae` handles both kinds: a *terminal* gets no
  bootstrap, a *truncation* bootstraps from the value of the cut state, and the advantage
  trace resets at either because the next transition belongs to a different episode.

  ARENA itself only ever produces terminals. The step cap **is** a terminal state here,
  not a Gymnasium `TimeLimit` truncation: the remaining budget is observable
  (`step_index / max_steps` is in every call row and in Red's task vector), and
  `compute_rewards` settles the whole outcome at the cap, so there is no future to
  bootstrap. The truncation path is kept because it is correct and tested, and becomes
  live the moment anyone chunks episodes.

  > This is the corrected account. Until the M1–M9 audit the env raised `terminated` and
  > `truncated` **together** at the step cap, which made both `terminated and not
  > truncated` and `truncated and not terminated` false — so ~90% of episode ends were
  > recorded as no boundary at all and GAE bled advantages across episodes. Every M5–M8
  > training run was affected. See [audit-m1-m9.md](audit-m1-m9.md).
- **Dict observations.** Both sides observe dicts; the buffer stores them per-key and
  batches with `stack_obs`.

Device default is **cpu**: for nets this small the MPS kernel-launch overhead outweighs
the parallelism. `mps` remains an explicit opt-in for scaled-up configs.

## Results (small.yaml, Apple M3)

**Red vs a passive Blue**, single fixed scenario, 15k steps (~6 s):

| family | attack success before → after |
|---|---|
| impersonation | 0.22 → 1.00 |
| direct_prompt_injection | 0.05 → 0.52 |
| indirect_prompt_injection | 0.13 → 0.55 |

**Blue vs the adaptive scripted Red**, full scenario distribution, 60k steps (~20 s),
5 seeds — all learn to discriminate:

| seed | return | quarantine rate | attack success |
|--:|--:|--:|--:|
| 0 | +0.605 | 0.550 | 0.000 |
| 1 | +0.518 | 0.537 | 0.000 |
| 2 | +0.515 | 0.547 | 0.000 |
| 3 | +0.531 | 0.587 | 0.000 |
| 4 | +0.610 | 0.523 | 0.007 |

Reference policies on the same episodes: **passive −0.487**, **paranoid −0.537**, **M4
causal monitor −0.092**. The theoretical ceiling at a 0.49 adversarial rate is ≈ **+0.49**,
so the trained Blue is at or slightly above the reference ceiling, and its quarantine rate
tracks the adversarial rate (~0.5) instead of saturating at 1.0 — it is discriminating, not
blanket-blocking. **The RL-trained sequence-aware defender beats the CASPIAN-style
baseline by ~0.6 reward.**

> **On the results above:** they were measured before the M1–M9 audit fixed the
> GAE-boundary bug. The *ordering* was re-verified after the fix
> (`test_m5_integration` still passes: trained Blue positive, causal monitor /
> passive / paranoid negative), but the exact figures were not re-measured.

## Three real bugs this module surfaced

### 1. Detection credit needed evidence (reward semantics)

Blue originally got the full detection reward for *any* quarantine in an adversarial
episode that didn't complete — including quarantining at step 0, before an attack existed.
A defender that blocks the first call of every session therefore "detected" 100% of
attacks, and PPO went straight to blanket paranoia.

Fixed with `TaintTracker.would_advance_attack`: a quarantine only earns detection credit
when the blocked call was demonstrably part of an attack in flight — it would complete the
objective, or it carries/extends taint from a designated source. `EpisodeOutcome` gained
`caught_in_flight` to carry this. Blocking an unrelated call now earns nothing.

Also: `attack_path` is empty when the attack never completes — i.e. precisely when Blue
*won* — so Blue lost its on-chain flag credit for succeeding. Added
`TaintTracker.live_chain_steps`, the in-flight chain, which the env now falls back to.

### 2. `w_false_quarantine` default 1.0 → 2.0

With `w_false_quarantine == w_detect`, blanket quarantine had EV 0 vs allow-always's −0.5,
so paranoia was the *best* trivial strategy — contradicting the proposal's "precision, not
paranoia" (§3.3). The default is now `w_detect + w_miss`, which makes the two degenerate
defenders score **exactly equally badly** (−0.5 each at a 50/50 ratio, vs +0.5 for a
discriminating Blue), so the learning signal has to come from discrimination. Asserted in
`rewards._demo`.

### 3. `scenario_id` collides across resets

`env.reset(seed=…)` rebuilds the scenario generator, restarting its counter — so scenario
ids repeat. `AdaptiveScriptedRed` keyed off that id and silently reused a spent attacker
that emitted filler actions: episodes looked adversarial while no attack was played, which
corrupted every seeded evaluation loop. Added `ARENAEnv.episode_index` (monotonic across
resets) and keyed the opponent off that.

## Two training-stability findings

**Learning rate.** At the usual 3e-4 Blue reliably converged into the paranoid basin; at
1e-3 it escapes and learns. `PPOConfig.lr` defaults to 1e-3 with the reason in the field
comment.

**Permissive initialisation.** Quarantine is an *absorbing* action — it ends the episode —
so a trigger-happy Blue only ever sees length-1 episodes and never collects the long
sequences carrying the evidence it needs. Measured: 2 of 5 seeds collapsed into that basin.
Biasing Blue's actor toward `ALLOW` at init (`[1, 0, -1]`) makes the defender start
permissive and learn when to intervene: **5 of 5 seeds** then learn. This is a real
property of the game, not a hyperparameter tweak, and is worth a sentence in the paper.

## Reproducibility

Two sources of hidden nondeterminism were removed, because an experiment whose result
depends on ambient global RNG state is not reproducible:

- `make_policy(..., seed=)` seeds initialisation explicitly. Without it, weights depended
  on whatever last touched the global torch RNG.
- `PPOTrainer` owns an `np.random.Generator` for minibatch shuffling instead of using
  `np.random` globally.

`test_training_does_not_depend_on_ambient_global_rng` locks both in. The full suite passes
in both randomised and fixed test order.

## Tests

- `tests/test_policies.py` (16) — shapes, action masking, gradient flow to every parameter,
  `evaluate`/`act` log-prob agreement, deterministic argmax, config-driven sizing,
  zero-length observation at episode start.
- `tests/test_ppo.py` (16) — GAE terminal vs truncation vs boundary reset against
  hand-computed values, buffer, device resolution, config validation, MPS smoke test,
  parameters actually move, reproducibility, ambient-RNG independence.
- `tests/test_m5_integration.py` (5, marked `slow`) — Red learns; Blue learns to
  discriminate rather than saturate; trained Blue beats both scripted references on the
  same episodes; trained policies plug in as opponents; **a frozen opponent's weights do
  not move during training** (which self-play correctness depends on).
