# M7 — Opponent-checkpoint league

Module: [`arena/league.py`](../arena/league.py), wired into
[`arena/selfplay.py`](../arena/selfplay.py). Config: `SelfPlayConfig.use_league` (+
`league_pool_max`, `league_p_latest`).

## The problem it fixes

M6, no league — `small.yaml`, seed 0, 4 generations × 40k steps/side:

```
gen 0: exploitability=0.035  blue_quarantine=0.545   <- discriminating
gen 1: exploitability=0.015  blue_quarantine=0.560   <- discriminating
gen 2: exploitability=0.050  blue_quarantine=0.145   <- drifting
gen 3: exploitability=0.030  blue_quarantine=0.010   <- collapsed to passive
```

Once Red is a *learned* policy (not the always-succeeding scripted attacker) and Blue has
beaten it, the frozen Red the next Blue trains against is weak. Against a weak attacker,
relaxing is cheap — so Blue unlearns quarantining, and Red, retrained against a passive
Blue, has no pressure to sharpen either. Non-transitive drift toward `(passive Blue,
mediocre Red)`, exactly as the proposal predicts (architecture.md §7).

## The league

**`League`** — per-side pools of past checkpoints. Pure bookkeeping: holds detached CPU
`state_dict`s, bounded by `pool_max` (oldest evicted, latest always kept), round-trips
through its own `state_dict`.

**Sampling** — `sample(side)` returns the latest checkpoint with probability `p_latest`,
otherwise draws uniformly over the whole pool (which includes the latest, so it is never
excluded). `sample_probs(side)` gives the exact induced distribution; tests check the
empirical draws match it.

**`FrozenPolicySampler`** — a plain `obs -> int` opponent that draws a checkpoint from a
pool and **resamples it whenever the env's episode index changes**, so one training phase
faces a *distribution* of past opponents rather than one fixed policy. Materialised nets
are cached by `(side, generation)`. `fallback` covers the empty pool at generation 0.
Frozen: `TorchPolicyAdapter.act` is under `no_grad`, the sampled parameters never reach an
optimiser — `test_m7_integration` re-checks weights don't move.

## Wired into self-play

`SelfPlayTrainer` builds a `League` when `use_league` is on (default). Each generation:

1. Train Red against `FrozenPolicySampler("blue", fallback=self.blue)` — the current Blue
   while the pool is empty, then a sample of every past Blue.
2. Train Blue against `AdaptiveRed(attacker=FrozenPolicySampler("red", fallback=self.red))`
   — league Reds on attack episodes, `BenignRoller` on benign ones.
3. Evaluate, then snapshot both policies into their pools.

`train()` can be called repeatedly; generation numbering continues and the pools keep
growing. `state_dict()` includes the league, so a run checkpoints and resumes whole.

## Result

`small.yaml`, seed 0, 4 generations × 40k steps/side, `use_league` off vs on:

| generation | 0 | 1 | 2 | 3 |
|---|--:|--:|--:|--:|
| **off** — Blue quarantine rate | 0.545 | 0.560 | 0.145 | **0.010** |
| **off** — Blue return | +0.325 | +0.416 | +0.032 | **−0.024** |
| **on** — Blue quarantine rate | 0.545 | 0.540 | 0.565 | **0.460** |
| **on** — Blue return | +0.325 | +0.464 | +0.428 | **+0.308** |

Without the league, Blue drifts to passive by generation 3 (quarantine rate 0.01, return
negative — worse than doing nothing). With it, Blue stays discriminating across all four
generations (quarantine rate ~0.46–0.57, return solidly positive) because it keeps facing
strong past Reds.

> **Audit correction.** The numbers in the table above were measured before the
> M1–M9 audit, which found a PPO bug that corrupted every training run
> ([audit-m1-m9.md](audit-m1-m9.md)). The **qualitative** claim was re-verified
> after the fix — with the league on, Blue's quarantine rate at generation 3 is
> **0.585** and it stops >92% of the league's attacks, so it is still
> discriminating rather than drifting to passive. The exact per-generation
> figures above, and the league-**off** column in particular, have not been
> re-measured and should be read as indicative.

The doc previously reported a league-on exploitability trace of
`0.035 → 0.400 → 0.205 → 0.140` and explained the generation-1 spike as Red
exploiting a *sampled past Blue*. The audit found that explanation was
diagnosing a bug: with a league, `GenerationStats.exploitability` was measuring
Red against the pool average rather than the current Blue, on a distribution
capped at `adversarial_ratio`. It now measures Red vs the **current, greedy**
Blue on an all-adversarial distribution, the same scale as the M8 metric. That
trace is therefore withdrawn rather than reinterpreted.

The clean metric — a fresh best-response Red vs the frozen latest Blue — is the
M8 harness's job, and after the fix it is **flat** across generations
(`0.720 → 0.765 → 0.740 → 0.710`). The league prevents drift; it does not, at
this scale, reduce exploitability. See [m8-evaluation.md](m8-evaluation.md).

`test_league_keeps_blue_discriminating_where_m6_drifts` asserts the final quarantine rate
stays above 0.20 (it lands ~0.59), does not tip past 0.95 into blanket paranoia, and that
Blue stops most of the league's attacks. Its former `exploitability < 0.30` assertion was
removed — it passed only because the bug crippled the attacker.

## Config

| field | default | meaning |
|---|---|---|
| `use_league` | `True` | off → the plain M6 loop |
| `league_pool_max` | `8` | checkpoints kept per side; `None` = unbounded |
| `league_p_latest` | `0.35` | P(opponent sampler returns the newest checkpoint) |

## Tests

- `tests/test_league.py` (19) — pool grows / evicts oldest / keeps latest; `sample` raises
  on empty, always returns the sole element of a size-1 pool; `sample_probs` sums to 1 and
  matches 6k empirical draws; `p_latest` extremes; construction validation; `state_dict`
  round-trips bit-identically; stored checkpoints are detached CPU copies (mutating the
  live policy does not touch them); `FrozenPolicySampler` uses the fallback while empty,
  raises without one, resamples per episode, stays frozen, caches materialised nets;
  `net_factory` shape.
- `tests/test_m7_integration.py` (`slow`) — league on by default, pools grow one per side
  per generation; frozen opponent still doesn't move with a league in play; generation
  numbering continues across `train()` calls; trainer+league checkpoint round-trips;
  `use_league=False` reproduces the M6 shape; reproducible; **the payoff** — a
  4-generation run keeps Blue's quarantine rate > 0.20 (M6 ended near 0.01) without
  tipping into blanket paranoia.
