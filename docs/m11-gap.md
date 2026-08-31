# M11 — Closing the gap: which levers actually worked

Baseline: [`docs/m11-multiseed.md`](m11-multiseed.md) · Harness:
[`scripts/run_multiseed.py`](../scripts/run_multiseed.py),
[`scripts/compare_levers.py`](../scripts/compare_levers.py) ·
Raw data: `report/lev_*.json`

Four levers, each an arm of the same experiment (config-only, 5 seeds, fresh
self-play Blue trained per seed). Judged on the **paired per-seed** difference,
not the means — the multi-seed run established that a mean shift smaller than
the spread is not a result.

## `arena_blue` across the arms

```
lever          exploitability            auroc            TPR@5%FPR
baseline       0.565 +/- 0.275     0.933 +/- 0.041     0.572 +/- 0.212
dense          0.675 +/- 0.154     0.948 +/- 0.035     0.625 +/- 0.250
causal         0.313 +/- 0.199     0.943 +/- 0.049     0.697 +/- 0.173
pfsp           0.687 +/- 0.126     0.933 +/- 0.050     0.591 +/- 0.240
gen8           0.520 +/- 0.137     0.905 +/- 0.028     0.434 +/- 0.122
```

## The result: causal features close the gap

Paired per-seed exploitability, **arm − causal_monitor** (negative = the
co-evolved defender is less exploitable):

```
causal   mean -0.323   [-0.553, -0.333, -0.053, -0.173, -0.503]   CONSISTENT 5/5
gen8     mean -0.116   [-0.023, -0.170, -0.230, -0.087, -0.070]   CONSISTENT 5/5
dense    mean +0.039   [-0.063, -0.290, +0.277, +0.090, +0.183]   sign flips
pfsp     mean +0.051   [+0.097, +0.020, -0.060, +0.080, +0.120]   sign flips
```

**Claim (separated, 5/5 seeds): a co-evolved defender given the same
hand-engineered causal features as the CASPIAN-style baseline is less
exploitable than that baseline** — 0.313 vs 0.636, every seed agreeing. Its
TPR@5%FPR also rises to 0.697 (from 0.572) and its seed spread narrows
(0.275 → 0.199).

**Claim (separated, 5/5 seeds): more generations also helps**, more modestly
(mean −0.116 vs the causal monitor).

**Not claimed:** that causal features beat *plain* `arena_blue`. That comparison
sign-flips (seed 2 is +0.15), so despite a −0.253 mean it is not separated and
is not banked.

**Did not work:** dense flag credit and PFSP league sampling both moved the mean
the *wrong* way and sign-flip. Both narrowed the spread, which is interesting on
its own, but neither is a win and neither becomes a default.

## What this changes

The M1–M9 audit's replacement claim — "at small.yaml scale co-evolution does not
beat the static baselines" — was withdrawn once by the multi-seed run (it was
one seed) and is now superseded properly: **with causal features, co-evolution
does beat the strongest static baseline, consistently.**

The honest reading of the whole arc:

1. The original M8 headline was an artifact of a PPO bug.
2. Its replacement was an artifact of a single seed.
3. The real gap was **representational**, not a failure of co-evolution: the GRU
   could not recover `sensitive_read_before_sink` from raw call rows at this
   scale. Handed the feature, the co-evolved policy overtakes the baseline that
   was built around it — and keeps the adaptive-attacker robustness the baseline
   lacks.

Seed variance remains real (±0.199 is still the widest in the table). Reducing
it further is the open problem; `paper.yaml` scale is the next lever.

## Reproduce

```bash
python3 scripts/run_multiseed.py small.yaml --seeds 0 1 2 3 4 --out report/multiseed_baseline.json
python3 scripts/run_multiseed.py small.yaml --seeds 0 1 2 3 4 --causal-features --out report/lev_causal.json
python3 scripts/compare_levers.py report/multiseed_baseline.json report/lev_*.json
```

~19 min per arm on an M3.
