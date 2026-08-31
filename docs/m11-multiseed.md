# M11 — Multi-seed evaluation, and what it does to the headline

Module: [`arena/eval/multiseed.py`](../arena/eval/multiseed.py) ·
Script: [`scripts/run_multiseed.py`](../scripts/run_multiseed.py) ·
Raw data: [`report/multiseed_baseline.json`](../report/multiseed_baseline.json)

## Why

Every headline this project has produced came from **one seed**. The M8 claim
("~3x less exploitable") was withdrawn in the M1–M9 audit as a measurement
artifact. Its replacement — *"arena_blue exploitability 0.780 vs the causal
monitor's 0.690, so co-evolution does not beat the static baselines"* — was
also a single seed, and this run shows **it does not survive either.**

## Result (`small.yaml`, 5 seeds, fresh self-play Blue trained per seed)

```
defender                       AUROC       TPR@5%FPR    exploitability        op.FPR
------------------------------------------------------------------------------------
single_shot            0.726+/-0.006   0.160+/-0.020     0.508+/-0.098 0.039+/-0.009
arena_blue             0.933+/-0.041   0.572+/-0.213     0.565+/-0.274 0.031+/-0.039
causal_monitor         0.975+/-0.007   0.832+/-0.165     0.636+/-0.081 0.047+/-0.005
static_allowlist       0.506+/-0.006   0.023+/-0.009     0.783+/-0.024 0.011+/-0.003
```

Paired per-seed differences — the comparison that actually has the statistical
power, since inside one seed every defender sees the same scenarios, the same
decision set and the same best-response budget:

```
exploitability: arena_blue vs causal_monitor   mean -0.071 +/- 0.204   SIGN FLIPS
    per-seed: [+0.090, -0.007, -0.203, +0.123, -0.357]

exploitability: causal_monitor vs static_allowlist  mean -0.147 +/- 0.085  consistent 5/5
    per-seed: [-0.087, -0.043, -0.250, -0.150, -0.207]
```

## What this establishes, and what it does not

**Established (seed-stable, 5/5):** the CASPIAN-style causal monitor beats the
static allow-list on both exploitability and AUROC. The allow-list's 0.783
matches TAMAS's reported ~80% static-baseline failure.

**NOT established:** anything about `arena_blue` vs `causal_monitor`. The sign
of the difference flips across seeds — 3 of 5 favour the co-evolved defender,
2 favour the causal monitor — and the spread (0.204) is three times the mean
difference (0.071). **Both** the earlier claim that co-evolution loses *and*
the mean in the table above, which now happens to favour co-evolution, are
inside the noise. Neither is a result.

## The actual finding: variance, not a gap

`arena_blue`'s exploitability spread is **+/-0.274**, against +/-0.081 for the
causal monitor and +/-0.024 for the allow-list — an order of magnitude more
seed-sensitivity than any static baseline, and the same story on TPR@5%FPR
(+/-0.213 vs +/-0.009).

So the open problem is not "co-evolution produces a worse defender". It is
**"co-evolution produces an unreliable one"**: the same loop, same config, same
budget, different seed, yields anywhere from clearly-better to clearly-worse
than the strongest hand-engineered baseline. That reframes the remaining work —
the levers in `docs/m11-gap.md` are now judged on whether they *reduce spread*,
not only on whether they move the mean.

## Method note

`PairedDelta.separated()` is a unanimous sign test, not a p-value. With 5 seeds
a t-test would be theatre; "every seed agrees on the direction" is a claim a
reader can verify from the per-seed list, which is printed. A mean difference
whose sign flips is reported as `SIGN FLIPS ACROSS SEEDS` and must not be
quoted as a finding.

`scripts/run_multiseed.py` trains a **fresh self-play Blue per seed** through a
defender factory. Reusing one trained net across seeds would hold that arm
fixed and make the reported spread meaningless —
`test_multiseed_calls_factory_per_seed` pins that.

Reproduce:

```bash
python3 scripts/run_multiseed.py small.yaml --seeds 0 1 2 3 4 --out report/multiseed_baseline.json
```

~19 min on an M3 (the per-seed self-play training dominates).
