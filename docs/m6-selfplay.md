# M6 — Alternating self-play

Module: [`arena/selfplay.py`](../arena/selfplay.py), [`scripts/train_selfplay.py`](../scripts/train_selfplay.py).
Config: `SelfPlayConfig`.

## The loop

One generation:

1. **Freeze Blue, train Red** to best-respond to it — Red goes looking for exploits.
2. **Freeze Red, train Blue** to best-respond to *that* Red — Blue patches them.
3. **Evaluate**: exploitability (best-response attack success vs the current Blue), Blue's
   return / quarantine rate / miss rate vs the current Red.

Repeat. `SelfPlayTrainer` holds one Red and one Blue `ActorCritic` and warm-starts each
generation from the previous weights.

### Frozen means frozen

The opponent is queried through `TorchPolicyAdapter` — `act` is under `torch.no_grad`, and
the opponent's parameters are never given to an optimiser. `test_m6_integration` asserts
Blue's weights do not move while Red trains, and vice versa.

### Blue's opponent is a mixture, not just the frozen Red

When Blue trains, it faces `AdaptiveRed(attacker=frozen_red)`: the frozen learned Red on
adversarial episodes, `BenignRoller` on benign ones. A Blue trained only against attacks
learns to quarantine everything (the M5 failure mode); it needs realistic benign traffic
in the same distribution to keep its false-positive cost real.

## What a run shows (small.yaml, seed 0, 4 generations × 40k steps/side, ~3 min on M3)

```
gen 0: exploitability=0.035  blue_return=+0.325  blue_quarantine=0.545  blue_miss=0.035
gen 1: exploitability=0.015  blue_return=+0.416  blue_quarantine=0.560  blue_miss=0.010
gen 2: exploitability=0.050  blue_return=+0.032  blue_quarantine=0.145  blue_miss=0.045
gen 3: exploitability=0.030  blue_return=-0.024  blue_quarantine=0.010  blue_miss=0.035
```

Generations 0–1 are exactly the intended dynamic: Blue discriminates (quarantine rate
~0.55 ≈ the adversarial rate), exploitability falls, Blue's return climbs.

**Generations 2–3 drift.** Blue's quarantine rate collapses to ~0.01 — it has become
passive. The cause is structural, not a bug: once Red is a *learned* policy rather than the
always-succeeding scripted attacker, and Blue has already learned to stop it, the frozen
Red that the next generation of Blue trains against is weak. Against a weak attacker,
relaxing is cheap, so Blue unlearns quarantining — and then Red, retrained against a
passive Blue, has no pressure to get sharper either. The pair drifts toward the
`(passive Blue, mediocre Red)` corner.

This is the classic non-transitive / cyclic self-play failure the proposal calls out
(architecture.md S7), and the fix is the **league / opponent-checkpoint pool (M7)**: each
side is periodically evaluated and trained against *past* versions of its opponent, not
only the current one, so Blue cannot forget how to handle a strong attacker and Red cannot
coast against a weak defender.

**M6's deliverable is the alternation mechanism**, shown to run, alternate, freeze
correctly, and reproduce. A monotone exploitability curve is M7 + M8, not M6.

## Config

`SelfPlayConfig`: `n_generations` (4), `steps_per_side` (40k — the wall-clock lever;
Blue needs roughly this much per generation to hold discrimination from a warm start),
`eval_episodes` (200), `stochastic_opponent` (True — the learner should face the
opponent's actual sampled distribution, not its greedy action).

## Tests

- `tests/test_selfplay.py` (11) — trainer builds both policies at config sizes;
  `state_dict` round-trips; `AdaptiveRed(attacker=…)` uses the fixed attacker on
  adversarial episodes and `BenignRoller` on benign ones; `AdaptiveScriptedRed` alias
  still works; `SelfPlayConfig` validation.
- `tests/test_m6_integration.py` (7, `slow`) — one generation completes and records
  finite stats; both policies change across a generation; **frozen Blue does not move
  while Red trains** and **frozen Red does not move while Blue trains**; Blue does not
  collapse to blanket quarantine (the M5 failure mode) once the opponent is a learned Red;
  a run is reproducible; a trained pair checkpoints and reloads.
