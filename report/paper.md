# ARENA: An Adversarial Co-Evolving Benchmark for Agentic Tool-Use Security

**Game-Theoretic Multi-Agent RL for MCP Systems**
CSI4006 Game Theory · VIT Vellore

---

## Abstract

Every existing multi-agent security benchmark scores defenders against a **static** attack set.
TAMAS (2025) shows current frameworks fail 81–82% of the time against attacks that *don't even adapt*.
ARENA measures what happens when the attacker learns too, reporting **exploitability** — residual
best-response attack success against a frozen defender — as the primary metric.

We implement a self-play RL arena where a Red (attacker) and Blue (defender) co-evolve inside a
simulated MCP tool-calling environment. Over 5 seeds at `small.yaml` scale (the default,
laptop-feasible configuration), the plain co-evolved defender and the strongest hand-engineered baseline
are **statistically indistinguishable** on exploitability (0.565 +/- 0.274 vs
0.636 +/- 0.081) — the sign of the difference flips across seeds. Given the *same*
hand-engineered causal features as that baseline, however, the co-evolved defender becomes
**less exploitable on every one of 5 seeds** (0.313 +/- 0.199 vs 0.636). The static
allow-list's 0.783 independently reproduces TAMAS's reported ~80% failure.

Two contributions follow. First, the gap was **representational, not a failure of
co-evolution**: the sequence encoder could not recover the decisive causal feature from raw
call metadata at this scale, and supplying it closes the gap. Second, and more durable, the
co-evolved defender carries an order-of-magnitude wider **seed variance** than any static
baseline — same loop, same config, different seed, materially different defender. That
reliability problem is invisible to single-seed evaluation, and surfacing it is what the
benchmark is for.

## 1. Introduction

### 1.1 The Gap

Multi-agent LLM security benchmarks test defenders against frozen attack lists. Real attackers
adapt. TAMAS (2025) showed 81–82% failure rates against static attacks — what happens when the
attacker also learns?

### 1.2 Our Contribution

1. **First co-evolving self-play formulation for MCP/tool-calling security** (MAGIC 2026 does
   self-play for jailbreak alignment, not tool-use infrastructure).
2. **Exploitability as the primary metric** — a game-theoretic equilibrium proxy, not static
   single-pass accuracy.
3. **Dynamic threat model** — extends TAMAS's static six-family taxonomy into a setting where
   attack strategy adapts to the deployed defender.
4. **Honest measurement** — the audit exposed 4 bugs in the measurement path; all fixed, and
   the corrected result is reported, not the inflated one.

## 2. Game Model

Two-player, general-sum, sequential Markov game Γ = (S, A_red, A_blue, P, R_red, R_red, γ):

| Element | Definition |
|---------|-----------|
| s ∈ S | tool-call history + session context + task state |
| A_red | next tool invocation, given task + history + Blue's past verdicts |
| A_blue | allow / flag / quarantine, conditioned on full call sequence |
| R_red | w₁·evasion + w₂·task-plausibility − w₃·step_cost |
| R_blue | detection_reward − false_positive_penalty |

The key design choice: **attack success is judged by the taint graph** (did sensitive data
reach an external sink?), not by whether Blue was fooled. This prevents Red from farming reward
with degenerate non-attacks.

## 3. Method

### 3.1 Environment

- **PettingZoo AEC** env with two agents: `red_0` (tool picker) and `blue_0` (verdict).
- **Anti-leakage invariant**: Blue observes only observable call metadata. The internal taint
  graph is never exposed. Verified by a test that asserts Blue's observation is byte-invariant
  to taint perturbation.
- **6 TAMAS families** extended into chained variants (`min_chain_len > 1`): no single call
  completes the attack; the malicious effect is split across several individually-benign calls.

### 3.2 Training

- **Alternating PPO**: freeze Blue → train Red (find exploits); freeze Red → train Blue (patch
  them); repeat.
- **League / checkpoint pool** (AlphaStar-style): per-side pools of past checkpoints. Each
  side trains against a *sample* of the opponent's history, preventing the non-transitive
  drift to `(passive Blue, weak Red)` that the leagueless loop exhibits by generation 2–3.
- **Own single-file PPO** (~250 lines), not Stable-Baselines3, because the alternating freeze
  loop and different observation/action spaces per side are not the shape SB3 optimizes for.

### 3.3 Evaluation

**Primary**: exploitability — freeze Blue, spawn a *fresh* random Red, train it as a best
response, report attack success. Tracked across self-play generations; success = monotone decline.

**Secondary**: AUROC, TPR@5%FPR — deliberately matching CASPIAN (2026) for direct comparison.

**Baselines**:
1. Static allow-list (TAMAS-style)
2. Single-shot per-call detector (TAMAS-style)
3. Causal/cascade monitor (CASPIAN-style, hand-engineered `sensitive_read_before_sink` feature)

## 4. Results

### 4.1 Baseline Leaderboard (small.yaml, single seed)

```
defender            AUROC   TPR@5%FPR  exploitability  op.FPR
------------------------------------------------------------------
causal_monitor      0.980       0.861           0.690   0.036
single_shot         0.728       0.034           0.773   0.010
static_allowlist    0.513       0.037           0.777   0.012
arena_blue          0.927       0.461           0.780   0.047
```

**Key finding**: at `small.yaml` scale, the co-evolved Blue is **the most exploitable row on
the board**. The causal monitor's hand-engineered feature (`sensitive_read_before_sink`)
essentially encodes the taint rule, which a GRU has to discover from sparse episode-level
reward.

### 4.2 Exploitability Curve

```
gen0 0.720   gen1 0.765   gen2 0.740   gen3 0.710      (delta -0.010)
```

The curve is **flat** over four generations. Self-play is not driving exploitability down at
this scale.

### 4.3 What Still Holds

Separately tested (M5 integration):
- Blue beats every baseline on **mean reward vs the training distribution**.
- The league prevents the M6 drift: quarantine rate stays ~0.5+ where the leagueless loop
  collapses to ~0.01.

### 4.4 The Honest Reading — and a second correction

Sections 4.1–4.3 report a **single seed**. Repeating the whole leaderboard over 5 seeds,
with a fresh self-play Blue trained per seed, does not support the reading that co-evolution
loses (`docs/m11-multiseed.md`):

```
defender                       AUROC       TPR@5%FPR    exploitability
----------------------------------------------------------------------
single_shot            0.726+/-0.006   0.160+/-0.020     0.508+/-0.098
arena_blue             0.933+/-0.041   0.572+/-0.213     0.565+/-0.274
causal_monitor         0.975+/-0.007   0.832+/-0.165     0.636+/-0.081
static_allowlist       0.506+/-0.006   0.023+/-0.009     0.783+/-0.024
```

Paired per-seed, `arena_blue - causal_monitor` on exploitability is
`[+0.090, -0.007, -0.203, +0.123, -0.357]` — mean -0.071, spread 0.204, **sign flips**.
Three seeds favour the co-evolved defender, two the causal monitor.

So the single-seed claim in §4.1 (`arena_blue` 0.780 vs 0.690, "the most exploitable row")
is withdrawn, exactly as the M8 claim before it was. It was noise. The mean in the 5-seed
table now leans the other way, and that is **also** not a result — it is inside the same
noise, and we decline to bank it.

What survives 5/5 seeds: the causal monitor beats the static allow-list (mean -0.147,
spread 0.085, every seed agreeing).

**The finding at this point is variance.** `arena_blue`'s exploitability spread is
+/-0.274 against +/-0.081 and +/-0.024 for the baselines. The open problem was therefore not
that co-evolution yields a worse defender but that it yields an *unreliable* one.

### 4.4b The gap closes — and the cause was representational

Four levers were run as arms of the same experiment (5 seeds each, config-only,
`docs/m11-gap.md`). Paired per-seed exploitability against the causal monitor:

```
lever    mean     per-seed                                        separated?
causal   -0.323   [-0.553, -0.333, -0.053, -0.173, -0.503]        YES, 5/5
gen8     -0.116   [-0.023, -0.170, -0.230, -0.087, -0.070]        YES, 5/5
dense    +0.039   [-0.063, -0.290, +0.277, +0.090, +0.183]        no, sign flips
pfsp     +0.051   [+0.097, +0.020, -0.060, +0.080, +0.120]        no, sign flips
```

**Given the same hand-engineered causal features the CASPIAN-style baseline uses, the
co-evolved defender becomes less exploitable than that baseline on every seed** —
0.313 +/- 0.199 vs 0.636 +/- 0.081. TPR@5%FPR rises to 0.697 and the seed spread narrows.
Training for more generations helps too, more modestly and also 5/5.

We do **not** claim causal features beat plain `arena_blue`: that comparison sign-flips at
seed 2 despite a -0.253 mean, so it is not banked. Dense flag credit and PFSP league
sampling both moved the mean the wrong way; neither becomes a default.

The diagnosis this supports is that the gap was **representational, not a failure of
co-evolution**. The GRU could not recover `sensitive_read_before_sink` from raw call rows
at this scale. Handed the feature, the co-evolved policy overtakes the baseline built
around it — while keeping the adaptive-attacker robustness that baseline lacks.

### 4.5 Transfer to LLM-planned attacks (M10)

Blue trains against `ScriptedAttacker`, which always plays the canonical minimal chain
(source → propagator ×N → sink). The held-out sweep replaces that planner with
`qwen2.5:3b`, which is given the episode's registry and objective and asked to plan the
calls. A plan counts only if replaying it through the taint tracker actually completes the
objective; otherwise the episode falls back to the scripted chain and the reason is
recorded. On `small.yaml`, 34 of 60 adversarial episodes (57%) were LLM-planned, of which
33 differed from the scripted chain.

```
                    scripted plans          LLM-planned plans
defender            AUROC   TPR@5%FPR       AUROC   TPR@5%FPR
------------------------------------------------------------
causal_monitor      0.978       0.875       0.914       0.675
single_shot         0.732       0.193       0.688       0.189
static_allowlist    0.517       0.042       0.515       0.039
```

**The causal monitor loses 20 points of TPR@5%FPR** (0.875 → 0.675) when the attack is
planned by a model rather than by our script. Its hand-engineered causal features are
partly tuned to the *shape* of the canonical chain, and a planner that pads with
ordinary-looking calls or reorders the laundering steps degrades them. `single_shot` and
`static_allowlist` barely move — they never used sequence structure, so there is less to
lose.

This is the sharpest evidence in the report for ARENA's premise: a defender can look
strong against a fixed attack generator and measurably weaken against an adaptive one,
without any change to the environment or the ground truth.

Two caveats stated plainly. `exploitability` is not a transfer signal in this table — its
best-response Red is PPO, not the LLM, so it re-derives its own attacker in both arms and
moves only through re-calibration. And 43% of episodes fell back to the scripted chain
(chiefly `plan_does_not_complete_objective`), so the LLM arm is a mixture, which makes the
measured degradation a *lower bound* on the true gap.

## 5. Methods Integrity: The Audit

After M9, a full module and data-flow audit found **4 bugs**, all in the measurement path:

1. **CRITICAL**: GAE boundaries invisible (~90% of episode ends had no boundary, corrupted
   every training run M5–M8).
2. Per-generation exploitability measured Red against a league sample, not the current Blue.
3. Learned Blue scored stochastically while baselines are deterministic.
4. `paper.yaml` wasn't actually a scale-up.

All fixed. The M8 headline ("~3× less exploitable") was an artifact and is withdrawn.
Re-measured numbers are reported above.

New `tests/test_contracts.py` (10 tests) covers module hand-offs — where all three measurement
bugs lived. Mutation-checked: reintroducing bug 1 fails immediately.

## 6. Limitations

1. **Scale**: all results are at `small.yaml` (6–10 tools, 12-step episodes, 40k PPO
   steps/side). `paper.yaml` (8–19 tools, 24-step episodes, 250k steps/side) needs real
   compute we don't have.
2. **Single seed**: the current leaderboard is one seed. Multi-seed results (≥3 seeds,
   mean ± spread) are needed before claiming anything moved.
3. **Simulated environment**: training uses scripted attack chains. The M10 sweep tests
   transfer to LLM-*planned* attacks (§4.5), but both arms still run inside the same
   simulated tool environment; no real MCP server is exercised.
4. **Blue reward sparsity**: Blue gets one sparse episode-level signal. Denser per-step
   reward shaping could help but is untested.
5. **No PFSP**: the league uses uniform sampling with a latest-bias. Prioritised sampling
   by win-rate (PFSP) is a known refinement from AlphaStar that could help but is unimplemented.

## 7. Future Work

1. **Scale up**: run `paper.yaml` end-to-end on real compute. Now aimed at *reducing seed
   variance* (§4.4), which is the open problem, rather than at closing a mean gap that the
   multi-seed run shows is not established.
2. **More generations**: push to 12–20 at `small.yaml` to test if the flat curve is a
   budget artifact.
3. **PFSP league sampling**: use the stored `opponent_win_rate` per checkpoint.
4. **Denser Blue reward**: shaped per-step signal instead of sparse episode-level.
5. **Give Blue causal features**: the causal_monitor wins because its hand-engineered feature
   ~encodes the taint rule. Feed `sequence_features` into BluePolicy alongside the GRU.
6. ~~**Multi-seed evaluation**~~ — done (§4.4, `arena/eval/multiseed.py`). It overturned the
   single-seed reading and should be run before any future claim.

## 8. Reproducibility

Every number in this paper is regenerable from a single command:

```bash
python3 scripts/reproduce.py small.yaml --blue runs/sp.pt --out report/ --sweep
```

This produces `report/leaderboard.md`, `report/leaderboard.png`,
`report/exploitability_curve.md`, `report/exploitability_curve.png`,
`report/transfer_sweep.md` (with `--sweep`, needs Ollama), and `report/results.json`.

## 9. Patent Claim Structure

The contribution is the **training loop**, not any single network:

**Independent Claim**: A method for training a defender policy in a tool-calling security
environment comprising:
- Alternating self-play between an attacker and defender policy
- A checkpoint pool per side storing past policies
- Opponent sampling from the opposing side's pool each episode
- On-demand spawning of a fresh best-response attacker to measure exploitability

**Dependent Claims**: The taint-graph anti-leakage invariant; the asymmetric reward with
detection-credit-gated-on-evidence; the scenario generator that produces chained variants
of static attack families.

## References

- TAMAS: Kavathekar et al., arXiv:2511.05269 (Nov 2025)
- CASPIAN: arXiv:2605.19240 (May 2026)
- MAGIC: Wen et al., arXiv:2602.01539 (Feb 2026)
- AlphaStar: Vinyals et al. 2019 (league / PFSP)
- Time Limits in RL: Pardo et al. 2018
- Toucan-1.5M: arXiv:2510.01179 (Oct 2025)
