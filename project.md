# ARENA — Project Requirements

**An Adversarial Co-Evolving Benchmark for Agentic Tool-Use Security**
Game-Theoretic Multi-Agent RL for MCP Systems · CSI4006 Game Theory · VIT Vellore

Status: **DRAFT — awaiting sign-off (Gate 1)**
Source: `ARENA_Project_Proposal.docx` (July 2026)

---

## 1. One-liner

A self-play RL arena where an attacker (Red) and defender (Blue) co-evolve inside a
simulated MCP tool-calling environment, producing a defender battle-tested against an
*adapting* adversary rather than a frozen attack list.

## 2. The gap being filled

Every existing multi-agent security benchmark scores a defender against a **static** attack
set. TAMAS (2025) shows current frameworks fail 81–82% of the time against attacks that
*don't even adapt*. No benchmark measures what happens when the attacker is also learning.
ARENA measures exactly that, and reports **exploitability** (residual best-response attack
success against a frozen defender) as the primary metric.

## 3. Formal game model

Two-player, general-sum, sequential Markov game Γ = (S, A_red, A_blue, P, R_red, R_blue, γ):

| Element | Definition |
|---|---|
| `s ∈ S` | tool-call history + session context + task state |
| `A_red` | next tool invocation / crafted payload, given task + history |
| `A_blue` | allow / flag / quarantine, conditioned on **full call sequence** |
| `R_red` | w1·evasion + w2·task-plausibility (sequence still looks legitimate) |
| `R_blue` | detection reward − false-positive penalty on benign sessions |

Objective: approximate equilibrium of Γ via self-play, not a one-shot classifier fit.

## 4. Confirmed scope decisions (user sign-off, this session)

| Decision | Choice | Consequence |
|---|---|---|
| Blue feature backbone | **Build fresh** — no CSI3022 Sentinel dependency | ARENA is self-contained; sequence feature extractor written here |
| Simulation fidelity | **Hybrid** — scripted/templated tool calls in the RL training loop; LLM-driven payloads only in a held-out eval sweep | Training is fast and CPU/MPS-feasible; LLM sweep demonstrates transfer to real payloads |
| Scope | **Full pipeline, small scale first** — all 9 stages ship and work end-to-end; small nets, short runs | Scale-up is a config change later, not a rewrite |
| Compute | **Local Mac only** (Apple M3, torch 2.10, MPS available) | Every default config must complete on M3 in minutes, not days |
| Eval LLM | **qwen2.5:3b via local Ollama** (confirmed installed; `qwen2.5:1.5b`, `tinyllama` also present) | Held-out payload sweep is free, local, offline, reproducible. No API keys, no cost, no external data egress |

## 5. Attack taxonomy

Six TAMAS families, each extended into **sequential/chained** MCP variants:

1. Impersonation
2. Direct Prompt Injection
3. Indirect Prompt Injection
4. Contradicting Agents
5. Byzantine Agent
6. Colluding Agents

Core chained pattern to reproduce: a single malicious effect (e.g. credential exfiltration)
split across several individually-benign tool calls — the case static per-call permission
systems provably cannot catch.

Canonical worked example (from proposal §8):
`read .env` → `summarize "for changelog"` → `POST to external webhook`.
Every call is in-policy alone; the trajectory is exfiltration.

## 6. Components to build

| Component | Role |
|---|---|
| Scenario Generator | Samples task + tool registry (benign + adversarial) from the 6-family taxonomy |
| Red Agent | RL attacker policy π_red; picks tool-call sequences maximizing evasion + impact |
| Blue Agent | RL defender policy π_blue over sequence features; allow / flag / quarantine |
| MCP Environment Simulator | PettingZoo / Gymnasium wrapper: task layer, tool registry, session call history |
| Reward Engine | Asymmetric rewards for Red (evasion + impact) and Blue (detection − FP penalty) |
| Self-Play League Orchestrator | Alternating PPO; checkpoint pool of past policies to sample opponents from |
| Evaluation Harness | Exploitability curves, AUROC, TPR@5%FPR vs static baselines |
| Benchmark Release | Packaged scenarios, leaderboard, trained checkpoints |

## 7. Training methodology

- **PPO**, alternating self-play: freeze Blue → train Red to find exploits; freeze Red →
  train Blue to patch them; repeat.
- **League / checkpoint pool** (AlphaStar-style) so each side is periodically evaluated
  against *past* opponents — the standard guard against cyclic, non-transitive strategies.
- Opponent sampling from the pool, not just the current opponent.

## 8. Evaluation

**Primary:** exploitability — best-response attacker success rate against a *frozen*
defender, tracked across self-play generations. Success = monotone decline.

**Secondary:** AUROC, TPR@5%FPR — deliberately matching CASPIAN (2026) so numbers are
directly comparable to published work.

**Baselines (all three must be implemented):**
1. Static allow-list
2. TAMAS-style single-shot per-call detector
3. CASPIAN-style causal/cascade monitor

**Reference anchor:** TAMAS reports ~80–82% static-baseline attack success. ARENA's
exploitability curve starts near there and must be shown to fall.

## 9. Deliverables

- [ ] Open-source MCP self-play environment (PettingZoo / Gymnasium API)
- [ ] Trained Red and Blue checkpoints across multiple self-play generations
- [ ] Exploitability leaderboard: ARENA-trained defenders vs the three static baselines
- [ ] LLM-payload transfer eval sweep (held out from training)
- [ ] Written report suitable for cs.CR / cs.GT
- [ ] Patent claim structure around the sequence-aware adaptive permission mechanism

## 10. Novelty claims to defend in the report

1. First co-evolving self-play formulation for **MCP / tool-calling** security specifically
   (MAGIC 2026 does self-play for general jailbreak alignment, not tool-use infrastructure).
2. First to report **exploitability** — a game-theoretic equilibrium proxy — for agentic
   tool security, instead of static single-pass accuracy.
3. Extends TAMAS's static six-family taxonomy into a **dynamic** threat model where attack
   strategy adapts to the deployed defender.
4. Operationalizes the "dynamic, sequence-aware permissioning for tool-chaining attacks"
   open problem named in Schroeder de Witt et al. (2026).
5. Answers Syrnikov et al. (2026): prompt-level constitutional constraints had no measurable
   effect on adversarial LLM behavior; only runtime adaptive enforcement worked. ARENA is
   that enforcement layer.

## 11. Risks & open items

| Risk | Mitigation |
|---|---|
| **Scale.** League self-play at paper scale is normally multi-GPU. | Build small-scale first: every stage correct and runnable on M3 in minutes. All sizes/steps live in config; scaling up is a config edit + a bigger machine. Report honest scale limits rather than overclaiming. |
| Self-play collapse / cyclic strategies | League checkpoint pool + periodic evaluation vs past opponents (§7). Exploitability vs the *pool*, not just current opponent. |
| Reward hacking by Red (degenerate evasion that isn't a real attack) | Task-plausibility term in R_red; separate "did the attack objective actually complete" check in the env, independent of Blue's verdict. |
| Simulated env too easy → results don't transfer | Held-out LLM-payload eval sweep (§4) is the transfer evidence. |
| Baselines implemented as strawmen | Baselines get their own tests and are tuned on a validation split before comparison. |
| LLM eval provider | **RESOLVED** — `qwen2.5:3b` via local Ollama, eval-only, responses cached to disk so sweeps are reproducible and re-runnable offline. |
| 3B model too weak to generate convincing attack payloads | It only has to *render* an attack plan the trained Red policy already chose into natural language — it is not doing the strategy. Prompt templates + rejection sampling on a validity check. If quality is still poor, fall back to templated payloads and report it. |

## 12. Non-goals

- Not a production MCP proxy or a shippable security product.
- No live/real MCP servers touched; no real credentials, no real network exfiltration.
  All tools are simulated; all "sensitive" data is synthetic.
- Not full frontier-model inference per RL step (explicitly ruled out for feasibility).

## 13. Datasets

No external training dataset is required — the environment *generates* its own scenarios.
Any external data pulled in (e.g. TAMAS instances for baseline comparison, tool-name
corpora for realism) must be **public, licensed, and recorded here** with source URL,
license, version/date, and justification before use.

| Dataset | Source URL | License | Version/date | Why it fits |
|---|---|---|---|---|
| **TAMAS** | https://github.com/microsoft/TAMAS | **MIT** (code) / **CDLA-Permissive-2.0** (data) — both permissive, redistributable | v1, arXiv:2511.05269 (Nov 2025); ACL 2026 | The baseline anchor. 300 adversarial instances, 6 attack families, 211 tools, 100 harmless tasks, 5 domains (news/education/finance/healthcare/legal). Gives ARENA (a) the taxonomy grounding, (b) the ~80–82% static-failure reference number, (c) a directly comparable single-shot detector baseline. |
| **Toucan-1.5M** | https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M | **Apache-2.0** | 1.5M trajectories, arXiv:2510.01179 | Realistic **benign** traffic + tool-registry realism. 495 real MCP servers, 2000+ tools, multi-turn/sequential/parallel real tool calls. Blue's false-positive penalty is meaningless without a realistic benign distribution — this supplies it. Subsampled (~5–20k) for local scale. |

**Rejected:** `obaydata/mcp-agent-trajectory-benchmark` (Apache-2.0, but only ~38–49 trajectories —
fully subsumed by Toucan). `MCP-Atlas` — harness is Apache-2.0 but it measures tool-use
*competency*, not security; out of scope.

**Rules:** licenses above were read from the source, not assumed. Nothing is vendored into the
repo — datasets are fetched by a script into an ignored `data/` dir, with a manifest recording
checksum + retrieval date. Attribution goes in the report and in `THIRD_PARTY.md`.

## 14. Key sources

- Kavathekar et al. — TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems, arXiv:2511.05269
- Schroeder de Witt et al. — Open Challenges in Multi-Agent Security, arXiv:2505.02077
- Wen et al. — MAGIC: Co-Evolving Attacker–Defender Adversarial Game for LLM Safety, arXiv:2602.01539
- CASPIAN: Online Detection and Attribution of Cascade Attacks…, arXiv:2605.19240
- Hao et al. — Game-Theoretic Lens on LLM-based Multi-Agent Systems, arXiv:2601.15047
- Zhu, Q. — Game Theory Meets LLM and Agentic AI, arXiv:2507.10621
- IBM — Cost of a Data Breach Report 2025

## 15. Environment

- macOS (Darwin 25.5.0), Apple M3
- Python 3.13.9 (Anaconda), torch 2.10.0, MPS available
- Git: local commits only this session — **no pushes** (user instruction)
