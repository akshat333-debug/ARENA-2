# M3 — The environment

Module: [`arena/env.py`](../arena/env.py). New deps: `gymnasium`, `pettingzoo`.

## Shape

`ARENAEnv` is a PettingZoo **AEC** env with two agents, `red_0` and `blue_0`.
One "MCP step" is two AEC turns:

1. `red_0` returns a tool index → the env holds it as the *pending proposal*.
2. `blue_0` observes the whole call sequence including that proposal and returns
   `ALLOW (0)` / `FLAG (1)` / `QUARANTINE (2)`.
   - `ALLOW`/`FLAG` → the call executes via `TaintTracker.step(..., input_refs=None)`;
     `FLAG` also records the step.
   - `QUARANTINE` → episode ends now; the pending call never runs.

Episode ends on quarantine, on `objective_completed` flipping true, or at
`scenario.max_steps` (a truncation). At the end the env builds an
`EpisodeOutcome` from the taint graph + Blue's verdicts and calls
`compute_rewards`; rewards are assigned once, on the terminal transition.

`env.last_outcome` / `env.last_breakdown` expose the result after the episode —
`env.rewards` is pruned once PettingZoo drains the dead agents.

## Actions

| Agent | Space | Notes |
|---|---|---|
| `red_0` | `Discrete(n_tools_max)` | Fixed size; a given registry may be smaller. `registry_mask` in Red's obs marks valid indices `0..len-1`; out-of-range wraps (`% len`). `ponytail:` revisit if a trained Red exploits the wrap. |
| `blue_0` | `Discrete(3)` | allow / flag / quarantine. |

Red's action is **tool index only** — no data-flow reference. The env applies M2's
"consume the most recent artifact" default, which is exactly the canonical
`read → summarize → send` chain. `tests/test_m3_integration.py` confirms all six
families are winnable this way, so M5's Red does not need a structured action
space. Deliberate interleaving attacks (a benign call spliced mid-chain) are a
later action-space extension.

## Observations — and the anti-leakage invariant

`blue_0` observation is a `Dict`:

```
domain : Box(5)                one-hot, observable
calls  : Box(max_steps, 21)    per-call feature rows, zero-padded
length : Box(1) int            filled rows incl. the pending proposal
```

Each call row (`_encode_call`) is **observable metadata only**: side-effect
one-hot (6), category one-hot (8), `sensitivity/2`, `is_untrusted_read`,
`is_privileged`, `position`, `resolved?`, and Blue's own past verdict one-hot
(allow, flag). It never reads `TaintTracker`.

Two tests lock this in (the M3 gate from architecture.md §4):

- **`test_blue_observation_is_invariant_to_taint_state_perturbation`** — corrupt
  `env._tracker` with fake artifacts and a fake `EXFIL` event mid-episode; Blue's
  observation is byte-identical before and after.
- **`test_blue_observation_identical_for_same_calls_regardless_of_objective`** —
  same registry and tool sequence, one episode adversarial and one benign; Blue
  sees exactly the same thing at every step.

`red_0` observation is a richer `Dict` (`task`, `registry`, `registry_mask`,
`objective` as multi-hot over the registry, `history` with Blue's full verdict
appended — Red's feedback channel — and `length`). Red is allowed to see the
objective and Blue's verdicts; Blue is not allowed to see the taint graph.

## Gymnasium single-agent view

`SingleAgentARENA(learner, opponent_policy, ...)` wraps the AEC env as a standard
`gymnasium.Env`, driving the other side with a fixed policy. `learner=RED` is
"train Red vs frozen Blue"; `learner=BLUE` the reverse. Reward for a gym
transition is read from `_cumulative_rewards[learner]`, which accumulates every
sub-step since the learner last acted and is zeroed when it acts again — so the
terminal reward is counted exactly once
(`test_single_agent_wrapper_reward_not_double_counted`).

This wrapper is the primitive the M8 exploitability sweep is built on: freeze
Blue at generation *g*, train a fresh best-response Red, report its success rate.
`test_exploitability_style_sweep_against_a_frozen_blue` demonstrates the loop
end-to-end with scripted policies.

## Tests

- `tests/test_env.py` — PettingZoo `api_test`, seeded reproducibility, the three
  termination paths, reward wiring, `FLAG` semantics, the two leakage tests,
  invalid-verdict rejection, action wrap, and the Gym wrapper (both directions,
  no double-count, post-done guard).
- `tests/test_m3_integration.py` — per family: winnable with tool-index actions,
  reward signs match outcome; a sequence-aware Blue beats the split chain a
  per-call Blue misses (headline claim, demonstrated); benign episodes don't
  punish a calm Blue but do punish a paranoid one; the exploitability sweep;
  full-stack seeded reproducibility.

`tests/__init__.py` was added so `test_m3_integration` can reuse helpers from
`test_env`.
