# M10 — LLM-Planned Attacks & Transfer Sweep

## What it does

M10 adds an **eval-only** transfer sweep. Blue is trained against
`ScriptedAttacker`, which always plays the same canonical minimal chain
(source → propagator ×N → sink). The sweep asks the held-out question:

> Does a defender trained on our canonical chains still catch an attack whose
> call sequence was planned by a real LLM?

A local `qwen2.5:3b` (via Ollama) is handed the episode's tool registry and
objective and asked to **plan the calls**. Training stays scripted and fast.

## What the LLM must change, and why

Blue observes tool-call metadata only — `{domain, calls, length}`, built in
`arena/features.py`. It never sees task text or payload strings.

So the held-out variable has to be the **attack plan**, not the payload prose.
The first cut of M10 appended an LLM payload string to `Scenario.task` and
compared two leaderboards; because nothing observes `task`, the two arms were
byte-identical (**0 of 183 decisions differed**) and the "transfer" table was
the same measurement printed twice. `arena/llm/attacker.py` is the fix.

`arena/llm/payloads.py` still renders payload *text*, but only for realism and
report illustration — it is explicitly not what the sweep measures.

## Validity is ground truth, not keywords

A proposed plan is accepted only if replaying it through `arena.taint.TaintTracker`
**actually completes the objective**. An LLM plan that does not is not an attack,
and that episode falls back to the scripted chain with the reason recorded.

This replaced a keyword validator whose rules were conjunctions lifted from the
fallback template wording (`"read"` AND `".env"`). Real model output —
*"read the sensitive file … send its content to the external server"* — scored
zero matches. Measured at the audit: **1 of 6 families** survived; the other five
silently became templates while reporting themselves as LLM-rendered. The
payload-text validator is now an intent vocabulary plus a refusal check (6/6).

## Files

| File | Role |
|------|------|
| `arena/llm/__init__.py` | Package exports |
| `arena/llm/client.py` | Cached Ollama client with disk cache |
| `arena/llm/payloads.py` | Payload **text** for realism/reports (not measured) |
| `arena/llm/attacker.py` | **LLM attack planner** — the held-out variable |
| `arena/llm/sweep.py` | Transfer sweep (scripted plans vs LLM plans) |
| `scripts/run_sweep.py` | One-command sweep script |
| `tests/test_llm.py` | Fast tests, client/payloads (mocked backend) |
| `tests/test_llm_attacker.py` | Fast tests, planner + the byte-identical regression |
| `tests/test_m10_integration.py` | Slow tests (real Ollama, skipped if unavailable) |

## Config

New `LLMConfig` section in `arena/config.py`:

```yaml
llm:
  model: qwen2.5:3b
  temperature: 0.7
  timeout: 60
  max_retries: 2
  cache_dir: .cache/llm
```

## How the cache works

- Keyed by `(model, prompt, options_hash)` → SHA-256 prefix
- Stored in `.cache/llm/<key>.json` (git-ignored)
- Cache hit returns without calling Ollama → sweep re-runnable offline
- Corrupt/empty cache entries treated as misses

## How validation works

Each attack family has keyword groups. At least one group must have all keywords
present (case-insensitive) for the LLM output to be accepted. On failure, the
M1 template fallback is used.

## Degradation

If Ollama is absent or the model is not pulled:
- `client.is_available()` returns `False`
- `render_payload()` returns the template fallback
- `run_sweep()` runs with templates only, labels output accordingly

## Running

```bash
# Full sweep (templated + LLM-rendered if available)
python scripts/run_sweep.py small.yaml

# With an ARENA-trained Blue
python scripts/run_sweep.py small.yaml --blue runs/sp.pt

# Tests
python -m pytest tests/test_llm.py -q           # fast (~13s)
python -m pytest tests/test_m10_integration.py -q  # slow, needs Ollama
```

## Anti-leakage invariant

Intact, and note it is *not* the same claim as before. The LLM chooses which
tools Red calls; it adds **no new channel to any observation**. Blue's view is
built by the env from observable metadata exactly as in M3, and the taint tracker
still adjudicates success from data flow — so there is no route for the
"is this episode adversarial" label to leak.

The earlier version of this file claimed, as a success criterion, that Blue's
observation is *byte-identical* between the two arms. That was true, and it was
precisely the bug: an identical observation means nothing was measured. What must
hold is the narrower property — no payload-derived feature reaches Blue — while
the *plan* is free to move, and does (99 of 230 decisions differ).

Fallbacks are counted and reported (`PlanStats`); if no plan differs, the sweep
prints a warning that the run is not a result.
