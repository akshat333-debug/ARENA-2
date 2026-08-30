# M10 — LLM Payload Rendering & Transfer Sweep

## What it does

M10 adds an **eval-only** LLM transfer sweep: attack payloads are rendered by a
local LLM (`qwen2.5:3b` via Ollama) instead of templates, to demonstrate the
trained defender transfers to LLM-authored attacks. Training stays
scripted/templated (fast).

## Files

| File | Role |
|------|------|
| `arena/llm/__init__.py` | Package exports |
| `arena/llm/client.py` | Cached Ollama client with disk cache |
| `arena/llm/payloads.py` | Per-family prompt templates + validation + fallback |
| `arena/llm/sweep.py` | Held-out transfer sweep (templated vs LLM-rendered) |
| `scripts/run_sweep.py` | One-command sweep script |
| `tests/test_llm.py` | 30 fast tests (mocked backend) |
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

A payload only changes the **content string** on a call — it never alters the
tool sequence or the taint graph. Blue's observation for an LLM-payload episode
is byte-identical to the templated one with the same tool sequence. Verified in
`test_payload_injection_does_not_change_taint_graph`.
