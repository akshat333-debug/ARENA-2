# ARENA Console

The product surface for ARENA: a landing page that explains why the benchmark
exists, and an operator console that runs it.

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # -> dist/, fully static
```

## The simulation is real

`src/lib/` is a faithful TypeScript mirror of the Python research code, not a
mock layer:

| Console module | Mirrors |
|---|---|
| `lib/domain.ts` | `arena/tools.py`, `arena/scenarios.py` — the six side-effect classes, six TAMAS families, all chained |
| `lib/taint.ts` | `arena/taint.py` — tag minting, propagation, flow events, `wouldAdvanceAttack` |
| `lib/rewards.ts` | `arena/rewards.py` — the asymmetric reward, evidence-gated detection credit |
| `lib/engine.ts` | `arena/env.py` — one episode as two-turn Red/Blue adjudication |
| `lib/policies.ts` | `arena/scripted.py`, `arena/baselines/` — the three baselines + the co-evolved policy |
| `lib/pipeline.ts` | the seven-stage execution graph, one node per Python module |

So clicking through the Arena genuinely runs the taint tracker and the reward
engine. The numbers quoted in the Analysis view are the repository's own 5-seed
results (`report/multiseed_baseline.json`, `report/lev_*.json`).

**The anti-leakage invariant holds here too.** `Episode.blueObservation()`
returns observable call metadata only — no task text, no objective, no taint
graph — and every defender's signature takes exactly that. Ground truth is
sealed in the Arena view until the episode ends, so you cannot play the defender
with the answers.

## Demo Mode is the default, on purpose

Everything above runs in the browser. No network, no API key, no backend, no
model call. `npm run build` produces a `dist/` that opens from the filesystem
(`base: "./"`), which is the answer to "the jury is waiting and the wifi is down".

`Run Demo` loads one of four preconfigured scenarios and drives the real
pipeline through it — the same components, the same state, not a slideshow.
`Reset` returns everything to a clean presentation state in one click.

**Live Mode** is optional and genuine. It probes a Python bridge at
`127.0.0.1:8000`; if that is up, episodes come from the actual implementation:

```bash
pip install fastapi uvicorn
python3 scripts/serve_api.py
```

If the probe fails the console falls back to Demo Mode, shows a dismissible
notice, and keeps working. There is no error screen in that path by design.

## Architecture

- **One store** (`store/useStore.ts`). Every view is a projection of it, so an
  action in the Arena moves the Simulation graph, the Evidence table, the Logs
  and the Metrics together. `rev` is an explicit revision counter: `Episode` is a
  mutable class, so without it Zustand hands subscribers the same reference and
  the UI silently stops updating while the logic runs.
- **One run driver** (`components/layout/RunDriver.tsx`), mounted in the Shell,
  not per page — so playback keeps advancing while the presenter navigates.
- **Hand-rolled SVG charts** (`components/viz/`). No chart library: full control,
  no runtime dependency, and error bars that are actually legible — the spread is
  the point of most of these figures.

## Routes

`/` landing · `/app` overview · `/app/arena` two-player game ·
`/app/simulation` live pipeline · `/app/workflow` architecture ·
`/app/analysis` metrics · `/app/evidence` call ledger · `/app/results` case
report · `/app/logs` execution monitor · `/app/settings` config
