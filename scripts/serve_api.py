"""Optional Live-Mode bridge for the console (frontend/).

    python3 scripts/serve_api.py           # needs: pip install fastapi uvicorn

The console does NOT need this. Its simulation engine is a faithful TypeScript
mirror of arena/ and runs entirely in the browser — that is Demo Mode, and it is
the default precisely so a presentation never depends on a process being up.

This server exists so Live Mode can be genuine when it is available: episodes
are then generated and adjudicated by the actual Python implementation, not the
mirror. If it is not running the console probes it once, falls back, and says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:  # pragma: no cover
    raise SystemExit(
        "Live Mode bridge needs FastAPI:  pip install fastapi uvicorn\n"
        "(The console works fully without it — Demo Mode is the default.)"
    )

from arena.config import load_config  # noqa: E402
from arena.env import ARENAEnv, BLUE, RED  # noqa: E402
from arena.baselines import CausalMonitor, SingleShotDetector, StaticAllowList  # noqa: E402
from arena.baselines.collect import collect_decisions, split  # noqa: E402
from arena.scenarios import ScenarioGenerator  # noqa: E402
from arena.scripted import BenignRoller, ScriptedAttacker  # noqa: E402

app = FastAPI(title="ARENA bridge", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

CFG = load_config("small.yaml")
_gen = ScenarioGenerator.from_config(CFG.scenario, seed=0)
_defenders: dict = {}


def _fit():
    if _defenders:
        return _defenders
    tr, _ = split(collect_decisions(120, 120, config=CFG, seed=1), seed=0)
    _defenders.update({
        "static_allowlist": StaticAllowList(),
        "single_shot": SingleShotDetector().fit(tr),
        "causal_monitor": CausalMonitor().fit(tr),
    })
    return _defenders


@app.get("/health")
def health():
    return {"ok": True, "config": CFG.name, "impl": "python"}


@app.get("/episode")
def episode(defender: str = "causal_monitor", adversarial: bool | None = None):
    """Roll one full episode through the REAL engine and return its trace."""
    blue = _fit().get(defender) or _fit()["causal_monitor"]
    for _ in range(200):
        sc = _gen.sample()
        if adversarial is None or sc.is_adversarial == adversarial:
            break
    env = ARENAEnv(CFG, scenario=sc)
    red = ScriptedAttacker(sc) if sc.is_adversarial else BenignRoller(sc, seed=0)
    env.reset(seed=0)
    calls = []
    for agent in env.agent_iter():
        _, _, term, trunc, _ = env.last()
        if term or trunc:
            env.step(None)
            continue
        if agent == RED:
            env.step(red(env.observe(RED)))
            continue
        pend = env._pending()
        v = int(blue(env.observe(BLUE)))
        calls.append({"step": pend[1], "tool": pend[0].name,
                      "sideEffect": pend[0].side_effect.value, "verdict": v})
        env.step(v)

    o, b = env.last_outcome, env.last_breakdown
    return {
        "id": sc.scenario_id, "domain": sc.domain.value, "task": sc.task,
        "adversarial": sc.is_adversarial,
        "family": sc.objective.family.value if sc.objective else None,
        "calls": calls,
        "outcome": {
            "objectiveCompleted": o.objective_completed, "quarantined": o.quarantined,
            "caughtInFlight": o.caught_in_flight, "maliciousSteps": list(o.malicious_steps),
            "nSteps": o.n_steps,
        },
        "rewards": {"red": b.r_red, "blue": b.r_blue},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
