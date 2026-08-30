"""Held-out transfer sweep — eval with LLM-rendered payloads.

Runs ``evaluate_defenders`` on a scenario set whose payloads come from the LLM
instead of templates.  Reports the same leaderboard columns, side by side with
the templated numbers.

If ``client.available()`` is ``False`` the sweep runs with templates and
clearly labels the output "LLM unavailable, templates used".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arena.baselines.collect import Decision, collect_decisions
from arena.config import ArenaConfig, LLMConfig
from arena.eval.harness import LeaderboardRow, evaluate_defenders, format_leaderboard
from arena.llm.client import LLMClient
from arena.llm.payloads import render_payload
from arena.scenarios import ScenarioGenerator
from arena.features import ALLOW


@dataclass(frozen=True)
class SweepResult:
    label: str
    llm_available: bool
    templated_rows: list[LeaderboardRow]
    llm_rows: list[LeaderboardRow] | None = None

    def format(self) -> str:
        lines = [
            f"=== Sweep: {self.label} ===",
            f"LLM available: {self.llm_available}",
            "",
        ]
        if self.llm_available and self.llm_rows:
            lines.append("--- Templated payloads ---")
            lines.append(format_leaderboard(self.templated_rows))
            lines.append("")
            lines.append("--- LLM-rendered payloads ---")
            lines.append(format_leaderboard(self.llm_rows))
            lines.append("")
            lines.append("Compare the two columns to measure transfer.")
        else:
            label = "templates (LLM unavailable)" if not self.llm_available else "templates"
            lines.append(f"--- {label} ---")
            lines.append(format_leaderboard(self.templated_rows))
        return "\n".join(lines)


def run_sweep(
    defenders: dict[str, object],
    config: ArenaConfig | None = None,
    *,
    llm_cfg: LLMConfig | None = None,
    n_decision_adv: int | None = None,
    n_decision_benign: int | None = None,
    br_steps: int | None = None,
    n_eval: int | None = None,
    seed: int = 0,
) -> SweepResult:
    """Run the held-out transfer sweep.

    For each defender, runs ``evaluate_defenders`` twice:
    1. With templated payloads (the standard path)
    2. With LLM-rendered payloads (if Ollama is available)

    Both runs use the same ``decisions`` list to keep the comparison fair.
    """
    cfg = config or ArenaConfig()
    llm = LLMClient(llm_cfg or cfg.llm)
    llm_ok = llm.is_available()

    # --- Templated run (always) ---
    na = n_decision_adv if n_decision_adv is not None else cfg.eval.n_decision_adv
    nb = n_decision_benign if n_decision_benign is not None else cfg.eval.n_decision_benign
    templated_decisions = collect_decisions(na, nb, config=cfg, seed=seed + 7)
    templated_rows = evaluate_defenders(
        defenders, cfg,
        decisions=templated_decisions,
        br_steps=br_steps, n_eval=n_eval, seed=seed,
    )

    if not llm_ok:
        return SweepResult(
            label=cfg.name,
            llm_available=False,
            templated_rows=templated_rows,
        )

    # --- LLM-rendered run ---
    llm_decisions = _collect_llm_decisions(na, nb, config=cfg, llm=llm, seed=seed + 7)
    llm_rows = evaluate_defenders(
        defenders, cfg,
        decisions=llm_decisions,
        br_steps=br_steps, n_eval=n_eval, seed=seed,
    )

    return SweepResult(
        label=cfg.name,
        llm_available=True,
        templated_rows=templated_rows,
        llm_rows=llm_rows,
    )


def _collect_llm_decisions(
    n_adversarial: int,
    n_benign: int,
    *,
    config: ArenaConfig,
    llm: LLMClient,
    seed: int = 0,
) -> list[Decision]:
    """Roll episodes with LLM-rendered payloads on adversarial scenarios.

    For each adversarial scenario, renders a family-specific payload via the
    LLM.  The payload is injected into the scenario's task text (as a suffix),
    which is what the ScriptedAttacker's tool calls reference.  The taint graph
    and tool sequence are unchanged — only the content string differs.

    Benign episodes are rolled normally (no payload injection).
    """
    gen = ScenarioGenerator.from_config(config.scenario, seed=seed)
    rows: list[Decision] = []
    made = 0
    want_adv, want_ben = n_adversarial, n_benign

    while want_adv > 0 or want_ben > 0:
        sc = gen.sample()
        if sc.is_adversarial and want_adv > 0:
            want_adv -= 1
            if sc.objective is not None:
                rp = render_payload(sc.objective.family, llm, objective_kind=sc.objective.kind)
                # Inject payload into the task — the ScriptedAttacker reads
                # tools by name from the registry, so this doesn't break the
                # chain; it only changes what Blue observes in the task text.
                import arena.scenarios as _sc
                sc = sc.model_copy(
                    update={"task": sc.task + "\n\n[Payload]: " + rp.payload}
                )
        elif (not sc.is_adversarial) and want_ben > 0:
            want_ben -= 1
        else:
            continue

        from arena.env import ARENAEnv, BLUE, RED
        from arena.scripted import ScriptedAttacker, BenignRoller

        env = ARENAEnv(config, scenario=sc)
        red = (
            ScriptedAttacker(sc)
            if sc.is_adversarial
            else BenignRoller(sc, seed=seed + made)
        )

        # Roll the episode, capturing Blue's decisions
        env.reset(seed=seed + made)
        captures: list[dict] = []
        for agent in env.agent_iter():
            _, _, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                continue
            if agent == RED:
                env.step(red(env.observe(RED)))
                continue
            obs = env.observe(BLUE)
            pend = env._pending()
            executed = tuple(rc.tool for rc in env._history)
            captures.append({
                "blue_obs": obs,
                "pending_tool": pend[0],
                "tools_incl_pending": executed + (pend[0],),
                "step_index": pend[1],
            })
            env.step(ALLOW)  # let the full chain play out

        from arena.tools import ToolSpec

        malicious = set(env._tracker.attack_path(sc.objective)) if sc.objective else set()
        completed = env.last_outcome.objective_completed if env.last_outcome else False
        for c in captures:
            rows.append(Decision(
                blue_obs=c["blue_obs"],
                pending_tool=c["pending_tool"],
                tools_incl_pending=c["tools_incl_pending"],
                step_index=c["step_index"],
                on_attack_path=c["step_index"] in malicious,
                adversarial=sc.is_adversarial,
                episode_completed=completed,
            ))
        made += 1

    return rows
