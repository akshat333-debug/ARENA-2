"""Held-out transfer sweep — eval with LLM-rendered payloads.

Runs ``evaluate_defenders`` on a scenario set whose payloads come from the LLM
instead of templates.  Reports the same leaderboard columns, side by side with
the templated numbers.

If ``client.available()`` is ``False`` the sweep runs with templates and
clearly labels the output "LLM unavailable, templates used".
"""

from __future__ import annotations

from dataclasses import dataclass

from arena.baselines.collect import Decision, _benign_profile, collect_decisions
from arena.config import ArenaConfig, LLMConfig
from arena.env import BLUE, RED, ARENAEnv
from arena.eval.harness import LeaderboardRow, evaluate_defenders, format_leaderboard
from arena.features import ALLOW
from arena.llm.attacker import LLMAttacker, plan_attack
from arena.llm.client import LLMClient
from arena.scenarios import ScenarioGenerator
from arena.scripted import BenignRoller


@dataclass(frozen=True)
class PlanStats:
    """How many adversarial episodes the LLM actually planned.

    Reported, never hidden: if this is 0 the "LLM" arm is the scripted arm and
    the comparison is vacuous. Saying so is the difference between a measurement
    and a formality.
    """

    n_adversarial: int = 0
    n_llm_planned: int = 0
    n_differing_from_scripted: int = 0
    reasons: dict[str, int] | None = None

    @property
    def llm_rate(self) -> float:
        return self.n_llm_planned / self.n_adversarial if self.n_adversarial else 0.0

    def summary(self) -> str:
        r = ", ".join(f"{k}={v}" for k, v in sorted((self.reasons or {}).items()))
        return (
            f"LLM planned {self.n_llm_planned}/{self.n_adversarial} attacks "
            f"({self.llm_rate:.0%}); {self.n_differing_from_scripted} differed from the "
            f"scripted chain." + (f"  fallbacks: {r}" if r else "")
        )


@dataclass(frozen=True)
class SweepResult:
    label: str
    llm_available: bool
    templated_rows: list[LeaderboardRow]
    llm_rows: list[LeaderboardRow] | None = None
    plan_stats: PlanStats | None = None

    def format(self) -> str:
        lines = [
            f"=== Transfer sweep: {self.label} ===",
            f"LLM available: {self.llm_available}",
        ]
        if self.plan_stats is not None:
            lines.append(self.plan_stats.summary())
        lines.append("")
        if self.llm_available and self.llm_rows:
            lines.append("--- Scripted attack plans (what Blue trained against) ---")
            lines.append(format_leaderboard(self.templated_rows))
            lines.append("")
            lines.append("--- LLM-planned attacks (held out) ---")
            lines.append(format_leaderboard(self.llm_rows))
            lines.append("")
            if self.plan_stats is not None and self.plan_stats.n_differing_from_scripted == 0:
                lines.append(
                    "WARNING: no LLM plan differed from the scripted chain, so the two "
                    "tables are the same measurement. Treat this as no result."
                )
            else:
                lines.append(
                    "Transfer = the change in AUROC / TPR@5%FPR between the two tables. "
                    "A defender that only works on the left has memorised our scripted "
                    "chain."
                )
                lines.append(
                    "Note: `exploitability` is NOT a transfer signal here. Its "
                    "best-response Red is PPO, not the LLM, so it re-derives its own "
                    "attacker in both arms; it shifts only because each defender is "
                    "re-calibrated on the arm's own decision set."
                )
        else:
            label = "scripted (LLM unavailable)" if not self.llm_available else "scripted"
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

    Each defender is scored twice on the same scenario stream:

    1. against :class:`~arena.scripted.ScriptedAttacker`'s canonical chains —
       what Blue was trained on;
    2. against :class:`~arena.llm.attacker.LLMAttacker`'s plans — held out.

    The difference is transfer. Both arms share the seed, so the scenarios,
    registries and benign traffic are identical and the *only* thing that moves
    is the attack plan.
    """
    cfg = config or ArenaConfig()
    llm = LLMClient(llm_cfg or cfg.llm)
    llm_ok = llm.is_available()

    na = n_decision_adv if n_decision_adv is not None else cfg.eval.n_decision_adv
    nb = n_decision_benign if n_decision_benign is not None else cfg.eval.n_decision_benign

    # --- Scripted arm (always) ---
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

    # --- LLM-planned arm ---
    llm_decisions, stats = _collect_llm_decisions(na, nb, config=cfg, llm=llm, seed=seed + 7)
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
        plan_stats=stats,
    )


def _collect_llm_decisions(
    n_adversarial: int,
    n_benign: int,
    *,
    config: ArenaConfig,
    llm: LLMClient,
    seed: int = 0,
) -> tuple[list[Decision], PlanStats]:
    """Roll episodes whose *attack plans* come from the LLM.

    Mirrors :func:`arena.baselines.collect.collect_decisions` exactly — same
    generator seed, same benign source, same allow-everything Blue — except that
    adversarial episodes are driven by :class:`LLMAttacker` instead of
    :class:`~arena.scripted.ScriptedAttacker`. That single substitution is the
    held-out variable.

    Ground truth and anti-leakage are untouched: labels still come from
    ``TaintTracker.attack_path``, and Blue's observation is built by the env from
    observable metadata as always.
    """
    gen = ScenarioGenerator.from_config(config.scenario, seed=seed)
    profile = _benign_profile(config)
    rows: list[Decision] = []
    made = 0
    want_adv, want_ben = n_adversarial, n_benign
    n_adv = n_llm = n_diff = 0
    reasons: dict[str, int] = {}

    while want_adv > 0 or want_ben > 0:
        sc = gen.sample()
        if sc.is_adversarial and want_adv > 0:
            want_adv -= 1
        elif (not sc.is_adversarial) and want_ben > 0:
            want_ben -= 1
        else:
            continue

        if sc.is_adversarial:
            from arena.llm.attacker import _scripted_indices

            plan = plan_attack(sc, llm)
            red = LLMAttacker(sc, plan)
            n_adv += 1
            n_llm += plan.via_llm
            if plan.reject_reason:
                reasons[plan.reject_reason] = reasons.get(plan.reject_reason, 0) + 1
            scripted = tuple(sc.registry[i].name for i in _scripted_indices(sc))
            n_diff += plan.tool_names != scripted
        else:
            red = BenignRoller(sc, seed=seed + made, profile=profile)

        env = ARENAEnv(config, scenario=sc)
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

    stats = PlanStats(
        n_adversarial=n_adv,
        n_llm_planned=n_llm,
        n_differing_from_scripted=n_diff,
        reasons=reasons,
    )
    return rows, stats
