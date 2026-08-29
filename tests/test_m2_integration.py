"""M2 integration: scenario -> scripted attack -> taint graph -> reward.

The unit tests prove the tracker and the reward engine behave in isolation.
These prove the pieces compose: for *every* one of the six attack families, a
minimal winning tool-call sequence can be constructed inside a generated
scenario's own registry, the taint graph recognises it as objective completion,
and the reward engine pays out the way the game model says it should.

If a family's objective were unsatisfiable in its registry, the exploitability
metric would be silently poisoned — Red could never win those episodes no matter
how well it played — so this test is a load-bearing guard, not a nicety.
"""

from __future__ import annotations

import pytest

from arena.config import RewardConfig, load_config
from arena.rewards import EpisodeOutcome, compute_rewards, heuristic_plausibility
from arena.scenarios import ALL_FAMILIES, ObjectiveKind, Scenario, ScenarioGenerator
from arena.taint import TaintTracker

W = RewardConfig()


def _scripted_attack(sc: Scenario) -> list[tuple[str, list[int] | None]]:
    """Build a minimal winning (tool_name, input_refs) chain for ``sc``.

    Shape: source -> propagator (repeated to pad) -> sink, spanning exactly
    ``max(min_chain_len, 3)`` distinct calls. Explicit refs thread the flow so it
    does not depend on the naive most-recent default.
    """
    obj = sc.objective
    assert obj is not None
    reg = sc.registry

    source_name = obj.source_tools[0]
    prop_name = reg.propagators[0].name
    if obj.kind is ObjectiveKind.INJECTED_CONTROL:
        # sink need not be in obj.sink_tools for this kind; any sink is a
        # privileged call the untrusted content can steer.
        sink_name = reg.sinks[0].name
    else:
        sink_name = obj.sink_tools[0]

    n_props = max(1, obj.min_chain_len - 2)
    calls: list[tuple[str, list[int] | None]] = [(source_name, None)]
    ref = 0
    for _ in range(n_props):
        calls.append((prop_name, [ref]))
        ref += 1
    calls.append((sink_name, [ref]))
    return calls


def _run(sc: Scenario) -> tuple[TaintTracker, list[tuple[str, list[int] | None]]]:
    reg = sc.registry
    tr = TaintTracker()
    calls = _scripted_attack(sc)
    for i, (name, refs) in enumerate(calls):
        tr.step(reg[reg.index(name)], i, refs)
    return tr, calls


@pytest.mark.parametrize("family", list(ALL_FAMILIES))
def test_every_family_is_actually_completable_in_its_own_registry(family):
    gen = ScenarioGenerator(seed=0)
    for _ in range(25):  # several registries per family, not just one lucky draw
        sc = gen.sample(force_family=family)
        tr, calls = _run(sc)
        assert tr.objective_completed(sc.objective), (
            f"{family.value}: scripted attack did not complete\n"
            f"  registry={sc.registry.names}\n  objective={sc.objective}\n  calls={calls}"
        )
        path = tr.attack_path(sc.objective)
        assert len(path) >= sc.objective.min_chain_len


@pytest.mark.parametrize("family", list(ALL_FAMILIES))
def test_reward_pays_out_for_a_completed_attack(family):
    gen = ScenarioGenerator(seed=1)
    sc = gen.sample(force_family=family)
    tr, calls = _run(sc)
    tools = [sc.registry[sc.registry.index(n)] for n, _ in calls]

    outcome = EpisodeOutcome(
        is_adversarial=True,
        objective_completed=True,
        completion_step=len(calls) - 1,
        n_steps=len(calls),
        plausibility=heuristic_plausibility(tools),
        malicious_steps=tr.attack_path(sc.objective),
    )
    rb = compute_rewards(outcome, W)
    assert rb.red_evade == W.w_evade
    assert rb.blue_miss == W.w_miss
    assert rb.r_red > rb.r_blue


@pytest.mark.parametrize("family", list(ALL_FAMILIES))
def test_blue_quarantine_at_the_sink_denies_the_reward(family):
    gen = ScenarioGenerator(seed=2)
    sc = gen.sample(force_family=family)
    tr, calls = _run(sc)
    sink_step = len(calls) - 1

    # Blue quarantines the final call: the sink never executes, nothing completes.
    outcome = EpisodeOutcome(
        is_adversarial=True,
        objective_completed=False,
        n_steps=len(calls),
        plausibility=0.5,
        quarantined=True,
        quarantine_step=sink_step,
        flagged_steps=(sink_step,),
        malicious_steps=tr.attack_path(sc.objective),
    )
    rb = compute_rewards(outcome, W)
    assert rb.blue_detect == W.w_detect
    assert rb.red_evade == 0.0
    assert rb.r_blue > rb.r_red


def test_a_benign_rollout_completes_nothing_and_is_cheap_for_blue_to_pass():
    gen = ScenarioGenerator(seed=3, adversarial_ratio=0.0)
    sc = gen.sample()
    assert sc.objective is None
    reg = sc.registry
    # benign work: list -> read a normal file -> summarize -> write locally
    benign = ["list_directory", "read_file", "summarize_text", "write_file"]
    tr = TaintTracker()
    for i, name in enumerate(n for n in benign if n in reg.names):
        tr.step(reg[reg.index(name)], i)
    assert not any(a.tainted for a in tr.artifacts)

    outcome = EpisodeOutcome(is_adversarial=False, objective_completed=False,
                             n_steps=len(tr.artifacts), plausibility=1.0)
    assert compute_rewards(outcome, W).r_blue == 0.0


def test_shipped_config_reward_weights_load():
    for name in ("small.yaml", "paper.yaml"):
        cfg = load_config(name)
        assert cfg.reward == RewardConfig()  # neither file overrides; defaults apply


def test_swapping_the_source_for_a_benign_read_breaks_completion():
    """Sanity: the attack completes *because* it moves sensitive data, not just
    because it has the right shape."""
    gen = ScenarioGenerator(seed=4)
    sc = gen.sample(force_family=ALL_FAMILIES[1])  # direct prompt injection, exfil
    reg = sc.registry
    if "read_file" not in reg.names:
        pytest.skip("registry lacks a benign read to substitute")
    calls = _scripted_attack(sc)
    calls[0] = ("read_file", None)  # neutralise the source
    tr = TaintTracker()
    for i, (name, refs) in enumerate(calls):
        tr.step(reg[reg.index(name)], i, refs)
    assert not tr.objective_completed(sc.objective)
