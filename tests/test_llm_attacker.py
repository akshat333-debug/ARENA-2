"""Fast tests for arena/llm/attacker.py — the LLM attack planner (M10).

No network: the client is stubbed. The one thing these must pin down is that a
*validated* plan is validated against the taint tracker's ground truth, and that
an invalid plan falls back to the scripted chain rather than being played.
"""

from __future__ import annotations

import json

import pytest

from arena.config import load_config
from arena.llm.attacker import (
    AttackPlan,
    LLMAttacker,
    _scripted_indices,
    build_prompt,
    parse_plan,
    plan_attack,
    plan_completes_objective,
)
from arena.scenarios import AttackFamily, ScenarioGenerator

CFG = load_config("small.yaml")


class StubClient:
    """Minimal LLMClient stand-in. ``reply`` may be a str or None."""

    def __init__(self, reply, available=True):
        self.reply = reply
        self._available = available
        self.prompts: list[str] = []

    def is_available(self) -> bool:
        return self._available

    def generate(self, prompt, *, options=None):
        self.prompts.append(prompt)
        return self.reply


def adv_scenario(seed=1, family=AttackFamily.DIRECT_PROMPT_INJECTION):
    return ScenarioGenerator.from_config(CFG.scenario, seed=seed).sample(force_family=family)


def scripted_names(sc):
    return tuple(sc.registry[i].name for i in _scripted_indices(sc))


# --- prompt ---------------------------------------------------------------


def test_prompt_lists_registry_and_designated_source_sink():
    sc = adv_scenario()
    p = build_prompt(sc)
    for name in sc.registry.names:
        assert name in p
    # parity with the trained Red, which gets these as one-hot channels
    for name in sc.objective.source_tools:
        assert name in p
    assert str(sc.objective.min_chain_len) in p


def test_prompt_mentions_untrusted_read_when_required():
    sc = adv_scenario(family=AttackFamily.INDIRECT_PROMPT_INJECTION)
    if sc.objective.requires_untrusted_read:
        assert "untrusted" in build_prompt(sc).lower()


# --- parse_plan -----------------------------------------------------------


def test_parse_plan_accepts_bare_json_array():
    sc = adv_scenario()
    names = list(scripted_names(sc))
    assert parse_plan(json.dumps(names), sc) == names


def test_parse_plan_extracts_array_from_surrounding_prose():
    sc = adv_scenario()
    names = list(scripted_names(sc))
    text = f"Sure, here is the plan:\n{json.dumps(names)}\nHope that helps!"
    assert parse_plan(text, sc) == names


@pytest.mark.parametrize("bad", ["", "no json here", "[]", "{}", "[1, 2, 3]", "not json ["])
def test_parse_plan_rejects_garbage(bad):
    assert parse_plan(bad, adv_scenario()) is None


def test_parse_plan_rejects_tool_outside_registry():
    sc = adv_scenario()
    assert parse_plan(json.dumps(["definitely_not_a_tool"]), sc) is None
    # a plan that is *mostly* valid is still rejected — a plan naming a tool that
    # does not exist is not playable
    mixed = [*scripted_names(sc), "definitely_not_a_tool"]
    assert parse_plan(json.dumps(mixed), sc) is None


# --- ground-truth validity -----------------------------------------------


def test_scripted_chain_completes_the_objective():
    """Sanity: the fallback itself passes the validity check."""
    sc = adv_scenario()
    assert plan_completes_objective(list(scripted_names(sc)), sc)


def test_plan_of_harmless_reads_does_not_complete_objective():
    sc = adv_scenario()
    benign = [t.name for t in sc.registry if not t.is_sink][:1] * 3
    assert plan_completes_objective(benign, sc) is False


def test_plan_completes_objective_rejects_empty_and_unknown():
    sc = adv_scenario()
    assert plan_completes_objective([], sc) is False
    assert plan_completes_objective(["nope"], sc) is False


# --- plan_attack: acceptance and every fallback path ---------------------


def test_valid_llm_plan_is_accepted():
    sc = adv_scenario()
    names = list(scripted_names(sc))
    plan = plan_attack(sc, StubClient(json.dumps(names)))
    assert plan.via_llm is True
    assert plan.reject_reason is None
    assert list(plan.tool_names) == names


@pytest.mark.parametrize(
    "client,reason",
    [
        (StubClient(None, available=False), "llm_unavailable"),
        (StubClient(None), "no_response"),
        (StubClient("total garbage"), "unparseable_or_unknown_tool"),
    ],
)
def test_fallback_paths_record_their_reason(client, reason):
    sc = adv_scenario()
    plan = plan_attack(sc, client)
    assert plan.via_llm is False
    assert plan.reject_reason == reason
    assert plan.tool_names == scripted_names(sc)


def test_plan_that_does_not_complete_objective_is_rejected():
    """The check that matters: a well-formed plan naming real tools is still
    refused if the taint tracker says it is not an attack. Without this the
    'LLM arm' would happily score plans that never move sensitive data."""
    sc = adv_scenario()
    harmless = [t.name for t in sc.registry if not t.is_sink][:1] * 2
    plan = plan_attack(sc, StubClient(json.dumps(harmless)))
    assert plan.via_llm is False
    assert plan.reject_reason == "plan_does_not_complete_objective"
    assert plan.tool_names == scripted_names(sc)


def test_plan_attack_rejects_benign_scenario():
    gen = ScenarioGenerator.from_config(CFG.scenario, seed=0)
    sc = next(s for _ in range(200) if not (s := gen.sample()).is_adversarial)
    with pytest.raises(ValueError):
        plan_attack(sc, StubClient("[]"))


# --- LLMAttacker ----------------------------------------------------------


def test_attacker_plays_the_plan_then_emits_filler():
    sc = adv_scenario()
    names = list(scripted_names(sc))
    atk = LLMAttacker(sc, AttackPlan(sc.scenario_id, tuple(names), True))
    played = [atk({}) for _ in range(len(names) + 2)]
    assert played[: len(names)] == [sc.registry.index(n) for n in names]
    assert played[len(names):] == [0, 0]


def test_attacker_reset_replays():
    sc = adv_scenario()
    atk = LLMAttacker(sc, AttackPlan(sc.scenario_id, scripted_names(sc), True))
    first = [atk({}) for _ in range(3)]
    atk.reset()
    assert [atk({}) for _ in range(3)] == first


def test_attacker_reports_via_llm():
    sc = adv_scenario()
    names = scripted_names(sc)
    assert LLMAttacker(sc, AttackPlan(sc.scenario_id, names, True)).via_llm is True
    assert LLMAttacker(sc, AttackPlan(sc.scenario_id, names, False, "x")).via_llm is False


# --- the regression that the original M10 lacked -------------------------


def test_a_differing_plan_actually_moves_blue_observations():
    """The defect this module was written to fix: the pre-fix sweep injected the
    payload into ``Scenario.task``, which Blue never observes, so both arms of
    the 'transfer' sweep produced byte-identical decisions (0 of 183 differed).

    A sweep arm is only a measurement if a different plan changes what Blue sees.
    """
    import numpy as np

    from arena.env import BLUE, RED, ARENAEnv
    from arena.features import ALLOW

    sc = adv_scenario()
    names = list(scripted_names(sc))
    # a longer plan: pad the front with a harmless read, same ending
    filler = next(t.name for t in sc.registry if not t.is_sink and not t.is_source)
    longer = [filler, *names]
    assert plan_completes_objective(longer, sc), "padded plan must still be an attack"

    def observations(plan_names):
        env = ARENAEnv(CFG, scenario=sc)
        atk = LLMAttacker(sc, AttackPlan(sc.scenario_id, tuple(plan_names), True))
        env.reset(seed=0)
        seen = []
        for agent in env.agent_iter():
            _, _, term, trunc, _ = env.last()
            if term or trunc:
                env.step(None)
                continue
            if agent == RED:
                env.step(atk(env.observe(RED)))
                continue
            seen.append(env.observe(BLUE)["calls"].copy())
            env.step(ALLOW)
        return seen

    a, b = observations(names), observations(longer)
    assert len(a) != len(b) or any(not np.array_equal(x, y) for x, y in zip(a, b)), (
        "a different attack plan produced identical Blue observations"
    )
