"""
agents.dispatcher — the branch decision (blend 04).

`choose()` is a pure function, so the whole decision table is a parametrised
test rather than a claim in a docstring. The agent's control flow (which
branch actually runs, and that the no-data branches cost zero calls) runs on a
real ADK InvocationContext with both destinations replaced by recorders.
"""

from __future__ import annotations

import json

import pytest
from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

from gub_agent.agents import dispatcher as dp
from gub_agent.agents.answers import (
    intent_options,
    no_access_payload,
    smalltalk_payload,
)
from gub_agent.agents.format_gate import gate_problems
from gub_agent.agents.formatter import ANSWER_STATE_KEY
from gub_agent.schemas import RouterDecision
from tests.helpers import invocation_ctx

INV = "inv-dispatch"


def _decision(**over) -> RouterDecision:
    base = {"intent": "campaign_status", "confidence": 0.9, "language": "en"}
    base.update(over)
    return RouterDecision.model_validate(base)


# ── the decision table, pure ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        # The two no-data intents win over everything, at any confidence.
        (_decision(intent="workspace_personal", confidence=0.95), dp.ABSTAIN),
        (_decision(intent="workspace_personal", confidence=0.20), dp.ABSTAIN),
        (_decision(intent="smalltalk", confidence=0.97), dp.SMALLTALK),
        (_decision(intent="smalltalk", confidence=0.55), dp.SMALLTALK),
        # FACT intents with an entity, above the floor → the fast path.
        (_decision(intent="campaign_status", entity_surface="Silverado"), dp.FAST),
        (_decision(intent="campaign_facts", entity_surface="Silverado"), dp.FAST),
        (_decision(intent="account_facts", entity_surface="chevy"), dp.FAST),
        (_decision(intent="staff_lookup", entity_surface="Alex"), dp.FAST),
        # A count carries its subject in `slots`, not as an entity.
        (_decision(intent="count_or_rank"), dp.FAST),
        # A FACT intent with nothing to look up is the executor's problem.
        (_decision(intent="campaign_status"), dp.DEEP),
        (_decision(intent="account_facts"), dp.DEEP),
        # The intents that need judgement never take the fast path.
        (_decision(intent="assessment", confidence=0.95, entity_surface="Chevy"), dp.DEEP),
        (_decision(intent="exploratory", confidence=0.95), dp.DEEP),
        (_decision(intent="market_enrichment", confidence=0.95), dp.DEEP),
        # Below the floor: ask, rather than look up the wrong thing…
        (
            _decision(intent="campaign_facts", confidence=0.62, entity_surface="Silverado"),
            dp.CLARIFY,
        ),
        (_decision(intent="assessment", confidence=0.50), dp.CLARIFY),
        # …and `exploratory` is not clarifiable: it IS the catch-all.
        (_decision(intent="exploratory", confidence=0.30), dp.DEEP),
        # …unless the bot already resolved the entity this turn (no second card).
        (_decision(intent="campaign_status", confidence=0.40, entity_id="c1"), dp.DEEP),
        # Exactly at the floor is above it.
        (_decision(intent="campaign_status", confidence=0.70, entity_surface="X"), dp.FAST),
        (_decision(intent="campaign_status", confidence=0.699, entity_surface="X"), dp.CLARIFY),
    ],
)
def test_the_decision_table(decision: RouterDecision, expected: str):
    assert dp.choose(decision) == expected


def test_the_router_fallback_decision_lands_on_the_deep_path():
    """A missing or schema-invalid router output arrives as `exploratory` at
    confidence 0 — the deep path, NOT a clarification (blend 04's edge table:
    a bad router costs latency, never the turn)."""
    from gub_agent.schemas.router import FALLBACK_DECISION

    assert dp.choose(FALLBACK_DECISION) == dp.DEEP


# ── the deterministic payloads ground themselves ──────────────────────────────


@pytest.mark.parametrize("language", ["ru", "en"])
def test_the_templates_pass_the_gates_checks_against_an_empty_index(language):
    """These texts ship without any retrieval, so every number and every
    capitalised word in them would be "ungrounded" — they are written to
    contain none. Pinned here because a later edit to the wording is exactly
    how that would break."""
    assert gate_problems(smalltalk_payload(language), {}) == []
    assert gate_problems(no_access_payload(language, "Silverado 2026 Q3"), {}) == []


def test_a_clarification_offers_the_confusable_intents():
    payload = dp.clarify_intent_payload("campaign_facts", "ru", "Silverado")
    assert payload.kind == "clarify"
    assert len(payload.candidates) == len(intent_options("campaign_facts")) == 3
    assert all(c.entity_type == "intent" for c in payload.candidates)
    # The bot's renderer shows the bullets, not the candidates — both carry the
    # same options (`chat/render-answer.ts:renderAnswerText`).
    assert payload.blocks[0].items == [c.name for c in payload.candidates]
    assert gate_problems(payload, {}) == []


# ── the agent's control flow ──────────────────────────────────────────────────


class Recorder(BaseAgent):
    """A destination that records it ran and emits one payload event."""

    runs: list = []

    async def _run_async_impl(self, ctx):
        self.runs.append(ctx.invocation_id)
        payload = {"kind": "answer", "headline": self.name, "citations": [], "facts": []}
        yield Event(
            invocation_id=ctx.invocation_id,
            author="format_gate",
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=json.dumps(payload))]
            ),
            actions=EventActions(state_delta={ANSWER_STATE_KEY: payload}),
        )


@pytest.fixture
def branches(monkeypatch):
    """The dispatcher with both destinations recorded. `fast_path` keeps its
    name — `_deep_agent()` finds the deep agent by elimination — and
    `outcome()` is patched alongside it, since a recorder does not write the
    fast path's real outcome store."""
    fast = Recorder(name="fast_path", runs=[])
    deep = Recorder(name="gub_pipeline", runs=[])
    monkeypatch.setattr(dp, "fast_path", fast)
    state = {"outcome": "answered"}
    monkeypatch.setattr(dp, "outcome", lambda _inv: state["outcome"])
    agent = dp.Dispatcher(name="dispatcher", sub_agents=[fast, deep])
    return agent, fast, deep, state


async def _ctx(decision: RouterDecision | None):
    state = {} if decision is None else {"router_decision": decision.model_dump()}
    return await invocation_ctx(state=state, invocation_id=INV, user_text="какой-то вопрос")


async def test_workspace_personal_abstains_with_no_model_and_no_tool_call(branches):
    agent, fast, deep, _ = branches
    ctx = await _ctx(_decision(intent="workspace_personal", confidence=0.95))

    events = [e async for e in agent.run_async(ctx)]

    assert (fast.runs, deep.runs) == ([], [])  # zero tool calls, zero model calls
    assert len(events) == 1
    payload = events[0].actions.state_delta[ANSWER_STATE_KEY]
    assert (payload["kind"], payload["headline"]) == ("abstain", "NO_COMPANY_RECORDS")
    # The author the bot's answer channel accepts — `dispatcher` is ignored.
    assert events[0].author == "format_gate"


async def test_smalltalk_answers_from_the_template(branches):
    agent, fast, deep, _ = branches
    ctx = await _ctx(_decision(intent="smalltalk", confidence=0.97, language="ru"))

    events = [e async for e in agent.run_async(ctx)]

    assert (fast.runs, deep.runs) == ([], [])
    assert "Привет" in events[0].actions.state_delta[ANSWER_STATE_KEY]["headline"]


async def test_a_low_confidence_turn_asks_which_question_was_meant(branches):
    agent, fast, deep, _ = branches
    ctx = await _ctx(
        _decision(intent="campaign_facts", confidence=0.55, entity_surface="Silverado")
    )

    events = [e async for e in agent.run_async(ctx)]

    assert (fast.runs, deep.runs) == ([], [])
    assert events[0].actions.state_delta[ANSWER_STATE_KEY]["kind"] == "clarify"


async def test_the_fast_path_answering_ends_the_turn(branches):
    agent, fast, deep, _ = branches
    ctx = await _ctx(_decision(intent="campaign_status", entity_surface="Silverado"))

    events = [e async for e in agent.run_async(ctx)]

    assert fast.runs == [INV]
    assert deep.runs == []  # no executor, no critic — that is the whole point
    assert len(events) == 1


async def test_the_fast_path_declining_falls_through_to_the_deep_path(branches):
    agent, fast, deep, state = branches
    state["outcome"] = "deep"
    ctx = await _ctx(_decision(intent="campaign_status", entity_surface="Silverado"))

    events = [e async for e in agent.run_async(ctx)]

    assert fast.runs == [INV]
    assert deep.runs == [INV]  # one fallback, same turn
    assert [e.actions.state_delta[ANSWER_STATE_KEY]["headline"] for e in events] == [
        "fast_path",
        "gub_pipeline",
    ]


async def test_a_schema_invalid_router_output_goes_deep(branches):
    """`intent: "vibes"` is not in the literal set — the decision does not
    validate, so the dispatcher treats it as exploratory at confidence 0."""
    agent, fast, deep, _ = branches
    ctx = await invocation_ctx(
        state={"router_decision": {"intent": "vibes", "confidence": 0.99}},
        invocation_id=INV,
        user_text="статус Silverado",
    )

    [e async for e in agent.run_async(ctx)]

    assert (fast.runs, deep.runs) == ([], [INV])


async def test_no_router_decision_at_all_goes_deep(branches):
    agent, fast, deep, _ = branches
    ctx = await _ctx(None)

    [e async for e in agent.run_async(ctx)]

    assert (fast.runs, deep.runs) == ([], [INV])
