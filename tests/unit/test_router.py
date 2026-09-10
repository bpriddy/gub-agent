"""
agents.router + the new root wiring (blend 04).

Three things are worth pinning here: the routing contract's shape, how the
decision is READ back (state, then the router's own event text, then the
fallback — a bad router must cost the deep path and nothing else), and the
pipeline structure the bot and Agentspace see at the boundary.
"""

from __future__ import annotations

import json

import pytest
from google.adk.events import Event
from google.genai import types as genai_types
from pydantic import ValidationError

from gub_agent.agents.router import ROUTER_STATE_KEY, decision_from, router_agent, user_text
from gub_agent.config import AGENT_NAME
from gub_agent.schemas.router import FALLBACK_DECISION, FAST_INTENTS, RouterDecision
from tests.helpers import invocation_ctx

INV = "inv-router"


def _decision_dict(**over) -> dict:
    base = {
        "intent": "campaign_status",
        "confidence": 0.93,
        "entity_surface": "Silverado 2026 Q3",
        "slots": {"complete": True},
        "missing_slots": [],
        "language": "ru",
    }
    base.update(over)
    return base


def _router_event(text: str) -> Event:
    return Event(
        invocation_id=INV,
        author="router",
        content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]),
    )


# ── the contract ──────────────────────────────────────────────────────────────


def test_the_decision_validates_and_keeps_the_surface_verbatim():
    decision = RouterDecision.model_validate(_decision_dict())
    assert decision.entity_surface == "Silverado 2026 Q3"
    assert decision.entity_id is None
    assert decision.slots.entity is None and decision.missing_slots == []


def test_confidence_is_bounded_and_the_intent_is_closed():
    with pytest.raises(ValidationError):
        RouterDecision.model_validate(_decision_dict(confidence=1.4))
    with pytest.raises(ValidationError):
        RouterDecision.model_validate(_decision_dict(intent="vibes"))


def test_the_fast_intents_are_the_five_fact_shaped_ones():
    assert FAST_INTENTS == {
        "campaign_status",
        "campaign_facts",
        "account_facts",
        "staff_lookup",
        "count_or_rank",
    }


def test_slots_are_named_fields_the_builder_validates_later():
    """The router fills the named fields; `fast_path.org_query_args` decides
    whether they assemble into a query. A free `dict[str, str]` was the spec's
    shape and is NOT used: gemini-3.5-flash leaves such a map empty (see
    `schemas/router.py`), which killed every count question."""
    decision = RouterDecision.model_validate(
        _decision_dict(
            intent="count_or_rank",
            slots={"entity": "campaigns", "status": "live", "complete": True},
        )
    )
    assert (decision.slots.entity, decision.slots.status) == ("campaigns", "live")
    assert decision.slots.complete is True


def test_slots_complete_defaults_to_false():
    """An unset flag must cost the deep path, never a confidently wrong count."""
    bare = {"intent": "count_or_rank", "confidence": 0.9, "language": "ru"}
    assert RouterDecision.model_validate(bare).slots.complete is False


# ── reading the decision back ─────────────────────────────────────────────────


async def test_the_decision_comes_from_state():
    ctx = await invocation_ctx(state={ROUTER_STATE_KEY: _decision_dict()}, invocation_id=INV)
    assert decision_from(ctx).intent == "campaign_status"


async def test_a_json_string_in_state_is_parsed():
    """`output_key` normally lands a dict; a string is the shape an older ADK
    (and a hand-seeded session) can produce."""
    ctx = await invocation_ctx(
        state={ROUTER_STATE_KEY: json.dumps(_decision_dict())}, invocation_id=INV
    )
    assert decision_from(ctx).intent == "campaign_status"


async def test_the_routers_own_event_is_the_fallback_source():
    """No state commit — the decision is still readable off the event the
    router emitted, fences and all."""
    ctx = await invocation_ctx(
        events=[_router_event("```json\n" + json.dumps(_decision_dict()) + "\n```")],
        invocation_id=INV,
    )
    assert decision_from(ctx).entity_surface == "Silverado 2026 Q3"


async def test_the_newest_router_event_wins():
    ctx = await invocation_ctx(
        events=[
            _router_event(json.dumps(_decision_dict(intent="assessment"))),
            _router_event(json.dumps(_decision_dict(intent="campaign_facts"))),
        ],
        invocation_id=INV,
    )
    assert decision_from(ctx).intent == "campaign_facts"


@pytest.mark.parametrize(
    "raw",
    [
        {"intent": "vibes", "confidence": 0.9},  # not in the literal set
        {"confidence": 0.9},  # no intent at all
        "not json at all",
        {},
    ],
)
async def test_anything_unusable_becomes_the_deep_path_never_an_error(raw):
    ctx = await invocation_ctx(state={ROUTER_STATE_KEY: raw}, invocation_id=INV)
    decision = decision_from(ctx)
    assert (decision.intent, decision.confidence) == ("exploratory", 0.0)
    assert decision == FALLBACK_DECISION


async def test_no_decision_at_all_becomes_the_deep_path():
    ctx = await invocation_ctx(invocation_id=INV)
    assert decision_from(ctx) == FALLBACK_DECISION


# ── the question ──────────────────────────────────────────────────────────────


async def test_user_text_prefers_user_content_and_falls_back_to_the_events():
    ctx = await invocation_ctx(invocation_id=INV, user_text="статус Silverado 2026 Q3")
    assert user_text(ctx) == "статус Silverado 2026 Q3"

    seeded = await invocation_ctx(
        invocation_id=INV,
        events=[
            Event(
                invocation_id=INV,
                author="user",
                content=genai_types.Content(
                    role="user", parts=[genai_types.Part(text="сколько live кампаний")]
                ),
            )
        ],
    )
    assert user_text(seeded) == "сколько live кампаний"


# ── the agent + the root wiring ───────────────────────────────────────────────


def test_the_router_is_a_no_tool_typed_classifier():
    assert router_agent.name == "router"  # the author the bot ignores
    assert router_agent.tools == []
    assert router_agent.output_schema is RouterDecision
    assert router_agent.output_key == ROUTER_STATE_KEY


def test_the_root_is_echo_then_router_then_dispatcher():
    """The boundary contract: same engine, same stream_query, and sandbox_echo
    still first so the provenance event precedes any work."""
    from gub_agent.agent import deep_agent, root_agent

    assert [a.name for a in root_agent.sub_agents] == ["sandbox_echo", "router", "dispatcher"]
    # The deep path is the pipeline as it was, minus the echo that moved up.
    assert [a.name for a in deep_agent.sub_agents] == [
        AGENT_NAME,
        "format_gate",
        "critic_gate",
        "loop_escalator",
    ]
    assert deep_agent.max_iterations == 2


def test_the_executor_keeps_its_guardrail_callbacks():
    """The per-pass budget resets and the round limiter stay attached to the
    executor — the fast path must not have moved them onto the new root."""
    from gub_agent.agent import executor_agent

    assert executor_agent.before_agent_callback is not None
    assert executor_agent.before_model_callback is not None
    assert executor_agent.before_tool_callback is not None
    assert executor_agent.after_tool_callback is not None


def test_both_destinations_are_in_the_dispatchers_subtree():
    from gub_agent.agent import dispatcher

    assert [a.name for a in dispatcher.sub_agents] == ["fast_path", "gub_pipeline"]
