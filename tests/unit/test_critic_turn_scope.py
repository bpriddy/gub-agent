"""
The critic's deterministic facts are about THIS turn (thread-topics).

`_executor_made_tool_call` and `_last_executor_text` read session events, and
session events are never trimmed: a thread keeps every turn it has ever had.
Unfiltered, both scans reached back into earlier turns —

- a tool call from ANY earlier turn made the critic read "TOOL CALL THIS TURN:
  yes", which switches off its "you answered from memory" guard for the rest of
  the session;
- a turn whose executor produced no text read the PREVIOUS answer instead, so
  the format gate could ship it as this turn's and CriticGate could skip the
  critic because an earlier turn had abstained.

Both now filter on the current invocation id. Built on a REAL ADK
InvocationContext (tests/helpers.py), because the invocation id is the thing
under test and a fake would supply whatever the test assumed.
"""

from __future__ import annotations

from google.adk.agents import BaseAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event
from google.genai import types as genai_types

from gub_agent.agents.critic import (
    CriticGate,
    _critic_instruction,
    _executor_made_tool_call,
    _last_executor_text,
)
from gub_agent.config import AGENT_NAME
from tests.helpers import invocation_ctx

EARLIER = "inv-earlier"
NOW = "inv-now"


def _text(author: str, text: str, inv: str) -> Event:
    role = "user" if author == "user" else "model"
    return Event(
        invocation_id=inv,
        author=author,
        content=genai_types.Content(role=role, parts=[genai_types.Part(text=text)]),
    )


def _call(inv: str) -> Event:
    return Event(
        invocation_id=inv,
        author=AGENT_NAME,
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(function_call=genai_types.FunctionCall(name="find", args={}))],
        ),
    )


def _earlier_turn_with_a_tool_call() -> list[Event]:
    return [_text("user", "how is chevy doing?", EARLIER), _call(EARLIER)]


# ── "was a tool called this turn" ────────────────────────────────────────────


async def test_an_earlier_turns_tool_call_is_not_this_turns():
    ctx = await invocation_ctx(
        invocation_id=NOW,
        events=[
            *_earlier_turn_with_a_tool_call(),
            _text(AGENT_NAME, "draft from an earlier turn", EARLIER),
            _text("user", "and how many campaigns are live?", NOW),
            _text(AGENT_NAME, "There are 12 live campaigns.", NOW),  # from memory
        ],
    )
    assert _executor_made_tool_call(ctx) is False
    # …and the critic is told so, which is what keeps its guard armed.
    assert "TOOL CALL THIS TURN: no" in _critic_instruction(ReadonlyContext(ctx))


async def test_this_turns_tool_call_still_counts():
    ctx = await invocation_ctx(
        invocation_id=NOW,
        events=[*_earlier_turn_with_a_tool_call(), _text("user", "and Q3?", NOW), _call(NOW)],
    )
    assert _executor_made_tool_call(ctx) is True
    assert "TOOL CALL THIS TURN: yes" in _critic_instruction(ReadonlyContext(ctx))


# ── the executor's text ──────────────────────────────────────────────────────


async def test_the_last_executor_text_is_this_turns_or_nothing():
    ctx = await invocation_ctx(
        invocation_id=NOW,
        events=[
            _text(AGENT_NAME, "the earlier answer", EARLIER),
            _text("user", "and Q3?", NOW),
            _call(NOW),  # the executor died after its call: no text this turn
        ],
    )
    # Not "the earlier answer": the format gate treats "" as "emit nothing",
    # which is the honest trace of a failed run.
    assert _last_executor_text(ctx) == ""

    ctx_ok = await invocation_ctx(
        invocation_id=NOW,
        events=[_text(AGENT_NAME, "the earlier answer", EARLIER), _text(AGENT_NAME, "Q3 is", NOW)],
    )
    assert _last_executor_text(ctx_ok) == "Q3 is"


class _RecordingCritic(BaseAgent):
    ran: list = []

    async def _run_async_impl(self, ctx):
        self.ran.append(ctx.invocation_id)
        yield Event(invocation_id=ctx.invocation_id, author=self.name)


async def test_an_earlier_abstention_does_not_skip_this_turns_critic():
    """CriticGate's deterministic pass is for THIS turn's NO_COMPANY_RECORDS.
    An earlier turn's must not buy the current draft a free "sufficient"."""
    critic = _RecordingCritic(name="critic", ran=[])
    gate = CriticGate(name="critic_gate", sub_agents=[critic])
    ctx = await invocation_ctx(
        invocation_id=NOW,
        events=[
            _text(AGENT_NAME, "NO_COMPANY_RECORDS", EARLIER),
            _text("user", "and the other account?", NOW),
            _call(NOW),
        ],
    )

    [e async for e in gate.run_async(ctx)]

    assert critic.ran == [NOW]


async def test_this_turns_abstention_still_skips_the_critic():
    critic = _RecordingCritic(name="critic", ran=[])
    gate = CriticGate(name="critic_gate", sub_agents=[critic])
    ctx = await invocation_ctx(
        invocation_id=NOW,
        events=[
            _text(AGENT_NAME, "the earlier answer", EARLIER),
            _text(AGENT_NAME, "NO_COMPANY_RECORDS", NOW),
        ],
    )

    events = [e async for e in gate.run_async(ctx)]

    assert critic.ran == []
    assert events[0].actions.state_delta["critic_verdict"]["sufficient"] is True
