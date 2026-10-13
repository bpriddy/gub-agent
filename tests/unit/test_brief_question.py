"""
The question and the intent in the formatter's brief (BE-4) — `compose_brief`
and `FormatGate` in `agents/format_gate.py`.

THE PROBLEM. The deep path's brief opened with the executor's answer and never
said what was asked: the formatter rendered a rundown the way it rendered a
single fact, and a rule that depends on the kind of question (BE-5) had nothing
to read. The fast path's draft already carries its own `QUESTION:` line and its
`intent=`; only the deep path was blind.

THE FIX, pinned here:
  * the deep path's brief opens with `QUESTION:` (the reader's words, one line,
    capped) and `INTENT:` (the router's label) and then, unchanged, the
    executor's answer and the evidence;
  * the label is the router's REAL decision. A fallback (the router said
    nothing, or a speculative fork before its keep) shows no label — never a
    guessed `exploratory` — and neither does a stale one from an earlier turn;
  * no question or no decision costs nothing: the brief is what it was, byte
    for byte, and no line is written for the missing half;
  * the fast path adds nothing (its draft already says both), and the retry
    briefs keep both lines;
  * reading the decision for the brief writes no log line.

The gate-flow tests run the real `FormatGate` over a scripted formatter and read
the brief it was handed, so each reads the same on the old code (no such lines).
The kept-speculation half lives in `test_speculation.py`.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

from gub_agent import config
from gub_agent.agents.evidence_index import reset_evidence_index, set_answer_draft
from gub_agent.agents.format_gate import (
    MAX_BRIEF_QUESTION_CHARS,
    FormatGate,
    compose_brief,
)
from gub_agent.config import AGENT_NAME
from gub_agent.schemas.router import FALLBACK_DECISION

from .test_format_gate import INV, _scripted, _seed_index, _valid_dict

EXECUTOR = "chevy is in good shape — the Q3 push is live."

# What the bot sends when the reader picked a card: a prefix, then the question.
PREFIXED = (
    'User selected campaign 0f1bd315-6c13-4fea-8dd6-6a93bb0fb6da "Chevy | Red Tag 2026" '
    "(Chevy). Original question: how is it going?"
)


@pytest.fixture(autouse=True)
def _no_evidence_rows_event(monkeypatch):
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", False)
    yield
    reset_evidence_index(SimpleNamespace(invocation_id=INV))  # the draft store is process-wide


async def _ctx(
    question: str | None = "how is chevy doing?",
    decision: dict | None = None,
    *,
    as_event: bool = False,
) -> InvocationContext:
    """A gate context: the executor's answer is in the session; the question is
    the runner's `user_content` (or, with `as_event`, only a user event — the
    fallback `user_text` reads for a hand-built context); `decision` is
    `state["router_decision"]`."""
    service = InMemorySessionService()
    session = await service.create_session(
        app_name="gub",
        user_id="u",
        state={"router_decision": decision} if decision is not None else {},
    )
    if question is not None and as_event:
        await service.append_event(
            session,
            Event(
                invocation_id=INV,
                author="user",
                content=genai_types.Content(role="user", parts=[genai_types.Part(text=question)]),
            ),
        )
    await service.append_event(
        session,
        Event(
            invocation_id=INV,
            author=AGENT_NAME,
            content=genai_types.Content(role="model", parts=[genai_types.Part(text=EXECUTOR)]),
        ),
    )
    content = (
        genai_types.Content(role="user", parts=[genai_types.Part(text=question)])
        if question is not None and not as_event
        else None
    )
    return InvocationContext(
        session_service=service,
        invocation_id=INV,
        agent=BaseAgent(name="host"),
        session=session,
        user_content=content,
    )


def _real(intent: str = "assessment", confidence: float = 0.9) -> dict:
    return {"intent": intent, "confidence": confidence, "language": "en"}


async def _briefs(
    ctx: InvocationContext, script: list | None = None, draft: str | None = None
) -> list[str]:
    """Run the real gate; the brief each formatter run was handed. `draft` is
    what the fast path leaves for it, set after the index reset."""
    _seed_index()
    if draft is not None:
        set_answer_draft(INV, draft)
    formatter = _scripted(script or [_valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    [_ async for _ in gate.run_async(ctx)]
    return formatter.briefs


# ── the deep path's brief says what was asked ────────────────────────────────


async def test_the_deep_brief_opens_with_the_question_and_the_intent():
    [brief] = await _briefs(await _ctx("how is chevy doing?", _real("assessment")))

    assert brief.startswith(
        "QUESTION: how is chevy doing?\nINTENT: assessment\n\nEXECUTOR ANSWER (render this"
    )
    # …and the rest of the brief is where it was
    assert f"EXECUTOR ANSWER (render this — do not add facts):\n{EXECUTOR}\n" in brief
    assert "ALLOWED_EVIDENCE (the ONLY citable ids" in brief
    assert "RULES (checked in code" in brief


async def test_the_question_is_the_whole_message_the_bot_sent_prefix_included():
    """`user_text` is what the runner was called with — the bot's card-pick
    prefix is part of it, and the brief carries it, entity name and all."""
    [brief] = await _briefs(await _ctx(PREFIXED, _real("campaign_facts")))

    assert f"QUESTION: {PREFIXED}\n" in brief
    assert "INTENT: campaign_facts\n" in brief


async def test_a_question_that_only_exists_as_a_user_event_is_still_asked():
    """`user_text`'s fallback: a context built without `user_content`."""
    [brief] = await _briefs(await _ctx("how is chevy doing?", _real(), as_event=True))

    assert brief.startswith("QUESTION: how is chevy doing?\n")


async def test_every_intent_label_is_the_routers_own():
    for intent in ("exploratory", "assessment", "campaign_status", "file_lookup", "count_or_rank"):
        [brief] = await _briefs(await _ctx("how is chevy doing?", _real(intent)))
        assert f"\nINTENT: {intent}\n" in brief


# ── no decision, no label ────────────────────────────────────────────────────


async def test_a_fallback_decision_writes_no_intent_not_a_guessed_one():
    """FALLBACK_DECISION is `exploratory` at confidence 0 — what an unreadable
    decision costs, and what a speculative fork carries until its keep. As a
    brief line it would be a guess presented as a classification."""
    for fallback in (FALLBACK_DECISION.model_dump(exclude_none=True), None):
        [brief] = await _briefs(await _ctx("how is chevy doing?", fallback))

        assert brief.startswith("QUESTION: how is chevy doing?\n\nEXECUTOR ANSWER")
        assert "INTENT:" not in brief


async def test_a_garbled_decision_writes_no_intent():
    [brief] = await _briefs(await _ctx("how is chevy doing?", {"intent": "weather"}))

    assert brief.startswith("QUESTION: how is chevy doing?\n\nEXECUTOR ANSWER")
    assert "INTENT:" not in brief


async def test_a_low_confidence_real_decision_still_names_its_intent():
    """Only the fallback is `unknown`: a router that classified at 0.3 said
    something, and the label is its."""
    [brief] = await _briefs(await _ctx("how is chevy doing?", _real("assessment", 0.3)))

    assert "INTENT: assessment\n" in brief


# ── no question, no line ─────────────────────────────────────────────────────


async def test_no_question_writes_no_question_line():
    [brief] = await _briefs(await _ctx(None, _real("exploratory")))

    assert "QUESTION:" not in brief
    assert brief.startswith("INTENT: exploratory\n\nEXECUTOR ANSWER")


async def test_a_blank_question_writes_no_question_line():
    [brief] = await _briefs(await _ctx("   \n  ", _real()))

    assert "QUESTION:" not in brief


async def test_neither_question_nor_decision_is_the_brief_before_this_change():
    """The prefix-less, question-less case: no stray `QUESTION:` / `INTENT:` /
    `None`, and the brief begins where it always began."""
    [brief] = await _briefs(await _ctx(None, None))

    assert brief.startswith("EXECUTOR ANSWER (render this — do not add facts):\n")
    assert "QUESTION" not in brief and "INTENT" not in brief and "None" not in brief
    assert brief == compose_brief(EXECUTOR, _seed_index(), "")


# ── the question is one short line ───────────────────────────────────────────


async def test_a_multi_line_question_is_one_line():
    [brief] = await _briefs(await _ctx("how is chevy\n\n  doing,\tthis quarter?", _real()))

    assert "QUESTION: how is chevy doing, this quarter?\n" in brief


async def test_a_pasted_document_does_not_ride_along_on_every_attempt():
    huge = "how is chevy doing? " + "context " * 5000
    [brief] = await _briefs(await _ctx(huge, _real()))

    [line] = [line for line in brief.splitlines() if line.startswith("QUESTION: ")]
    assert len(line) == len("QUESTION: ") + MAX_BRIEF_QUESTION_CHARS
    assert line.endswith("…") and line.startswith("QUESTION: how is chevy doing? context")
    assert len(brief) < 5_000  # the 40 KB message is not in it


def test_compose_brief_takes_the_two_lines_as_keywords():
    brief = compose_brief("x", {}, "", question="  why?  ", intent="exploratory")

    assert brief.startswith("QUESTION: why?\nINTENT: exploratory\n\nEXECUTOR ANSWER")
    assert compose_brief("x", {}, "", question="", intent=None) == compose_brief("x", {}, "")


# ── the retries, the fast path, the logs ─────────────────────────────────────


async def test_the_retry_briefs_carry_the_question_and_the_intent_too():
    bad = _valid_dict()
    bad["citations"] = ["org_query:nope"]
    bad["facts"] = [{"evidence_id": "org_query:nope", "value": "x"}]

    first, retry = await _briefs(
        await _ctx("how is chevy doing?", _real("assessment")), [bad, _valid_dict()]
    )

    for brief in (first, retry):
        assert brief.startswith("QUESTION: how is chevy doing?\nINTENT: assessment\n\n")
    assert "FORMAT_FEEDBACK" not in first and "FORMAT_FEEDBACK" in retry


async def test_the_fast_path_draft_already_asks_and_the_gate_adds_nothing():
    """The fast path leaves its draft — which opens with the question and says
    `intent=` — and calls this gate; a second pair of lines would be a
    duplicate, and the router's decision is the same one the draft quotes."""
    draft = (
        "DETERMINISTIC LOOKUP (fast path) — intent=campaign_status, tool=get_campaign. "
        "Answer the question from these fields; nothing else was retrieved.\n"
        "QUESTION: how is Red Tag doing?\nRESULT:\nstatus: live"
    )
    ctx = await _ctx("how is Red Tag doing?", _real("campaign_status"))
    [brief] = await _briefs(ctx, draft=draft)

    assert brief.startswith("EXECUTOR ANSWER (render this — do not add facts):\nDETERMINISTIC")
    assert brief.count("QUESTION:") == 1
    assert "\nINTENT:" not in brief


async def test_reading_the_decision_for_the_brief_writes_no_router_line(caplog):
    """`decision_from` logs `router: no decision …` for a turn with none, and the
    dispatcher already did: one more per deep turn would double that count."""
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.router"):
        await _briefs(await _ctx("how is chevy doing?", None))
        await _briefs(await _ctx("how is chevy doing?", {"intent": "weather"}))

    assert [r.getMessage() for r in caplog.records if r.name == "gub_agent.agents.router"] == []
