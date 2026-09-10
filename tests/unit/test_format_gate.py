"""
agents.format_gate — the deterministic checks and the retry machinery.

Pure-function halves (gate_problems, compose_brief, template_payload) are
tested directly; the gate's control flow (feedback event → formatter re-run →
template after two failed retries; deterministic abstention with NO formatter
LLM run) runs against a scripted formatter stand-in on a real ADK
InvocationContext, the way test_sandbox.py drives CriticGate.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from pydantic import ValidationError

from gub_agent.agents.evidence_index import (
    evidence_index,
    record_evidence,
    reset_evidence_index,
)
from gub_agent.agents.format_gate import (
    FormatGate,
    compose_brief,
    gate_problems,
    template_payload,
)
from gub_agent.config import AGENT_NAME
from gub_agent.schemas import AnswerPayload

INV = "inv-1"


def _seed_index() -> dict:
    """A believable turn: an account row, a campaign row, an aggregate total."""
    ctx = SimpleNamespace(invocation_id=INV)
    reset_evidence_index(ctx)
    record_evidence(
        SimpleNamespace(name="org_query"),
        {},
        ctx,
        {
            "results": [{"id": "a1", "name": "chevy", "status": "active"}],
            "total": 1,
        },
    )
    record_evidence(
        SimpleNamespace(name="get_campaign"),
        {},
        ctx,
        {"id": "c1", "name": "Q3 push", "status": "live", "budget": 1200000},
    )
    return evidence_index(INV)


def _payload(**over) -> AnswerPayload:
    base = {
        "kind": "answer",
        "headline": "chevy is in good shape",
        "blocks": [{"kind": "text", "text": "the Q3 push is live with budget 1200000."}],
        "citations": ["get_campaign:c1:status"],
        "facts": [
            {
                "evidence_id": "get_campaign:c1:status",
                "entity_id": "c1",
                "field": "status",
                "value": "live",
            }
        ],
    }
    base.update(over)
    return AnswerPayload.model_validate(base)


# ── gate_problems, pure ───────────────────────────────────────────────────────


async def test_a_grounded_payload_passes():
    assert gate_problems(_payload(), _seed_index()) == []


async def test_unknown_citation_fails_hard():
    index = _seed_index()
    payload = _payload(
        citations=["org_query:nope"],
        facts=[{"evidence_id": "org_query:nope", "value": "x"}],
    )
    problems = gate_problems(payload, index)
    assert len(problems) == 1
    assert "unknown citation" in problems[0]
    assert "org_query:nope" in problems[0]


async def test_ungrounded_number_is_named():
    problems = gate_problems(
        _payload(blocks=[{"kind": "text", "text": "the Q3 push is live with budget 999."}]),
        _seed_index(),
    )
    assert any("ungrounded number" in p and "999" in p for p in problems)


async def test_grounded_numbers_survive_reformatting_commas():
    """1,200,000 in the answer grounds against 1200000 in the evidence."""
    problems = gate_problems(
        _payload(blocks=[{"kind": "text", "text": "the Q3 push budget is 1,200,000."}]),
        _seed_index(),
    )
    assert problems == []


async def test_ungrounded_entity_is_the_critics_old_hard_fail():
    """'Chevrolet' when the tool returned 'chevy' — the exact case the critic
    used to hard-fail on, now caught in code."""
    problems = gate_problems(
        _payload(blocks=[{"kind": "text", "text": "latest from Chevrolet is strong."}]),
        _seed_index(),
    )
    assert any("ungrounded entity" in p and "Chevrolet" in p for p in problems)


async def test_grounded_entity_and_sentence_openers_pass():
    problems = gate_problems(
        _payload(
            headline="chevy is healthy",
            blocks=[
                {
                    "kind": "bullets",
                    # "Overall" / "Nothing" open their bullets — grammar, not
                    # names; "Q3 push" is grounded (case-insensitive).
                    "items": ["Overall the Q3 Push is live", "Nothing else moved recently"],
                }
            ],
        ),
        _seed_index(),
    )
    assert problems == []


async def test_abstain_and_clarify_skip_grounding_but_not_citation_checks():
    index = _seed_index()
    abstain = AnswerPayload.model_validate({"kind": "abstain", "headline": "NO_COMPANY_RECORDS"})
    assert gate_problems(abstain, index) == []
    clarify = AnswerPayload.model_validate(
        {"kind": "clarify", "headline": "Which Chevrolet do you mean?"}
    )
    assert gate_problems(clarify, index) == []  # echoes the user's word, legitimately
    bad_abstain = AnswerPayload.model_validate(
        {
            "kind": "abstain",
            "headline": "NO_COMPANY_RECORDS",
            "citations": ["org_query:nope"],
            "facts": [{"evidence_id": "org_query:nope"}],
        }
    )
    assert any("unknown citation" in p for p in gate_problems(bad_abstain, index))


async def test_table_source_id_cells_do_not_trip_number_grounding():
    index = _seed_index()
    payload = _payload(
        blocks=[
            {
                "kind": "table",
                "columns": ["Campaign", "Status", "Source"],
                "rows": [["Q3 push", "live", "get_campaign:c1:status"]],
            }
        ],
    )
    assert gate_problems(payload, index) == []


# ── brief + template, pure ────────────────────────────────────────────────────


async def test_brief_carries_answer_evidence_and_feedback():
    index = _seed_index()
    brief = compose_brief("chevy is fine.", index, feedback="ungrounded number: 999")
    assert "EXECUTOR ANSWER" in brief
    assert "chevy is fine." in brief
    assert "- get_campaign:c1:budget = 1200000" in brief
    assert "FORMAT_FEEDBACK" in brief and "999" in brief
    assert "FORMAT_FEEDBACK" not in compose_brief("x", index, feedback="")


async def test_template_render_is_bullets_of_evidence_rows_with_ids():
    index = _seed_index()
    payload = template_payload("Here is the long executor answer\nmore detail", index)
    assert payload.kind == "answer"
    assert payload.headline == "Here is the long executor answer"  # verbatim — no filler veto
    bullets = payload.blocks[0]
    assert bullets.kind == "bullets"
    assert all("[" in item for item in bullets.items)
    assert set(payload.citations) == {f.evidence_id for f in payload.facts}
    assert all(cid in index for cid in payload.citations)
    # entity rows preferred over per-field rows
    assert "org_query:a1" in payload.citations


async def test_template_with_no_evidence_falls_back_to_executor_text():
    reset_evidence_index(SimpleNamespace(invocation_id=INV))
    payload = template_payload("nothing was retrieved this turn", {})
    assert payload.blocks[0].kind == "text"
    assert payload.citations == []


# ── the gate's control flow, on a real InvocationContext ─────────────────────


class ScriptedFormatter(BaseAgent):
    """Stand-in for the formatter LLM: plays back one scripted outcome per run
    — a payload dict (emitted the way LlmAgent's output_schema does: JSON text
    + state_delta) or a ValidationError to raise. The last item repeats."""

    script: list = []
    runs: list = []  # one entry per run — mutated in place, pydantic-safe

    async def _run_async_impl(self, ctx):
        item = self.script[min(len(self.runs), len(self.script) - 1)]
        self.runs.append(ctx.invocation_id)
        if isinstance(item, Exception):
            raise item
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=json.dumps(item))]
            ),
            actions=EventActions(state_delta={"answer_payload": item}),
        )


def _scripted(script: list) -> ScriptedFormatter:
    return ScriptedFormatter(name="formatter", script=script, runs=[])


async def _gate_ctx(executor_text: str | None) -> InvocationContext:
    service = InMemorySessionService()
    session = await service.create_session(app_name="gub", user_id="u", state={})
    if executor_text is not None:
        await service.append_event(
            session,
            Event(
                invocation_id=INV,
                author=AGENT_NAME,
                content=genai_types.Content(
                    role="model", parts=[genai_types.Part(text=executor_text)]
                ),
            ),
        )
    return InvocationContext(
        session_service=service,
        invocation_id=INV,
        agent=BaseAgent(name="host"),
        session=session,
    )


def _valid_dict() -> dict:
    return _payload().model_dump(exclude_none=True)


async def test_gate_passes_a_valid_payload_through_untouched():
    _seed_index()
    formatter = _scripted([_valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx("chevy is in good shape — the Q3 push is live.")

    events = [e async for e in gate.run_async(ctx)]

    assert len(formatter.runs) == 1
    assert [e.author for e in events] == ["formatter"]
    assert events[0].actions.state_delta["answer_payload"]["headline"] == "chevy is in good shape"


async def test_gate_retries_with_feedback_then_accepts():
    _seed_index()
    bad = _payload(
        citations=["org_query:nope"],
        facts=[{"evidence_id": "org_query:nope", "value": "x"}],
    ).model_dump(exclude_none=True)
    formatter = _scripted([bad, _valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx("chevy is in good shape — the Q3 push is live.")

    events = [e async for e in gate.run_async(ctx)]

    assert len(formatter.runs) == 2
    feedback_events = [
        e
        for e in events
        if e.author == "format_gate"
        and e.actions
        and (e.actions.state_delta or {}).get("format_feedback")
    ]
    assert len(feedback_events) == 1
    assert "unknown citation" in feedback_events[0].actions.state_delta["format_feedback"]
    # last payload-bearing event is the accepted formatter one, not a template
    assert events[-1].author == "formatter"


async def test_gate_emits_template_after_two_failed_retries():
    _seed_index()
    bad = _payload(
        citations=["org_query:nope"],
        facts=[{"evidence_id": "org_query:nope", "value": "x"}],
    ).model_dump(exclude_none=True)
    formatter = _scripted([bad])  # every attempt returns the same bad payload
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx("chevy is in good shape — the Q3 push is live.")

    events = [e async for e in gate.run_async(ctx)]

    assert len(formatter.runs) == 3  # initial + two retries
    last = events[-1]
    assert last.author == "format_gate"
    template = last.actions.state_delta["answer_payload"]
    assert template["kind"] == "answer"
    assert json.loads(last.content.parts[0].text) == template


async def test_pydantic_validation_errors_take_the_same_retry_path():
    _seed_index()
    try:
        AnswerPayload.model_validate({"kind": "answer", "headline": "Sure, here it is"})
    except ValidationError as exc:
        boom = exc
    formatter = _scripted([boom, _valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx("chevy is in good shape — the Q3 push is live.")

    events = [e async for e in gate.run_async(ctx)]

    assert len(formatter.runs) == 2
    feedback = next(
        (e.actions.state_delta or {}).get("format_feedback", "")
        for e in events
        if e.author == "format_gate" and e.actions and e.actions.state_delta
    )
    assert "invalid payload" in feedback


async def test_exact_abstention_becomes_a_payload_with_no_formatter_run():
    _seed_index()
    formatter = _scripted([_valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx("NO_COMPANY_RECORDS")

    events = [e async for e in gate.run_async(ctx)]

    assert formatter.runs == []  # deterministic — no LLM for one marker word
    assert len(events) == 1
    payload = events[0].actions.state_delta["answer_payload"]
    assert payload["kind"] == "abstain"
    assert events[0].author == "format_gate"


async def test_no_executor_text_means_no_events_and_no_formatter_run():
    _seed_index()
    formatter = _scripted([_valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _gate_ctx(None)

    events = [e async for e in gate.run_async(ctx)]

    assert events == []
    assert formatter.runs == []


async def test_critic_gate_recognises_the_abstain_payload():
    """The pipeline invariant: NO_COMPANY_RECORDS keeps skipping the critic
    LLM whether it arrives as the bare marker or as the typed abstain payload
    already written to state by the format gate."""
    from gub_agent.agents.critic import CriticGate

    class _RecordingCritic(BaseAgent):
        ran: list = []

        async def _run_async_impl(self, ctx):
            self.ran.append(ctx.invocation_id)
            yield Event(invocation_id=ctx.invocation_id, author=self.name)

    critic = _RecordingCritic(name="critic", ran=[])
    gate = CriticGate(name="critic_gate", sub_agents=[critic])

    service = InMemorySessionService()
    session = await service.create_session(
        app_name="gub",
        user_id="u",
        state={"answer_payload": {"kind": "abstain", "headline": "NO_COMPANY_RECORDS"}},
    )
    # Executor text that is NOT the bare marker — only the payload says abstain.
    await service.append_event(
        session,
        Event(
            invocation_id=INV,
            author=AGENT_NAME,
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text="I cannot answer that from GUB.")]
            ),
        ),
    )
    ctx = InvocationContext(
        session_service=service, invocation_id=INV, agent=BaseAgent(name="host"), session=session
    )

    events = [e async for e in gate.run_async(ctx)]

    assert critic.ran == []
    verdict = events[0].actions.state_delta["critic_verdict"]
    assert verdict["sufficient"] is True
    assert "abstain" in verdict["reason"]
