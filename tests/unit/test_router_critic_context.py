"""
The router's and the critic's view of earlier turns (thread-topics).

Only the executor used to be bounded. ADK shows an agent its OWN tool traffic
as function_call / function_response parts, which `strip_prior_turn_tool_parts`
removes; every OTHER agent gets that traffic flattened into text —
"[gub_agent] `get_account_overview` tool returned result: {…}" — so for the
router and the critic that pruner removed nothing, and both re-read every
earlier turn's full tool results on every call (measured offline at ~41k tokens
per earlier heavy turn). With a thread keeping up to 200 turns that is
quadratic cost and, eventually, a request over the model's input limit — which
inside Agent Engine reaches the caller as an EMPTY 200 stream.

Every fixture here goes through ADK's own `_get_contents`, for the same reason
test_context_window.py does: the rendering is ADK's, and a hand-written
fixture would encode today's wording instead of catching the day it changes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.adk.events import Event
from google.adk.flows.llm_flows.contents import _get_contents
from google.genai import types as genai_types

from gub_agent.agents import context_pruning
from gub_agent.agents.context_pruning import (
    WINDOW_STATE_KEY,
    _is_foreign_context,
    _is_foreign_tool_text,
    _turn_starts,
    strip_prior_turn_tool_text,
)
from gub_agent.agents.critic import _critic_before_model
from gub_agent.agents.router import _router_before_model
from gub_agent.config import AGENT_NAME

TOOL = "get_account_overview"


@pytest.fixture(autouse=True)
def _default_window(monkeypatch):
    """A known default whatever the environment says (see test_context_window)."""
    monkeypatch.setattr(context_pruning, "CONTEXT_TURN_WINDOW", 10)


def _ev(author: str, role: str, part: genai_types.Part, inv: str) -> Event:
    return Event(
        invocation_id=inv, author=author, content=genai_types.Content(role=role, parts=[part])
    )


def _text(author: str, role: str, text: str, inv: str) -> Event:
    return _ev(author, role, genai_types.Part(text=text), inv)


def _payload(n: int) -> dict:
    """A heavy tool result: rows plus `_sources` plumbing, and a per-turn
    marker so a test can say exactly WHICH turn's payload survived."""
    return {
        "marker": f"payload-{n}",
        "rows": [f"row {n}.{i} " + "x" * 80 for i in range(40)],
        "_sources": [f"file-{n}-{i:03d}" for i in range(60)],
    }


def _tool_traffic(n: int, inv: str) -> list[Event]:
    return [
        _ev(
            AGENT_NAME,
            "model",
            genai_types.Part(function_call=genai_types.FunctionCall(name=TOOL, args={"n": n})),
            inv,
        ),
        _ev(
            AGENT_NAME,
            "user",
            genai_types.Part(
                function_response=genai_types.FunctionResponse(name=TOOL, response=_payload(n))
            ),
            inv,
        ),
    ]


def _heavy_turn(n: int) -> list[Event]:
    """One complete deep-path turn as this pipeline emits it, one invocation."""
    inv = f"inv-{n}"
    return [
        _text("user", "user", f"question {n}", inv),
        _text("router", "model", f'{{"intent": "account_facts", "turn": {n}}}', inv),
        *_tool_traffic(n, inv),
        _text(AGENT_NAME, "model", f"draft {n}", inv),
        _text("format_gate", "model", f'{{"kind": "answer", "headline": "formatted {n}"}}', inv),
        _text("critic", "model", f'{{"sufficient": true, "reason": "verdict {n}"}}', inv),
    ]


def _history(turns: int) -> list[Event]:
    events: list[Event] = []
    for n in range(turns):
        events.extend(_heavy_turn(n))
    return events


def _router_events(prior: int) -> list[Event]:
    """What the router sees: history, then this turn's question and nothing else."""
    return _history(prior) + [_text("user", "user", f"question {prior}", f"inv-{prior}")]


def _critic_events(prior: int) -> list[Event]:
    """What the critic sees: history, then THIS turn's question, the
    executor's tool work and draft, and the format gate's payload."""
    inv = f"inv-{prior}"
    return _history(prior) + [
        _text("user", "user", f"question {prior}", inv),
        _text("router", "model", f'{{"intent": "account_facts", "turn": {prior}}}', inv),
        *_tool_traffic(prior, inv),
        _text(AGENT_NAME, "model", f"draft {prior}", inv),
        _text(
            "format_gate", "model", f'{{"kind": "answer", "headline": "formatted {prior}"}}', inv
        ),
    ]


def _cb(state: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(state=state if state is not None else {}, invocation_id="inv-now")


def _req(events: list[Event], agent: str) -> SimpleNamespace:
    return SimpleNamespace(contents=_get_contents(None, events, agent))


def _all_text(contents: list) -> str:
    return "\n".join(p.text for c in contents for p in (c.parts or []) if getattr(p, "text", None))


def _tool_text_parts(contents: list) -> list[str]:
    return [
        p.text
        for c in contents
        for p in (c.parts or [])
        if getattr(p, "text", None) and _is_foreign_tool_text(p)
    ]


# ── the premise ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("agent", ["router", "critic"])
def test_adk_hands_other_agents_tool_traffic_as_text(agent):
    """The tripwire. If ADK stops flattening another agent's tool parts into
    text, or rewords the flattening, this fails — and the stripper needs a
    look before the router and critic quietly re-inflate."""
    req = _req(_router_events(3), agent)
    fn_parts = [
        p
        for c in req.contents
        for p in (c.parts or [])
        if getattr(p, "function_call", None) or getattr(p, "function_response", None)
    ]
    assert fn_parts == [], "no function parts reach another agent — that is the whole problem"
    rendered = _tool_text_parts(req.contents)
    assert len(rendered) == 6, "3 turns x (one call + one result), recognised as tool text"
    assert sum("tool returned result:" in t for t in rendered) == 3
    assert sum("called tool" in t for t in rendered) == 3
    assert "payload-0" in _all_text(req.contents)


@pytest.mark.parametrize(
    ("text", "is_tool"),
    [
        # 2.9.2 (_fencing.py): the payload fenced on the following lines.
        (
            f"[gub_agent] `{TOOL}` tool returned result:\n<<<BEGIN_QUOTED_AGENT_CONTENT>>>\n{{}}",
            True,
        ),
        (f"[gub_agent] called tool `{TOOL}` with parameters:\n<<<BEGIN_QUOTED", True),
        # 2.6.1 (contents.py): same openings, payload on the same line.
        (f"[gub_agent] `{TOOL}` tool returned result: {{'rows': []}}", True),
        (f"[gub_agent] called tool `{TOOL}` with parameters: {{'n': 1}}", True),
        # Prose and thinking are what the router needs — never tool text.
        ("[gub_agent] said:\n<<<BEGIN_QUOTED_AGENT_CONTENT>>>\ndraft", False),
        ("[critic] thought:\n<<<BEGIN_QUOTED_AGENT_CONTENT>>>\nhmm", False),
        ("[gub_agent] said: I called tool `x` and it returned result: 3", False),
        ("For context: below is a transcript", False),
    ],
)
def test_tool_text_is_recognised_in_both_adk_wordings(text, is_tool):
    assert _is_foreign_tool_text(genai_types.Part(text=text)) is is_tool


# ── the router ───────────────────────────────────────────────────────────────


def test_the_router_keeps_earlier_prose_and_drops_earlier_tool_results():
    req = _req(_router_events(4), "router")
    _router_before_model(_cb(), req)

    kept = _all_text(req.contents)
    assert _tool_text_parts(req.contents) == []
    for n in range(4):
        assert f"payload-{n}" not in kept
        assert f"file-{n}-000" not in kept  # the `_sources` plumbing went with it
        # …and everything "and Q3?" is resolved against is still there.
        for prose in (f"question {n}", f"draft {n}", f"formatted {n}", f"verdict {n}"):
            assert prose in kept, prose
        assert f'"turn": {n}' in kept  # the router's own earlier decisions
    assert "question 4" in kept
    assert _turn_starts(req.contents)[-1] == len(req.contents) - 1


def test_the_router_is_windowed_to_the_sessions_turns(caplog):
    req = _req(_router_events(6), "router")
    with caplog.at_level("INFO", logger="gub_agent.agents.context_pruning"):
        _router_before_model(_cb({WINDOW_STATE_KEY: 2}), req)

    assert len(_turn_starts(req.contents)) == 2
    kept = _all_text(req.contents)
    assert "draft 5" in kept and "question 6" in kept
    assert "question 4" not in kept
    assert "turn_window: agent=router window=2 source=state turns=7" in caplog.text
    # The executor-only rollout line stays executor-only.
    assert "context_window:" not in caplog.text


def test_a_thread_window_leaves_every_earlier_turns_prose():
    # 200 in state: nothing is cut, but the tool text still goes.
    req = _req(_router_events(12), "router")
    _router_before_model(_cb({WINDOW_STATE_KEY: 200}), req)
    assert len(_turn_starts(req.contents)) == 13
    assert _tool_text_parts(req.contents) == []
    assert "question 0" in _all_text(req.contents)


def test_what_the_stripping_saves_is_most_of_the_request():
    """Not a benchmark — a floor. The fixture's payloads are small next to a
    real account overview (~165k characters of function parts per turn,
    measured on the sandbox), and still they dominate the request."""
    req = _req(_router_events(4), "router")
    before = len(_all_text(req.contents))
    _router_before_model(_cb(), req)
    after = len(_all_text(req.contents))
    assert after < before * 0.35, (before, after)


# ── the critic ───────────────────────────────────────────────────────────────


def test_the_critic_keeps_this_turns_tool_evidence_whole():
    """The critic judges THIS turn's evidence. Everything from the user's
    question on passes through untouched — byte for byte, not merely
    "something about payload-4 is still there"."""
    req = _req(_critic_events(4), "critic")
    current_start = _turn_starts(req.contents)[-1]
    current_before = [_all_text([c]) for c in req.contents[current_start:]]

    _critic_before_model(_cb(), req)

    current_start_after = _turn_starts(req.contents)[-1]
    assert [_all_text([c]) for c in req.contents[current_start_after:]] == current_before
    kept = _all_text(req.contents)
    assert "payload-4" in kept and "file-4-059" in kept
    assert sum("payload-" in t for t in _tool_text_parts(req.contents)) == 1
    for n in range(4):
        assert f"payload-{n}" not in kept
        assert f"draft {n}" in kept


def test_a_one_turn_window_still_keeps_the_critics_whole_current_turn():
    """memory-00 §3.5 kept the critic unwindowed for fear a turn window would
    cut inside the current turn and remove the user's question. It cannot:
    foreign-context contents are not boundaries, so the harshest window there
    is still starts at the question."""
    req = _req(_critic_events(5), "critic")
    _critic_before_model(_cb({WINDOW_STATE_KEY: 1}), req)

    assert len(_turn_starts(req.contents)) == 1
    kept = _all_text(req.contents)
    assert "question 5" in kept and "draft 5" in kept and "payload-5" in kept
    assert "called tool" in kept  # this turn's call, as the critic reads it
    assert "question 4" not in kept and "draft 4" not in kept


# ── the stripper's own edges ─────────────────────────────────────────────────


def test_a_foreign_content_left_with_only_its_preamble_is_dropped():
    req = _req(_router_events(3), "router")
    strip_prior_turn_tool_text(None, req)
    for content in req.contents:
        if _is_foreign_context(content):
            assert len(content.parts) > 1, "a bare preamble with no transcript after it"


def test_a_single_turn_request_is_an_identity():
    contents = _get_contents(None, _router_events(0), "router")
    req = SimpleNamespace(contents=contents)
    strip_prior_turn_tool_text(None, req)
    assert req.contents is contents

    empty = SimpleNamespace(contents=None)
    strip_prior_turn_tool_text(None, empty)
    assert empty.contents is None


def test_history_with_no_tool_traffic_is_an_identity():
    # Fast-path and smalltalk turns leave no tool text behind: no copies.
    events = [
        _text("user", "user", "hi", "inv-0"),
        _text("format_gate", "model", '{"kind": "smalltalk"}', "inv-0"),
        _text("user", "user", "status of Silverado?", "inv-1"),
    ]
    contents = _get_contents(None, events, "router")
    req = SimpleNamespace(contents=contents)
    strip_prior_turn_tool_text(None, req)
    assert req.contents is contents


def test_a_user_message_shaped_like_a_tool_line_is_never_edited():
    # Only foreign-context contents are touched.
    pasted = f"[gub_agent] `{TOOL}` tool returned result: what does this mean?"
    events = [
        _text("user", "user", pasted, "inv-0"),
        _text("format_gate", "model", '{"kind": "answer"}', "inv-0"),
        _text("user", "user", "and now?", "inv-1"),
    ]
    req = _req(events, "router")
    strip_prior_turn_tool_text(None, req)
    assert pasted in _all_text(req.contents)


def test_the_session_history_is_not_mutated():
    """The contents ADK builds share Part objects with session history
    (`_copy_content_for_request` copies shallowly). Copy-on-write: the
    request's contents are replaced, the originals keep every part."""
    req = _req(_router_events(2), "router")
    originals = [(c, list(c.parts or [])) for c in req.contents]
    strip_prior_turn_tool_text(None, req)
    for content, parts in originals:
        assert list(content.parts or []) == parts
