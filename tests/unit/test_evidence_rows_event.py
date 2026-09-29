"""
EVIDENCE_ROWS_EVENT (latency-04 A4) — the format gate streams its evidence
index, once per run, as one content-less partial event.

The index the formatter's brief is composed from is final when the gate reads
it, long before the payload, and exists only in this process. The bot's early
claim filter (its B3 part ii) needs the ids the formatter will cite, so the
gate yields ONE `partial=True` event authored `format_gate` with
`custom_metadata={"evidence_rows": [...]}` before the formatter's first event.
Pinned here:

  * flag on: exactly one such event, before anything the formatter yields;
    one row per index entry, in index order, each carrying the entry's
    evidence id (the index KEY), as new dicts;
  * the abstain and no-text paths emit none;
  * on a real Runner, the persisted session events are identical with the
    flag on and off — a partial is streamed, never appended;
  * flag off: the gate's event list is today's;
  * the log line, `tenant=` last.
"""

from __future__ import annotations

import json
import logging

from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from gub_agent import config
from gub_agent.agents.evidence_index import (
    evidence_index,
    evidence_rows,
    record_evidence,
    reset_evidence_index,
)
from gub_agent.agents.format_gate import FormatGate
from gub_agent.config import AGENT_NAME

from .test_format_gate import _gate_ctx, _scripted, _seed_index, _valid_dict

TEXT = "chevy is in good shape — the Q3 push is live."


def _rows_events(events: list[Event]) -> list[Event]:
    return [e for e in events if e.custom_metadata and "evidence_rows" in e.custom_metadata]


async def test_flag_on_one_partial_rows_event_before_the_formatter(monkeypatch):
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", True)
    index = _seed_index()
    formatter = _scripted([_valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    events = [e async for e in gate.run_async(await _gate_ctx(TEXT))]

    [rows_event] = _rows_events(events)
    assert events[0] is rows_event  # before the formatter's first event
    assert rows_event.partial is True
    assert rows_event.author == "format_gate"
    assert rows_event.content is None
    assert not (rows_event.actions and rows_event.actions.state_delta)
    rows = rows_event.custom_metadata["evidence_rows"]
    # One row per entry, in index order, keyed by the entry's evidence id.
    assert [r["evidence_id"] for r in rows] == list(index)
    for row in rows:
        entry = index[row["evidence_id"]]
        assert row == {
            "evidence_id": row["evidence_id"],
            "entity_id": entry["entity_id"],
            "field": entry["field"],
            "value": entry["value"],
        }
        assert row is not entry
    assert {"get_campaign:c1:status", "org_query:a1", "org_query:total"} <= {
        r["evidence_id"] for r in rows
    }
    # The rest is today's: the formatter's payload, relayed as before.
    assert [e.author for e in events[1:]] == ["formatter"]


async def test_one_event_per_gate_run_even_across_retries(monkeypatch):
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", True)
    _seed_index()
    bad = {**_valid_dict(), "citations": ["nope:1"]}
    formatter = _scripted([bad, _valid_dict()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    events = [e async for e in gate.run_async(await _gate_ctx(TEXT))]

    assert len(formatter.runs) == 2  # a retry happened
    assert len(_rows_events(events)) == 1


async def test_rows_are_new_dicts(monkeypatch):
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", True)
    index = _seed_index()
    before = json.dumps(index, sort_keys=True)
    gate = FormatGate(name="format_gate", sub_agents=[_scripted([_valid_dict()])])
    [rows_event] = _rows_events([e async for e in gate.run_async(await _gate_ctx(TEXT))])

    for row in rows_event.custom_metadata["evidence_rows"]:
        row["value"] = "changed after the yield"
    assert json.dumps(evidence_index("inv-1"), sort_keys=True) == before


async def test_the_abstain_and_no_text_paths_emit_none(monkeypatch):
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", True)
    for text in ("NO_COMPANY_RECORDS", None):
        _seed_index()
        gate = FormatGate(name="format_gate", sub_agents=[_scripted([_valid_dict()])])
        events = [e async for e in gate.run_async(await _gate_ctx(text))]
        assert _rows_events(events) == []


async def test_flag_off_the_gate_yields_todays_events(monkeypatch):
    runs = {}
    for flag in (False, True):
        monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", flag)
        _seed_index()
        gate = FormatGate(name="format_gate", sub_agents=[_scripted([_valid_dict()])])
        runs[flag] = [e async for e in gate.run_async(await _gate_ctx(TEXT))]

    shape = lambda es: [  # noqa: E731
        (e.author, bool(e.partial), e.content.model_dump() if e.content else None) for e in es
    ]
    assert _rows_events(runs[False]) == []
    assert shape(runs[False]) == shape([e for e in runs[True] if e not in _rows_events(runs[True])])


async def test_the_log_line(monkeypatch, caplog):
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", True)
    index = _seed_index()
    gate = FormatGate(name="format_gate", sub_agents=[_scripted([_valid_dict()])])
    ctx = await _gate_ctx(TEXT)
    ctx.session.state["tenant"] = "chevy"
    with caplog.at_level(logging.INFO, logger="gub_agent"):
        [e async for e in gate.run_async(ctx)]

    rows = evidence_rows(index)
    size = len(json.dumps(rows, ensure_ascii=False).encode("utf-8"))
    records = [r for r in caplog.records if r.getMessage().startswith("evidence_rows:")]
    assert [r.getMessage() for r in records] == [
        f"evidence_rows: n={len(rows)} bytes={size} inv=inv-1 tenant=chevy"
    ]
    # Not a format_gate line: `textPayload:"format_gate"` reads the gate's
    # drop and repair rates, and engine lines carry their file name.
    [record] = records
    assert "format_gate" not in record.pathname and "format_gate" not in record.getMessage()


# ── on a real Runner: nothing of it is stored ─────────────────────────────────


class _Executor(BaseAgent):
    """The executor's part, for the gate: this invocation's evidence, then its prose."""

    async def _run_async_impl(self, ctx):
        reset_evidence_index(ctx)
        record_evidence(
            type("T", (), {"name": "get_campaign"})(),
            {},
            ctx,
            {"id": "c1", "name": "Q3 push", "status": "live", "budget": 1200000},
        )
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=genai_types.Content(role="model", parts=[genai_types.Part(text=TEXT)]),
        )


async def _run(flag: bool, monkeypatch) -> tuple[list[Event], list[Event], dict]:
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", flag)
    root = SequentialAgent(
        name="root",
        sub_agents=[
            _Executor(name=AGENT_NAME),
            FormatGate(name="format_gate", sub_agents=[_scripted([_valid_dict()])]),
        ],
    )
    runner = InMemoryRunner(agent=root, app_name="gub")
    session = await runner.session_service.create_session(app_name="gub", user_id="u")
    message = genai_types.Content(role="user", parts=[genai_types.Part(text="how is chevy?")])
    streamed = [
        e async for e in runner.run_async(user_id="u", session_id=session.id, new_message=message)
    ]
    stored = await runner.session_service.get_session(
        app_name="gub", user_id="u", session_id=session.id
    )
    return streamed, list(stored.events), dict(stored.state)


def _stored_shape(events: list[Event]) -> list[tuple]:
    return [
        (
            e.author,
            bool(e.partial),
            e.content.model_dump() if e.content else None,
            json.dumps((e.actions.state_delta or {}) if e.actions else {}, sort_keys=True),
            e.custom_metadata,
        )
        for e in events
    ]


async def test_the_persisted_session_is_identical_with_the_flag_on_and_off(monkeypatch):
    off_stream, off_stored, off_state = await _run(False, monkeypatch)
    on_stream, on_stored, on_state = await _run(True, monkeypatch)

    # Streamed: the one extra event, right after the executor's prose.
    [rows_event] = _rows_events(on_stream)
    assert on_stream.index(rows_event) == 1
    assert _rows_events(off_stream) == []
    assert [e.author for e in on_stream if e is not rows_event] == [e.author for e in off_stream]
    # Stored: nothing of it — events, state and all.
    assert _stored_shape(on_stored) == _stored_shape(off_stored)
    assert on_state == off_state
    assert _rows_events(on_stored) == []
