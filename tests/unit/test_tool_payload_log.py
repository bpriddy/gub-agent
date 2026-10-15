"""
`tool_payload:` telemetry (latency-05 W1-4) — one line per tool response, from
`evidence_index.record_evidence`, giving the response's size as JSON (`chars`)
and its size after the plumbing keys the executor never reads are dropped
(`seen_chars`). Pure telemetry: the response passes through untouched, and the
line carries neither `format_gate` nor a logger named after it.
"""

from __future__ import annotations

import json
import logging
import re
from types import SimpleNamespace

from gub_agent.agents.evidence_index import record_evidence, reset_evidence_index

EVIDENCE_LOGGER = "gub_agent.agents.evidence_index"

LINE = re.compile(
    r"^tool_payload: tool=(?P<tool>[a-z_]+) chars=(?P<chars>\d+) seen_chars=(?P<seen>\d+)"
    r" inv=(?P<inv>\S+) tenant=(?P<tenant>[a-z0-9-]+)$"
)


def _lines(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage().startswith("tool_payload:")]


def _ctx(inv: str, tenant: str | None = None) -> SimpleNamespace:
    state = {"tenant": tenant} if tenant else {}
    ctx = SimpleNamespace(invocation_id=inv, state=state)
    reset_evidence_index(ctx)
    return ctx


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


RESPONSE = {
    "id": "a1",
    "name": "Acme",
    "status": "live",
    "_sources": [{"fileId": "f" * 33, "name": "deck.pdf"}] * 20,
    "_sourcesTotal": 20,
    "_cited": {"f" * 33: "deck.pdf"},
}


def test_line_format_is_anchored_and_tenant_is_last(caplog):
    ctx = _ctx("inv-tp-1", "acme")
    with caplog.at_level(logging.INFO, logger=EVIDENCE_LOGGER):
        record_evidence(_tool("get_account_overview"), {}, ctx, RESPONSE)
    [line] = _lines(caplog)
    m = LINE.match(line.getMessage())
    assert m, line.getMessage()
    assert m["tool"] == "get_account_overview"
    assert m["inv"] == "inv-tp-1"
    assert m["tenant"] == "acme"
    assert line.getMessage().endswith("tenant=acme")


def test_chars_is_the_raw_json_length_and_seen_chars_drops_the_plumbing(caplog):
    ctx = _ctx("inv-tp-2")
    with caplog.at_level(logging.INFO, logger=EVIDENCE_LOGGER):
        record_evidence(_tool("get_account_overview"), {}, ctx, RESPONSE)
    m = LINE.match(_lines(caplog)[0].getMessage())
    visible = {k: v for k, v in RESPONSE.items() if not k.startswith("_")}
    assert int(m["chars"]) == len(json.dumps(RESPONSE, ensure_ascii=False))
    assert int(m["seen"]) == len(json.dumps(visible, ensure_ascii=False))
    assert int(m["seen"]) < int(m["chars"])


def test_a_turn_without_a_tenant_reads_anomaly(caplog):
    ctx = _ctx("inv-tp-3")
    with caplog.at_level(logging.INFO, logger=EVIDENCE_LOGGER):
        record_evidence(_tool("org_query"), {}, ctx, {"results": [], "total": 0})
    assert LINE.match(_lines(caplog)[0].getMessage())["tenant"] == "anomaly"


def test_every_response_gets_a_line_a_bare_array_included(caplog):
    ctx = _ctx("inv-tp-4", "acme")
    with caplog.at_level(logging.INFO, logger=EVIDENCE_LOGGER):
        record_evidence(_tool("find"), {}, ctx, [{"id": "h1", "name": "x"}])
        record_evidence(_tool("org_query"), {}, ctx, {"results": [], "total": 0})
    tools = [LINE.match(r.getMessage())["tool"] for r in _lines(caplog)]
    assert tools == ["find", "org_query"]


def test_the_response_passes_through_unmodified_and_the_line_never_names_the_gate(caplog):
    ctx = _ctx("inv-tp-5")
    before = json.dumps(RESPONSE, sort_keys=True)
    with caplog.at_level(logging.INFO, logger=EVIDENCE_LOGGER):
        assert record_evidence(_tool("get_account_overview"), {}, ctx, RESPONSE) is None
    assert json.dumps(RESPONSE, sort_keys=True) == before
    [line] = _lines(caplog)
    assert "format_gate" not in line.getMessage()
    assert "format_gate" not in line.name
    assert "format_gate" not in line.pathname
