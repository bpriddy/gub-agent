"""
READER_TEXT_SCRUB (latency-05 W1-5) — no record ids and no raw ISO timestamps in
what a reader sees.

The incident (2026-10-15, "tell me more about acme"): the fast path's draft
copied every scalar of the account record, `id`, `createdAt` and `updatedAt`
among them; the formatter kept those three and wrote "The Acme account
(id: <uuid>) was created on 2026-06-04T18:33:52.846Z and was last updated on
2026-10-15T11:12:22.426Z." The fixtures below carry that payload shape.

Pinned:

  * the format gate's emitted payload has no UUID and no ISO timestamp in its
    headline, text blocks, bullets or table cells, on the accepted path AND the
    template path; `facts`, `citations`, `[evidence id]` markers and the table's
    source-id column are byte-identical;
  * with the flag off the gate emits exactly what it did (one event, the
    formatter's own);
  * `reader_text: scrubbed ids=%d dates=%d inv=%s tenant=%s`, anchored, only
    when something was scrubbed, never from format_gate.py;
  * the fast path's draft leaves a record id out (flag on) and is today's
    otherwise;
  * the formatter prompt states the rule, in a date form the gate grounds
    ("June 4, 2026", never "Jun 4, 2026": the gate reads "Jun" as an ungrounded
    name);
  * the flag's code default is off, and both deploy env files say 1.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gub_agent import config
from gub_agent.agents import answers
from gub_agent.agents.evidence_index import (
    evidence_index,
    record_evidence,
    reset_evidence_index,
)
from gub_agent.agents.fast_path import Lookup, compose_draft
from gub_agent.agents.format_gate import FormatGate, gate_problems
from gub_agent.prompts.formatter import FORMATTER_INSTRUCTION
from gub_agent.schemas import AnswerPayload
from gub_agent.schemas.router import FALLBACK_DECISION

from .test_format_gate import INV, _gate_ctx, _scripted

REPO = Path(__file__).resolve().parents[2]

UUID = "fdb3f9ff-1c2d-4e5f-8a9b-0c1d2e3f4a5b"
CAMPAIGN_UUID = "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"
CREATED = "2026-06-04T18:33:52.846Z"
UPDATED = "2026-10-15T11:12:22.426Z"
ACCOUNT = {
    "id": UUID,
    "name": "Acme",
    "createdAt": CREATED,
    "updatedAt": UPDATED,
    "status": "active",
}
ENTITY = f"get_account_overview:{UUID}"
CREATED_ID = f"{ENTITY}:createdAt"
UPDATED_ID = f"{ENTITY}:updatedAt"

UUID_ANYWHERE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
ISO_ANYWHERE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")

INCIDENT_TEXT = (
    f"The Acme account (id: {UUID}) was created on {CREATED} and was last updated on {UPDATED}."
)
INCIDENT_CLEAN = "The Acme account was created on Jun 4, 2026 and was last updated on Oct 15, 2026."


@pytest.fixture(autouse=True)
def _gate_without_the_rows_event(monkeypatch):
    # The event lists below are pinned without it (the deploy env turns it on).
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", False)


def _seed() -> dict:
    ctx = SimpleNamespace(invocation_id=INV)
    reset_evidence_index(ctx)
    record_evidence(SimpleNamespace(name="get_account_overview"), {}, ctx, dict(ACCOUNT))
    return evidence_index(INV)


def _payload(**over) -> dict:
    base = {
        "kind": "answer",
        "headline": "Acme account",
        "blocks": [{"kind": "text", "text": INCIDENT_TEXT}],
        "citations": [CREATED_ID, UPDATED_ID],
        "facts": [
            {"evidence_id": CREATED_ID, "entity_id": UUID, "field": "createdAt", "value": CREATED},
            {"evidence_id": UPDATED_ID, "entity_id": UUID, "field": "updatedAt", "value": UPDATED},
        ],
    }
    base.update(over)
    return base


def _as_emitted(payload: dict) -> dict:
    """`payload` as the gate serialises it (defaults filled in)."""
    return AnswerPayload.model_validate(payload).model_dump(exclude_none=True)


async def _emitted(payload: dict, executor_text: str = INCIDENT_TEXT):
    """(events, the last payload the gate put on the wire) for a formatter that
    answers `payload` every time."""
    _seed()
    formatter = _scripted([payload])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    events = [e async for e in gate.run_async(await _gate_ctx(executor_text))]
    wire = [json.loads(e.content.parts[0].text) for e in events if e.content]
    return events, wire[-1]


def _flag(monkeypatch, on: bool) -> None:
    # raising=False: the old tree has no such attribute, and the assertions
    # below, not the setattr, are what must fail there.
    monkeypatch.setattr(config, "READER_TEXT_SCRUB", on, raising=False)


# ── the gate's emitted payload ────────────────────────────────────────────────


async def test_incident_payload_loses_the_id_and_the_raw_timestamps(monkeypatch):
    _flag(monkeypatch, True)
    original = _payload()
    events, wire = await _emitted(original)

    # The formatter's own event, then the gate's corrected one (last wins).
    assert [e.author for e in events] == ["formatter", "format_gate"]
    assert wire["blocks"][0]["text"] == INCIDENT_CLEAN
    assert not UUID_ANYWHERE.search(json.dumps([wire["headline"], wire["blocks"]]))
    assert not ISO_ANYWHERE.search(json.dumps([wire["headline"], wire["blocks"]]))
    # Machine fields: byte-identical.
    assert json.dumps(wire["facts"], sort_keys=True) == json.dumps(
        _as_emitted(original)["facts"], sort_keys=True
    )
    assert json.dumps(wire["citations"]) == json.dumps(original["citations"])
    assert events[-1].actions.state_delta["answer_payload"] == wire


async def test_flag_off_the_gate_emits_what_it_always_did(monkeypatch):
    _flag(monkeypatch, False)
    original = _payload()
    events, wire = await _emitted(original)

    assert [e.author for e in events] == ["formatter"]
    assert wire == original
    assert UUID in wire["blocks"][0]["text"] and CREATED in wire["blocks"][0]["text"]


async def test_a_date_inside_a_sentence_keeps_its_final_period(monkeypatch):
    _flag(monkeypatch, True)
    text = f"Acme was created on {CREATED}."
    _, wire = await _emitted(
        _payload(
            blocks=[{"kind": "text", "text": text}],
            citations=[CREATED_ID],
            facts=_payload()["facts"][:1],
        ),
        executor_text=text,
    )
    assert wire["blocks"][0]["text"] == "Acme was created on Jun 4, 2026."


async def test_a_bullet_keeps_its_evidence_id_marker_and_a_table_its_source_column(monkeypatch):
    _flag(monkeypatch, True)
    bullet = f"Acme (id: {UUID}) was created {CREATED} [{CREATED_ID}]"
    table = {
        "kind": "table",
        "columns": ["Field", "Value", "Source"],
        "rows": [
            ["Created", CREATED, CREATED_ID],
            ["Account", UUID, ENTITY],
        ],
    }
    original = _payload(
        blocks=[{"kind": "bullets", "items": [bullet]}, table],
        citations=[CREATED_ID, ENTITY],
        facts=[
            {"evidence_id": CREATED_ID, "entity_id": UUID, "field": "createdAt", "value": CREATED},
            {"evidence_id": ENTITY, "entity_id": UUID, "value": json.dumps(ACCOUNT)},
        ],
    )
    _, wire = await _emitted(original, executor_text=bullet)

    assert wire["blocks"][0]["items"] == [f"Acme was created Jun 4, 2026 [{CREATED_ID}]"]
    assert wire["blocks"][1]["rows"] == [
        ["Created", "Jun 4, 2026", CREATED_ID],  # the source id is not read
        ["Account", "—", ENTITY],  # a cell that was only an id says so
    ]
    assert wire["blocks"][1]["columns"] == table["columns"]
    # The whole-entity fact's value keeps its id: machine field.
    assert wire["facts"] == _as_emitted(original)["facts"]


async def test_the_template_the_gate_falls_back_to_is_scrubbed_too(monkeypatch):
    _flag(monkeypatch, True)
    # An answer the gate rejects three times: it cites an id that is not in the index.
    bad = _payload(
        citations=["org_query:nope"], facts=[{"evidence_id": "org_query:nope", "value": "x"}]
    )
    executor_text = f"{INCIDENT_TEXT}\nStatus: active, updated {UPDATED}"
    events, wire = await _emitted(bad, executor_text=executor_text)

    assert events[-1].author == "format_gate"
    shown = json.dumps([wire["headline"], wire["blocks"]])
    assert not UUID_ANYWHERE.search(shown) and not ISO_ANYWHERE.search(shown)
    assert "Oct 15, 2026" in shown


async def test_no_id_no_timestamp_no_second_event_and_no_log_line(monkeypatch, caplog):
    _flag(monkeypatch, True)
    clean = _payload(blocks=[{"kind": "text", "text": "The Acme account is active."}])
    with caplog.at_level(logging.INFO):
        events, wire = await _emitted(clean, executor_text="The Acme account is active.")
    assert [e.author for e in events] == ["formatter"]
    assert wire == clean
    assert not [r for r in caplog.records if r.getMessage().startswith("reader_text:")]


async def test_log_line_is_anchored_comes_from_answers_and_never_names_the_gate(
    monkeypatch, caplog
):
    _flag(monkeypatch, True)
    with caplog.at_level(logging.INFO):
        await _emitted(_payload())
    [line] = [r for r in caplog.records if r.getMessage().startswith("reader_text:")]
    assert re.fullmatch(
        r"reader_text: scrubbed ids=1 dates=2 inv=inv-1 tenant=anomaly", line.getMessage()
    )
    assert line.name == "gub_agent.agents.answers"
    assert "format_gate" not in line.getMessage() + line.name + line.pathname


# ── the scrub itself: what it will and will not touch ─────────────────────────


def _scrubbed(text: str, **over) -> str:
    base = {
        "kind": "answer",
        "headline": "x",
        "blocks": [{"kind": "text", "text": text}],
        "citations": ["t:a"],
        "facts": [{"evidence_id": "t:a"}],
    }
    base.update(over)
    payload = AnswerPayload.model_validate(base)
    return answers.scrub_reader_text(payload)[0].blocks[0].text


@pytest.mark.parametrize(
    ("text", "want"),
    [
        (f"Acme (id: {UUID}) is live.", "Acme is live."),
        (f"Acme (ID: {UUID}) is live.", "Acme is live."),
        (f"Acme (id = {UUID}) is live.", "Acme is live."),
        (f"Acme id: {UUID} is live.", "Acme is live."),
        (f"Acme id:{UUID} is live.", "Acme is live."),
        (f"Acme {UUID.upper()} is live.", "Acme is live."),
        (f"Acme is live ({UUID}).", "Acme is live."),
        (f"Acme ({UUID}, active) is live.", "Acme (active) is live."),
        ("Created 2026-06-04T18:33Z.", "Created Jun 4, 2026."),
        ("Created 2026-06-04T18:33:52Z.", "Created Jun 4, 2026."),
        ("Created 2026-06-04T18:33:52.846Z.", "Created Jun 4, 2026."),
        ("Created 2026-06-04T18:33:52.846+02:00 sharp.", "Created Jun 4, 2026 sharp."),
        ("Created 2026-06-04T18:33:52-0500 sharp.", "Created Jun 4, 2026 sharp."),
        ("Created 2026-06-04T18:33:52 sharp.", "Created Jun 4, 2026 sharp."),
        ("Created 2026-06-04T23:30:00-05:00.", "Created Jun 4, 2026."),  # day as written
        ("Created 2026-12-01T00:00:00Z.", "Created Dec 1, 2026."),
    ],
)
def test_the_forms_it_removes_and_rewrites(text, want):
    assert _scrubbed(text) == want


@pytest.mark.parametrize(
    "text",
    [
        "Created 2026-06-04.",  # a date-only string names no instant
        "Created 2026-13-04T18:33:52Z.",  # not a date: left as written
        f"See https://app.example.com/accounts/{UUID} for more.",  # a URL
        f"See [the account](https://app.example.com/accounts/{UUID}) for more.",  # a link
        f"Open /accounts/{UUID} next.",  # a path
        f"Acme [{ENTITY}] is live.",  # a citation marker
        "Acme [1] and [2, 3] are live now.",  # source-number markers
        f"Cited as {ENTITY} here.",  # an evidence id outside brackets
        f"Ref x{UUID} stays glued.",  # part of a longer token
    ],
)
def test_what_it_leaves_alone(text):
    assert _scrubbed(text) == text


def test_a_string_that_was_only_an_id_is_not_blanked_but_a_table_cell_says_so():
    assert _scrubbed(UUID) == UUID
    payload = AnswerPayload.model_validate(
        {
            "kind": "answer",
            "headline": "x",
            "blocks": [{"kind": "table", "columns": ["A", "B"], "rows": [[UUID, "ok"]]}],
            "citations": ["t:a"],
            "facts": [{"evidence_id": "t:a"}],
        }
    )
    assert answers.scrub_reader_text(payload)[0].blocks[0].rows == [["—", "ok"]]


def test_the_citations_own_ids_stay_wherever_they_are_written():
    cited = f"odd_tool_name-{UUID}"  # not shaped like an evidence id, but cited
    text = f"Acme ({UUID}) is live; see {cited}."
    out = _scrubbed(text, citations=[cited], facts=[{"evidence_id": cited}])
    assert out == f"Acme is live; see {cited}."


def test_machine_fields_and_unlisted_text_are_untouched_and_nothing_changed_is_the_same_object():
    raw = {
        "kind": "answer",
        "headline": "x",
        "blocks": [{"kind": "text", "text": f"Acme ({UUID}) {CREATED}"}],
        "citations": [CREATED_ID],
        "facts": [{"evidence_id": CREATED_ID, "entity_id": UUID, "value": CREATED}],
        "assumptions": [f"Looked for: {UUID}"],
        "follow_ups": [f"What changed on {UUID}?"],
    }
    payload = AnswerPayload.model_validate(raw)
    out, ids, dates = answers.scrub_reader_text(payload)
    assert (ids, dates) == (1, 1)
    for field in ("facts", "citations", "assumptions", "follow_ups"):
        assert getattr(out, field) == getattr(payload, field)
    assert payload.blocks[0].text.endswith(CREATED)  # the input is not mutated
    clean = AnswerPayload.model_validate({**raw, "blocks": [{"kind": "text", "text": "ok"}]})
    assert answers.scrub_reader_text(clean)[0] is clean


# ── the draft: the source ─────────────────────────────────────────────────────

ROW = {"id": CAMPAIGN_UUID, "name": "Holiday push", "status": "live", "createdAt": CREATED}


def _draft() -> str:
    lookup = Lookup(
        evidence=[
            ("get_account_overview", {**ACCOUNT, "campaigns": [ROW], "ownerId": CAMPAIGN_UUID})
        ],
        tool="get_account_overview",
    )
    return compose_draft("tell me more about acme", FALLBACK_DECISION, lookup)


def test_the_draft_leaves_record_ids_out_and_keeps_every_other_field(monkeypatch):
    _flag(monkeypatch, True)
    draft = _draft()
    assert UUID not in draft and CAMPAIGN_UUID not in draft
    assert f"createdAt: {CREATED}" in draft and "name: Acme" in draft
    assert '"name": "Holiday push"' in draft and '"status": "live"' in draft
    assert "Never write a record id or UUID" in draft.splitlines()[0]


def test_flag_off_the_draft_is_what_it_was(monkeypatch):
    _flag(monkeypatch, False)
    draft = _draft()
    assert f"id: {UUID}" in draft and f"ownerId: {CAMPAIGN_UUID}" in draft
    assert json.dumps(ROW) in draft
    assert "Never write a record id" not in draft


# ── the prompt, the flag, the env ─────────────────────────────────────────────


def test_the_formatter_prompt_states_the_rule_in_a_date_form_the_gate_grounds():
    flat = " ".join(FORMATTER_INSTRUCTION.split())
    assert "Never write a record id or a UUID into prose" in flat
    assert "never as an ISO timestamp" in flat
    # "Jun" is an ungrounded name to the gate; "June" is a month it knows.
    assert '("June 4, 2026")' in flat and "Jun 4" not in flat
    _seed()
    index = evidence_index(INV)
    june = AnswerPayload.model_validate(
        _payload(blocks=[{"kind": "text", "text": "The Acme account was created on June 4, 2026."}])
    )
    assert gate_problems(june, index) == []


def test_the_code_default_is_off():
    env = {k: v for k, v in os.environ.items() if k != "READER_TEXT_SCRUB"}
    out = subprocess.run(
        [sys.executable, "-c", "from gub_agent import config; print(config.READER_TEXT_SCRUB)"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


def _stated(name: str, key: str) -> list[str]:
    return [
        line.partition("=")[2].strip()
        for line in (REPO / name).read_text().splitlines()
        if line.startswith(f"{key}=")
    ]


def test_both_deploy_env_files_state_the_flag_with_one_value():
    prod = _stated("deploy-prod.env", "READER_TEXT_SCRUB")
    sandbox = _stated("deploy-sandbox.env", "READER_TEXT_SCRUB")
    assert len(prod) == 1 and len(sandbox) == 1, (prod, sandbox)
    assert prod == sandbox == ["1"]


# ── what the adversarial review found ─────────────────────────────────────────


def test_an_impossible_date_is_left_as_written_and_is_not_counted():
    payload = AnswerPayload.model_validate(
        {
            "kind": "answer",
            "headline": "x",
            "blocks": [{"kind": "text", "text": "Ref 2026-13-04T18:33:52Z and 2026-02-30T10:00Z."}],
            "citations": ["t:a"],
            "facts": [{"evidence_id": "t:a"}],
        }
    )
    scrubbed, ids, dates = answers.scrub_reader_text(payload)
    assert (ids, dates) == (0, 0) and scrubbed is payload


def test_an_id_label_without_a_colon_goes_with_its_parentheses():
    assert (
        _scrubbed(f"The Acme account (ID {UUID}) was created.") == "The Acme account was created."
    )
    assert _scrubbed(f"Acme (uuid {UUID}) is live.") == "Acme is live."


@pytest.mark.parametrize(
    "text",
    [
        f"No company records found for {UUID}.",
        f"The ID of the Acme account is {UUID}.",
        f"Here it is: {UUID}",
    ],
)
def test_a_sentence_the_id_ended_is_kept_as_written(text):
    """An id the reader asked about, or their own words echoed back, is not a
    leak, and "found for." is worse than an id they can read in their question."""
    assert _scrubbed(text) == text


@pytest.mark.parametrize(
    ("text", "want"),
    [
        (f"Created by Acme (id: {UUID})", "Created by Acme"),
        (f"{UUID} Acme is live.", "Acme is live."),
        (f"Acme is live, **{UUID}**.", "Acme is live."),
        (f"Acme is live, `{UUID}`.", "Acme is live."),
    ],
)
def test_what_an_id_leaves_behind_is_tidied_at_the_edges_and_in_emphasis(text, want):
    assert _scrubbed(text) == want


def test_the_tidy_only_runs_where_an_id_came_out():
    text = "Voulez-vous continuer ? Oui : tout va bien !"  # French spacing is not ours to fix
    assert _scrubbed(text) == text


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("Updated [2026-06-04T18:33:52Z]", "Updated [Jun 4, 2026]"),
        (
            f"Acme [draft (id: {UUID}) created 2026-06-04T18:33Z, see [1]",
            "Acme [draft created Jun 4, 2026, see [1]",
        ),
        (f"Acme [id: {UUID}] is live.", "Acme [] is live."),
    ],
)
def test_bracketed_prose_is_scrubbed_only_a_marker_is_protected(text, want):
    """The format gate treats every `[...]` as a citation marker and never looks
    inside one, so an id written as bracketed prose would otherwise hide there."""
    assert _scrubbed(text) == want


@pytest.mark.parametrize(
    "text",
    [
        "Acme [1] and [2, 3] and [4-6] are live.",
        f"Acme [{ENTITY}] and [{CREATED_ID}] are live.",
        f"See [the account](https://app.example.com/accounts/{UUID}) now.",
        f"See [the account {UUID}](https://app.example.com/a) now.",
    ],
)
def test_the_markers_and_links_stay(text):
    assert _scrubbed(text) == text


@pytest.mark.parametrize(
    "hostile",
    [
        " " * 100_000,
        "\t " * 50_000,
        "[" * 100_000,
        "(id: " * 20_000,
        " " * 60_000 + "(id:",
        "a " * 60_000,
    ],
    ids=["spaces", "tabs", "brackets", "labels", "spaces-then-label", "words"],
)
def test_a_hostile_string_cannot_hold_the_event_loop(hostile):
    """No regex of the scrub is quadratic on a run of spaces, tabs, `[` or
    `(id:`, and a string longer than any answer text is left as written."""
    start = time.monotonic()
    out = answers._Scrub(None).text(hostile)
    assert out == hostile
    assert time.monotonic() - start < 2.0


def test_the_regexes_stay_linear_below_the_length_cap():
    """The cap is a safety net, not the fix: the longest string the scrub does
    search (10,000 characters) of each hostile shape is fast."""
    for shape in (" ", "\t ", "[", "(id: ", f"x {UUID} ", " (", ", ) "):
        text = (shape * (answers._MAX_SCRUB_CHARS // len(shape)))[: answers._MAX_SCRUB_CHARS]
        start = time.monotonic()
        answers._Scrub(None).text(text)
        assert time.monotonic() - start < 1.0, repr(shape)


def test_a_scrub_that_raises_costs_nothing_and_says_so(monkeypatch, caplog):
    _flag(monkeypatch, True)

    def boom(payload):
        raise RuntimeError("boom")

    monkeypatch.setattr(answers, "scrub_reader_text", boom)
    payload = AnswerPayload.model_validate(_payload())
    with caplog.at_level(logging.INFO):
        out, changed = answers.scrub_for_emit(payload, "inv-9", "acme")
    assert out is payload and changed is False
    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("reader_text:")]
    assert line == (
        "reader_text: scrub failed (RuntimeError) — payload emitted as written "
        "inv=inv-9 tenant=acme"
    )


async def test_a_gate_whose_scrub_raises_still_emits_the_formatters_answer(monkeypatch):
    _flag(monkeypatch, True)

    def boom(payload):
        raise RuntimeError("boom")

    monkeypatch.setattr(answers, "scrub_reader_text", boom)
    original = _payload()
    events, wire = await _emitted(original)
    assert [e.author for e in events] == ["formatter"] and wire == original


async def test_the_accepted_path_scrubs_a_payload_once(monkeypatch, caplog):
    """The gate scrubs before it logs the payload's shape, then emits what it
    scrubbed: a second pass over it would be work, and for a few inputs a second
    `reader_text:` line for one payload."""
    _flag(monkeypatch, True)
    calls = []
    real = answers.scrub_reader_text

    def counting(payload):
        calls.append(1)
        return real(payload)

    monkeypatch.setattr(answers, "scrub_reader_text", counting)
    with caplog.at_level(logging.INFO):
        await _emitted(_payload())
    assert len(calls) == 1
    assert len([r for r in caplog.records if r.getMessage().startswith("reader_text:")]) == 1


async def test_the_template_path_logs_the_shape_of_what_ships(monkeypatch):
    """`answer_shape` describes the payload the gate shipped: on the template
    path that is the scrubbed one, as on the accepted path."""
    from gub_agent.agents import format_gate as gate_module

    _flag(monkeypatch, True)
    shapes = []
    real = gate_module.log_answer_shape

    def capture(inv, tenant, payload, executor_text):
        shapes.append(payload)
        return real(inv, tenant, payload, executor_text)

    monkeypatch.setattr(gate_module, "log_answer_shape", capture)
    bad = _payload(
        citations=["org_query:nope"], facts=[{"evidence_id": "org_query:nope", "value": "x"}]
    )
    events, wire = await _emitted(
        bad, executor_text=f"{INCIDENT_TEXT}\nStatus: active, updated {UPDATED}"
    )

    assert events[-1].author == "format_gate"
    [template] = shapes
    shown = json.dumps([template.headline, [b.model_dump() for b in template.blocks]])
    assert not UUID_ANYWHERE.search(shown) and not ISO_ANYWHERE.search(shown)
    assert template.model_dump(exclude_none=True) == wire  # what was logged is what shipped
