"""
Engine telemetry (BE-1) — lines that make the evidence cap and the answer's
shape measurable from the logs, and the `tenant=` label on the lines that had
none. Nothing here changes what a turn answers.

Pinned:

  * `evidence_index: cap …` — once per gate run, `dropped_by_tool` a SORTED
    `tool:count,…` string, `-` when nothing was refused; the index it counts is
    admitted exactly as before;
  * `answer_shape: …` — once per gate run, when a payload is accepted or the
    template is emitted (deep path and fast path alike), with a retention that
    is the share of the executor's capitalised runs the payload kept;
  * neither new line carries the substring `format_gate` — the gate's drop and
    repair rates are read with `textPayload:"format_gate"` and an engine line
    carries its logger's file name — in its text, its logger name or its file;
  * the five existing lines keep their old text byte for byte and end in
    `tenant=<label>`; the 403/404 shortcut has a line of its own;
  * BE-6 adds two fast-path lines and changes no old one: a 403/404 on an id the
    bot supplied (`bot-supplied id refused …`) and an id that is not a campaign
    (`bot-supplied id is not a campaign …`); the `shortcut` line still counts
    only the refusals the fast path answered itself.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from google.adk.events import Event
from google.genai import types as genai_types

from gub_agent.agents import evidence_index as ei
from gub_agent.agents import fast_path as fp
from gub_agent.agents.evidence_index import (
    capitalised_runs,
    evidence_index,
    log_answer_shape,
    log_cap,
    record_evidence,
    reset_evidence_index,
)
from gub_agent.agents.format_gate import FormatGate
from gub_agent.agents.round_limiter import MAX_ROUNDS, reset_rounds, round_limit
from gub_agent.config import AGENT_NAME
from gub_agent.schemas import AnswerPayload, RouterDecision
from tests.helpers import invocation_ctx

from .test_format_gate import _scripted

INV = "inv-telemetry"
EVIDENCE_LOGGER = "gub_agent.agents.evidence_index"


def _lines(caplog, prefix: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage().startswith(prefix)]


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _seed(inv: str = INV) -> dict:
    """A small believable turn; names the payloads below are grounded in."""
    ctx = SimpleNamespace(invocation_id=inv)
    reset_evidence_index(ctx)
    record_evidence(
        _tool("get_campaign"),
        {},
        ctx,
        {"id": "c1", "name": "Q3 push", "status": "live", "budget": 1200000},
    )
    record_evidence(_tool("org_query"), {}, ctx, {"results": [{"id": "a1", "name": "chevy"}]})
    return evidence_index(inv)


def _answer(**over) -> AnswerPayload:
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


async def _gate_ctx(executor_text: str, tenant: str | None = None):
    state = {"tenant": tenant} if tenant else {}
    return await invocation_ctx(
        state=state,
        invocation_id=INV,
        events=[
            Event(
                invocation_id=INV,
                author=AGENT_NAME,
                content=genai_types.Content(
                    role="model", parts=[genai_types.Part(text=executor_text)]
                ),
            )
        ],
    )


# Capitalised runs: "Chevy", "Q3", "Holiday Rundown" — the payload keeps two.
EXECUTOR = "Chevy is in good shape. Q3 push is live; Holiday Rundown ended."


# ── the cap line ──────────────────────────────────────────────────────────────


def test_cap_line_counts_what_the_cap_refused_by_tool_sorted(monkeypatch, caplog):
    monkeypatch.setattr(ei, "MAX_ENTRIES", 10)
    ctx = SimpleNamespace(invocation_id="inv-cap")
    reset_evidence_index(ctx)
    # 5 rows of {id, name, status}: an entity entry + 2 fields = 3 entries each.
    rows = [{"id": f"r{i}", "name": f"n{i}", "status": "s"} for i in range(5)]
    # Rows 0-2 fill 9 entries; row 3 takes the 10th (its entity entry) and
    # loses both fields (2); row 4 loses entity + both fields (3); the
    # response-level `total` loses 1  =>  org_query 6.
    record_evidence(_tool("org_query"), {}, ctx, {"results": rows, "total": 5})
    # Full: a detail row (entity + 3 fields) loses 4; an overview row loses 2.
    record_evidence(
        _tool("get_campaign"), {}, ctx, {"id": "c", "name": "x", "status": "y", "budget": 1}
    )
    record_evidence(_tool("get_account_overview"), {}, ctx, {"id": "a", "name": "z"})

    with caplog.at_level(logging.INFO, logger=EVIDENCE_LOGGER):
        log_cap("inv-cap", "chevy", evidence_index("inv-cap"))

    [line] = _lines(caplog, "evidence_index: cap")
    assert line.getMessage() == (
        "evidence_index: cap kept=10 dropped=12"
        " dropped_by_tool=get_account_overview:2,get_campaign:4,org_query:6"
        " inv=inv-cap tenant=chevy"
    )


def test_cap_line_says_dash_when_nothing_was_refused(caplog):
    _seed("inv-nodrop")
    with caplog.at_level(logging.INFO, logger=EVIDENCE_LOGGER):
        log_cap("inv-nodrop", "anomaly", evidence_index("inv-nodrop"))
    [line] = _lines(caplog, "evidence_index: cap")
    assert line.getMessage() == (
        "evidence_index: cap kept=6 dropped=0 dropped_by_tool=- inv=inv-nodrop tenant=anomaly"
    )


def test_the_drop_count_resets_with_the_index(monkeypatch, caplog):
    monkeypatch.setattr(ei, "MAX_ENTRIES", 2)
    ctx = SimpleNamespace(invocation_id="inv-reset")
    reset_evidence_index(ctx)
    record_evidence(_tool("org_query"), {}, ctx, {"results": [{"id": "a", "name": "n", "s": "t"}]})
    assert ei.dropped_by_tool("inv-reset") == {"org_query": 1}
    reset_evidence_index(ctx)  # a critic retry's fresh executor pass
    assert ei.dropped_by_tool("inv-reset") == {}


def test_counting_the_drops_does_not_change_what_is_admitted(monkeypatch):
    """The same responses indexed under a cap give the same keys as before the
    counters existed: first-come, whole row or nothing past the cap."""
    monkeypatch.setattr(ei, "MAX_ENTRIES", 5)
    ctx = SimpleNamespace(invocation_id="inv-admit")
    reset_evidence_index(ctx)
    record_evidence(
        _tool("org_query"),
        {},
        ctx,
        {"results": [{"id": "a", "x": "1", "y": "2"}, {"id": "b", "x": "3", "y": "4"}]},
    )
    assert list(evidence_index("inv-admit")) == [
        "org_query:a",
        "org_query:a:x",
        "org_query:a:y",
        "org_query:b",
        "org_query:b:x",
    ]


# ── the shape line ────────────────────────────────────────────────────────────


def test_capitalised_runs_are_distinct_and_ignore_markers_and_links():
    text = (
        "Silverado HD and Red Tag 2026 are live. Red Tag again [src: 1AbCdEf]. "
        "See https://drive.google.com/file/d/1XyZ for Holiday Push."
    )
    assert capitalised_runs(text) == ["Silverado HD", "Red Tag", "See", "Holiday Push"]


def test_shape_line_counts_blocks_words_cites_and_retention(caplog):
    payload = _answer(
        blocks=[
            {"kind": "text", "text": "the Q3 push is live with budget 1200000."},
            {"kind": "bullets", "items": ["Chevy spend is on plan", "two weeks left"]},
            {"kind": "table", "columns": ["name", "status"], "rows": [["a", "b"], ["c", "d"]]},
        ],
        follow_ups=["what is next?"],
        assumptions=["live means active"],
    )
    with caplog.at_level(logging.INFO, logger=EVIDENCE_LOGGER):
        log_answer_shape(INV, "chevy", payload, EXECUTOR)
    [line] = _lines(caplog, "answer_shape:")
    # words: headline 5 + text 8 + bullets 5+3 + assumption 3 + follow-up 3 = 27
    assert line.getMessage() == (
        f"answer_shape: kind=answer words=27 text_blocks=1 bullets=2 table_rows=2 cites=1"
        f" exec_chars={len(EXECUTOR)} retention=0.67 inv={INV} tenant=chevy"
    )


def test_retention_is_a_dash_without_executor_text_or_runs(caplog):
    with caplog.at_level(logging.INFO, logger=EVIDENCE_LOGGER):
        log_answer_shape(INV, "anomaly", _answer(), "")
        log_answer_shape(INV, "anomaly", _answer(), "nothing capitalised here at all")
    messages = [r.getMessage() for r in _lines(caplog, "answer_shape:")]
    assert [m.split(" retention=")[1].split(" ")[0] for m in messages] == ["-", "-"]
    assert "exec_chars=0 " in messages[0]


def test_retention_counts_cells_and_bullets_not_just_text():
    payload = _answer(
        headline="nothing here",
        blocks=[
            {"kind": "bullets", "items": ["Chevy and Q3 are fine"]},
            {"kind": "table", "columns": ["name", "x"], "rows": [["Holiday Rundown", "y"]]},
        ],
    )
    assert ei.retention(EXECUTOR, payload) == "1.00"


def test_retention_ignores_one_letter_runs_and_matches_whole_words_only():
    payload = _answer(
        blocks=[{"kind": "text", "text": "nothing matched; Reduced spend on the iPhone push"}]
    )
    # "A" and "I" are no names; "Red" is not inside "Reduced"; "Phone" is not
    # a run of "iPhone"; Tahoe and Silverado are not kept.
    assert capitalised_runs("A. I found Red, iPhone and Tahoe Silverado") == [
        "Red",
        "Tahoe Silverado",
    ]
    assert ei.retention("A. I found Red, iPhone and Tahoe Silverado", payload) == "0.00"
    assert (
        ei.retention(
            "Reduced spend, Tahoe",
            _answer(blocks=[{"kind": "text", "text": "Reduced spend on the tahoe"}]),
        )
        == "1.00"
    )


def test_retention_does_not_search_the_headline_or_the_column_names():
    payload = _answer(
        headline="Holiday Rundown and Chevy",
        blocks=[{"kind": "table", "columns": ["Holiday Rundown", "Chevy"], "rows": [["x", "y"]]}],
    )
    assert ei.retention("Holiday Rundown and Chevy", payload) == "0.00"


def test_the_cap_does_not_count_a_repeat_of_an_admitted_key_as_dropped(monkeypatch):
    monkeypatch.setattr(ei, "MAX_ENTRIES", 3)
    ctx = SimpleNamespace(invocation_id="inv-repeat")
    reset_evidence_index(ctx)
    row = {"id": "c", "name": "x", "status": "y"}
    record_evidence(_tool("get_campaign"), {}, ctx, row)
    # The same entity again, at the cap: it would only overwrite admitted keys.
    record_evidence(_tool("get_campaign"), {}, ctx, row)
    assert ei.dropped_by_tool("inv-repeat") == {}
    # A different entity is a real loss: entity entry + 2 fields.
    record_evidence(_tool("get_campaign"), {}, ctx, {"id": "d", "name": "x", "status": "y"})
    assert ei.dropped_by_tool("inv-repeat") == {"get_campaign": 3}


async def test_gate_logs_cap_and_shape_once_on_an_accepted_payload(caplog):
    _seed()
    formatter = _scripted([_answer().model_dump(exclude_none=True)])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    with caplog.at_level(logging.INFO):
        [e async for e in gate.run_async(await _gate_ctx(EXECUTOR, "chevy"))]

    [cap] = _lines(caplog, "evidence_index: cap")
    assert cap.getMessage() == (
        f"evidence_index: cap kept=6 dropped=0 dropped_by_tool=- inv={INV} tenant=chevy"
    )
    [shape] = _lines(caplog, "answer_shape:")
    assert shape.getMessage() == (
        "answer_shape: kind=answer words=13 text_blocks=1 bullets=0 table_rows=0 cites=1"
        f" exec_chars={len(EXECUTOR)} retention=0.33 inv={INV} tenant=chevy"
    )


async def test_gate_logs_the_shape_of_the_template_when_it_is_emitted(caplog):
    _seed()
    bad = _answer(
        citations=["org_query:nope"], facts=[{"evidence_id": "org_query:nope", "value": "x"}]
    ).model_dump(exclude_none=True)
    formatter = _scripted([bad])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    with caplog.at_level(logging.INFO):
        events = [e async for e in gate.run_async(await _gate_ctx(EXECUTOR))]

    emitted = AnswerPayload.model_validate(events[-1].actions.state_delta["answer_payload"])
    [shape] = _lines(caplog, "answer_shape:")  # once: the rejected attempts log none
    assert shape.getMessage().startswith(f"answer_shape: kind={emitted.kind} ")
    assert shape.getMessage().endswith(f"inv={INV} tenant=anomaly")
    assert f"cites={len(emitted.citations)} " in shape.getMessage()
    assert len(_lines(caplog, "evidence_index: cap")) == 1


async def test_the_fast_path_gate_run_logs_both_lines_too(caplog):
    """The fast path hands its draft to the same gate, so its accepted payload
    is logged by the same call; exec_chars is the draft's length."""
    _seed()
    from gub_agent.agents.evidence_index import set_answer_draft

    draft = "DETERMINISTIC LOOKUP\nQUESTION: how is chevy?\nRESULT:\nstatus: live"
    set_answer_draft(INV, draft)
    formatter = _scripted([_answer().model_dump(exclude_none=True)])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await invocation_ctx(state={"tenant": "chevy"}, invocation_id=INV)
    with caplog.at_level(logging.INFO):
        [e async for e in gate.run_async(ctx)]
    [shape] = _lines(caplog, "answer_shape:")
    assert f"exec_chars={len(draft)} " in shape.getMessage()
    assert len(_lines(caplog, "evidence_index: cap")) == 1


async def test_no_new_line_carries_the_gate_name(caplog):
    """`textPayload:"format_gate"` is the gate's drop/repair rate query. The
    new lines must not inflate it: not in the text, not in the logger name, and
    not in the file name an engine line is prefixed with."""
    _seed()
    formatter = _scripted([_answer().model_dump(exclude_none=True)])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    with caplog.at_level(logging.INFO):
        [e async for e in gate.run_async(await _gate_ctx(EXECUTOR))]

    new = _lines(caplog, "evidence_index: cap") + _lines(caplog, "answer_shape:")
    assert len(new) == 2
    for record in new:
        assert "format_gate" not in record.getMessage()
        assert "format_gate" not in record.name
        assert "format_gate" not in record.filename
        assert "format_gate" not in record.pathname
        # The formatted record, however a handler lays it out.
        formatted = logging.Formatter(
            "%(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
        ).format(record)
        assert "format_gate" not in formatted


# ── the lines that gained tenant= ─────────────────────────────────────────────


async def test_repaired_and_dropped_lines_keep_their_prefix_and_end_in_tenant(caplog):
    _seed()
    # "c1:status" is the evidence id minus its tool prefix (repairable); the
    # source id is one no row of the turn mentions (pruned, never a rejection).
    payload = _answer(
        citations=["c1:status"],
        facts=[
            {
                "evidence_id": "c1:status",
                "entity_id": "c1",
                "field": "status",
                "value": "live",
                "source_file_ids": ["1NeverSeenByTheTurn0000000000000000"],
            }
        ],
    ).model_dump(exclude_none=True)
    gate = FormatGate(name="format_gate", sub_agents=[_scripted([payload])])
    with caplog.at_level(logging.INFO):
        [e async for e in gate.run_async(await _gate_ctx(EXECUTOR, "chevy"))]

    [repaired] = _lines(caplog, "format_gate: repaired citation(s)")
    assert repaired.getMessage() == (
        f"format_gate: repaired citation(s) (inv={INV}) — c1:status -> get_campaign:c1:status"
        " tenant=chevy"
    )
    [dropped] = _lines(caplog, "format_gate: dropped")
    assert dropped.getMessage() == (
        f"format_gate: dropped 1 unknown source_file_id(s) (inv={INV}) tenant=chevy"
    )
    # Accepted after the repair: the shape line describes the CORRECTED payload.
    [shape] = _lines(caplog, "answer_shape:")
    assert "cites=1 " in shape.getMessage()


async def test_round_limit_hit_keeps_its_prefix_and_ends_in_tenant(caplog):
    ctx = SimpleNamespace(invocation_id="inv-rounds", state={"tenant": "chevy"})
    reset_rounds(ctx)
    with caplog.at_level(logging.WARNING, logger="gub_agent.agents.round_limiter"):
        for _ in range(MAX_ROUNDS + 1):
            round_limit(
                ctx,
                SimpleNamespace(config=SimpleNamespace(tools=["t"]), tools_dict={}, contents=[]),
            )
    [hit] = _lines(caplog, "round_limit: HIT")
    assert hit.getMessage() == (
        f"round_limit: HIT (round {MAX_ROUNDS + 1}, inv=inv-rounds) — tools removed tenant=chevy"
    )


async def test_round_limit_label_defaults_to_anomaly(caplog):
    ctx = SimpleNamespace(invocation_id="inv-rounds2")  # no state at all
    reset_rounds(ctx)
    with caplog.at_level(logging.WARNING, logger="gub_agent.agents.round_limiter"):
        for _ in range(MAX_ROUNDS + 1):
            round_limit(
                ctx,
                SimpleNamespace(config=SimpleNamespace(tools=["t"]), tools_dict={}, contents=[]),
            )
    [hit] = _lines(caplog, "round_limit: HIT")
    assert hit.getMessage().endswith("— tools removed tenant=anomaly")


# ── the fast path's lines ─────────────────────────────────────────────────────


def _decision(**over) -> RouterDecision:
    base = {
        "intent": "campaign_status",
        "confidence": 0.93,
        "entity_surface": "Silverado 2026 Q3",
        "language": "en",
    }
    base.update(over)
    return RouterDecision.model_validate(base)


@pytest.mark.parametrize("status", [403, 404])
async def test_the_403_404_shortcut_has_its_own_line(status, caplog):
    ctx = await invocation_ctx(state={"tenant": "chevy"}, invocation_id=INV, user_text="x")
    shim = fp._ToolShim(ctx)
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.fast_path"):
        lookup = fp._shortcut({"error": True, "status": status}, _decision(), shim)
    assert lookup is not None and lookup.payload is not None
    assert [r.getMessage() for r in caplog.records] == [
        f"fast_path: shortcut status={status} intent=campaign_status inv={INV} tenant=chevy"
    ]


@pytest.mark.parametrize("status", [403, 404])
async def test_a_refused_bot_supplied_id_has_its_own_line_and_no_shortcut_line(status, caplog):
    """BE-6. The `shortcut` line counts the turns the fast path ANSWERED on a
    refusal; a refusal about an id the bot handed over is answered by the deep
    path, so it is logged apart and the old line keeps its meaning."""
    ctx = await invocation_ctx(state={"tenant": "chevy"}, invocation_id=INV, user_text="x")
    shim = fp._ToolShim(ctx)
    decision = _decision(entity_id="c1", entity_type="campaign")
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.fast_path"):
        assert (
            fp._shortcut({"error": True, "status": status}, decision, shim, bot_supplied=True)
            is None
        )
    [line] = [r.getMessage() for r in caplog.records]
    assert line == (
        f"fast_path: bot-supplied id refused status={status} entity_type=campaign"
        f" intent=campaign_status — deep path (inv={INV}) tenant=chevy"
    )
    assert "fast_path: shortcut" not in line


async def test_a_bot_supplied_id_with_any_other_error_logs_the_usual_deep_path_line(caplog):
    ctx = await invocation_ctx(state={"tenant": "chevy"}, invocation_id=INV, user_text="x")
    decision = _decision(entity_id="c1", entity_type="campaign")
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.fast_path"):
        assert (
            fp._shortcut(
                {"error": True, "status": 500}, decision, fp._ToolShim(ctx), bot_supplied=True
            )
            is None
        )
    assert [r.getMessage() for r in caplog.records] == [
        f"fast_path: 500 from the lookup — deep path (inv={INV}) tenant=chevy"
    ]


async def test_the_untyped_id_line_says_dash_and_ends_in_tenant(caplog):
    ctx = await invocation_ctx(
        state={"gub_jwt": "jwt", "tenant": "chevy"}, invocation_id=INV, user_text="x"
    )
    decision = _decision(entity_id="c1")  # an id no prefix vouched for
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.fast_path"):
        assert await fp._campaign(decision, fp._ToolShim(ctx), "x") is None
    assert [r.getMessage() for r in caplog.records] == [
        "fast_path: bot-supplied id is not a campaign entity_type=- intent=campaign_status"
        f" — deep path (inv={INV}) tenant=chevy"
    ]


async def test_the_new_fast_path_lines_never_match_the_format_gate_query(caplog):
    """`textPayload:"format_gate"` is the gate's drop/repair rate query (see
    the module docstring): neither BE-6 line may match it, in its text, its
    logger name or its file."""
    ctx = await invocation_ctx(state={"tenant": "chevy"}, invocation_id=INV, user_text="x")
    shim = fp._ToolShim(ctx)
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.fast_path"):
        fp._shortcut(
            {"error": True, "status": 404},
            _decision(entity_id="c1", entity_type="campaign"),
            shim,
            bot_supplied=True,
        )
        await fp._campaign(_decision(entity_id="c1", entity_type="piece"), shim, "x")
    assert len(caplog.records) == 2
    for record in caplog.records:
        assert "format_gate" not in record.getMessage()
        assert "format_gate" not in record.name
        assert "format_gate" not in record.filename
        assert "format_gate" not in record.pathname
        assert record.getMessage().endswith("tenant=chevy")


async def test_other_statuses_log_only_the_deep_path_line(caplog):
    ctx = await invocation_ctx(state={"tenant": "chevy"}, invocation_id=INV, user_text="x")
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.fast_path"):
        assert fp._shortcut({"error": True, "status": 500}, _decision(), fp._ToolShim(ctx)) is None
    assert [r.getMessage() for r in caplog.records] == [
        f"fast_path: 500 from the lookup — deep path (inv={INV}) tenant=chevy"
    ]


async def test_lookup_raised_and_no_citable_evidence_lines_end_in_tenant(monkeypatch, caplog):
    async def boom(*a, **kw):
        raise RuntimeError("exploded")

    async def empty(decision, shim, question):
        return fp.Lookup(evidence=[("find", {})], tool="find")

    ctx = await invocation_ctx(
        state={"gub_jwt": "jwt", "tenant": "chevy", "router_decision": _decision().model_dump()},
        invocation_id=INV,
        user_text="q",
    )
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.fast_path"):
        monkeypatch.setitem(fp._LOOKUPS, "campaign_status", boom)
        [e async for e in fp.fast_path.run_async(ctx)]
        monkeypatch.setitem(fp._LOOKUPS, "campaign_status", empty)
        [e async for e in fp.fast_path.run_async(ctx)]

    [raised] = _lines(caplog, "fast_path: lookup raised")
    assert raised.getMessage() == f"fast_path: lookup raised (inv={INV}) — deep path tenant=chevy"
    [none] = _lines(caplog, "fast_path: no citable evidence")
    assert none.getMessage() == (
        f"fast_path: no citable evidence (inv={INV}) — deep path tenant=chevy"
    )
