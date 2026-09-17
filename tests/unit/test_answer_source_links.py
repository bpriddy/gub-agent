"""
blend 08 §5.2 — the source ids a fact carries, and what the gate does with them.

Three rules, and they are all about the SAME asymmetry: a link is a nice to
have, an answer is not. So an id the evidence does not support is removed
quietly, never rejected; the gate has three attempts before the turn drops to
a template, and spending one on provenance would trade the answer for a
footnote.
"""

from __future__ import annotations

from types import SimpleNamespace

from gub_agent.agents.evidence_index import (
    evidence_index,
    record_evidence,
    reset_evidence_index,
    source_file_ids,
)
from gub_agent.agents.format_gate import compose_brief, gate_problems, prune_source_file_ids
from gub_agent.schemas.answer import AnswerPayload, Fact, TextBlock

AAA = "1rRAWMbDltWlshELSJb3QELxNiSBylximIAzpNrd1L1w"
BBB = "1RsPYpRiFaV6T1ClhnREl9DC2V4X2qssz_r3FtvbfJQY"

STATUS = (
    "## Context\n"
    f"- The campaign was paused in October 2024. [src: {AAA}]\n"
    "## Transient\n"
    f"- Assets delivered on October 15, 2024 [expires: 2026-11-30] [src: {BBB}]\n"
)


def _indexed() -> dict:
    ctx = SimpleNamespace(invocation_id="inv-links")
    reset_evidence_index(ctx)
    record_evidence(
        SimpleNamespace(name="get_campaign"),
        {},
        ctx,
        {"id": "c1", "name": "EV Social", "budget": 1200000, "statusMarkdown": STATUS},
    )
    return evidence_index("inv-links")


# ── the parser ────────────────────────────────────────────────────────────────


def test_source_file_ids_is_total_and_ordered():
    assert source_file_ids(STATUS) == [AAA, BBB]
    assert source_file_ids(f"- a [src: {AAA}]\n- b [src: {AAA}]") == [AAA]
    assert source_file_ids("- a [expires: 2026-11-30]") == []
    assert source_file_ids("- see [the brief](https://example.com/x)") == []
    assert source_file_ids(None) == []
    assert source_file_ids(1200000) == []


def test_every_index_entry_answers_source_file_ids():
    index = _indexed()
    # The field that CARRIES the markers.
    assert index["get_campaign:c1:statusMarkdown"]["source_file_ids"] == [AAA, BBB]
    # A sibling field of the same entity inherits them — D1(a). A budget cites
    # no document of its own, but the campaign it belongs to does.
    assert index["get_campaign:c1:budget"]["source_file_ids"] == [AAA, BBB]
    # And the whole-entity row, whose own value was compacted.
    assert index["get_campaign:c1"]["source_file_ids"] == [AAA, BBB]
    # Uniform shape: never a missing key.
    assert all("source_file_ids" in e for e in index.values())


# ── the gate ──────────────────────────────────────────────────────────────────


def _payload(source_ids: list[str]) -> AnswerPayload:
    return AnswerPayload(
        kind="answer",
        headline="EV Social is paused",
        blocks=[TextBlock(text="The campaign was paused in October 2024.")],
        citations=["get_campaign:c1:statusMarkdown"],
        facts=[
            Fact(
                evidence_id="get_campaign:c1:statusMarkdown",
                entity_id="c1",
                field="statusMarkdown",
                value=STATUS,
                source_file_ids=source_ids,
            )
        ],
        follow_ups=["What changed on EV Social this week?"],
    )


def test_supported_source_ids_pass_through_untouched():
    index = _indexed()
    payload, dropped = prune_source_file_ids(_payload([AAA, BBB]), index)
    assert dropped == 0
    assert payload.facts[0].source_file_ids == [AAA, BBB]


def test_an_invented_id_is_dropped_not_rejected():
    """The single most important behaviour here. A hallucinated Drive id would
    become `drive.google.com/file/d/<hallucination>` — a 404 published under
    the answer's authority — so it must not survive; but it must also not cost
    the turn its answer."""
    index = _indexed()
    invented = "1INVENTEDbyTheModel_000000000000000000"
    payload, dropped = prune_source_file_ids(_payload([AAA, invented]), index)
    assert dropped == 1
    assert payload.facts[0].source_file_ids == [AAA]
    # Never a gate problem: the payload still passes every check.
    assert gate_problems(payload, index) == []


def test_ids_borrowed_from_another_evidence_row_are_dropped():
    index = _indexed()
    # A real file id, but one THIS fact's evidence row never mentions.
    other = "0AMkVeL1ln2XhUk9PVA"
    payload, dropped = prune_source_file_ids(_payload([other]), index)
    assert dropped == 1
    assert payload.facts[0].source_file_ids == []
    assert gate_problems(payload, index) == []


def test_a_payload_with_no_source_ids_is_returned_unchanged():
    """`dropped == 0` is what tells the gate not to re-emit — an engine that
    never fills the field must not start emitting an extra event."""
    index = _indexed()
    original = _payload([])
    payload, dropped = prune_source_file_ids(original, index)
    assert dropped == 0
    assert payload is original


def test_an_unknown_evidence_id_takes_its_source_ids_with_it():
    index = _indexed()
    payload = _payload([AAA])
    unknown = "get_campaign:NOPE:statusMarkdown"
    payload = payload.model_copy(
        update={
            "citations": [unknown],
            "facts": [payload.facts[0].model_copy(update={"evidence_id": unknown})],
        }
    )
    pruned, dropped = prune_source_file_ids(payload, index)
    assert dropped == 1
    assert pruned.facts[0].source_file_ids == []
    # The unknown CITATION is still a rejection — that check is unchanged.
    assert any("unknown citation" in p for p in gate_problems(pruned, index))


# ── the brief ─────────────────────────────────────────────────────────────────


def test_the_brief_offers_the_ids_only_where_there_are_any():
    brief = compose_brief("EV Social is paused.", _indexed(), "")
    assert f"sources: {AAA}, {BBB}" in brief
    # `name` carries no markers of its own but inherits the row's, so the only
    # entries WITHOUT a sources line are ones whose row cites nothing.
    lines = brief.splitlines()
    assert sum(1 for line in lines if line.strip().startswith("sources:")) > 0
    # The rule the formatter is judged by must be in the brief, not only in the
    # system prompt: the brief is what a sandbox formatter_variant keeps.
    assert "source_file_ids" in brief
    assert "NEVER write one into a sentence" in brief


def test_the_brief_says_nothing_about_sources_when_nothing_cites_a_file():
    ctx = SimpleNamespace(invocation_id="inv-nosrc")
    reset_evidence_index(ctx)
    record_evidence(
        SimpleNamespace(name="org_query"),
        {},
        ctx,
        {"results": [{"id": "c1", "budget": 5}], "total": 1},
    )
    brief = compose_brief("five", evidence_index("inv-nosrc"), "")
    # No evidence line carries one, AND the rule about them is left out: the
    # brief is re-sent on every formatter attempt, so a rule about a field
    # that will stay empty is input cost on the turn's hottest path.
    assert not any(line.strip().startswith("sources:") for line in brief.splitlines())
    assert "source_file_ids" not in brief
