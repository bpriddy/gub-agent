"""
Provenance must survive the evidence-index cap — replayed from a real turn.

`tests/fixtures/recorded_chevy_turn_2026-09-17.json` is the Chevy account turn
of 2026-09-17 as the Vertex session persisted it (business text redacted,
marker ids and row counts kept): `find` (20 rows), `get_account_overview`
(47 campaign stubs, no statusMarkdown on any; the account's own status under
`currentState.status_markdown`, 345 markers, `_cited` 87), then six
`get_campaign` details with 21–185 markers and `_cited` 6–20.

What happened live: the index hit MAX_ENTRIES on the overview; the six detail
responses indexed NOTHING; 0/400 entries carried a source id; the brief had no
`sources:` line; the formatter cited five `get_account_overview:<cid>:budget`
stub rows and — correctly — copied no ids. Every link the feature promised was
impossible on that turn, with every flag on.

These pin the fix: provenance is recorded UNCAPPED per entity, the stub row the
formatter did cite inherits its campaign's ids, the brief offers them once per
entity, and the gate admits them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

from gub_agent.agents.evidence_index import (
    MAX_ENTRIES,
    entry_source_ids,
    evidence_index,
    provenance,
    record_evidence,
    reset_evidence_index,
)
from gub_agent.agents.format_gate import compose_brief, gate_problems, prune_source_file_ids
from gub_agent.schemas.answer import AnswerPayload, Fact, TextBlock

FIXTURE = Path(__file__).parents[1] / "fixtures" / "recorded_chevy_turn_2026-09-17.json"
CITED_STUB = "get_account_overview:0f1bd315-6c13-4fea-8dd6-6a93bb0fb6da:budget"
CITED_CAMPAIGN = "0f1bd315-6c13-4fea-8dd6-6a93bb0fb6da"
ACCOUNT = "fdb3f9ff-8094-4851-8ad4-a62f82761ad9"


def _replay(invocation_id: str = "chevy") -> tuple[dict, dict]:
    ctx = SimpleNamespace(invocation_id=invocation_id)
    reset_evidence_index(ctx)
    for r in json.loads(FIXTURE.read_text(encoding="utf-8")):
        record_evidence(SimpleNamespace(name=r["name"]), {}, ctx, r["response"])
    return evidence_index(invocation_id), provenance(invocation_id)


def _detail_cited_ids(campaign_id: str) -> list[str]:
    for r in json.loads(FIXTURE.read_text(encoding="utf-8")):
        if r["name"] == "get_campaign" and r["response"]["id"] == campaign_id:
            return list(r["response"]["_cited"].keys())
    raise AssertionError("campaign not in fixture")


async def test_the_recorded_turn_really_does_overflow_the_cap():
    """The premise, pinned so a future MAX_ENTRIES change is a conscious one."""
    index, _ = _replay()
    assert len(index) == MAX_ENTRIES
    assert not any(k.startswith("get_campaign:") for k in index), (
        "the six detail responses were expected to be dropped by the cap; if they now "
        "fit, this test's premise changed — re-read the module docstring"
    )
    assert CITED_STUB in index  # the row the live formatter cited


async def test_provenance_is_recorded_for_rows_the_cap_dropped():
    _, prov = _replay()
    # The campaign the formatter cited was seen twice: as a stub (no status)
    # and as a detail row the index never took. Its provenance is the detail
    # row's `_cited` — backend-resolved, so every id names a real file.
    assert prov[CITED_CAMPAIGN] == _detail_cited_ids(CITED_CAMPAIGN)
    # All six detail campaigns are in the map, not one.
    detail_ids = {
        r["response"]["id"]
        for r in json.loads(FIXTURE.read_text(encoding="utf-8"))
        if r["name"] == "get_campaign"
    }
    assert detail_ids <= set(prov)


async def test_the_account_own_status_is_read_from_its_nested_snake_case_home():
    _, prov = _replay()
    # 345 markers under currentState.status_markdown; `_cited` (87) is
    # preferred because it is the resolvable subset.
    assert len(prov[ACCOUNT]) == 87


async def test_the_cited_stub_row_inherits_its_campaign_ids():
    index, prov = _replay()
    entry = index[CITED_STUB]
    assert entry["source_file_ids"] == []  # the row itself still cites nothing…
    assert entry_source_ids(entry, prov) == _detail_cited_ids(CITED_CAMPAIGN)  # …its entity does


async def test_the_brief_offers_sources_once_per_entity_and_carries_the_rule():
    index, prov = _replay()
    brief = compose_brief("Chevy is in good shape.", index, "", prov)
    lines = brief.splitlines()
    src_lines = [i for i, line in enumerate(lines) if line.strip().startswith("sources:")]
    assert src_lines, "no sources: line — the live failure, verbatim"
    # Under the ENTITY row, not under every field row: the line before each
    # `sources:` is a `- <tool>:<id> = {` entity line, never a `:field =` line.
    for i in src_lines:
        prev = lines[i - 1]
        assert prev.startswith("- ") and " = {" in prev, prev
        assert not re.match(r"^- [a-z_]+:[^:]+:[A-Za-z_]+ = ", prev), prev
    # Once per entity, so the brief stays a brief.
    entities_with_prov = [
        k for k, e in index.items() if e.get("field") is None and entry_source_ids(e, prov)
    ]
    assert len(src_lines) == len(entities_with_prov)
    assert "source_file_ids" in brief  # _SOURCE_RULE joined, because there is something to copy
    assert "EVERY `<tool>:<id>:<field>` row" in brief


async def test_the_gate_admits_inherited_ids_and_still_drops_invented_ones():
    index, prov = _replay()
    ids = _detail_cited_ids(CITED_CAMPAIGN)
    payload = AnswerPayload(
        kind="answer",
        headline="Five campaigns carry budgets",
        blocks=[TextBlock(text="Chevy | Holiday 2026 has a budget of 1500000.")],
        citations=[CITED_STUB],
        facts=[
            Fact(
                evidence_id=CITED_STUB,
                entity_id=CITED_CAMPAIGN,
                field="budget",
                value="1500000",
                source_file_ids=[ids[0], ids[1], "1INVENTEDbyTheModel_00000000000000000"],
            )
        ],
        follow_ups=["What changed on Chevy this week?"],
    )
    pruned, dropped = prune_source_file_ids(payload, index, prov)
    assert dropped == 1
    assert pruned.facts[0].source_file_ids == [ids[0], ids[1]]
    # Without the provenance map the same ids are "unknown" — the live state.
    pruned_blind, dropped_blind = prune_source_file_ids(payload, index)
    assert dropped_blind == 3 and pruned_blind.facts[0].source_file_ids == []
    # And none of this is a rejection.
    assert gate_problems(pruned, index) == []


async def test_reset_clears_provenance_with_the_index():
    ctx = SimpleNamespace(invocation_id="chevy")
    _replay("chevy")
    assert provenance("chevy")
    reset_evidence_index(ctx)
    assert provenance("chevy") == {}
    assert evidence_index("chevy") == {}


async def test_the_template_fallback_still_carries_attribution():
    """The turn a reader is most likely to distrust — three burnt formatter
    attempts, raw rows rendered mechanically. No model is involved, so there is
    no id to invent and nothing to gain by withholding one."""
    from gub_agent.agents.format_gate import template_payload

    index, prov = _replay()
    payload = template_payload("Chevy is in good shape.", index, prov)
    assert payload.facts, "template payload cites nothing at all"
    with_ids = [f for f in payload.facts if f.source_file_ids]
    assert with_ids, "template fallback dropped every source id"
    for fact in with_ids:
        assert set(fact.source_file_ids) <= set(prov.get(fact.entity_id, []))
