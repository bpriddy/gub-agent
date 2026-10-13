"""
agents.evidence_index — priority-aware admission (BE-2).

The index holds MAX_ENTRIES entries (1 per entity + 1 per scalar field), not
records, and it used to admit first-come. On a Chevy deep turn the order is
always the same — a broad read (`find`, an account overview with 47 campaign
stubs), then four to six `get_campaign` calls in the last round — so the broad
read filled the index and the detail rows, which carry `statusMarkdown`, were
refused. Pinned here:

  * a DETAIL entry (the top-level row of a detail tool, a campaign's pieces, the
    rows of `get_piece`, nested status text) evicts the OLDEST stub FIELD row
    when the index is full, and is refused only when none is left;
  * a stub row is still first-come, never evicts, and its whole-entity row is
    never evicted;
  * the BE-1 cap line stays true: an evicted stub is dropped for ITS tool, and
    `kept` is the post-eviction size;
  * the rows event is the post-eviction index, in index order, in its old shape;
  * the same turn indexes the same way every time.

The six `get_campaign` responses of `recorded_chevy_turn_2026-09-17.json` (the
turn that started this) are replayed in the last tests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from gub_agent.agents import evidence_index as ei
from gub_agent.agents.evidence_index import (
    MAX_ENTRIES,
    dropped_by_tool,
    evidence_index,
    evidence_rows,
    log_cap,
    provenance,
    record_evidence,
    reset_evidence_index,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "recorded_chevy_turn_2026-09-17.json"

SIX_CAMPAIGNS = {
    "T1-1 HD MCP",
    "Chevy | Red Tag 2026",
    "Chevy | Holiday 2026",
    "Chevy | BHAC",
    "Chevy | Q1 2027 SUV",
    "Chevy | See the USA",
}


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _start(monkeypatch, cap: int | None, invocation_id: str = "inv-p") -> SimpleNamespace:
    if cap is not None:
        monkeypatch.setattr(ei, "MAX_ENTRIES", cap)
    ctx = SimpleNamespace(invocation_id=invocation_id)
    reset_evidence_index(ctx)
    return ctx


def _stub_rows(n: int, prefix: str = "r") -> list[dict]:
    return [{"id": f"{prefix}{i}", "name": f"n{i}", "status": f"s{i}"} for i in range(n)]


def _fill_with_stubs(ctx: SimpleNamespace, rows: int = 3) -> None:
    """`rows` org_query rows of {id, name, status} (3 entries each) plus the
    response-level `total`: 3 * rows + 1 entries, six of them stub FIELD rows
    when rows == 3."""
    record_evidence(_tool("org_query"), {}, ctx, {"results": _stub_rows(rows), "total": rows})


# ── admission order ───────────────────────────────────────────────────────────


def test_a_detail_row_evicts_the_oldest_stub_field_rows_when_the_index_is_full(monkeypatch):
    ctx = _start(monkeypatch, 10)
    _fill_with_stubs(ctx)  # 3 rows x 3 + total = 10: full
    assert len(evidence_index("inv-p")) == 10
    status = "Launch is on track. [src: 1DriveFileA]"

    record_evidence(
        _tool("get_campaign"),
        {},
        ctx,
        {"id": "c", "name": "Red Tag", "status": "live", "budget": 900, "statusMarkdown": status},
    )

    index = evidence_index("inv-p")
    assert len(index) == 10
    for key in (
        "get_campaign:c",
        "get_campaign:c:name",
        "get_campaign:c:status",
        "get_campaign:c:budget",
        "get_campaign:c:statusMarkdown",
    ):
        assert key in index, key
    assert index["get_campaign:c:statusMarkdown"]["value"] == status
    assert index["get_campaign:c:statusMarkdown"]["source_file_ids"] == ["1DriveFileA"]
    # Exactly the five OLDEST stub field rows went, in index order…
    assert [k for k in index if k.startswith("org_query:") and k.count(":") == 2] == [
        "org_query:r2:status"
    ]
    # …the stubs' whole-entity rows and the response scalar stayed.
    for key in ("org_query:r0", "org_query:r1", "org_query:r2", "org_query:total"):
        assert key in index, key
    # An evicted stub is a loss for ITS tool, and the detail row lost nothing.
    assert dropped_by_tool("inv-p") == {"org_query": 5}


def test_a_stub_row_is_still_first_come_and_never_evicts(monkeypatch):
    """Unchanged behaviour, pinned: a list result arriving at a full index is
    refused whole, even when stub field rows are sitting in it."""
    ctx = _start(monkeypatch, 10)
    _fill_with_stubs(ctx)
    before = list(evidence_index("inv-p"))

    record_evidence(_tool("org_query"), {}, ctx, {"results": _stub_rows(2, "late")})
    record_evidence(_tool("find"), {}, ctx, {"result": [{"id": "f", "name": "x"}]})

    assert list(evidence_index("inv-p")) == before
    assert dropped_by_tool("inv-p") == {"find": 2, "org_query": 6}


def test_detail_rows_are_never_evicted_by_later_detail_rows(monkeypatch):
    ctx = _start(monkeypatch, 6)
    record_evidence(
        _tool("get_campaign"), {}, ctx, {"id": "a", "name": "A", "status": "live"}
    )  # 3 entries, detail
    record_evidence(
        _tool("org_query"), {}, ctx, {"results": [{"id": "s", "x": "1", "y": "2"}]}
    )  # 3 entries, stub: full

    record_evidence(_tool("get_campaign"), {}, ctx, {"id": "b", "name": "B", "status": "live"})

    index = evidence_index("inv-p")
    # B's entity entry and its first field fit by evicting the stub's two fields;
    # its last field is refused because everything left is detail or an entity.
    assert list(index) == [
        "get_campaign:a",
        "get_campaign:a:name",
        "get_campaign:a:status",
        "org_query:s",
        "get_campaign:b",
        "get_campaign:b:name",
    ]
    assert dropped_by_tool("inv-p") == {"get_campaign": 1, "org_query": 2}


def test_a_repeated_detail_row_neither_evicts_nor_counts(monkeypatch):
    ctx = _start(monkeypatch, 10)
    _fill_with_stubs(ctx)
    row = {"id": "c", "name": "Red Tag", "status": "live"}
    record_evidence(_tool("get_campaign"), {}, ctx, row)
    first = dict(evidence_index("inv-p"))
    drops = dropped_by_tool("inv-p")

    record_evidence(_tool("get_campaign"), {}, ctx, row)

    assert evidence_index("inv-p") == first
    assert dropped_by_tool("inv-p") == drops


def test_the_overview_lists_stubs_but_its_account_row_is_detail(monkeypatch):
    ctx = _start(monkeypatch, 12)
    record_evidence(
        _tool("get_account_overview"),
        {},
        ctx,
        {"id": "acct", "name": "Chevy", "status": "active", "campaigns": _stub_rows(3, "c")},
    )  # account row 3 entries + 3 stubs x 3 = 12: full
    assert len(evidence_index("inv-p")) == 12

    record_evidence(
        _tool("get_campaign"), {}, ctx, {"id": "d", "name": "D", "status": "live", "budget": 5}
    )

    index = evidence_index("inv-p")
    # The oldest stub fields went — the campaign stubs', never the account's.
    for key in ("get_account_overview:acct", "get_account_overview:acct:name"):
        assert key in index, key
    assert "get_account_overview:c0:name" not in index
    assert "get_account_overview:c0:status" not in index
    assert "get_account_overview:c1:name" not in index
    assert "get_account_overview:c1:status" not in index
    assert "get_account_overview:c1" in index  # the stub's entity row stays
    assert all(f"get_campaign:d{suffix}" in index for suffix in ("", ":name", ":status", ":budget"))
    assert dropped_by_tool("inv-p") == {"get_account_overview": 4}


def test_a_campaigns_pieces_are_admitted_with_it_but_stay_evictable(monkeypatch):
    ctx = _start(monkeypatch, 12)
    record_evidence(_tool("org_query"), {}, ctx, {"results": _stub_rows(4)})  # 12: full

    # A campaign and one piece stub: 4 entries. A refusal here would show up as a
    # `get_campaign` loss on every turn whose campaign has pieces.
    record_evidence(
        _tool("get_campaign"),
        {},
        ctx,
        {"id": "c1", "name": "C1", "pieces": [{"id": "p1", "name": "Spot"}]},
    )
    index = evidence_index("inv-p")
    assert all(f"get_campaign:{k}" in index for k in ("c1", "c1:name", "p1", "p1:name"))
    assert "get_campaign" not in dropped_by_tool("inv-p")

    # The next detail row takes the older org_query stub fields first…
    record_evidence(_tool("get_campaign"), {}, ctx, {"id": "c2", "a": "1", "b": "2", "c": "3"})
    index = evidence_index("inv-p")
    assert not any(k.startswith("org_query:r") and k.count(":") == 2 for k in index)
    assert "get_campaign:p1:name" in index
    assert "get_campaign:c2:c" in index

    # …and only when none is left does it take the piece's field row. The piece's
    # entity row and the campaign's own fields stay (the last field is refused).
    record_evidence(_tool("get_campaign"), {}, ctx, {"id": "c3", "x": "1"})
    index = evidence_index("inv-p")
    assert "get_campaign:p1:name" not in index
    assert "get_campaign:p1" in index
    assert "get_campaign:c1:name" in index
    assert "get_campaign:c3" in index
    assert "get_campaign:c3:x" not in index
    assert dropped_by_tool("inv-p") == {"get_campaign": 2, "org_query": 8}


# ── what counts as a detail row ───────────────────────────────────────────────


def test_the_accounts_nested_status_text_is_indexed_whole_as_a_field_row():
    ctx = SimpleNamespace(invocation_id="inv-nested")
    reset_evidence_index(ctx)
    status = "Chevy leads on volume. [src: 1AccountDoc] Spend is pacing. [src: 1Deck]"
    record_evidence(
        _tool("get_account_overview"),
        {},
        ctx,
        {
            "id": "acct",
            "name": "Chevy",
            "currentState": {"industry": "auto", "status_markdown": status},
        },
    )
    index = evidence_index("inv-nested")
    entry = index["get_account_overview:acct:currentState.status_markdown"]
    assert entry["field"] == "currentState.status_markdown"
    assert entry["entity_id"] == "acct"
    assert entry["value"] == status
    assert entry["source_file_ids"] == ["1AccountDoc", "1Deck"]
    # The whole-entity row is scalars only, as before.
    assert "Chevy leads" not in index["get_account_overview:acct"]["value"]


def test_nested_status_text_of_a_stub_row_is_not_indexed():
    ctx = SimpleNamespace(invocation_id="inv-nested-stub")
    reset_evidence_index(ctx)
    record_evidence(
        _tool("org_query"),
        {},
        ctx,
        {"results": [{"id": "x", "name": "X", "currentState": {"status_markdown": "long text"}}]},
    )
    assert list(evidence_index("inv-nested-stub")) == ["org_query:x", "org_query:x:name"]


def test_get_piece_rows_are_indexed_and_admitted_ahead_of_stubs(monkeypatch):
    ctx = _start(monkeypatch, 10)
    _fill_with_stubs(ctx)  # 10: full
    status = "Piece shipped. [src: 1PieceDoc]"

    record_evidence(
        _tool("get_piece"),
        {},
        ctx,
        {
            "piece": {"id": "p1", "name": "Spot", "statusMarkdown": status},
            "campaign": {"id": "c1", "name": "Holiday"},
        },
    )

    index = evidence_index("inv-p")
    assert index["get_piece:p1:statusMarkdown"]["value"] == status
    assert index["get_piece:p1:statusMarkdown"]["entity_id"] == "p1"
    assert index["get_piece:c1:name"]["value"] == "Holiday"
    assert "get_piece" not in dropped_by_tool("inv-p")
    assert dropped_by_tool("inv-p") == {"org_query": 5}


def test_nested_dicts_of_other_tools_are_still_not_rows():
    ctx = SimpleNamespace(invocation_id="inv-wrapper")
    reset_evidence_index(ctx)
    record_evidence(
        _tool("org_query"), {}, ctx, {"wrapped": {"id": "w", "name": "not a row"}, "total": 1}
    )
    assert list(evidence_index("inv-wrapper")) == ["org_query:total"]


def test_eviction_does_not_touch_provenance(monkeypatch):
    ctx = _start(monkeypatch, 6)
    record_evidence(
        _tool("org_query"),
        {},
        ctx,
        {"results": [{"id": "s", "name": "S", "statusMarkdown": "x [src: 1StubFile]"}]},
    )  # entity + 2 fields: 3 entries
    record_evidence(_tool("org_query"), {}, ctx, {"results": [{"id": "t", "a": "1", "b": "2"}]})
    assert len(evidence_index("inv-p")) == 6

    record_evidence(
        _tool("get_campaign"), {}, ctx, {"id": "c", "name": "C", "status": "live"}
    )  # evicts s:name and s:statusMarkdown, then t:a

    index = evidence_index("inv-p")
    assert "org_query:s:statusMarkdown" not in index
    assert provenance("inv-p")["s"] == ["1StubFile"]


# ── the cap line and the rows event ──────────────────────────────────────────


def test_the_cap_line_counts_an_evicted_stub_for_its_tool_and_kept_after_eviction(
    monkeypatch, caplog
):
    ctx = _start(monkeypatch, 10)
    _fill_with_stubs(ctx)
    record_evidence(_tool("find"), {}, ctx, {"result": [{"id": "f", "name": "x"}]})  # refused: 2
    record_evidence(
        _tool("get_campaign"), {}, ctx, {"id": "c", "name": "C", "status": "live", "budget": 1}
    )  # evicts 4

    with caplog.at_level(logging.INFO, logger="gub_agent.agents.evidence_index"):
        log_cap("inv-p", "chevy", evidence_index("inv-p"))

    [line] = [r for r in caplog.records if "evidence_index: cap" in r.getMessage()]
    assert line.getMessage() == (
        "evidence_index: cap kept=10 dropped=6 dropped_by_tool=find:2,org_query:4"
        " inv=inv-p tenant=chevy"
    )
    assert "format_gate" not in line.getMessage()


def test_the_rows_event_is_the_post_eviction_index_in_its_old_shape(monkeypatch):
    ctx = _start(monkeypatch, 10)
    _fill_with_stubs(ctx)
    record_evidence(
        _tool("get_campaign"), {}, ctx, {"id": "c", "name": "C", "status": "live", "budget": 1}
    )
    index = evidence_index("inv-p")

    rows = evidence_rows(index)

    assert [r["evidence_id"] for r in rows] == list(index)
    assert len(rows) == 10
    assert all(set(r) == {"evidence_id", "entity_id", "field", "value"} for r in rows)
    assert "org_query:r0:name" not in {r["evidence_id"] for r in rows}  # evicted, so not sent
    assert next(r for r in rows if r["evidence_id"] == "get_campaign:c:budget") == {
        "evidence_id": "get_campaign:c:budget",
        "entity_id": "c",
        "field": "budget",
        "value": "1",
    }


# ── the recorded Chevy turn ───────────────────────────────────────────────────


def _calls() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _replay(invocation_id: str = "chevy-replay") -> tuple[dict, dict[str, int]]:
    ctx = SimpleNamespace(invocation_id=invocation_id)
    reset_evidence_index(ctx)
    for call in _calls():
        record_evidence(_tool(call["name"]), {}, ctx, call["response"])
    return evidence_index(invocation_id), dropped_by_tool(invocation_id)


def test_each_of_the_six_recorded_get_campaign_responses_indexes_rows():
    index, dropped = _replay()
    campaigns = [c["response"] for c in _calls() if c["name"] == "get_campaign"]
    assert {c["name"] for c in campaigns} == SIX_CAMPAIGNS
    for campaign in campaigns:
        key = f"get_campaign:{campaign['id']}"
        assert key in index, f"{campaign['name']}: no whole-entity row"
        # The status narrative the executor calls its primary evidence, whole.
        assert index[f"{key}:statusMarkdown"]["value"] == campaign["statusMarkdown"]
        assert index[f"{key}:budget"]["value"] == str(campaign["budget"])
    assert "get_campaign" not in dropped  # the live acceptance number
    assert len(index) == MAX_ENTRIES  # the cap still binds: it is a prompt budget


def test_the_recorded_accounts_nested_status_is_indexed():
    index, _ = _replay()
    account = next(c["response"] for c in _calls() if c["name"] == "get_account_overview")
    entry = index[f"get_account_overview:{account['id']}:currentState.status_markdown"]
    assert entry["value"] == account["currentState"]["status_markdown"]


def test_the_recorded_replay_evicts_the_oldest_stubs_first_and_keeps_every_entity_row():
    index, dropped = _replay()
    calls = _calls()
    find_rows = next(c["response"] for c in calls if c["name"] == "find")["result"]
    find_fields = [
        f"find:{row['id']}:{k}"
        for row in find_rows
        for k, v in row.items()
        if k != "id" and isinstance(v, (str, int, float, bool))
    ]
    survivors = [k for k in find_fields if k in index]
    # `find` is the oldest read: what survives of it is a SUFFIX of its field rows.
    assert survivors and len(survivors) < len(find_fields)
    assert survivors == find_fields[-len(survivors) :]
    # The evicted stubs are counted against the tool that wrote them.
    assert dropped["find"] == len(find_fields) - len(survivors)

    # No whole-entity row is ever evicted: every one the broad reads admitted
    # (before any detail response arrived) is still there.
    ctx = SimpleNamespace(invocation_id="broad-only")
    reset_evidence_index(ctx)
    for call in calls[:2]:  # find, get_account_overview
        record_evidence(_tool(call["name"]), {}, ctx, call["response"])
    entity_rows = {k for k, e in evidence_index("broad-only").items() if e["field"] is None}
    assert len(entity_rows) > 40
    assert entity_rows <= set(index)


def test_the_recorded_replay_is_deterministic():
    first_index, first_dropped = _replay("run-1")
    second_index, second_dropped = _replay("run-2")
    assert list(first_index) == list(second_index)
    assert first_index == second_index
    assert first_dropped == second_dropped
    # Insertion order is call order: the six detail rows come in the order the
    # executor fetched them, after the broad reads they were admitted ahead of.
    expected = [
        f"get_campaign:{c['response']['id']}" for c in _calls() if c["name"] == "get_campaign"
    ]
    assert [k for k in first_index if k in expected] == expected
    assert list(first_index).index(expected[0]) > list(first_index).index(
        next(k for k in first_index if k.startswith("get_account_overview:"))
    )


@pytest.mark.parametrize("cap", [60, 150, 399])
def test_the_index_never_exceeds_the_cap_and_never_loses_a_detail_row_it_had(monkeypatch, cap):
    monkeypatch.setattr(ei, "MAX_ENTRIES", cap)
    invocation_id = f"cap-{cap}"
    ctx = SimpleNamespace(invocation_id=invocation_id)
    reset_evidence_index(ctx)
    held: set[str] = set()
    for call in _calls():
        record_evidence(_tool(call["name"]), {}, ctx, call["response"])
        index = evidence_index(invocation_id)
        assert len(index) <= cap
        if call["name"] == "get_campaign":
            prefix = f"get_campaign:{call['response']['id']}"
            held |= {k for k in index if k == prefix or k.startswith(prefix + ":")}
        assert held <= set(index), "a detail row was evicted"
