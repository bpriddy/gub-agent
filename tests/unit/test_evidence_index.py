"""
agents.evidence_index — the turn's citable evidence, built from tool results.

Pins the id scheme the whole contract addresses ("<tool>:<row id>" /
"<tool>:<row id>:<field>" / "<tool>:<key>" for top-level scalars like
org_query's `total`), the shapes of the fleet's tools (results lists, detail
dicts with nested lists, aggregate rows without ids), and the per-pass reset.
"""

from __future__ import annotations

from types import SimpleNamespace

from gub_agent.agents.evidence_index import (
    evidence_index,
    record_evidence,
    reset_evidence_index,
    set_format_feedback,
    take_format_feedback,
)


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _ctx(invocation_id: str = "inv-1") -> SimpleNamespace:
    return SimpleNamespace(invocation_id=invocation_id)


def _fresh(invocation_id: str = "inv-1") -> SimpleNamespace:
    ctx = _ctx(invocation_id)
    reset_evidence_index(ctx)
    return ctx


async def test_org_query_rows_and_total_are_addressable():
    ctx = _fresh()
    record_evidence(
        _tool("org_query"),
        {},
        ctx,
        {
            "results": [
                {"id": "c1", "name": "Q3 push", "status": "live", "budget": 1200000},
            ],
            "total": 7,
            "truncated": False,
        },
    )
    index = evidence_index("inv-1")
    assert index["org_query:c1"]["entity_id"] == "c1"
    assert '"name": "Q3 push"' in index["org_query:c1"]["value"]
    assert index["org_query:c1:budget"] == {
        "tool": "org_query",
        "entity_id": "c1",
        "field": "budget",
        "value": "1200000",
    }
    assert index["org_query:total"]["value"] == "7"


async def test_aggregate_rows_without_ids_get_positional_ids():
    ctx = _fresh()
    record_evidence(
        _tool("org_query"),
        {},
        ctx,
        {"results": [{"accountName": "chevy", "campaignCount": 9}], "total": 1},
    )
    index = evidence_index("inv-1")
    assert index["org_query:results0:campaignCount"]["value"] == "9"
    assert index["org_query:results0"]["entity_id"] is None


async def test_detail_dict_with_nested_lists_indexes_both_levels():
    ctx = _fresh()
    record_evidence(
        _tool("get_account_overview"),
        {},
        ctx,
        {
            "id": "a1",
            "name": "chevy",
            "status": "active",
            "campaigns": [{"id": "c1", "name": "Q3 push", "status": "live"}],
        },
    )
    index = evidence_index("inv-1")
    assert index["get_account_overview:a1:name"]["value"] == "chevy"
    assert index["get_account_overview:c1:status"]["value"] == "live"


async def test_error_responses_and_plumbing_fields_are_not_evidence():
    ctx = _fresh()
    record_evidence(_tool("get_campaign"), {}, ctx, {"error": True, "message": "403"})
    record_evidence(
        _tool("get_campaign"),
        {},
        ctx,
        {"id": "c1", "name": "Q3 push", "_sources": [{"fileId": "f1"}]},
    )
    index = evidence_index("inv-1")
    assert not any("403" in str(e["value"]) for e in index.values())
    assert "get_campaign:c1:_sources" not in index
    assert "_sources" not in index["get_campaign:c1"]["value"]


async def test_reset_clears_index_and_feedback_per_invocation():
    ctx = _fresh()
    record_evidence(_tool("org_query"), {}, ctx, {"results": [{"id": "c1"}], "total": 1})
    set_format_feedback("inv-1", "unknown citation: x")
    assert evidence_index("inv-1")
    assert take_format_feedback("inv-1")

    reset_evidence_index(ctx)
    assert evidence_index("inv-1") == {}
    assert take_format_feedback("inv-1") == ""


async def test_invocations_do_not_bleed_into_each_other():
    ctx1, ctx2 = _fresh("inv-a"), _fresh("inv-b")
    record_evidence(_tool("org_query"), {}, ctx1, {"results": [{"id": "c1"}], "total": 1})
    record_evidence(_tool("org_query"), {}, ctx2, {"results": [{"id": "c2"}], "total": 1})
    assert "org_query:c1" in evidence_index("inv-a")
    assert "org_query:c1" not in evidence_index("inv-b")
