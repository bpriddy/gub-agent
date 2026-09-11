"""
agents.fast_path — the deterministic lookup (blend 04).

The pure halves (`resolve_hits`, `org_query_args`, `period_range`,
`compose_draft`) are tested directly. The agent itself runs on a real ADK
InvocationContext with `gub_get` / `gub_post` mocked at the tool modules'
import sites — the same functions the deep path calls, which is the point of
reusing them — and with the format gate replaced by a scripted stand-in
wherever the test is not about the gate itself.
"""

from __future__ import annotations

import json
from importlib import import_module

import pytest
from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

from gub_agent.agents import fast_path as fp
from gub_agent.agents.answers import not_found_payload
from gub_agent.agents.evidence_index import answer_draft, evidence_index
from gub_agent.agents.format_gate import FormatGate, gate_problems
from gub_agent.agents.formatter import ANSWER_STATE_KEY
from gub_agent.schemas import RouterDecision
from tests.helpers import invocation_ctx

INV = "inv-fast"


# ── doubles ───────────────────────────────────────────────────────────────────


class FakeGub:
    """(method, path) → payload, plus a call log. Registered paths match by
    prefix so `/org/campaigns/<uuid>` needs no uuid in the test."""

    def __init__(self) -> None:
        self.routes: list[tuple[str, str, dict]] = []
        self.calls: list[tuple[str, str]] = []

    def on(self, method: str, path: str, payload: dict) -> FakeGub:
        self.routes.append((method, path, payload))
        return self

    def _answer(self, method: str, path: str) -> dict:
        self.calls.append((method, path))
        for m, prefix, payload in self.routes:
            if m == method and path.startswith(prefix):
                return payload
        return {"error": True, "status": 404, "message": "not found"}

    async def get(self, path: str, tool_context=None, **params) -> dict:
        return self._answer("GET", path)

    async def post(self, path: str, body: dict, tool_context=None) -> dict:
        self.calls.append(("BODY", json.dumps(body, sort_keys=True, ensure_ascii=False)))
        return self._answer("POST", path)


@pytest.fixture
def gub(monkeypatch) -> FakeGub:
    """Patch the tool modules' own `gub_get` / `gub_post` bindings — they
    imported the names, so patching `_client` would not be seen.

    Addressed through `import_module`, not a dotted string:
    `gub_agent.tools.org_query` as an ATTRIBUTE is the function re-exported by
    `tools/__init__.py`, which shadows the module of the same name.
    """
    fake = FakeGub()
    for name in ("accounts", "discovery", "staff"):
        monkeypatch.setattr(import_module(f"gub_agent.tools.{name}"), "gub_get", fake.get)
    monkeypatch.setattr(import_module("gub_agent.tools.org_query"), "gub_post", fake.post)
    return fake


class ScriptedGate(BaseAgent):
    """Stand-in for the format gate: records that it ran and emits one payload
    event, the way the real gate's deterministic branch does."""

    runs: list = []

    async def _run_async_impl(self, ctx):
        self.runs.append(ctx.invocation_id)
        payload = {"kind": "answer", "headline": "scripted", "citations": [], "facts": []}
        yield Event(
            invocation_id=ctx.invocation_id,
            author="format_gate",
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=json.dumps(payload))]
            ),
            actions=EventActions(state_delta={ANSWER_STATE_KEY: payload}),
        )


@pytest.fixture
def gate(monkeypatch) -> ScriptedGate:
    scripted = ScriptedGate(name="format_gate", runs=[])
    monkeypatch.setattr(fp, "format_gate", scripted)
    return scripted


def _decision(**over) -> RouterDecision:
    base = {
        "intent": "campaign_status",
        "confidence": 0.93,
        "entity_surface": "Silverado 2026 Q3",
        "language": "ru",
    }
    base.update(over)
    return RouterDecision.model_validate(base)


async def _ctx(decision: RouterDecision | None = None, question: str = "статус Silverado 2026 Q3"):
    """A context carrying the router's decision in state and a JWT in the
    session, so no token exchange is attempted.

    `question` matters for `count_or_rank`: the builder reads the user's own
    words to tell a count from a list or a sum (`_asked_shape`).
    """
    state = {"gub_jwt": "test-jwt"}
    if decision is not None:
        state["router_decision"] = decision.model_dump()
    return await invocation_ctx(state=state, invocation_id=INV, user_text=question)


def _hit(**over) -> dict:
    base = {"type": "campaign", "id": "c1", "name": "Silverado 2026 Q3", "similarity": 0.9}
    base.update(over)
    return base


CAMPAIGN = {
    "id": "c1",
    "name": "Silverado 2026 Q3",
    "status": "live",
    "budget": 1200000,
    "accountName": "chevy",
}


# ── resolve_hits: blend 02's rules ────────────────────────────────────────────


def test_a_single_typed_hit_resolves():
    r = fp.resolve_hits([_hit()], "Silverado 2026 Q3", ("campaign",))
    assert (r.kind, r.entity_id) == ("one", "c1")


def test_an_exact_name_wins_over_a_narrow_margin():
    hits = [
        _hit(id="a", name="Silverado", similarity=0.9),
        _hit(id="b", name="Silverado 2026 Q3", similarity=0.89),
    ]
    r = fp.resolve_hits(hits, "Silverado 2026 Q3", ("campaign",))
    assert (r.kind, r.entity_id) == ("one", "b")


def test_two_hits_with_the_same_exact_name_are_ambiguous():
    hits = [_hit(id="a", name="Silverado", similarity=1.0), _hit(id="b", name="Silverado")]
    assert fp.resolve_hits(hits, "silverado", ("campaign",)).kind == "ambiguous"


def test_a_narrow_margin_below_the_exact_floor_is_ambiguous():
    hits = [
        _hit(id="a", name="EV One", similarity=0.62),
        _hit(id="b", name="EV Two", similarity=0.60),
    ]
    assert fp.resolve_hits(hits, "EV", ("campaign",)).kind == "ambiguous"


def test_a_wide_margin_resolves_to_the_top_hit():
    hits = [
        _hit(id="a", name="EV Everywhere", similarity=0.7),
        _hit(id="b", name="Evergreen", similarity=0.3),
    ]
    r = fp.resolve_hits(hits, "EV", ("campaign",))
    assert (r.kind, r.entity_id) == ("one", "a")


def test_hits_of_the_wrong_type_do_not_count():
    hits = [_hit(type="idea", id="i1", similarity=0.99), _hit(type="staff", id="s1")]
    assert fp.resolve_hits(hits, "Silverado", ("campaign",)).kind == "none"


# ── the org_query builder ─────────────────────────────────────────────────────


def test_period_range_takes_explicit_years_and_quarters_only():
    assert fp.period_range("2026") == ["2026-01-01", "2026-12-31"]
    assert fp.period_range("2026 Q3") == ["2026-07-01", "2026-09-30"]
    assert fp.period_range("Q1 2025") == ["2025-01-01", "2025-03-31"]
    assert fp.period_range("this year") is None  # a judgement, not a regex


async def test_a_count_becomes_an_aggregate_query(gub):
    args = await fp.org_query_args(
        _decision(
            intent="count_or_rank",
            slots={"entity": "campaigns", "status": "live", "complete": True},
        ),
        fp._ToolShim(await _ctx()),
        "how many campaigns are live?",
    )
    assert args == {
        "entity": "campaigns",
        "filter": {"status": {"eq": "live"}},
        "aggregate": {"count": {"op": "count"}},
    }


async def test_a_metric_becomes_a_ranking_with_a_default_limit(gub):
    args = await fp.org_query_args(
        _decision(
            intent="count_or_rank",
            slots={"entity": "campaigns", "metric": "budget", "complete": True},
        ),
        fp._ToolShim(await _ctx()),
        "which campaign has the largest budget?",
    )
    assert args["sort"] == [{"field": "budget", "direction": "desc"}]
    assert args["limit"] == 5
    assert "aggregate" not in args
    # NULL budgets sort FIRST, so a ranking that does not exclude them returns
    # a row with no budget as the "largest" — live regression `fact-15`.
    assert args["filter"]["budget"] == {"is_null": False}


async def test_a_named_account_is_resolved_to_an_id_never_similar_to(gub):
    gub.on("GET", "/org/search", {"hits": [_hit(type="account", id="a1", name="chevy")]})
    args = await fp.org_query_args(
        _decision(
            intent="count_or_rank",
            slots={
                "entity": "campaigns",
                "status": "live",
                "account": "chevy",
                "complete": True,
            },
        ),
        fp._ToolShim(await _ctx()),
        "how many live campaigns does chevy have?",
    )
    # id filter, not `similar_to` — which v1 requires to be a call's SOLE
    # filter and could therefore never combine with the status above.
    assert args["filter"] == {"status": {"eq": "live"}, "accountId": {"eq": "a1"}}
    assert "similar_to" not in json.dumps(args)


@pytest.mark.parametrize(
    "slots",
    [
        # The router itself says the fields do not cover the sentence.
        {"entity": "campaigns", "status": "live", "complete": False},
        # An office needs an id this path cannot resolve deterministically.
        {"entity": "staff", "office": "London", "complete": True},
        {"entity": "campaigns", "status": "active", "complete": True},  # another entity's status
        {"entity": "campaigns", "period": "недавно", "complete": True},  # a relative period
        {"entity": "accounts", "period": "2026", "complete": True},  # no date field for accounts
        {"status": "live", "complete": True},  # no entity named
        {"entity": "campaigns", "metric": "engagement", "complete": True},  # no such field
        {"entity": "campaigns", "limit": 500, "complete": True},  # out of range
        {"entity": "campaigns", "group_by": "region", "complete": True},  # not groupable
        {"entity": "staff", "industry": "auto", "complete": True},  # industry is accounts-only
    ],
)
async def test_slots_that_do_not_assemble_deterministically_go_deep(gub, slots):
    args = await fp.org_query_args(
        _decision(intent="count_or_rank", slots=slots),
        fp._ToolShim(await _ctx()),
        "how many campaigns are live?",
    )
    assert args is None


def test_asked_shape_separates_the_eight_cells_the_eval_measured():
    """The blend-06 eval (2026-09-10) found the fast path served eight FACT
    cells and got four wrong — all `count_or_rank`, and the split is exactly
    by what the sentence asked for. `Slots` cannot see that difference, so the
    builder reads the user's words. This table IS the regression.
    """
    served_correctly = [
        "How many client accounts do we have?",  # fact-02
        "How many campaigns are currently live?",  # fact-04
        "How many active staff members do we have?",  # fact-08
        "How many campaigns are in pitch status?",  # fact-20
    ]
    answered_wrongly = [
        "Which client accounts do we have?",  # fact-01 — got a count
        "Which campaigns are in the awarded status?",  # fact-05 — got a count
        "What is the combined budget of all our campaigns?",  # fact-16 — got a ranking
    ]
    for question in served_correctly:
        assert fp._asked_shape(question) == "count", question
    for question in answered_wrongly:
        # "other" means the deep path, which answered 11 of 11 FACT cells right
        assert fp._asked_shape(question) == "other", question
    # fact-15 stays on the fast path — it IS a ranking — and is correct now
    # only because the builder excludes NULLs (tested above).
    assert fp._asked_shape("Which campaign has the largest budget?") == "rank"


def test_asked_shape_prefers_the_disqualifying_reading():
    """An aggregate or an enumeration wins over counting words in the same
    sentence — the builder can serve neither."""
    assert fp._asked_shape("list how many we have per office") == "other"
    assert fp._asked_shape("what is the total number of campaigns") == "other"
    assert fp._asked_shape("сколько кампаний live?") == "count"
    assert fp._asked_shape("какие кампании live?") == "other"
    assert fp._asked_shape("суммарный бюджет кампаний") == "other"
    assert fp._asked_shape("самая дорогая кампания") == "rank"
    # Nothing recognisable is a deep-path question, never a guessed count.
    assert fp._asked_shape("chevy campaigns") == "other"


# ── the agent ─────────────────────────────────────────────────────────────────


async def test_a_selected_campaign_id_skips_the_search_entirely(gub, gate):
    gub.on("GET", "/org/campaigns/", CAMPAIGN)
    ctx = await _ctx(_decision(entity_id="c1"))

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert [m for m, _ in gub.calls] == ["GET"]  # no /org/search round trip
    assert gub.calls[0][1] == "/org/campaigns/c1"
    assert fp.outcome(INV) == "answered"
    assert gate.runs == [INV]
    assert len(events) == 1


async def test_the_evidence_ids_are_indistinguishable_from_the_deep_paths(gub, gate):
    gub.on("GET", "/org/search", {"hits": [_hit()]}).on("GET", "/org/campaigns/", CAMPAIGN)
    ctx = await _ctx(_decision())

    [e async for e in fp.fast_path.run_async(ctx)]

    index = evidence_index(INV)
    assert "get_campaign:c1" in index  # <tool>:<row id>, exactly as after a tool call
    assert index["get_campaign:c1:status"]["value"] == "live"
    # The resolution step is internal — the search result is not offered as
    # citable evidence.
    assert not any(key.startswith("find:") for key in index)


async def test_the_draft_carries_the_question_and_verbatim_values(gub, gate):
    gub.on("GET", "/org/search", {"hits": [_hit()]}).on("GET", "/org/campaigns/", CAMPAIGN)
    ctx = await _ctx(_decision())

    [e async for e in fp.fast_path.run_async(ctx)]

    draft = answer_draft(INV)
    assert "статус Silverado 2026 Q3" in draft
    assert "status: live" in draft
    assert "budget: 1200000" in draft  # never reformatted — the gate grounds it


async def test_an_ambiguous_surface_goes_deep_with_no_card_and_no_detail_call(gub, gate):
    gub.on(
        "GET",
        "/org/search",
        {
            "hits": [
                _hit(id="a", name="EV One", similarity=0.62),
                _hit(id="b", name="EV Two", similarity=0.60),
            ]
        },
    )
    ctx = await _ctx(_decision(entity_surface="EV"))

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert events == []
    assert fp.outcome(INV) == "deep"
    assert gate.runs == []
    assert [path for _, path in gub.calls] == ["/org/search"]


async def test_an_empty_detail_result_falls_through_to_the_deep_path(gub, gate):
    gub.on("GET", "/org/search", {"hits": [_hit()]}).on("GET", "/org/campaigns/", {})
    ctx = await _ctx(_decision())

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert events == []
    assert fp.outcome(INV) == "deep"


async def test_a_403_is_answered_immediately_without_the_gate(gub, gate):
    gub.on("GET", "/org/campaigns/", {"error": True, "status": 403, "message": "denied"})
    ctx = await _ctx(_decision(entity_id="c1"))

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert fp.outcome(INV) == "answered"
    assert gate.runs == []  # no formatter LLM to hear the same 403
    assert len(events) == 1
    payload = events[0].actions.state_delta[ANSWER_STATE_KEY]
    assert payload["kind"] == "answer"
    assert "Нет доступа" in payload["headline"]
    assert events[0].author == "format_gate"  # the bot's answer channel


async def test_a_404_names_what_was_looked_for_in_the_assumption_line(gub, gate):
    gub.on("GET", "/org/campaigns/", {"error": True, "status": 404, "message": "nope"})
    ctx = await _ctx(_decision(entity_id="c9"))

    events = [e async for e in fp.fast_path.run_async(ctx)]

    payload = events[0].actions.state_delta[ANSWER_STATE_KEY]
    assert payload["assumptions"] == ["Искал: Silverado 2026 Q3"]
    # The surface rides in `assumptions` precisely so the gate's grounding
    # checks pass against an EMPTY index — the headline names no entity.
    assert gate_problems(not_found_payload("ru", "Silverado 2026 Q3"), {}) == []


async def test_a_500_falls_through_to_the_deep_path(gub, gate):
    gub.on("GET", "/org/campaigns/", {"error": True, "status": 500, "message": "boom"})
    ctx = await _ctx(_decision(entity_id="c1"))

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert events == []
    assert fp.outcome(INV) == "deep"


async def test_an_unexpected_exception_costs_the_deep_path_not_the_turn(monkeypatch, gate):
    async def boom(*a, **kw):
        raise RuntimeError("httpx exploded")

    monkeypatch.setattr(import_module("gub_agent.tools.accounts"), "gub_get", boom)
    ctx = await _ctx(_decision(entity_id="c1"))

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert events == []
    assert fp.outcome(INV) == "deep"


async def test_staff_falls_back_to_search_staff_when_the_cross_search_misses(gub, gate):
    gub.on("GET", "/org/search", {"hits": []}).on(
        "GET", "/org/staff", {"staff": [{"id": "s1", "name": "Alex Kim", "title": "strategist"}]}
    )
    ctx = await _ctx(_decision(intent="staff_lookup", entity_surface="Alex"))

    [e async for e in fp.fast_path.run_async(ctx)]

    assert fp.outcome(INV) == "answered"
    assert "search_staff:s1" in evidence_index(INV)


async def test_a_count_answers_from_total_not_from_row_arithmetic(gub, gate):
    gub.on("POST", "/org/query", {"results": [{"count": 7}], "total": 1})
    ctx = await _ctx(
        _decision(
            intent="count_or_rank",
            entity_surface=None,
            slots={"entity": "campaigns", "status": "live", "complete": True},
        ),
        question="сколько кампаний сейчас live?",
    )

    [e async for e in fp.fast_path.run_async(ctx)]

    assert fp.outcome(INV) == "answered"
    assert "org_query:results0:count" in evidence_index(INV)


# ── the invariant: nothing ships without the gate ─────────────────────────────


class ScriptedFormatter(BaseAgent):
    """The formatter LLM, scripted — emits whatever payload the test gives it
    the way `output_schema` does (JSON text + state_delta)."""

    payload: dict = {}
    runs: list = []

    async def _run_async_impl(self, ctx):
        self.runs.append(ctx.invocation_id)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=json.dumps(self.payload))]
            ),
            actions=EventActions(state_delta={ANSWER_STATE_KEY: self.payload}),
        )


async def test_the_real_gate_renders_the_fast_path_draft_and_grounds_it(gub, monkeypatch):
    """The fast path's payload goes through the SAME gate as the deep path's:
    the gate reads the draft (there is no executor text to read) and its
    grounding checks run against the fast path's evidence index."""
    gub.on("GET", "/org/search", {"hits": [_hit()]}).on("GET", "/org/campaigns/", CAMPAIGN)
    formatter = ScriptedFormatter(
        name="formatter",
        payload={
            "kind": "answer",
            "headline": "Silverado 2026 Q3 — live",
            "blocks": [{"kind": "text", "text": "budget 1200000."}],
            "citations": ["get_campaign:c1:status"],
            "facts": [
                {
                    "evidence_id": "get_campaign:c1:status",
                    "entity_id": "c1",
                    "field": "status",
                    "value": "live",
                }
            ],
        },
        runs=[],
    )
    monkeypatch.setattr(fp, "format_gate", FormatGate(name="format_gate", sub_agents=[formatter]))
    ctx = await _ctx(_decision())

    events = [e async for e in fp.fast_path.run_async(ctx)]

    assert formatter.runs == [INV]  # one formatter pass, no retry: it grounded
    assert [e.author for e in events] == ["formatter"]
    assert fp.outcome(INV) == "answered"
