"""
fast_path.py — one deterministic lookup instead of the ReAct loop (blend 04).

A `BaseAgent` with NO model call before the formatter. For a FACT intent with
an unambiguous entity the whole answer is one HTTP request, so this path makes
that request itself, feeds the result into the turn's evidence index, and runs
the format gate — the same gate, the same `AnswerPayload`, the same evidence
ids as the deep path. What it skips is the executor's 1-3 model rounds at
MEDIUM thinking and the critic pass after them: 20-45 s of model turns for
information one call already has.

Four choices worth their comments:

- **The tool functions are reused, not reimplemented.** `get_campaign` and
  friends are ordinary async functions with `tool_context: Any = None`
  (`tools/accounts.py:92`), and `_resolve_gub_jwt` only ever touches
  `tool_context.state` (`tools/_client.py:68-80`). `_ToolShim` supplies that
  one field from `ctx.session.state`, so the session's cached `gub_jwt` is
  reused with no re-exchange and a new HTTP layer never comes into existence.
- **The critic does not run here.** Grounding is the format gate's job (in
  code, blend 03) and information sufficiency is guaranteed by construction: a
  deterministic call for a known entity cannot have made the wrong tool
  choice. There is nothing left for the judge to judge.
- **It never guesses an entity.** Resolution mirrors the bot's own rules
  (`gub-gchat-bot/src/entity/resolve.ts`, blend 02) with the same thresholds;
  anything ambiguous goes to the deep path WITHOUT a clarification card — the
  bot may already have spent this turn's one card, and a second question is
  worse than a slower answer.
- **`similar_to` is structurally out of reach.** The `count_or_rank` builder
  resolves names through `/org/search` and filters by id, so it never emits
  the one operator that must be a call's sole filter
  (`tools/org_query.py` docstring). It cannot violate that rule by
  construction rather than by care.

Failure policy: an empty result, an unassemblable query or an unexpected
exception falls through to the deep path ONCE (the executor may still find an
answer); a 403 or a 404 is answered immediately, because paying 30 s of ReAct
to hear the same 403 helps nobody.
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from ..schemas import AnswerPayload
from ..schemas.router import RouterDecision
from ..tools.accounts import get_account_overview, get_campaign
from ..tools.discovery import find
from ..tools.org_query import org_query
from ..tools.staff import get_staff_profile, search_staff
from .answers import no_access_payload, not_found_payload, payload_event
from .evidence_index import (
    evidence_index,
    record_evidence,
    reset_evidence_index,
    set_answer_draft,
)
from .format_gate import format_gate
from .router import decision_from, user_text

logger = logging.getLogger(__name__)

# Blend 02's thresholds, same values, same meaning (the bot's
# CLARIFY_MARGIN_FLOOR / CLARIFY_EXACT_FLOOR defaults). Provisional until
# blend 06 ratifies them.
FAST_MARGIN_FLOOR = 0.15
FAST_EXACT_FLOOR = 0.85

# How much of one tool result the draft quotes. The formatter also receives
# every indexed value under ALLOWED_EVIDENCE, so the draft is orientation, not
# the data itself.
MAX_DRAFT_FIELD_CHARS = 600
MAX_DRAFT_ROWS = 10

# ── the fast path's outcome, for the dispatcher ───────────────────────────────
#
# An async generator cannot return a value alongside its events, and inferring
# "did it answer?" from the event stream would couple the dispatcher to the
# gate's internals. Same in-process, invocation-keyed store as
# `round_limiter.py` (flat `state` writes do not survive between calls).

_OUTCOME: OrderedDict[str, str] = OrderedDict()
_MAX_TRACKED = 256


def _set_outcome(invocation_id: str, outcome: str) -> None:
    _OUTCOME[invocation_id] = outcome
    _OUTCOME.move_to_end(invocation_id)
    while len(_OUTCOME) > _MAX_TRACKED:
        _OUTCOME.popitem(last=False)


def outcome(invocation_id: str) -> str:
    """`"answered"` when the fast path emitted a payload for this invocation,
    `"deep"` otherwise (including "it never ran")."""
    return _OUTCOME.get(invocation_id, "deep")


# ── the tool-context shim ─────────────────────────────────────────────────────


class _ToolShim:
    """The one field the tool functions actually use.

    `_resolve_gub_jwt` reads `state["gub_jwt"]` and, on a cold session,
    exchanges the injected OAuth token and writes the result back
    (`tools/_client.py:68-108`). That write goes to the live session-state
    object and is NOT committed as a state_delta — the same limitation the
    whole codebase works around — so a cold session that then falls through to
    the deep path pays one extra exchange. Correct either way, and the warm
    case (the bot's sessions are long-lived) is the common one.

    `invocation_id` is here for `record_evidence`, which reads it off the tool
    context to key the turn's index.
    """

    def __init__(self, ctx: InvocationContext) -> None:
        self.state = ctx.session.state
        self.invocation_id = ctx.invocation_id


# ── entity resolution ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Resolution:
    """`kind`: "one" (id resolved), "ambiguous", "none". Only "one" carries an
    id — an ambiguous surface is never picked from."""

    kind: str
    entity_id: str | None = None
    name: str | None = None


def resolve_hits(hits: list[dict], surface: str, wanted: tuple[str, ...]) -> Resolution:
    """The bot's disambiguation rules (`entity/resolve.ts:initialDecision`) as
    a pure function over `/org/search` hits, restricted to the types this
    intent can use:

    - one hit of a wanted type → that one;
    - exactly one hit whose NAME equals the surface → that one, whatever the
      margin says;
    - two or more hits with that same exact name → ambiguous (the canonical
      case: three campaigns all called "Silverado");
    - top − second < margin AND top < exact floor → ambiguous;
    - otherwise the top hit — a wide margin or a near-exact top hit means the
      search already has a clear winner.
    """
    typed = [
        h
        for h in hits
        if isinstance(h, dict) and h.get("type") in wanted and isinstance(h.get("id"), str)
    ]
    if not typed:
        return Resolution("none")
    typed.sort(key=lambda h: float(h.get("similarity") or 0.0), reverse=True)

    if len(typed) == 1:
        return Resolution("one", typed[0]["id"], typed[0].get("name"))

    exact = [h for h in typed if str(h.get("name", "")).lower() == surface.lower()]
    if len(exact) == 1:
        return Resolution("one", exact[0]["id"], exact[0].get("name"))
    if len(exact) >= 2:
        return Resolution("ambiguous")

    top = float(typed[0].get("similarity") or 0.0)
    second = float(typed[1].get("similarity") or 0.0)
    if top - second < FAST_MARGIN_FLOOR and top < FAST_EXACT_FLOOR:
        return Resolution("ambiguous")
    return Resolution("one", typed[0]["id"], typed[0].get("name"))


async def _resolve(surface: str | None, wanted: tuple[str, ...], shim: _ToolShim) -> Resolution:
    """Resolve a surface through the existing `find` tool (`GET /org/search`,
    `tools/discovery.py:49`) — the same endpoint, the same access scoping the
    deep path uses."""
    if not surface or not surface.strip():
        return Resolution("none")
    response = await find(surface, shim)
    if not isinstance(response, dict) or response.get("error"):
        return Resolution("none")
    hits = response.get("hits")
    if not isinstance(hits, list):
        hits = response if isinstance(response, list) else []
    return resolve_hits(hits, surface.strip(), wanted)


# ── the org_query builder (count_or_rank) ─────────────────────────────────────

# Status vocabularies are ENTITY-SPECIFIC and a value from the wrong entity
# silently matches nothing (`tools/org_query.py` docstring) — so a status the
# entity does not know is a deep-path question, not a filter.
_STATUS_VALUES = {
    "campaigns": {"pitch", "awarded", "live", "ended", "lost"},
    "accounts": {"active", "inactive", "prospect"},
    "staff": {"active", "on_leave", "former"},
    "pieces": set(),
}

# The only date field the builder will filter on, per entity.
_PERIOD_FIELD = {"campaigns": "awardedAt"}

# The only numeric metric it will rank by, per entity.
_METRIC_FIELDS = {"campaigns": {"budget": "budget"}}

# What the sentence asks FOR. Checked against the user's own words because
# `Slots` has no field for it: a count, a ranking and a sum are three different
# answers built from identical slots.
_COUNT_RE = re.compile(
    r"\b(how many|how much|number of|count of|сколько|количество)\b",
    re.I,
)
_RANK_RE = re.compile(
    r"\b(largest|biggest|highest|greatest|top|most expensive|smallest|lowest|cheapest"
    r"|самый|самая|самое|крупнейш\w*|наибольш\w*|наименьш\w*)\b",
    re.I,
)
# Aggregates this builder cannot express, and enumerations it must not answer
# with a number. Both belong to the deep path.
_AGGREGATE_RE = re.compile(
    r"\b(combined|total(?!\s+of\s+\d)|sum|average|mean|median|суммарн\w*|средн\w*|в сумме|итого)\b",
    re.I,
)
_ENUMERATE_RE = re.compile(
    r"\b(which|what are|list|name the|show me|give me the (?:names|list)"
    r"|какие|какой из|перечисл\w*|назови|покажи)\b",
    re.I,
)


def _asked_shape(question: str) -> str:
    """ "count" | "rank" | "other" — what the sentence wants back.

    Order matters. An enumeration or an aggregate is disqualifying even when
    counting words are also present ("list how many we have per office"), so
    those are tested first; a ranking beats a bare count ("which campaign has
    the largest budget" contains neither "how many" nor a plain list).
    """
    if _AGGREGATE_RE.search(question):
        return "other"
    if _RANK_RE.search(question):
        return "rank"
    if _ENUMERATE_RE.search(question):
        return "other"
    if _COUNT_RE.search(question):
        return "count"
    return "other"


# Fields it will group by, per entity — an FK id group carries its `*Name`
# companion in the result rows, so a group-by needs no follow-up query.
_GROUP_FIELDS = {
    "campaigns": {"status": "status", "account": "accountId", "accountid": "accountId"},
    "accounts": {"status": "status", "industry": "industry"},
    "staff": {"office": "officeId", "officeid": "officeId", "status": "status"},
    "pieces": {"campaign": "campaignId", "campaignid": "campaignId"},
}

_YEAR_RE = re.compile(r"^(\d{4})$")
_QUARTER_RE = re.compile(r"^(?:(\d{4})\s*[- ]?\s*q([1-4])|q([1-4])\s*[- ]?\s*(\d{4}))$")
_QUARTER_RANGES = {
    1: ("01-01", "03-31"),
    2: ("04-01", "06-30"),
    3: ("07-01", "09-30"),
    4: ("10-01", "12-31"),
}


def period_range(text: str) -> list[str] | None:
    """`["YYYY-MM-DD", "YYYY-MM-DD"]` for an explicit year or quarter, else
    None. Deliberately narrow: a relative period ("this year", "недавно")
    needs a judgement about what the user meant, which is the executor's job,
    not a regex's."""
    value = text.strip().lower()
    year = _YEAR_RE.match(value)
    if year:
        return [f"{year.group(1)}-01-01", f"{year.group(1)}-12-31"]
    quarter = _QUARTER_RE.match(value)
    if quarter:
        y = quarter.group(1) or quarter.group(4)
        q = int(quarter.group(2) or quarter.group(3))
        start, end = _QUARTER_RANGES[q]
        return [f"{y}-{start}", f"{y}-{end}"]
    return None


async def org_query_args(
    decision: RouterDecision, shim: _ToolShim, question: str
) -> dict[str, Any] | None:
    """`org_query` kwargs assembled from the router's `slots`, or None → deep
    path.

    Conservative in three layers, because a count that quietly dropped a
    constraint would answer a DIFFERENT question than the one asked — the one
    failure mode worse than being slow:

    1. `slots.complete` must be true — the router's own statement that the
       named fields express the whole sentence (`schemas/router.py`).
    2. `slots.office` is refused outright: an office needs an id this path
       cannot resolve deterministically.
    3. every remaining value must map to a field the NAMED entity actually
       has, with an entity-specific status vocabulary (a status from another
       entity silently matches nothing).
    """
    slots = decision.slots
    if not slots.complete:
        logger.info("fast_path: router says the slots do not cover the question — deep path")
        return None
    if slots.office:
        logger.info("fast_path: office constraint needs an id lookup — deep path")
        return None

    entity = slots.entity
    if entity is None:
        return None

    args: dict[str, Any] = {"entity": entity}
    filters: dict[str, dict[str, Any]] = {}

    if slots.status:
        status = slots.status.strip().lower()
        if status not in _STATUS_VALUES[entity]:
            return None
        filters["status"] = {"eq": status}

    if slots.industry:
        if entity != "accounts":
            return None
        filters["industry"] = {"eq": slots.industry.strip()}

    if slots.period:
        field_name = _PERIOD_FIELD.get(entity)
        window = period_range(slots.period)
        if field_name is None or window is None:
            return None
        filters[field_name] = {"between": window}

    if slots.account:
        if entity not in ("campaigns", "accounts"):
            return None
        # Resolve the name to an id and filter on the id — never `similar_to`,
        # which v1 requires to be a call's SOLE filter.
        resolution = await _resolve(slots.account, ("account",), shim)
        if resolution.kind != "one" or resolution.entity_id is None:
            return None
        filters["accountId" if entity == "campaigns" else "id"] = {"eq": resolution.entity_id}

    if filters:
        args["filter"] = filters

    if slots.group_by:
        field_name = _GROUP_FIELDS[entity].get(slots.group_by.strip().lower())
        if field_name is None:
            return None
        args["group_by"] = [field_name]

    if slots.limit is not None:
        if not 1 <= slots.limit <= 100:
            return None
        args["limit"] = slots.limit

    # What SHAPE of answer the sentence asked for. `slots` cannot tell these
    # apart — "how many campaigns", "which campaigns" and "the combined budget
    # of the campaigns" all arrive as the same entity/status/metric fields — and
    # answering one with another is how this builder produced four confidently
    # wrong answers out of eight in the 2026-09-10 eval: a count for "which
    # accounts do we have", a ranking for "the combined budget".
    shape = _asked_shape(question)
    if shape == "rank":
        if not slots.metric:
            return None
        field_name = _METRIC_FIELDS.get(entity, {}).get(slots.metric.strip().lower())
        if field_name is None:
            return None
        # A ranking: the DB sorts, we take the top rows. NULLs must be excluded
        # or they sort first and the "largest budget" is a row with no budget —
        # which is exactly what `fact-15` returned.
        filters[field_name] = {"is_null": False}
        args["filter"] = filters
        args["sort"] = [{"field": field_name, "direction": "desc"}]
        args.setdefault("limit", 5)
    elif shape == "count":
        if slots.metric:
            # A metric with a counting question is an aggregate ("the total
            # budget"), and this builder cannot express sum/avg — the deep path
            # can. Never serve a ranking in its place.
            return None
        # A count: `total` is the real DB count and the aggregate row makes it
        # citable evidence. Never count rows in Python.
        args["aggregate"] = {"count": {"op": "count"}}
    else:
        # "which/list" wants rows, an aggregate wants sum/avg, and anything
        # unrecognised is a shape this builder has not been shown to get right.
        # All three cost the deep path, per `schemas/router.py`: "an unset flag
        # must cost the deep path, never a confidently wrong count."
        return None

    return args


# ── the lookup ────────────────────────────────────────────────────────────────


@dataclass
class Lookup:
    """What one deterministic lookup produced: tool results to index (in call
    order), or a ready payload for the 403/404 shortcut. `None` from
    `_lookup` means "deep path"."""

    evidence: list[tuple[str, dict]] = field(default_factory=list)
    payload: AnswerPayload | None = None
    tool: str = ""


def _is_error(response: Any) -> bool:
    return isinstance(response, dict) and bool(response.get("error"))


def _shortcut(response: dict, decision: RouterDecision) -> Lookup | None:
    """A 403/404 answered on the spot; any other error → deep path."""
    status = response.get("status")
    if status == 403:
        return Lookup(payload=no_access_payload(decision.language, decision.entity_surface))
    if status == 404:
        return Lookup(payload=not_found_payload(decision.language, decision.entity_surface))
    logger.info("fast_path: %s from the lookup — deep path", status)
    return None


async def _campaign(decision: RouterDecision, shim: _ToolShim, question: str) -> Lookup | None:
    # `entity_id` is the bot's "User selected campaign <uuid>" prefix (blend
    # 02) — a campaign id, so only the campaign intents may use it.
    campaign_id = decision.entity_id
    if not campaign_id:
        resolution = await _resolve(decision.entity_surface, ("campaign",), shim)
        if resolution.kind != "one" or resolution.entity_id is None:
            return None
        campaign_id = resolution.entity_id
    response = await get_campaign(campaign_id, shim)
    if _is_error(response):
        return _shortcut(response, decision)
    if not isinstance(response, dict) or not response.get("id"):
        return None
    return Lookup(evidence=[("get_campaign", response)], tool="get_campaign")


async def _account(decision: RouterDecision, shim: _ToolShim, question: str) -> Lookup | None:
    resolution = await _resolve(decision.entity_surface, ("account",), shim)
    if resolution.kind != "one" or resolution.entity_id is None:
        return None
    response = await get_account_overview(resolution.entity_id, shim)
    if _is_error(response):
        return _shortcut(response, decision)
    if not isinstance(response, dict) or not response.get("id"):
        return None
    return Lookup(evidence=[("get_account_overview", response)], tool="get_account_overview")


async def _staff(decision: RouterDecision, shim: _ToolShim, question: str) -> Lookup | None:
    resolution = await _resolve(decision.entity_surface, ("staff",), shim)
    if resolution.kind == "ambiguous":
        return None
    if resolution.kind == "one" and resolution.entity_id:
        response = await get_staff_profile(resolution.entity_id, shim)
        if _is_error(response):
            return _shortcut(response, decision)
        if not isinstance(response, dict) or not response.get("id"):
            return None
        return Lookup(evidence=[("get_staff_profile", response)], tool="get_staff_profile")

    # Nothing in the cross-entity search: `search_staff` also matches titles,
    # emails and departments, which `/org/search` does not — one more
    # deterministic call before giving the turn to the executor.
    if not decision.entity_surface:
        return None
    response = await search_staff(decision.entity_surface, tool_context=shim)
    if _is_error(response):
        return _shortcut(response, decision)
    rows = response.get("staff") if isinstance(response, dict) else None
    if not rows:
        return None
    return Lookup(evidence=[("search_staff", response)], tool="search_staff")


async def _count_or_rank(decision: RouterDecision, shim: _ToolShim, question: str) -> Lookup | None:
    args = await org_query_args(decision, shim, question)
    if args is None:
        return None
    response = await org_query(**args, tool_context=shim)
    if _is_error(response):
        return _shortcut(response, decision)
    if not isinstance(response, dict) or not response.get("results"):
        return None
    return Lookup(evidence=[("org_query", response)], tool="org_query")


_LOOKUPS = {
    "campaign_status": _campaign,
    "campaign_facts": _campaign,
    "account_facts": _account,
    "staff_lookup": _staff,
    "count_or_rank": _count_or_rank,
}


# ── the draft the format gate renders ─────────────────────────────────────────


def _scalar_lines(row: dict) -> list[str]:
    lines = []
    for key, value in row.items():
        if key.startswith("_") or not isinstance(value, (str, int, float, bool)):
            continue
        text = str(value)
        if len(text) > MAX_DRAFT_FIELD_CHARS:
            text = text[:MAX_DRAFT_FIELD_CHARS] + "…"
        lines.append(f"{key}: {text}")
    return lines


def compose_draft(question: str, decision: RouterDecision, lookup: Lookup) -> str:
    """The gate's input for a fast-path turn — the question and the fields the
    one call returned, copied verbatim. Values are never reformatted here: the
    gate grounds every number against the indexed value, and a prettified
    number would be rejected as ungrounded (correctly)."""
    lines = [
        f"DETERMINISTIC LOOKUP (fast path) — intent={decision.intent}, tool={lookup.tool}. "
        "Answer the question from these fields; nothing else was retrieved.",
        f"QUESTION: {question.strip()}",
        "RESULT:",
    ]
    for _tool, response in lookup.evidence:
        lines.extend(_scalar_lines(response))
        for key, value in response.items():
            if key.startswith("_") or not isinstance(value, list):
                continue
            rows = [item for item in value if isinstance(item, dict)][:MAX_DRAFT_ROWS]
            if not rows:
                continue
            lines.append(f"{key}:")
            for item in rows:
                scalars = {
                    k: v
                    for k, v in item.items()
                    if isinstance(v, (str, int, float, bool)) and not k.startswith("_")
                }
                lines.append(f"- {json.dumps(scalars, ensure_ascii=False, default=str)}")
    return "\n".join(lines)


# ── the agent ─────────────────────────────────────────────────────────────────


class _ToolStub:
    """`record_evidence` reads `tool.name` to prefix the evidence ids
    (`evidence_index.py:record_evidence`) — using the real tool names keeps a
    fast-path `evidence_id` indistinguishable from a deep-path one, which is
    what blend 05's conflict filter and the eval reader rely on."""

    def __init__(self, name: str) -> None:
        self.name = name


class FastPath(BaseAgent):
    """One lookup, then the format gate. Emits nothing at all when it decides
    the question is not its business — the dispatcher reads `outcome()` and
    runs the deep path."""

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        _set_outcome(ctx.invocation_id, "deep")  # pessimistic until a payload ships
        decision = decision_from(ctx)
        shim = _ToolShim(ctx)
        # `Slots` cannot say whether the sentence wanted a count, a list or a
        # sum, so the count_or_rank builder reads the user's own words.
        question = user_text(ctx)

        # The fast path owns this turn's index: the executor's per-pass reset
        # (`agent.py:_before_agent`) has not run and must not be relied on.
        reset_evidence_index(shim)

        lookup_fn = _LOOKUPS.get(decision.intent)
        if lookup_fn is None:
            return
        try:
            lookup = await lookup_fn(decision, shim, question)
        except Exception:
            # Never fail the turn on the fast path — the deep path is always a
            # correct (if slower) answer to the same question.
            logger.exception("fast_path: lookup raised (inv=%s) — deep path", ctx.invocation_id)
            return
        if lookup is None:
            return

        if lookup.payload is not None:
            _set_outcome(ctx.invocation_id, "answered")
            yield payload_event(ctx, lookup.payload)
            return

        for tool_name, response in lookup.evidence:
            record_evidence(_ToolStub(tool_name), {}, shim, response)
        index = evidence_index(ctx.invocation_id)
        if not index:
            # Nothing citable came back — a payload would have to be
            # uncited, which is exactly what the contract forbids.
            logger.info("fast_path: no citable evidence (inv=%s) — deep path", ctx.invocation_id)
            return

        set_answer_draft(ctx.invocation_id, compose_draft(question, decision, lookup))
        _set_outcome(ctx.invocation_id, "answered")
        # The gate is the deep path's own object, invoked here rather than
        # adopted: ADK allows an agent ONE parent (`base_agent.py:135-145`),
        # and `run_async` needs nothing but a context. Same gate, same
        # validators, same retry-then-template guarantee — so the fast path
        # cannot ship a payload the deep path would have rejected.
        async for event in format_gate.run_async(ctx):
            yield event


fast_path = FastPath(name="fast_path")
