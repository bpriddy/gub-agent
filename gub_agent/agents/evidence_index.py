"""
evidence_index.py — after_tool_callback building the turn's citable evidence.

Every tool result the executor receives is broken into addressable rows:

    "<tool>:<row id>"          — a whole entity (value = its scalar fields)
    "<tool>:<row id>:<field>"  — one field of it
    "<tool>:<key>"             — a top-level response scalar (org_query `total`)

The formatter may cite ONLY these ids; the format gate verifies
`citations ⊆ index` and grounds every number and entity name in the answer
against the indexed values (`agents/format_gate.py`). This is "GROUND EVERY
ENTITY" moved from prompt prose into code.

State lives in an in-process dict keyed on invocation_id — flat
`callback_context.state` writes do not survive between calls (see the
docstring in `sandbox.py:29-36`), which is exactly why `round_limiter.py`
keeps the same shape. Reset runs in the executor's `_before_agent`
(`agent.py`), next to `reset_rounds` / `reset_tool_budget`: the index is
per-executor-pass, so a critic-requested retry grounds against what THAT pass
retrieved (the executor's own doctrine is to re-query, never to answer from
prior turns).

The gate's retry feedback travels the same way (`set_format_feedback` /
`take_format_feedback`): it also goes out as a `state_delta` event for trace
visibility, but the formatter's next run reads it from here, deterministically.

`set_answer_draft` is the fast path's entry point into the same machinery
(blend 04): the fast path has no executor prose for the gate to render, so it
leaves its deterministic draft here and the gate prefers it over the
executor's last text. It is cleared by the reset below, which means a fast
path that gave up and fell through to the deep path cannot leave its draft
behind for the executor's pass to render instead of its own answer.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Any

# The per-bullet Drive provenance the status-synthesis prompt writes after
# every bullet of a `statusMarkdown` (blend 08 §5.2). 47/47 campaigns carry
# these, 689 markers over 232 distinct files, and until now nothing read them.
#
# Parsed off the FIELD row, never the whole-entity row: `_compact_row` caps a
# row at MAX_ENTITY_VALUE_CHARS, 30 of the 47 campaigns have a status longer
# than that, and only 52.8 % of the markers fall inside the cap. The entity
# row's marker set is the head of the document, which is worse than none —
# it would attribute late claims to early files.
SRC_MARKER_RE = re.compile(r"\[src:\s*([A-Za-z0-9_-]+)\]")

# Rows are compact (scalar fields only), but the formatter instruction carries
# the whole index — cap the entries so a runaway fan-out cannot balloon the
# prompt. 400 comfortably covers MAX_TOOL_CALLS=8 × 50-row default limits.
MAX_ENTRIES = 400
# A whole-entity value is the row's scalars as JSON; cap it so statusMarkdown
# and friends do not turn one row into a page. Field-level values stay whole —
# they are what numbers and names are grounded against.
MAX_ENTITY_VALUE_CHARS = 2_000

_INDEX: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
# entity_id → Drive file ids its status text cites — UNCAPPED, beside the
# capped index (blend 08 §5.2, second attempt). Why a second store: MAX_ENTRIES
# protects the formatter BRIEF, and on the first live account turn it evicted
# exactly the rows that carry provenance. Replayed from the recorded session
# (tests/fixtures/recorded_chevy_turn_2026-09-17.json): `find` → 113 entries,
# `get_account_overview` (47 campaign stubs × 10 scalars) → cap hit at 400,
# then six `get_campaign` responses carrying 21–185 markers each indexed ZERO
# rows. 0/400 entries had a source id; the brief had no `sources:` line; the
# model copied nothing, correctly. A set of ids per entity is a few hundred
# bytes and needs no cap; keeping it here lets the stub row the formatter DID
# cite (`get_account_overview:<cid>:budget`) inherit the provenance the
# detail row it never saw would have carried.
_PROVENANCE: OrderedDict[str, dict[str, list[str]]] = OrderedDict()
_FEEDBACK: OrderedDict[str, str] = OrderedDict()
_BRIEF: OrderedDict[str, str] = OrderedDict()
_DRAFT: OrderedDict[str, str] = OrderedDict()
_MAX_TRACKED = 256


def _bucket(store: OrderedDict, invocation_id: str, default: Any) -> Any:
    value = store.get(invocation_id)
    if value is None:
        value = default
        store[invocation_id] = value
    store.move_to_end(invocation_id)
    while len(store) > _MAX_TRACKED:
        store.popitem(last=False)
    return value


def reset_evidence_index(callback_context: Any) -> None:
    """ADK before_agent_callback: clear this invocation's index (and any stale
    format feedback) so each executor pass grounds against its own results —
    see module docstring."""
    invocation_id = getattr(callback_context, "invocation_id", "") or "?"
    _INDEX.pop(invocation_id, None)
    _PROVENANCE.pop(invocation_id, None)
    _FEEDBACK.pop(invocation_id, None)
    _BRIEF.pop(invocation_id, None)
    _DRAFT.pop(invocation_id, None)
    return None


def evidence_index(invocation_id: str) -> dict[str, dict[str, Any]]:
    """This invocation's index (empty when no tool has returned yet)."""
    return _INDEX.get(invocation_id, {})


def provenance(invocation_id: str) -> dict[str, list[str]]:
    """entity_id → source file ids, over EVERY row any tool returned this
    invocation — including rows the index cap dropped."""
    return _PROVENANCE.get(invocation_id, {})


def entry_source_ids(entry: dict[str, Any], prov: dict[str, list[str]]) -> list[str]:
    """The file ids a fact citing this entry may carry: the entry's own, then
    its entity's (any tool, any row, cap or no cap). Deduped, order kept."""
    entity_id = entry.get("entity_id")
    inherited = prov.get(entity_id, []) if isinstance(entity_id, str) else []
    out: list[str] = []
    seen: set[str] = set()
    for file_id in [*(entry.get("source_file_ids") or []), *inherited]:
        if file_id not in seen:
            seen.add(file_id)
            out.append(file_id)
    return out


def set_format_feedback(invocation_id: str, feedback: str) -> None:
    _bucket(_FEEDBACK, invocation_id, "")
    _FEEDBACK[invocation_id] = feedback


def take_format_feedback(invocation_id: str) -> str:
    """The gate's feedback for the formatter's retry, if any (read, not popped:
    the retry may itself be retried and the gate overwrites each round)."""
    return _FEEDBACK.get(invocation_id, "")


def set_formatter_brief(invocation_id: str, brief: str) -> None:
    """The formatter's whole model input for this invocation — the executor's
    answer + ALLOWED_EVIDENCE — composed by the format gate before it runs the
    formatter (`agents/format_gate.py`) and injected as request CONTENT by the
    formatter's before_model_callback (`agents/formatter.py`). Content, not
    instruction, so a sandbox `formatter_variant` can replace the prompt
    without losing the data."""
    _bucket(_BRIEF, invocation_id, "")
    _BRIEF[invocation_id] = brief


def formatter_brief(invocation_id: str) -> str:
    return _BRIEF.get(invocation_id, "")


def set_answer_draft(invocation_id: str, text: str) -> None:
    """The text the format gate should render for this invocation INSTEAD of
    the executor's last message (blend 04). The fast path
    (`agents/fast_path.py`) writes the deterministic result of its one lookup
    here; nothing else writes it, so on the deep path the gate reads the
    executor exactly as before.

    Not an event: emitting the draft would put engine-internal prose on the
    wire, and the bot renders any author outside its answer channel as the
    user's bubble text (`gub-gchat-bot/src/agent/client.ts:209-217`)."""
    _bucket(_DRAFT, invocation_id, "")
    _DRAFT[invocation_id] = text


def answer_draft(invocation_id: str) -> str:
    return _DRAFT.get(invocation_id, "")


def record_evidence(tool: Any, args: dict, tool_context: Any, tool_response: Any) -> None:
    """ADK after_tool_callback on the executor: index this result's rows.
    Returns None so the tool response passes through unmodified."""
    invocation_id = getattr(tool_context, "invocation_id", "") or "?"
    if not isinstance(tool_response, dict):
        return None
    index = _bucket(_INDEX, invocation_id, {})
    prov = _bucket(_PROVENANCE, invocation_id, {})
    tool_name = getattr(tool, "name", None) or str(tool)
    # Provenance BEFORE the index, over every row: it must not depend on
    # whether the row survives MAX_ENTRIES.
    _record_provenance(tool_response, prov)
    _index_response(tool_name, tool_response, index)
    return None


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _compact_row(row: dict[str, Any]) -> str:
    """The row's scalar fields as one JSON line — what a whole-entity citation
    carries as its value (and what entity names are grounded against)."""
    scalars = {k: v for k, v in row.items() if _is_scalar(v) and not k.startswith("_")}
    text = json.dumps(scalars, ensure_ascii=False, default=str)
    if len(text) > MAX_ENTITY_VALUE_CHARS:
        text = text[:MAX_ENTITY_VALUE_CHARS]
    return text


def source_file_ids(text: Any) -> list[str]:
    """The `[src: <driveFileId>]` markers in a value, in order and deduped.

    Total by design: a non-string, an empty string or a value with no markers
    all give []. An entity with no Drive provenance is the common case (every
    `org_query` row), not an error."""
    if not isinstance(text, str) or "[src:" not in text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for file_id in SRC_MARKER_RE.findall(text):
        if file_id not in seen:
            seen.add(file_id)
            out.append(file_id)
    return out


# Where a row's status text lives on the wire. Campaign detail: top-level
# `statusMarkdown`. Account detail: NESTED and snake_case —
# `currentState.status_markdown` (345 markers on the recorded Chevy turn, all
# invisible to a reader of `row["statusMarkdown"]`).
_STATUS_KEYS = ("statusMarkdown", "status_markdown")


def _row_source_ids(row: dict[str, Any]) -> list[str]:
    """Every Drive file id this row's status text cites, wherever the wire put
    it. Prefers the backend's `_cited` map when present: it is the marker set
    already resolved against `drive_file_snapshots`, i.e. ONLY files the bot
    can name — so nothing offered here can render as nothing. Falls back to
    parsing markers (an older backend, or a row without `_cited`)."""
    cited = row.get("_cited")
    if isinstance(cited, dict) and cited:
        return [k for k in cited if isinstance(k, str)]
    out: list[str] = []
    seen: set[str] = set()

    def take(text: Any) -> None:
        for file_id in source_file_ids(text):
            if file_id not in seen:
                seen.add(file_id)
                out.append(file_id)

    for key in _STATUS_KEYS:
        take(row.get(key))
    for value in row.values():
        if isinstance(value, dict):
            for key in _STATUS_KEYS:
                take(value.get(key))
    return out


def _record_provenance(response: dict[str, Any], prov: dict[str, list[str]]) -> None:
    """entity_id → source ids for every id-bearing row in the response, top
    level and every top-level list. Uncapped; unions across tools, so a
    campaign seen as an overview stub and again as a detail row ends up with
    the detail row's ids on both."""

    def note(row: dict[str, Any]) -> None:
        entity_id = row.get("id")
        if not isinstance(entity_id, str):
            return
        ids = _row_source_ids(row)
        if not ids:
            return
        have = prov.setdefault(entity_id, [])
        for file_id in ids:
            if file_id not in have:
                have.append(file_id)

    if response.get("error"):
        return
    if isinstance(response.get("id"), str):
        note(response)
    for key, value in response.items():
        if key.startswith("_") or not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                note(item)


def _index_row(tool_name: str, row: dict[str, Any], fallback_id: str, index: dict) -> None:
    if len(index) >= MAX_ENTRIES:
        return
    row_id = row.get("id") if isinstance(row.get("id"), str) else fallback_id
    entity_id = row.get("id") if isinstance(row.get("id"), str) else None
    # Every source id this ROW mentions, whichever field carried it — so a
    # whole-entity citation can be attributed even though its own value was
    # compacted. D1(a): per-field granularity, which OVER-attributes a
    # multi-file status to any fact citing it. D1(b) — one evidence row per
    # bullet — is exact and costs an evidence-index rewrite; it is unratified,
    # and this shape is the one that ships without it.
    row_sources = _row_source_ids(row)
    index[f"{tool_name}:{row_id}"] = {
        "tool": tool_name,
        "entity_id": entity_id,
        "field": None,
        "value": _compact_row(row),
        "source_file_ids": row_sources,
    }
    for key, value in row.items():
        if len(index) >= MAX_ENTRIES:
            return
        if key == "id" or key.startswith("_") or not _is_scalar(value):
            continue
        index[f"{tool_name}:{row_id}:{key}"] = {
            "tool": tool_name,
            "entity_id": entity_id,
            "field": key,
            "value": str(value),
            # A field row carries the markers of its OWN value when it has
            # them (statusMarkdown), and otherwise the row's — a `budget`
            # field cites no file of its own, but the campaign it belongs to
            # does, and that is the honest attribution available at D1(a).
            "source_file_ids": source_file_ids(value) or row_sources,
        }


def _index_response(tool_name: str, response: dict[str, Any], index: dict) -> None:
    """Break one tool response into rows.

    Shapes in the fleet: `org_query` → {results: [...], total, truncated}
    (rows may be entity rows with `id` or aggregate/group rows without one);
    list tools → {accounts|staff|hits|ideas: [...]}; detail tools → the entity
    dict itself, with nested lists (an overview's `campaigns`, a campaign's
    `pieces`). Handled generically: the top-level dict is a row when it has an
    `id`; every top-level list of dicts contributes rows; remaining top-level
    scalars (org_query's `total`) index as "<tool>:<key>".
    """
    if response.get("error"):
        return  # an error body is not evidence

    if isinstance(response.get("id"), str):
        _index_row(tool_name, response, "self", index)

    row_counter = 0
    for key, value in response.items():
        if key.startswith("_"):
            continue  # _sources and friends are attribution plumbing, not evidence
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _index_row(tool_name, item, f"{key}{row_counter}", index)
                    row_counter += 1
        elif (
            _is_scalar(value)
            and not key.startswith("_")
            and not isinstance(response.get("id"), str)  # already indexed as row fields
        ):
            if len(index) < MAX_ENTRIES:
                index[f"{tool_name}:{key}"] = {
                    "tool": tool_name,
                    "entity_id": None,
                    "field": key,
                    "value": str(value),
                    # Uniform shape: every entry answers `source_file_ids`, so
                    # no reader has to guess whether the key is missing or the
                    # list is empty. A response-level scalar cites no document.
                    "source_file_ids": [],
                }
