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

The index is capped at MAX_ENTRIES and ADMISSION IS BY PRIORITY. A DETAIL row
is an entity the executor fetched on purpose — the top-level row of
`get_account_overview` (the account), `get_campaign` and `get_staff_profile`,
and the `piece` / `campaign` rows of `get_piece` — and its status text
(`statusMarkdown`, or the account's nested `currentState.status_markdown`) is
what an assessment is written from. A STUB is a row that came back as a list
member of a search or list (`find`, `org_query`, the list tools), of an overview
(its campaign stubs) or of a campaign (its `pieces`: name and dates, no status).
The one list a detail response carries as content is a profile's `metadata`:
its members are MEMBER rows, admitted like detail, but their field rows stay
evictable.

Detail rows are admitted ahead of stubs, status text first. When the index is
full a detail or member entry evicts the oldest stub FIELD row (a stub's
whole-entity row stays, so the entity is still named), then the oldest member
field row, and is refused only when none is left. A stub, and a
response-level scalar, is first-come as before. So an overview read before the
detail fetches cannot starve them, whatever the order of the turn.
`_PROVENANCE` is separate and uncapped.

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
import logging
import re
from collections import OrderedDict
from typing import Any

from ..schemas.answer import AnswerPayload, BulletBlock, TableBlock, TextBlock, count_words

logger = logging.getLogger(__name__)

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
# prompt. The cap counts ENTRIES, not records: a record costs one whole-entity
# entry plus one per scalar field (a campaign stub ≈ 10-12), so 400 entries is
# about 33-40 records, not the 8 × 50 rows of MAX_TOOL_CALLS × the default limit
# (an account overview alone is ~47 stubs). That is why admission is by
# priority (see the module docstring) and not first-come.
MAX_ENTRIES = 400
# The tools whose top-level row is a DETAIL row (module docstring).
DETAIL_TOOLS = frozenset({"get_account_overview", "get_campaign", "get_piece", "get_staff_profile"})
# `get_piece` answers `{piece, campaign}`: no top-level `id` and no list, so its
# rows are the nested dicts. Only this wrapper shape is read that way.
_WRAPPED_DETAIL_TOOLS = frozenset({"get_piece"})

# What a row may do when the index is full, and what may be done to its fields.
_STUB = 0  # first-come, whole row or nothing; its fields are evictable
_MEMBER = 1  # admitted ahead of stubs (evicts one); its fields are evictable, after the stubs'
_DETAIL = 2  # admitted ahead of stubs (evicts one); its fields are never evicted
# The lists a detail response carries ABOUT ITSELF, and what each member is. The
# tool contract calls a campaign's `pieces` STUBS (name, dates, no status), so
# they never take a slot from a detail row; a profile's `metadata` (skills,
# interests, certifications) is the person's own content. Every other list of
# any response is a stub list, counted under the tool's own name; an own list is
# counted as "<tool>.<list>", so `dropped_by_tool.get_campaign` is only ever
# DETAIL entries lost.
_OWN_LISTS: dict[str, dict[str, int]] = {
    "get_campaign": {"pieces": _STUB},
    "get_staff_profile": {"metadata": _MEMBER},
}
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
# Entries the cap refused, per invocation and tool: what `log_cap` reports. Reset
# with the index, so it describes the same executor pass the index does.
_DROPPED: OrderedDict[str, dict[str, int]] = OrderedDict()
_MAX_TRACKED = 256


class _EvictionPool:
    """The index keys a detail entry may evict when the index is full, oldest
    first, in two ranks: stub FIELD rows, then member field rows. Each key maps
    to the label its loss is counted under. Per invocation, reset with the
    index; an entity row is never in it."""

    __slots__ = ("members", "stubs")

    def __init__(self) -> None:
        self.stubs: dict[str, str] = {}
        self.members: dict[str, str] = {}

    def forget(self, key: str) -> None:
        self.stubs.pop(key, None)
        self.members.pop(key, None)

    def pop_oldest(self) -> tuple[str, str] | None:
        for rank in (self.stubs, self.members):
            if rank:
                key = next(iter(rank))
                return key, rank.pop(key)
        return None


_EVICTABLE: OrderedDict[str, _EvictionPool] = OrderedDict()


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
    _DROPPED.pop(invocation_id, None)
    _EVICTABLE.pop(invocation_id, None)
    return None


def evidence_index(invocation_id: str) -> dict[str, dict[str, Any]]:
    """This invocation's index (empty when no tool has returned yet)."""
    return _INDEX.get(invocation_id, {})


def evidence_rows(index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The index as the bot's early claim filter reads a cited fact
    (EVIDENCE_ROWS_EVENT, the format gate's rows event): one row per entry, in
    index order, keyed by the entry's evidence id — the index KEY, which is
    what a citation names. NEW dicts, never the entries themselves: a held
    speculative event is deep-copied only in its content and actions
    (speculation.py:_as_yielded), and `tool` / `source_file_ids` are left out
    because the filter reads neither."""
    return [
        {
            "evidence_id": evidence_id,
            "entity_id": entry.get("entity_id"),
            "field": entry.get("field"),
            "value": entry.get("value"),
        }
        for evidence_id, entry in index.items()
    ]


def log_evidence_rows(invocation_id: str, tenant: str, rows: list[dict[str, Any]]) -> None:
    """The rows event's one line. Logged HERE, not in format_gate.py: engine
    lines carry their file name, and `textPayload:"format_gate"` is how the
    gate's drop and repair rates are read — a line on every gate run would
    inflate every one of them."""
    logger.info(
        "evidence_rows: n=%d bytes=%d inv=%s tenant=%s",
        len(rows),
        len(json.dumps(rows, ensure_ascii=False).encode("utf-8")),
        invocation_id,
        tenant,
    )


def dropped_by_tool(invocation_id: str) -> dict[str, int]:
    """Entries the cap refused or evicted this invocation, per tool (empty =
    none). An evicted stub counts against ITS tool, not the detail row's, and
    the members of a detail response's own list count as "<tool>.<list>"
    (`get_campaign.pieces`), so a `get_campaign` count is only ever a detail
    entry the index had no room for."""
    return dict(_DROPPED.get(invocation_id, {}))


def log_cap(invocation_id: str, tenant: str, index: dict[str, dict[str, Any]]) -> None:
    """The cap's one line per gate run: how many entries the index kept (after
    any eviction) and how many MAX_ENTRIES refused or evicted, by tool.
    `dropped_by_tool` is a sorted `tool:count,…` string, `-` when nothing was
    refused or evicted.

    Like `log_evidence_rows`, logged from THIS module: an engine line carries
    its file name, and `textPayload:"format_gate"` is how the gate's drop and
    repair rates are read."""
    dropped = dropped_by_tool(invocation_id)
    logger.info(
        "evidence_index: cap kept=%d dropped=%d dropped_by_tool=%s inv=%s tenant=%s",
        len(index),
        sum(dropped.values()),
        ",".join(f"{tool}:{count}" for tool, count in sorted(dropped.items())) or "-",
        invocation_id,
        tenant,
    )


# A run of capitalised words — the proper nouns (campaign, model and account
# names) a rundown is made of. Latin and Cyrillic; digits and hyphens ride
# along inside a word ("Red Tag", "Q1", "V8").
# A run starts at a word boundary ("iPhone" is not "Phone") and a run shorter
# than two characters ("A", "I") is not a name.
_CAP_RUN_RE = re.compile(
    r"(?<![\w'’-])[A-ZА-ЯЁ][\w'’-]*(?:[ \t]+[A-ZА-ЯЁ][\w'’-]*)*",
)
_MIN_RUN_CHARS = 2
# What is not prose in an executor answer: source markers and links (a Drive
# id can open with a capital).
_NOT_PROSE_RE = re.compile(r"\[src:[^\]]*\]|https?://\S+")


def capitalised_runs(text: str) -> list[str]:
    """The distinct capitalised runs of `text`, in order of first appearance."""
    seen: dict[str, None] = {}
    for run in _CAP_RUN_RE.findall(_NOT_PROSE_RE.sub(" ", text)):
        if len(run) >= _MIN_RUN_CHARS:
            seen.setdefault(run, None)
    return list(seen)


def _payload_text(payload: AnswerPayload) -> str:
    """The payload's text blocks, bullets and table cells (not the headline or
    the column names)."""
    parts: list[str] = []
    for block in payload.blocks:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, BulletBlock):
            parts.extend(block.items)
        elif isinstance(block, TableBlock):
            for row in block.rows:
                parts.extend(row)
    return "\n".join(parts)


def retention(executor_text: str, payload: AnswerPayload) -> str:
    """The share of the executor's distinct capitalised runs that appear in the
    payload (2 decimals), or `-` when there is no executor text or no run. A
    run counts as kept when it occurs in the payload text as a whole word run,
    case-insensitively ("Red" is not found inside "Reduced")."""
    runs = capitalised_runs(executor_text or "")
    if not runs:
        return "-"
    haystack = _payload_text(payload).casefold()
    kept = sum(
        1 for run in runs if re.search(rf"(?<!\w){re.escape(run.casefold())}(?!\w)", haystack)
    )
    return f"{kept / len(runs):.2f}"


def log_answer_shape(
    invocation_id: str, tenant: str, payload: AnswerPayload, executor_text: str
) -> None:
    """The shape of the payload the gate shipped (accepted from the formatter,
    or the template render), next to how much of the executor's answer it kept.

    `words` is the contract's own word count (headline, text, bullets,
    assumptions, follow-ups; tables are data). Logged from THIS module for the
    reason `log_cap` is."""
    words = count_words(payload.headline)
    text_blocks = bullets = table_rows = 0
    for block in payload.blocks:
        if isinstance(block, TextBlock):
            text_blocks += 1
            words += count_words(block.text)
        elif isinstance(block, BulletBlock):
            bullets += len(block.items)
            words += sum(count_words(item) for item in block.items)
        elif isinstance(block, TableBlock):
            table_rows += len(block.rows)
    words += sum(count_words(a) for a in payload.assumptions)
    words += sum(count_words(f) for f in payload.follow_ups)
    logger.info(
        "answer_shape: kind=%s words=%d text_blocks=%d bullets=%d table_rows=%d cites=%d "
        "exec_chars=%d retention=%s inv=%s tenant=%s",
        payload.kind,
        words,
        text_blocks,
        bullets,
        table_rows,
        len(payload.citations),
        len(executor_text or ""),
        retention(executor_text, payload),
        invocation_id,
        tenant,
    )


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
    _record_provenance(tool_response, prov, tool_name)
    _index_response(
        tool_name,
        tool_response,
        index,
        _bucket(_DROPPED, invocation_id, {}),
        _bucket(_EVICTABLE, invocation_id, _EvictionPool()),
    )
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


def _record_provenance(
    response: dict[str, Any], prov: dict[str, list[str]], tool_name: str = ""
) -> None:
    """entity_id → source ids for every id-bearing row in the response, top
    level, every top-level list and (for `get_piece`, the one tool whose rows
    are its nested dicts) the `piece` / `campaign` rows. Uncapped, whatever the
    index admits; unions across tools, so a
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
        if key.startswith("_"):
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    note(item)
        elif isinstance(value, dict) and tool_name in _WRAPPED_DETAIL_TOOLS:
            note(value)


def _note_drop(dropped: dict[str, int] | None, tool_name: str, count: int = 1) -> None:
    """Count refused entries. Callers pass only keys not already in the index (a
    repeat of an admitted key is not a loss); a key refused twice is counted
    twice, so `dropped` is refused entries, not distinct keys."""
    if dropped is not None and count > 0:
        dropped[tool_name] = dropped.get(tool_name, 0) + count


def _evict_one(index: dict, pool: _EvictionPool, dropped: dict[str, int] | None) -> bool:
    """Free one slot by evicting the OLDEST stub field row (the oldest member
    field row only when no stub field is left), counted as dropped under the
    label its writer gave it. False when there is none. Oldest first: it is the
    broadest read of the turn (the executor lists, then narrows), and its
    whole-entity row stays, so the entity is still named in the index."""
    popped = pool.pop_oldest()
    if popped is None:
        return False
    key, label = popped
    if index.pop(key, None) is not None:
        _note_drop(dropped, label)
    return True


def _admit_ahead(
    label: str,
    key: str,
    entry: dict[str, Any],
    index: dict,
    pool: _EvictionPool,
    dropped: dict[str, int] | None,
    evictable: bool,
) -> None:
    """Admit one entry of a detail or member row, evicting stub field rows (then
    member field rows) to make room and refusing (counted under `label`) only
    when none is left. A key already in the index is overwritten in place, which
    needs no room."""
    if key in index:
        index[key] = entry
        if not evictable:
            pool.forget(key)  # promoted: a detail row now owns this key
        return
    while len(index) >= MAX_ENTRIES:
        if not _evict_one(index, pool, dropped):
            _note_drop(dropped, label)
            return
    index[key] = entry
    if evictable:
        pool.members[key] = label


def _nested_status_items(row: dict[str, Any]):
    """The status text a detail row carries one level down, as (`parent.key`,
    text) — the account's `currentState.status_markdown`, which `_field_items`
    cannot see because it reads scalars only. 24 KB on the recorded Chevy
    account; it is the account's own narrative."""
    for parent, value in row.items():
        if parent.startswith("_") or not isinstance(value, dict):
            continue
        for status_key in _STATUS_KEYS:
            text = value.get(status_key)
            if isinstance(text, str) and text.strip():
                yield f"{parent}.{status_key}", text


def _is_status_field(key: str) -> bool:
    """A status text field: `statusMarkdown`, `status_markdown`, or one level down
    (`currentState.status_markdown`)."""
    return key.rsplit(".", 1)[-1] in _STATUS_KEYS


def _index_row(
    tool_name: str,
    row: dict[str, Any],
    fallback_id: str,
    index: dict,
    dropped: dict[str, int] | None = None,
    pool: _EvictionPool | None = None,
    tier: int = _STUB,
    label: str | None = None,
) -> None:
    """Index one row at its tier (the constants above): a stub is first-come,
    whole row or nothing once the index is full; a member or detail row is
    admitted ahead of the stubs, its status text before its other fields.
    `label` is what a refusal or eviction of this row is counted under (the
    tool's name unless the row is a member of one of its own lists)."""
    if pool is None:
        pool = _EvictionPool()
    label = label or tool_name
    row_id = row.get("id") if isinstance(row.get("id"), str) else fallback_id
    entity_id = row.get("id") if isinstance(row.get("id"), str) else None
    if tier == _STUB and len(index) >= MAX_ENTRIES:
        # The whole row is refused: its entity entry plus one per field it
        # would have contributed. Counted only, so the cap log line can say
        # which tool lost how much.
        keys = [f"{tool_name}:{row_id}"]
        keys.extend(f"{tool_name}:{row_id}:{key}" for key, _ in _field_items(row))
        _note_drop(dropped, label, sum(1 for key in keys if key not in index))
        return
    # Every source id this ROW mentions, whichever field carried it — so a
    # whole-entity citation can be attributed even though its own value was
    # compacted. D1(a): per-field granularity, which OVER-attributes a
    # multi-file status to any fact citing it. D1(b) — one evidence row per
    # bullet — is exact and costs an evidence-index rewrite; it is unratified,
    # and this shape is the one that ships without it.
    row_sources = _row_source_ids(row)
    entity_key = f"{tool_name}:{row_id}"
    entity_entry = {
        "tool": tool_name,
        "entity_id": entity_id,
        "field": None,
        "value": _compact_row(row),
        "source_file_ids": row_sources,
    }
    fields = list(_field_items(row))
    field_evictable = tier != _DETAIL
    if tier == _STUB:
        index[entity_key] = entity_entry
    else:
        fields.extend(_nested_status_items(row))
        # The status text is what the entity is assessed from: when the room runs
        # out part way through a row it must be the metadata that is refused, not
        # the narrative (a stable sort — every other field keeps its wire order).
        fields.sort(key=lambda item: not _is_status_field(item[0]))
        # The entity entry is never evictable: the entity stays named.
        _admit_ahead(label, entity_key, entity_entry, index, pool, dropped, False)
    for position, (key, value) in enumerate(fields):
        field_key = f"{tool_name}:{row_id}:{key}"
        field_entry = {
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
        if tier != _STUB:
            _admit_ahead(label, field_key, field_entry, index, pool, dropped, field_evictable)
            continue
        if len(index) >= MAX_ENTRIES:
            _note_drop(
                dropped,
                label,
                sum(1 for k, _ in fields[position:] if f"{tool_name}:{row_id}:{k}" not in index),
            )
            return
        if field_key not in index:
            pool.stubs[field_key] = label
        index[field_key] = field_entry


def _field_items(row: dict[str, Any]):
    """The (key, value) pairs of a row that become field entries."""
    for key, value in row.items():
        if key == "id" or key.startswith("_") or not _is_scalar(value):
            continue
        yield key, value


def _index_response(
    tool_name: str,
    response: dict[str, Any],
    index: dict,
    dropped: dict[str, int] | None = None,
    pool: _EvictionPool | None = None,
) -> None:
    """Break one tool response into rows.

    Shapes in the fleet: `org_query` → {results: [...], total, truncated}
    (rows may be entity rows with `id` or aggregate/group rows without one);
    list tools → {accounts|staff|hits|ideas: [...]}; detail tools → the entity
    dict itself, with nested lists (an overview's `campaigns`, a campaign's
    `pieces`), or `{piece, campaign}` for `get_piece`. Handled generically: the
    top-level dict is a row when it has an `id` (a DETAIL row when the tool is in
    DETAIL_TOOLS); every top-level list of dicts contributes rows (stubs, or
    members for a profile's `metadata`); remaining top-level scalars (org_query's
    `total`) index as "<tool>:<key>".
    """
    if response.get("error"):
        return  # an error body is not evidence
    if pool is None:
        pool = _EvictionPool()

    if isinstance(response.get("id"), str):
        _index_row(
            tool_name,
            response,
            "self",
            index,
            dropped,
            pool,
            _DETAIL if tool_name in DETAIL_TOOLS else _STUB,
        )
    own_lists = _OWN_LISTS.get(tool_name, {})

    row_counter = 0
    for key, value in response.items():
        if key.startswith("_"):
            continue  # _sources and friends are attribution plumbing, not evidence
        if isinstance(value, list):
            # What a member of THIS list is: a stub, unless the entity owns the
            # list as content (a profile's metadata).
            member_tier = own_lists.get(key, _STUB)
            label = f"{tool_name}.{key}" if key in own_lists else tool_name
            for item in value:
                if isinstance(item, dict):
                    _index_row(
                        tool_name,
                        item,
                        f"{key}{row_counter}",
                        index,
                        dropped,
                        pool,
                        member_tier,
                        label,
                    )
                    row_counter += 1
        elif (
            isinstance(value, dict)
            and tool_name in _WRAPPED_DETAIL_TOOLS
            and isinstance(value.get("id"), str)
        ):
            _index_row(tool_name, value, key, index, dropped, pool, _DETAIL)
        elif (
            _is_scalar(value)
            and not key.startswith("_")
            and not isinstance(response.get("id"), str)  # already indexed as row fields
        ):
            if len(index) >= MAX_ENTRIES:
                _note_drop(dropped, tool_name, 0 if f"{tool_name}:{key}" in index else 1)
            else:
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
