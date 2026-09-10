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
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

# Rows are compact (scalar fields only), but the formatter instruction carries
# the whole index — cap the entries so a runaway fan-out cannot balloon the
# prompt. 400 comfortably covers MAX_TOOL_CALLS=8 × 50-row default limits.
MAX_ENTRIES = 400
# A whole-entity value is the row's scalars as JSON; cap it so statusMarkdown
# and friends do not turn one row into a page. Field-level values stay whole —
# they are what numbers and names are grounded against.
MAX_ENTITY_VALUE_CHARS = 2_000

_INDEX: OrderedDict[str, dict[str, dict[str, Any]]] = OrderedDict()
_FEEDBACK: OrderedDict[str, str] = OrderedDict()
_BRIEF: OrderedDict[str, str] = OrderedDict()
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
    _FEEDBACK.pop(invocation_id, None)
    _BRIEF.pop(invocation_id, None)
    return None


def evidence_index(invocation_id: str) -> dict[str, dict[str, Any]]:
    """This invocation's index (empty when no tool has returned yet)."""
    return _INDEX.get(invocation_id, {})


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


def record_evidence(tool: Any, args: dict, tool_context: Any, tool_response: Any) -> None:
    """ADK after_tool_callback on the executor: index this result's rows.
    Returns None so the tool response passes through unmodified."""
    invocation_id = getattr(tool_context, "invocation_id", "") or "?"
    if not isinstance(tool_response, dict):
        return None
    index = _bucket(_INDEX, invocation_id, {})
    tool_name = getattr(tool, "name", None) or str(tool)
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


def _index_row(tool_name: str, row: dict[str, Any], fallback_id: str, index: dict) -> None:
    if len(index) >= MAX_ENTRIES:
        return
    row_id = row.get("id") if isinstance(row.get("id"), str) else fallback_id
    entity_id = row.get("id") if isinstance(row.get("id"), str) else None
    index[f"{tool_name}:{row_id}"] = {
        "tool": tool_name,
        "entity_id": entity_id,
        "field": None,
        "value": _compact_row(row),
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
                }
