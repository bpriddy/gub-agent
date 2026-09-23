"""context_pruning.py — keep the conversation, drop the receipts.

Every model call re-reads the whole session transcript, tool payloads
included — so a prior turn's 47-campaign list taxes every round of every
LATER turn (tokens × remaining calls). But prior turns' tool payloads are
dead weight BY DOCTRINE: the executor must re-query for entity facts
("do NOT rely on prior turns"), and the critic verifies grounding against
the CURRENT turn's results only.

This before_model_callback strips function_call / function_response parts
from all content BEFORE the current turn (= everything before the last
user message that carries real text). Prior answers' prose survives, so
follow-ups still resolve ("what about the other campaign?"); the current
turn's raw tool data is untouched, so synthesis-over-raw-data and the
critic's grounding checks are unaffected.

`trim_to_recent_turns` (memory-00 §3) is the second, coarser bound in the
same place: a sliding window of the last CONTEXT_TURN_WINDOW real turns.
The pruner above keeps a long conversation from carrying its tool
payloads; the window keeps it from carrying every turn. Together they are
what let the bot's 5-minute idle session reset be relaxed to a day —
until one of them is live, that timer is the only thing bounding how much
transcript a model round re-reads.
"""

from __future__ import annotations

import logging
from typing import Any

from google.genai import types as genai_types

from ..config import CONTEXT_TURN_WINDOW

logger = logging.getLogger(__name__)


def _is_function_part(part: Any) -> bool:
    return bool(getattr(part, "function_call", None) or getattr(part, "function_response", None))


def _has_user_text(content: Any) -> bool:
    """True for a user-role content carrying real text (a typed question).

    Tool responses also arrive role="user" in the genai format, but as
    function_response parts, not text — they must not count as a turn
    boundary.
    """
    if content.role != "user":
        return False
    return any(getattr(p, "text", None) and not _is_function_part(p) for p in (content.parts or []))


# Keys that are pure tool plumbing — the executor/critic prompts already tell
# the model these are for grounding infrastructure and MUST NOT appear in prose.
# `_sources` (Drive file citations) measured at 512k chars / 128k tokens for ONE
# account overview (3,257 file refs) — re-sent every ReAct round. Dead weight.
#
# `_cited` and `_sourcesTotal` (blend 08 §5.1) are the same kind of thing: a
# fileId → name map and a count that exist so the BOT can bind links. The
# formatter gets the source ids it copies from the evidence brief's `sources:`
# lines, never from here — so to the model this is 87 opaque Drive ids per
# account overview, re-sent every round, that it is forbidden to use.
_PLUMBING_KEYS = ("_sources", "_sourcesTotal", "_cited")


def _without_plumbing(obj: Any) -> Any:
    """Return `obj` with plumbing keys dropped, COPY-ON-WRITE: a new dict/list is
    built only where a key was actually removed; unchanged subtrees (and all
    scalars) are shared. So this never mutates the input in place — the
    function_response payload is shared with ADK session history, and mutating a
    nested field there can corrupt it (safe under the eager model_dump() prod
    persistence uses today, but a landmine under InMemorySessionService and for
    trace consumers that read `_sources`)."""
    if isinstance(obj, dict):
        new: dict = {}
        changed = False
        for k, v in obj.items():
            if k in _PLUMBING_KEYS:
                changed = True
                continue
            nv = _without_plumbing(v)
            changed = changed or nv is not v
            new[k] = nv
        return new if changed else obj
    if isinstance(obj, list):
        rebuilt = [_without_plumbing(v) for v in obj]
        return rebuilt if any(n is not o for n, o in zip(rebuilt, obj)) else obj
    return obj


def strip_source_metadata(callback_context: Any, llm_request: Any) -> None:
    """Observation masking: drop `_sources` citation plumbing from every tool
    response in the request. The model is instructed to ignore it (see
    prompts/executor.py, prompts/critic.py), yet it dominates prompt size on
    portfolio questions — one account overview carried 128k tokens of file refs,
    re-sent each round. Stripping it is loss-free for the answer and roughly
    halves prompt tokens on the heavy questions. The scrubbed copy is assigned
    back to `function_response.response` — the original (session-shared) payload
    is never mutated in place."""
    for content in llm_request.contents or []:
        for part in content.parts or []:
            fr = getattr(part, "function_response", None)
            resp = getattr(fr, "response", None) if fr is not None else None
            if isinstance(resp, dict):
                scrubbed = _without_plumbing(resp)
                if scrubbed is not resp:
                    try:
                        fr.response = scrubbed
                    except Exception:  # noqa: BLE001 — best-effort across ADK versions
                        pass
    return None


def strip_prior_turn_tool_parts(callback_context: Any, llm_request: Any) -> None:
    """Drop function_call/function_response parts from pre-current-turn content."""
    contents = llm_request.contents or []
    if not contents:
        return None

    # The current turn starts at the LAST user content with real text.
    boundary = None
    for i in range(len(contents) - 1, -1, -1):
        if _has_user_text(contents[i]):
            boundary = i
            break
    if boundary is None or boundary == 0:
        return None  # single-turn request (or nothing to prune) — leave as-is

    pruned: list[Any] = []
    for i, content in enumerate(contents):
        if i >= boundary:
            pruned.append(content)
            continue
        kept_parts = [p for p in (content.parts or []) if not _is_function_part(p)]
        if kept_parts:
            pruned.append(genai_types.Content(role=content.role, parts=kept_parts))
        # A content that was ONLY tool payload disappears entirely.

    llm_request.contents = pruned
    return None


# ── Conversation window (memory-00 §3) ───────────────────────────────────────


def _is_foreign_context(content: Any) -> bool:
    """True for an ADK foreign-agent context content.

    ADK rewrites another agent's events into a role="user" content whose FIRST
    part is exactly "For context:" (adk/flows/llm_flows/contents.py:1006-1007),
    followed by "[author] said: ..." parts. Those satisfy _has_user_text but
    they are NOT turn boundaries.

    This matters more here than in a single-agent app: the pipeline runs
    router, executor, critic, formatter and format gate, so ONE completed turn
    emits several of these. Counting them inflates the turn count 4-7x and
    silently shrinks the window to a fraction of CONTEXT_TURN_WINDOW — no
    exception, no malformed request, just a model that can no longer resolve
    "the other one".

    A SUBTRACTIVE filter layered on _has_user_text rather than a second copy of
    the "is this a real user message" rule: tool responses also arrive
    role="user" (as function_response parts), and a second home for that rule
    is the first thing to drift.
    """
    parts = content.parts or []
    if not parts:
        return False
    return (getattr(parts[0], "text", None) or "").strip() == "For context:"


def _turn_starts(contents: list[Any]) -> list[int]:
    """Indices of REAL user turns — the only indices it is safe to cut at."""
    return [
        i
        for i, content in enumerate(contents)
        if _has_user_text(content) and not _is_foreign_context(content)
    ]


def trim_to_recent_turns(callback_context: Any, llm_request: Any) -> None:
    """Sliding window: show the model the last CONTEXT_TURN_WINDOW complete turns.

    This is the bound that replaces the bot's 5-minute idle session reset. Once
    a session lives as long as the conversation does, every model round of every
    later turn re-reads the whole transcript; strip_prior_turn_tool_parts already
    keeps that growth to prose rather than tool payloads, so it is a slow leak
    and not a runaway — but it is unbounded, and input volume is the primary
    cost driver on this workload.

    A window, not a TTL, because time was never the thing that mattered: a
    follow-up needs the last few exchanges ("the other campaign", "and Q3?"),
    and that is equally true of a 30-second pause and an overnight one.

    Per-request only. ADK rebuilds `contents` from the session's events on every
    round, so nothing is destroyed: raising CONTEXT_TURN_WINDOW restores the
    older turns on the very next call, and a badly chosen N is a config change
    rather than a lost conversation. That is the decisive advantage over a
    bot-owned truncated transcript.

    Orphan safety — why there is no repair pass here. A message-counted window
    can split a function_call from its function_response, which is not a smaller
    request but an INVALID one (Gemini answers 400 INVALID_ARGUMENT, which
    inside Agent Engine reaches the caller as an empty 200 stream). This design
    cannot produce that, for two independent reasons: the cut is always AT a
    turn boundary, and a call/response pair always lives between two consecutive
    boundaries; and strip_prior_turn_tool_parts runs immediately after and
    removes every pre-current-turn function part anyway. An earlier draft
    specified a _drop_orphan_responses helper — it would be dead code in all
    three chains as wired, and a later reader would trust it.
    """
    window = CONTEXT_TURN_WINDOW
    if window <= 0:
        return None  # explicitly disabled — the rollback, with no redeploy

    contents = llm_request.contents or []
    if not contents:
        return None

    starts = _turn_starts(contents)
    if len(starts) <= window:
        return None  # shorter than the window — no-op, and no copies made

    # starts[-1] IS the current turn and cut <= starts[-1] for any window >= 1,
    # so the current turn survives whole however much tool traffic it has
    # accumulated: the window bounds HISTORY, never the work in progress.
    cut = starts[-window]

    # Anything before the FIRST real boundary is preamble, not a turn. With
    # `instruction=` the system text rides in config.system_instruction and this
    # slice is empty; an agent configured with `static_instruction` has ADK put
    # instruction contents in `contents`, and those must keep leading the request.
    preamble = contents[: starts[0]]
    kept = preamble + contents[cut:]

    # INFO so the effective window is measurable from Cloud Logging at zero
    # cost. `turns` is the REAL turn count — the number this change exists to
    # get right, and the rollout gate: a count 4-7x higher than the
    # conversation's real length means _is_foreign_context is not working.
    logger.info(
        "context_window: turns=%d window=%d kept=%d dropped=%d",
        len(starts),
        window,
        len(kept),
        len(contents) - len(kept),
    )
    llm_request.contents = kept
    return None
