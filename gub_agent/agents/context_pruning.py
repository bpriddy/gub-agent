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
"""

from __future__ import annotations

from typing import Any

from google.genai import types as genai_types


def _is_function_part(part: Any) -> bool:
    return bool(
        getattr(part, "function_call", None) or getattr(part, "function_response", None)
    )


def _has_user_text(content: Any) -> bool:
    """True for a user-role content carrying real text (a typed question).

    Tool responses also arrive role="user" in the genai format, but as
    function_response parts, not text — they must not count as a turn
    boundary.
    """
    if content.role != "user":
        return False
    return any(
        getattr(p, "text", None) and not _is_function_part(p)
        for p in (content.parts or [])
    )


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
