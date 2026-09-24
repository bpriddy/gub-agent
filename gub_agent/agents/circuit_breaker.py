"""
circuit_breaker.py — before_tool_callback guardrails for the executor.

Three protections at the tool-call boundary via ADK's before_tool_callback
(return a dict → the tool is SKIPPED and the dict becomes its result; return
None → the tool runs normally):

1. LOOP DETECTION — a model that gets a thin/empty result sometimes repeats the
   exact same call. We hash (tool name + args); on a duplicate within the same
   executor pass we short-circuit with an instruction to reuse the earlier result.

2. QUERY WEAKENING (find_files only) — the same impulse, one step subtler: given
   nothing, the model drops words from the description and searches again until
   something matches. Measured doing exactly that four times in one turn, ending
   in a confident list of near-matches under a "not found" headline. A query
   whose words are a subset of one already tried this pass is refused.

3. CIRCUIT BREAKER — a hard cap on tool calls per executor pass. Analytical /
   resourcing questions were measured fanning out to 15-20 calls (each ≈ a full
   model round trip). Past the cap we stop dispatching and tell the executor to
   synthesize from what it has.

State lives in an in-process dict keyed on invocation_id — ADK session state
doesn't reliably persist plain writes between callbacks (same reason
round_limiter.py uses one). The budget is PER EXECUTOR PASS: reset_tool_budget()
runs as the executor's before_agent_callback, so each LoopAgent retry iteration
(the critic said "try again") starts with a fresh budget instead of inheriting
the first pass's count. Without the reset a retry could open already at the cap
and be unable to make the additional call the critic asked for.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from typing import Any

# Healthy questions use 2-8 calls; pathological fan-out hit 15-20. 8 leaves
# legitimate multi-round work untouched while cutting the runaway tail. Tune
# from logs — raise if it clips real questions, lower if fan-out persists.
MAX_TOOL_CALLS = 8

# invocation_id → {"count": int, "seen": set[str]}, reset per executor pass by
# reset_tool_budget(). Bounded so many turns don't grow it without limit.
_BUDGETS: OrderedDict[str, dict] = OrderedDict()
_MAX_TRACKED = 256


def _budget(invocation_id: str) -> dict:
    b = _BUDGETS.get(invocation_id)
    if b is None:
        b = {"count": 0, "seen": set(), "file_queries": []}
        _BUDGETS[invocation_id] = b
    _BUDGETS.move_to_end(invocation_id)
    while len(_BUDGETS) > _MAX_TRACKED:
        _BUDGETS.popitem(last=False)
    return b


def _query_tokens(query: Any) -> frozenset[str]:
    """A find_files query as a comparable bag of words.

    Deliberately NOT destopworded: this compares two queries against each other,
    not a query against a filename, so "the deck" versus "deck" is a narrowing
    like any other. Length >= 2 mirrors the backend's tokenizer.
    """
    if not isinstance(query, str):
        return frozenset()
    return frozenset(t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) >= 2)


def reset_tool_budget(callback_context: Any) -> None:
    """ADK before_agent_callback: clear this invocation's tool budget so each
    LoopAgent iteration (executor pass) starts fresh — see module docstring."""
    invocation_id = getattr(callback_context, "invocation_id", "") or "?"
    _BUDGETS.pop(invocation_id, None)
    return None


def circuit_breaker(tool: Any, args: dict, tool_context: Any) -> dict | None:
    """ADK before_tool_callback: dedupe repeats and cap calls per executor pass."""
    invocation_id = getattr(tool_context, "invocation_id", "") or "?"
    budget = _budget(invocation_id)
    tool_name = getattr(tool, "name", str(tool))

    # 1) Loop detection — identical (tool, args) already issued this pass.
    key = hashlib.md5(
        f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}".encode()
    ).hexdigest()
    if key in budget["seen"]:
        return {
            "error": True,
            "message": (
                f"You already called {tool_name} with these exact arguments in "
                f"this turn. Reuse the result you already have — do not repeat "
                f"the call."
            ),
        }

    # 2) Query-weakening detection, find_files only.
    #
    # Measured on the sandbox engine 2026-09-24, turn "the final OnStar pitch
    # pre-read doc": the endpoint correctly returned nothing, so the model
    # shortened the description and tried again, four times —
    #   "OnStar pitch pre-read" -> "OnStar pitch" -> "pitch pre-read" -> "pre-read"
    # — until "pre-read" matched EXT_PRE-READ Q4 Trucks at similarity 1.0. The
    # delivered answer then headlined "I was unable to find a file matching
    # final OnStar pitch pre-read doc by name" and listed five unrelated files
    # under it. That is the ORIGINAL reported complaint ("eight files that had
    # nothing to do with BHAC") rebuilt one layer up, and it defeats spec §6:
    # the gate holds for the endpoint and not for the system.
    #
    # Dropping words cannot make a file more findable — it can only make the
    # match less specific, so anything it newly returns is by construction a
    # worse answer than the nothing that came before. Adding words is fine and
    # stays allowed; only a subset (or a pure reordering, which carries no new
    # information either) is refused.
    #
    # This is a MECHANICAL guard on purpose. The find_files docstring already
    # forbids exactly this in plain language — "Name no file and offer no
    # near-match as a substitute" — and the model did it anyway, four times in
    # one turn. Prompt text is not a control here.
    if tool_name == "find_files":
        tokens = _query_tokens(args.get("query"))
        if tokens:
            for earlier in budget["file_queries"]:
                if tokens <= earlier:
                    return {
                        "error": True,
                        "message": (
                            "You already searched file names for a description that "
                            "included these words, and dropping words from it cannot "
                            "find more — a shorter description only matches less "
                            "specifically, so whatever it returns is a near-match, not "
                            "the file that was asked for. Do not retry with fewer "
                            "words. Say the file was not found BY NAME, name no "
                            "substitute, and let the turn rest there: a separate "
                            "system searches the user's own Workspace by CONTENT and "
                            "may still find it."
                        ),
                    }
            budget["file_queries"].append(tokens)

    # 3) Circuit breaker — hard per-pass tool budget.
    budget["count"] += 1
    if budget["count"] > MAX_TOOL_CALLS:
        return {
            "error": True,
            "message": (
                f"Tool-call budget ({MAX_TOOL_CALLS}) reached for this turn. "
                f"Stop calling tools and synthesize your answer from what you "
                f"have gathered so far. If something is genuinely missing, say "
                f"so rather than fetching more."
            ),
        }

    budget["seen"].add(key)
    return None
