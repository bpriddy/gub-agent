"""
circuit_breaker.py — before_tool_callback guardrails for the executor.

Two protections, both operating at the tool-call boundary via ADK's
before_tool_callback hook (return a dict → the tool is SKIPPED and the dict
becomes its result; return None → the tool runs normally):

1. LOOP DETECTION — a model that gets a thin/empty result sometimes repeats
   the exact same call. We hash (tool name + args); on a duplicate within the
   same turn we short-circuit with an instruction to reuse the earlier result.

2. CIRCUIT BREAKER — a hard cap on tool calls PER TURN. Analytical / resourcing
   questions were measured fanning out to 15-20 calls (each call ≈ a full model
   round trip). Past the cap we stop dispatching and tell the executor to
   synthesize from what it already has.

State note: tool_context.state persists for the whole SESSION, not one turn, so
the counters are keyed on invocation_id and reset when a new turn starts.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

# Healthy questions use 2-8 calls; pathological fan-out hit 15-20. 8 leaves
# legitimate multi-round work untouched while cutting the runaway tail. Tune
# from logs — raise if it clips real questions, lower if fan-out persists.
MAX_TOOL_CALLS = 8


def _reset_if_new_turn(state: Any, invocation_id: str) -> None:
    if state.get("_cb_inv") != invocation_id:
        state["_cb_inv"] = invocation_id
        state["_cb_count"] = 0
        state["_cb_seen"] = []


def circuit_breaker(tool: Any, args: dict, tool_context: Any) -> Optional[dict]:
    """ADK before_tool_callback: dedupe repeats and cap calls per turn."""
    state = tool_context.state
    invocation_id = getattr(tool_context, "invocation_id", "") or ""
    _reset_if_new_turn(state, invocation_id)

    tool_name = getattr(tool, "name", str(tool))

    # 1) Loop detection — identical (tool, args) already issued this turn.
    key = hashlib.md5(
        f"{tool_name}:{json.dumps(args, sort_keys=True, default=str)}".encode()
    ).hexdigest()
    if key in state["_cb_seen"]:
        return {
            "error": True,
            "message": (
                f"You already called {tool_name} with these exact arguments in "
                f"this turn. Reuse the result you already have — do not repeat "
                f"the call."
            ),
        }

    # 2) Circuit breaker — hard per-turn tool budget.
    state["_cb_count"] = state.get("_cb_count", 0) + 1
    if state["_cb_count"] > MAX_TOOL_CALLS:
        return {
            "error": True,
            "message": (
                f"Tool-call budget ({MAX_TOOL_CALLS}) reached for this turn. "
                f"Stop calling tools and synthesize your answer from what you "
                f"have gathered so far. If something is genuinely missing, say "
                f"so rather than fetching more."
            ),
        }

    state["_cb_seen"].append(key)
    return None
