"""
circuit_breaker.py — before_tool_callback guardrails for the executor.

Two protections at the tool-call boundary via ADK's before_tool_callback (return
a dict → the tool is SKIPPED and the dict becomes its result; return None → the
tool runs normally):

1. LOOP DETECTION — a model that gets a thin/empty result sometimes repeats the
   exact same call. We hash (tool name + args); on a duplicate within the same
   executor pass we short-circuit with an instruction to reuse the earlier result.

2. CIRCUIT BREAKER — a hard cap on tool calls per executor pass. Analytical /
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
from collections import OrderedDict
from typing import Any, Optional

# Healthy questions use 2-8 calls; pathological fan-out hit 15-20. 8 leaves
# legitimate multi-round work untouched while cutting the runaway tail. Tune
# from logs — raise if it clips real questions, lower if fan-out persists.
MAX_TOOL_CALLS = 8

# invocation_id → {"count": int, "seen": set[str]}, reset per executor pass by
# reset_tool_budget(). Bounded so many turns don't grow it without limit.
_BUDGETS: "OrderedDict[str, dict]" = OrderedDict()
_MAX_TRACKED = 256


def _budget(invocation_id: str) -> dict:
    b = _BUDGETS.get(invocation_id)
    if b is None:
        b = {"count": 0, "seen": set()}
        _BUDGETS[invocation_id] = b
    _BUDGETS.move_to_end(invocation_id)
    while len(_BUDGETS) > _MAX_TRACKED:
        _BUDGETS.popitem(last=False)
    return b


def reset_tool_budget(callback_context: Any) -> None:
    """ADK before_agent_callback: clear this invocation's tool budget so each
    LoopAgent iteration (executor pass) starts fresh — see module docstring."""
    invocation_id = getattr(callback_context, "invocation_id", "") or "?"
    _BUDGETS.pop(invocation_id, None)
    return None


def circuit_breaker(tool: Any, args: dict, tool_context: Any) -> Optional[dict]:
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

    # 2) Circuit breaker — hard per-pass tool budget.
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
