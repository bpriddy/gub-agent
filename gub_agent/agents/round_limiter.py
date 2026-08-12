"""
round_limiter.py — before_model_callback that caps ReAct ROUNDS (not tool calls).

The tool-call circuit breaker (circuit_breaker.py) blocks tool EXECUTION but the
model still spends a round trip to receive the block — so it doesn't cut latency.
Latency lives in the number of model rounds. This callback caps rounds: once the
executor has had MAX_ROUNDS chances to gather data, we strip the tool
declarations from the request so the model CANNOT call tools and is forced to
synthesize from what it already has.

Mechanism: before_model_callback fires once per model round. We count rounds per
invocation (ADK state doesn't persist plain writes between calls, so we key an
in-process dict on invocation_id). Past the cap we clear llm_request.config.tools
and tools_dict and append a directive to answer now.

The budget is PER EXECUTOR PASS: reset_rounds() runs as the executor's
before_agent_callback, so each LoopAgent retry iteration (the critic asked the
executor to try again) starts with a fresh MAX_ROUNDS budget. Without the reset,
a retry would inherit the first pass's count — often already at the cap — and
open with tools already stripped, unable to make the call the critic requested.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

from google.genai import types as genai_types

logger = logging.getLogger(__name__)

# Healthy questions finish in 2-4 rounds (find → detail → maybe refine → answer).
# Cap at 3 gathering rounds; the 4th model call has no tools and must synthesize.
MAX_ROUNDS = 3

_ROUNDS: "OrderedDict[str, int]" = OrderedDict()
_MAX_TRACKED = 256


def _tick(invocation_id: str) -> int:
    n = _ROUNDS.get(invocation_id, 0) + 1
    _ROUNDS[invocation_id] = n
    _ROUNDS.move_to_end(invocation_id)
    while len(_ROUNDS) > _MAX_TRACKED:
        _ROUNDS.popitem(last=False)
    return n


def reset_rounds(callback_context: Any) -> None:
    """ADK before_agent_callback: clear this invocation's round count so each
    LoopAgent iteration (executor pass) starts with a fresh MAX_ROUNDS budget —
    see module docstring."""
    invocation_id = getattr(callback_context, "invocation_id", "") or "?"
    _ROUNDS.pop(invocation_id, None)
    return None


def round_limit(callback_context: Any, llm_request: Any) -> None:
    """ADK before_model_callback: after MAX_ROUNDS, remove tools to force answer."""
    invocation_id = getattr(callback_context, "invocation_id", "") or "?"
    rnd = _tick(invocation_id)
    if rnd <= MAX_ROUNDS:
        return None

    # Past the budget: strip tool access so the model has to write the answer.
    cfg = getattr(llm_request, "config", None)
    if cfg is not None:
        try:
            cfg.tools = []
        except Exception:  # noqa: BLE001 — best-effort across ADK versions
            pass
    try:
        llm_request.tools_dict = {}
    except Exception:  # noqa: BLE001
        pass

    llm_request.contents = (llm_request.contents or []) + [
        genai_types.Content(
            role="user",
            parts=[
                genai_types.Part(
                    text=(
                        f"Tool-call limit reached ({MAX_ROUNDS} rounds of lookups). "
                        f"Do NOT call any tools now. Write your final answer from the "
                        f"data already gathered above. If something is genuinely "
                        f"missing, say what is missing — do not attempt more lookups."
                    )
                )
            ],
        )
    ]
    logger.warning("round_limit: HIT (round %d, inv=%s) — tools removed", rnd, invocation_id)
    return None
