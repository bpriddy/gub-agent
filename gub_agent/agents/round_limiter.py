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

The round past the budget is the one that writes the answer, and its first chunk
comes after ALL of its thinking (0.7-4.3k thought tokens, against 0.2-0.5k on a
tool round; ~13 s at the median, 25-30 s on a long one). Two things follow from
the round counter this module keeps, both after the tools are gone:

* `round_limit` marks the model call (`models.mark_synthesis_round`), so the
  executor's hedge can wait longer for it than for a tool round
  (config.EXECUTOR_SYNTHESIS_HEDGE_AFTER_MS);
* `synthesis_round_thinking` lowers that round's thinking when
  config.SYNTHESIS_ROUND_THINKING_LEVEL, or the sandbox override
  `synthesis_round_thinking_level`, says so. It reads the counter `round_limit`
  just ticked and never ticks.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

from google.genai import types as genai_types

from .. import config
from ..models import mark_synthesis_round
from ..sandbox import read_overrides, takes_named_thinking_level
from ..tenant import _state_of, label_of
from .critic import _passes_judged

logger = logging.getLogger(__name__)

# Healthy questions finish in 2-4 rounds (find → detail → maybe refine → answer).
# Cap at 3 gathering rounds; the 4th model call has no tools and must synthesize.
MAX_ROUNDS = 3

_ROUNDS: OrderedDict[str, int] = OrderedDict()
_MAX_TRACKED = 256

# The thinking states `synthesis_round_thinking` lowers to LOW (`_thinking_label`
# names them): MEDIUM, HIGH, dynamic, or no thinking config at all. Everything
# else is at or below LOW already, or not a level at all.
_LOWERABLE = ("MEDIUM", "HIGH", "DYNAMIC", "-")


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
    # This model call is the answer round: the hedge may wait longer for it.
    mark_synthesis_round()
    logger.warning(
        "round_limit: HIT (round %d, inv=%s) — tools removed tenant=%s",
        rnd,
        invocation_id,
        label_of(callback_context),
    )
    return None


def _effective_synthesis_thinking(callback_context: Any) -> str:
    """The switch for THIS call: the sandbox override when one was given (an
    explicit value wins over the env, either way, `AUTO` included), else the env
    flag. The sandbox is inert unless SANDBOX_ENABLED (`read_overrides`), so on a
    production engine this is the env flag alone. A bad sandbox value raised in
    `sandbox_before_model` already, earlier in the same chain."""
    given = read_overrides(_state_of(callback_context)).synthesis_round_thinking_level
    return given if given is not None else config.SYNTHESIS_ROUND_THINKING_LEVEL


def _thinking_label(thinking: Any) -> str:
    """What a request's thinking config asks for, as one word for the log line:
    the level's name, `OFF` (budget 0), `DYNAMIC` (budget -1), `budget=<n>`, or
    `-` for no config at all."""
    if thinking is None:
        return "-"
    level = getattr(thinking, "thinking_level", None)
    if level is not None:
        name = str(getattr(level, "name", level)).upper()
        if name != "THINKING_LEVEL_UNSPECIFIED":
            return name
    budget = getattr(thinking, "thinking_budget", None)
    if budget is None:
        return "-"
    return {0: "OFF", -1: "DYNAMIC"}.get(budget, f"budget={budget}")


def synthesis_round_thinking(callback_context: Any, llm_request: Any) -> None:
    """ADK before_model_callback: think less on the executor's round that writes
    the answer. Off (AUTO) unless SYNTHESIS_ROUND_THINKING_LEVEL or the sandbox
    override says LOW.

    Why: that round's first chunk comes after all of its thinking, 0.7-4.3k
    thought tokens against 0.2-0.5k on a tool round, which is most of its 13 s
    median — and the answer it writes is the executor's prose from data already
    gathered, the formatter's input, not a new investigation. Whether LOW costs
    the answer anything is what a paid replay of the round on both levels
    decides; until then this is off.

    Runs AFTER `round_limit` and reads the counter it just ticked — it never
    ticks, and it acts on the round past MAX_ROUNDS only. It writes
    `config.thinking_config` and nothing else: a FRESH config from the same
    constructor the executor's planner uses (never the planner's own object, which
    every later call shares), so a later round or pass starts from the planner's
    level again.

    Never raises the thinking: only a request that thinks at MEDIUM or HIGH,
    dynamically, or with no thinking config at all is lowered; one already at LOW
    or MINIMAL, or with thinking off, is left alone (a sandbox run may set the
    executor's level itself). A model that takes no named level (Claude, another
    Gemini id outside SANDBOX_THINKING_LEVEL_MODELS) is left alone too — a named
    level is a 400 there, and a 400 inside the engine reaches the caller as an
    empty stream.
    """
    # The round gate first: it is state only, so rounds 1-3 never reach the
    # sandbox reader (which warns about what it ignores) one more time.
    invocation_id = getattr(callback_context, "invocation_id", "") or "?"
    rnd = _ROUNDS.get(invocation_id, 0)
    if rnd <= MAX_ROUNDS:
        return None
    level = _effective_synthesis_thinking(callback_context)
    if level == "AUTO":
        return None
    cfg = getattr(llm_request, "config", None)
    if cfg is None or not takes_named_thinking_level(getattr(llm_request, "model", None)):
        return None

    was = _thinking_label(getattr(cfg, "thinking_config", None))
    if was not in _LOWERABLE:
        return None
    cfg.thinking_config = config.build_thinking_planner(thinking_level=level).thinking_config
    logger.info(
        "synthesis_thinking: level=%s was=%s round=%d pass=%d inv=%s tenant=%s",
        level,
        was,
        rnd,
        _passes_judged(callback_context) + 1,
        invocation_id,
        label_of(callback_context),
    )
    return None
