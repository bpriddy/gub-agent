"""
router.py — the intent router (blend 04, gub-agent#23).

One LlmAgent, no tools, `output_schema=RouterDecision`, thinking off: it
classifies the question and stops. The dispatcher (`agents/dispatcher.py`)
then picks a branch in plain code, so the routing decision is inspectable and
testable rather than implied by whatever the executor felt like doing.

Two wiring notes:

- It emits its JSON as TEXT, authored by `router`. The bot's answer channel
  routes by author and lists `router` among the pipeline internals it ignores
  (`gub-gchat-bot/src/agent/client.ts:209-217`, blend 03 step 0) — without
  that routing this JSON would land in the user's bubble, which is why 03
  ships first.
- The decision reaches the dispatcher as `state["router_decision"]`
  (`output_key`), committed the same way the critic's verdict reaches the
  escalator: ADK applies an event's `state_delta` when the runner consumes it,
  and the dispatcher runs after. The dispatcher ALSO re-reads the router's own
  event text as a fallback, so a state-commit surprise costs the deep path,
  never the turn.

Prior-turn tool payloads are pruned from its request for the same reason the
critic prunes them: the router classifies the CURRENT question, and a previous
turn's 50-row org_query result is nothing but tokens. That pruning has to work
on TEXT here (`strip_prior_turn_tool_text`) — the router never sees a function
part, only ADK's rendering of the executor's — and the router is windowed to
the same per-session number of turns as the executor.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from pydantic import ValidationError

from ..config import build_model
from ..instruction_utils import with_current_date
from ..models import bind_model_call
from ..prompts import ROUTER_INSTRUCTION
from ..sandbox import (
    ROUTER_THINKING_LEVEL,
    baseline_planner,
    sandbox_before_model,
    sandbox_instruction,
)
from ..schemas.router import FALLBACK_DECISION, RouterDecision
from .context_pruning import (
    strip_prior_turn_tool_parts,
    strip_prior_turn_tool_text,
    trim_to_recent_turns,
)

logger = logging.getLogger(__name__)

ROUTER_NAME = "router"
ROUTER_STATE_KEY = "router_decision"


def _router_before_model(callback_context, llm_request):
    """Sandbox overrides first (model / router_thinking_level / temperature — a
    no-op without state["sandbox"]), then the conversation window, then the
    prior-turn pruning.

    The window was DEFERRED here by memory-00 §3.5 until the executor's
    `context_window:` line had confirmed the real turn count in production —
    the router being the agent most exposed to boundary inflation. That line
    has since gated the executor's rollout, and thread-topics cannot leave the
    router unbounded: a thread keeps up to 200 turns, and the router runs on
    every one of them, fast path included. Same function, same per-session
    value, so "the last N exchanges" means one thing across the pipeline.

    The pruning is the content filter that note anticipated, narrowed: rather
    than dropping earlier foreign-context contents whole, which would take the
    executor's previous ANSWER with it — the very text "and Q3?" is resolved
    against — `strip_prior_turn_tool_text` drops only their tool-call and
    tool-result text and keeps the prose. `strip_prior_turn_tool_parts` stays
    in front of it and removes nothing today (the router has no tools, and ADK
    renders another agent's function parts as text); it is here so a future
    ADK that relays native function parts is still pruned.

    Order: the window first, so the pruners rebuild fewer contents — the same
    cost argument as the executor's chain (agent.py:_before_model)."""
    bind_model_call(callback_context)  # who is calling, for the model_call line
    sandbox_before_model(callback_context, llm_request, role="router")
    trim_to_recent_turns(callback_context, llm_request, role="router")
    strip_prior_turn_tool_parts(callback_context, llm_request)
    return strip_prior_turn_tool_text(callback_context, llm_request)


router_agent = LlmAgent(
    # Retry-with-backoff on 429/5xx, same as every other stage (config.build_model).
    model=build_model(),
    name=ROUTER_NAME,
    # Wrapped for the sandbox: state["sandbox"].router_instruction (or
    # .router_variant) replaces the prompt for that run only — the one
    # practical way to drive the misroute rate down without a redeploy.
    # InstructionProvider — appends today's date deterministically per request.
    instruction=sandbox_instruction(with_current_date(ROUTER_INSTRUCTION), role="router"),
    # Thinking off (thinking_budget=0; LOW with ROUTER_THINKING_OFF=0), from
    # sandbox.py so the provenance can't drift from what runs. The router exists
    # to save model turns; at LOW its ~99 thought tokens a call put TTFT p90 at
    # 3.2 s against 1.1 s off, for the same intent on 19 of 20 replayed requests.
    planner=baseline_planner(ROUTER_THINKING_LEVEL),
    # THE routing contract — a violation is a pydantic error the dispatcher
    # treats as "exploratory at confidence 0", i.e. the deep path.
    output_schema=RouterDecision,
    output_key=ROUTER_STATE_KEY,
    before_model_callback=_router_before_model,
)


# ── reading the decision back ────────────────────────────────────────────────


def _texts_of(event: Any) -> str:
    parts = event.content.parts if event.content and event.content.parts else []
    return "".join(
        part.text
        for part in parts
        if getattr(part, "text", None) and not getattr(part, "thought", False)
    )


def _from_events(ctx: InvocationContext) -> Any:
    """The router's own event text, newest first — the fallback for a decision
    that has not landed in state (a state_delta is committed when the runner
    consumes the event, and the ordering is ADK's business, not ours).

    THIS invocation's events only. Session events are never trimmed, so an
    unfiltered newest-first scan that finds no router text this turn walks
    straight back into an earlier turn and returns ITS decision — a fast path
    run against the previous question's entity. Absent is the honest answer
    here: `decision_from` turns it into the deep path."""
    for event in reversed(ctx.session.events):
        if event.invocation_id != ctx.invocation_id:
            continue
        if event.author != ROUTER_NAME:
            continue
        text = _texts_of(event).strip()
        if not text:
            continue
        # A model occasionally wraps structured output in a ```json fence.
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(text)
        except ValueError:
            continue
    return None


def decision_from(ctx: InvocationContext) -> RouterDecision:
    """This turn's RouterDecision — from `state["router_decision"]`, else the
    router's event text, else the fallback.

    A router that emitted nothing usable costs the DEEP PATH, never the turn
    (blend 04's edge table): `FALLBACK_DECISION` is `exploratory` at
    confidence 0, which every branch in the dispatcher reads as "the executor
    handles this".
    """
    raw: Any = None
    state = getattr(getattr(ctx, "session", None), "state", None)
    if state is not None:
        try:
            raw = state.get(ROUTER_STATE_KEY)
        except AttributeError:
            raw = None
    if raw is None:
        raw = _from_events(ctx)
    if raw is None:
        logger.info("router: no decision for inv=%s — deep path", ctx.invocation_id)
        return FALLBACK_DECISION
    if isinstance(raw, RouterDecision):
        return raw
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            logger.warning("router: decision is not JSON — deep path")
            return FALLBACK_DECISION
    try:
        return RouterDecision.model_validate(raw)
    except ValidationError as exc:
        logger.warning("router: decision failed validation (%s) — deep path", exc.error_count())
        return FALLBACK_DECISION


def user_text(ctx: InvocationContext) -> str:
    """The question this invocation is answering. `user_content` is what the
    runner was called with; the event scan is the fallback for contexts built
    by hand (tests, and any caller that seeds a session directly)."""
    content = getattr(ctx, "user_content", None)
    parts = content.parts if content and content.parts else []
    text = "".join(part.text for part in parts if getattr(part, "text", None))
    if text.strip():
        return text
    for event in reversed(ctx.session.events):
        if event.author == "user":
            text = _texts_of(event)
            if text.strip():
                return text
    return ""
