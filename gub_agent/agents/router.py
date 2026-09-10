"""
router.py — the intent router (blend 04, gub-agent#23).

One LlmAgent, no tools, `output_schema=RouterDecision`, thinking at LOW: it
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
turn's 50-row org_query result is nothing but tokens.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from pydantic import ValidationError

from ..config import build_model, build_thinking_planner
from ..instruction_utils import with_current_date
from ..prompts import ROUTER_INSTRUCTION
from ..sandbox import ROUTER_THINKING_LEVEL, sandbox_before_model, sandbox_instruction
from ..schemas.router import FALLBACK_DECISION, RouterDecision
from .context_pruning import strip_prior_turn_tool_parts

logger = logging.getLogger(__name__)

ROUTER_NAME = "router"
ROUTER_STATE_KEY = "router_decision"


def _router_before_model(callback_context, llm_request):
    """Sandbox overrides first (model / router_thinking_level / temperature — a
    no-op without state["sandbox"]), then the same prior-turn pruning the
    critic has always had."""
    sandbox_before_model(callback_context, llm_request, role="router")
    return strip_prior_turn_tool_parts(callback_context, llm_request)


router_agent = LlmAgent(
    # Retry-with-backoff on 429/5xx, same as every other stage (config.build_model).
    model=build_model(),
    name=ROUTER_NAME,
    # Wrapped for the sandbox: state["sandbox"].router_instruction (or
    # .router_variant) replaces the prompt for that run only — the one
    # practical way to drive the misroute rate down without a redeploy.
    # InstructionProvider — appends today's date deterministically per request.
    instruction=sandbox_instruction(with_current_date(ROUTER_INSTRUCTION), role="router"),
    # LOW, from sandbox.py so the provenance can't drift from what runs. The
    # router exists to save model turns; giving it a deliberation budget would
    # spend them again.
    planner=build_thinking_planner(thinking_level=ROUTER_THINKING_LEVEL),
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
    consumes the event, and the ordering is ADK's business, not ours)."""
    for event in reversed(ctx.session.events):
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
