"""
dispatcher.py — the branch decision (blend 04, gub-agent#23).

A `BaseAgent` of pure decision code: it reads the router's `RouterDecision`
and picks one of five branches. No LLM call, no tool call, nothing that can
fail slowly.

    workspace_personal            → abstain payload      (0 model, 0 tool calls)
    smalltalk                     → template payload     (0 model, 0 tool calls)
    confidence < CONFIDENCE_FLOOR → clarify payload      (which question?)
    FAST_INTENTS + an entity      → fast_path, deep path if it declines
    otherwise                     → deep_agent           (today's pipeline)

`choose()` is a pure function over the decision so the whole table is a unit
test (`tests/unit/test_dispatcher.py`) rather than a claim about the code.

Three details that are decisions, not accidents:

- **The floor does not apply when the bot already resolved the entity.** A
  `"User selected campaign <uuid>"` prefix means the user has JUST answered a
  disambiguation card (blend 02); asking them another question would be the
  second card the epic forbids. The prompt also raises confidence to 0.95 by
  rule in that case, so this is the belt to that suspenders — and it costs the
  deep path, not an answer.
- **The fast path declining is not an error.** It emits nothing when the
  entity is ambiguous, the slots do not assemble, or the result is empty, and
  the dispatcher then runs the deep path in the same turn: one fallback, no
  card, no second router call.
- **A broken router costs latency only.** `decision_from` returns
  `exploratory` at confidence 0 for a missing or schema-invalid decision, and
  `exploratory` is not a clarifiable intent (`CLARIFIABLE_INTENTS` below), so
  such a turn takes the deep path and behaves exactly as it does today — blend
  04's edge table, which asks for precisely that.

`CONFIDENCE_FLOOR = 0.70` is provisional — `blend-06-eval-and-thresholds.md`
ratifies it against a measured misroute rate.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from ..schemas.router import FAST_INTENTS, RouterDecision
from .answers import (
    abstain_payload,
    clarify_intent_payload,
    payload_event,
    smalltalk_payload,
)
from .fast_path import fast_path, outcome
from .router import decision_from

logger = logging.getLogger(__name__)

# Below this the router is guessing between intents — ask instead of guessing
# with it. Provisional; blend 06 ratifies it.
CONFIDENCE_FLOOR = 0.70

# Intents worth ASKING about when the router is unsure. `exploratory` is
# deliberately absent: it is already the catch-all — the intent the fallback
# decision carries and the one the executor is built to answer — so a
# clarification there would ask the user to choose between "explore this" and
# "explore this". Blend 04's edge table requires exactly this: a schema-invalid
# router output must reach the deep path, not a question.
CLARIFIABLE_INTENTS = FAST_INTENTS | {"assessment", "market_enrichment"}

# The five branches, as returned by `choose`.
ABSTAIN = "abstain"
SMALLTALK = "smalltalk"
CLARIFY = "clarify"
FAST = "fast"
DEEP = "deep"


def choose(decision: RouterDecision) -> str:
    """The decision table, as a pure function. Order matters: the two no-data
    intents win over the confidence floor (a greeting the router is only 60%
    sure is a greeting still needs no data), and the floor wins over the fast
    path (a lookup for the wrong intent is a confidently wrong answer, which
    is worse than a question)."""
    if decision.intent == "workspace_personal":
        return ABSTAIN
    if decision.intent == "smalltalk":
        return SMALLTALK
    if decision.confidence < CONFIDENCE_FLOOR:
        # See the module docstring: a resolved entity means the user has
        # already been asked once this turn.
        if decision.entity_id or decision.intent not in CLARIFIABLE_INTENTS:
            return DEEP
        return CLARIFY
    if decision.intent in FAST_INTENTS and (
        decision.entity_id
        or decision.entity_surface
        # A count needs no entity — its subject is in `slots`.
        or decision.intent == "count_or_rank"
    ):
        return FAST
    return DEEP


class Dispatcher(BaseAgent):
    """Runs one branch. `sub_agents` is `[fast_path, deep_agent]` — the fast
    path is invoked through the module singleton (it is the same object) and
    the deep path is found by elimination, so neither is keyed on an index."""

    def _deep_agent(self) -> BaseAgent:
        for agent in self.sub_agents:
            if agent.name != fast_path.name:
                return agent
        raise ValueError("dispatcher: no deep agent among sub_agents")

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        decision = decision_from(ctx)
        branch = choose(decision)
        logger.info(
            "dispatcher: intent=%s confidence=%.2f branch=%s (inv=%s)",
            decision.intent,
            decision.confidence,
            branch,
            ctx.invocation_id,
        )

        if branch == ABSTAIN:
            # GUB gets out of the way; the bot hides its section entirely
            # (`chat/cards.ts:660-675`) and the Workspace spoke owns the answer.
            yield payload_event(ctx, abstain_payload())
            return
        if branch == SMALLTALK:
            yield payload_event(ctx, smalltalk_payload(decision.language))
            return
        if branch == CLARIFY:
            yield payload_event(
                ctx,
                clarify_intent_payload(
                    decision.intent,
                    decision.language,
                    decision.entity_surface,
                ),
            )
            return

        if branch == FAST:
            async for event in fast_path.run_async(ctx):
                yield event
            if outcome(ctx.invocation_id) == "answered":
                return
            logger.info("dispatcher: fast path declined (inv=%s) — deep path", ctx.invocation_id)

        async for event in self._deep_agent().run_async(ctx):
            yield event
