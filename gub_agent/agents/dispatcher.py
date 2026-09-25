"""
dispatcher.py — the branch decision (blend 04, gub-agent#23).

A `BaseAgent` of pure decision code: it reads the router's `RouterDecision`
and picks one of five branches. No LLM call, no tool call, nothing that can
fail slowly.

    workspace_personal            → abstain payload      (0 model, 0 tool calls)
    smalltalk                     → template payload     (0 model, 0 tool calls)
    confidence < ROUTER_CONFIDENCE_FLOOR → clarify payload  (which question?)
    FAST_INTENTS + an entity      → fast_path, deep path if it declines
    otherwise                     → deep_agent           (today's pipeline)

`choose()` is a pure function over the decision so the whole table is a unit
test (`tests/unit/test_dispatcher.py`) rather than a claim about the code.

Three details that are decisions, not accidents:

- **The floor does not apply when the bot already resolved the entity.** A
  `"User selected <type> <uuid>"` prefix means the user has JUST answered a
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

`ROUTER_CONFIDENCE_FLOOR` (config.py, default 0.70) is still UNRATIFIED. It
is the one gate in `clarify-01-always-ask-when-unsure.md` that cannot be
measured without model spend, and the dispatcher log line below is what a
reliability diagram for it would be built from.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from ..config import ROUTER_CONFIDENCE_FLOOR
from ..schemas.router import FAST_INTENTS, RouterDecision
from ..tenant import label_of
from .answers import (
    abstain_payload,
    clarify_intent_payload,
    payload_event,
    smalltalk_payload,
)
from .fast_path import fast_path, outcome
from .router import decision_from

logger = logging.getLogger(__name__)

# Intents worth ASKING about when the router is unsure. `exploratory` is
# deliberately absent: it is already the catch-all — the intent the fallback
# decision carries and the one the executor is built to answer — so a
# clarification there would ask the user to choose between "explore this" and
# "explore this". Blend 04's edge table requires exactly this: a schema-invalid
# router output must reach the deep path, not a question.
#
# `file_lookup` (search-01) is absent for a different reason: it does not steer
# a branch at all, it only decides whether the executor is offered `find_files`
# (`agents/tool_gate.py`). Offering the user "did you mean: which file?" would
# put a card in front of a question that is already going to be answered the
# same way either way.
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
    if decision.confidence < ROUTER_CONFIDENCE_FLOOR:
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


# ── the per-turn pieces, shared with the speculative root ─────────────────────
#
# `agents/speculation.py` (SPECULATIVE_DEEP) makes the same decision and runs
# the same branches as `Dispatcher` below, with the deep path started before
# the router has answered. These are the parts it takes from here rather than
# restating, so the log lines stay one string each and a branch cannot drift
# between the two roots.


def log_decision(ctx: InvocationContext, decision: RouterDecision, branch: str) -> None:
    """The per-turn `dispatcher: intent` line."""
    # `tenant` is APPENDED, never interleaved: this line is the denominator
    # of every per-turn proportion measured from the logs, and existing
    # queries match on the `intent=`/`confidence=`/`branch=` substrings.
    # One engine serves more than one branded Chat app, so without the label
    # those proportions silently mix two bots' traffic (gub_agent/tenant.py).
    logger.info(
        "dispatcher: intent=%s confidence=%.2f branch=%s (inv=%s) tenant=%s",
        decision.intent,
        decision.confidence,
        branch,
        ctx.invocation_id,
        label_of(ctx),
    )


def log_fast_path_declined(ctx: InvocationContext) -> None:
    logger.info("dispatcher: fast path declined (inv=%s) — deep path", ctx.invocation_id)


def log_speculation(
    ctx: InvocationContext,
    result: str,
    *,
    lead_ms: int | None = None,
    buffered: int = 0,
    finished: bool = False,
    error: str | None = None,
) -> None:
    """The per-turn `speculation:` line — one per turn, next to the
    `dispatcher: intent` line, so the two count the same turns.

    `result` is `kept`, `cancelled:<branch>`, `restarted:<reason>` or `off`
    (`agents/speculation.py`); `off` is this dispatcher's, which runs only
    when nothing was started beside the router. `lead_ms` is how far the
    speculative deep run had got when the router's decision landed (`-` when
    there was none), `buffered` how many of its events were held at that
    moment, `finished` whether it had already run to its end, `error` the
    type of what it raised (`-` for nothing). `tenant` last, as on every
    per-turn line."""
    logger.info(
        "speculation: outcome=%s lead_ms=%s buffered=%d finished=%d error=%s inv=%s tenant=%s",
        result,
        "-" if lead_ms is None else lead_ms,
        buffered,
        1 if finished else 0,
        error or "-",
        ctx.invocation_id,
        label_of(ctx),
    )


def no_data_payload(ctx: InvocationContext, decision: RouterDecision, branch: str) -> Event | None:
    """The one event of a branch that needs no data — abstain, smalltalk,
    clarify — or None for the two that do (fast, deep)."""
    if branch == ABSTAIN:
        # GUB gets out of the way; the bot hides its section entirely
        # (`chat/cards.ts:660-675`) and the Workspace spoke owns the answer.
        return payload_event(ctx, abstain_payload())
    if branch == SMALLTALK:
        return payload_event(ctx, smalltalk_payload(decision.language))
    if branch == CLARIFY:
        return payload_event(
            ctx,
            clarify_intent_payload(
                decision.intent,
                decision.language,
                decision.entity_surface,
            ),
        )
    return None


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
        log_decision(ctx, decision, branch)
        # This dispatcher runs after the router, never beside it: nothing was
        # speculated on this turn (SPECULATIVE_DEEP=0, or a turn the
        # speculative root ran serially).
        log_speculation(ctx, "off")

        event = no_data_payload(ctx, decision, branch)
        if event is not None:
            yield event
            return

        if branch == FAST:
            async for event in fast_path.run_async(ctx):
                yield event
            if outcome(ctx.invocation_id) == "answered":
                return
            log_fast_path_declined(ctx)

        async for event in self._deep_agent().run_async(ctx):
            yield event
