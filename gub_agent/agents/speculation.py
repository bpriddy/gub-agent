"""
speculation.py — the deep path starts beside the router, not after it
(SPECULATIVE_DEEP).

The router is ONE model call in front of every turn, and on production traffic
it is slow in exactly the way a serial stage must not be: p50 5.8 s, p90 28.0 s,
max 67 s (n=212, router-era turns), because gemini-3.5-flash sometimes waits
10-67 s for its first token on the Vertex side — in day-dependent episodes,
0-84% of calls, and a paid replay showed it is not our request bytes. The deep
path is what the dispatcher picks on 155/212 of those turns (24/26 recently).
Starting it at the same moment as the router hides the router's WHOLE duration
on those turns, whatever the stall's cause: an upper bound of p50 -5.9 s,
mean -10.8 s, p90 -28 s per deep turn. What it costs on the other turns is the
speculative work thrown away: 1-2 executor rounds (~$0.015-0.03 each).

    SpeculativeDispatch("speculative_dispatch")      replaces [router, dispatcher]
      ├─ router       — run live; its events are yielded as they come
      └─ dispatcher   — its deep agent is run speculatively on a FORK; its
                        branches run exactly as `Dispatcher` runs them

How a turn goes:

1. The deep agent starts on a fork of the session (`_Speculation`): a copy of
   the state, the same event objects in a new list, its own session service.
   Every non-partial event it yields is appended to the fork, which is what the
   Runner does for the real session and what lets its sub-agents (the critic,
   the format gate, the loop's second pass) read each other's work. Every event
   — partials included — is held in a queue instead of being yielded.
2. The router runs live. The decision is read (`_decision_of_this_turn`) and
   `choose()`n exactly as the dispatcher does, and logged on the same
   `dispatcher: intent` line (`dispatcher.log_decision`).
3. The deep branch, and nothing the deep path read differs → KEPT: the held
   events are yielded in order — the Runner appends them to the real session
   and applies their state_delta, the way it would have had they been yielded
   live — and the rest are relayed as the fork produces them.
   Any other branch → the speculation is CANCELLED (task cancelled and awaited:
   in-flight model and tool calls end with it) before that branch runs, and
   nothing it produced reaches the stream, the session or the state. A deep
   branch whose inputs differ → RESTARTED: cancelled, then a fresh deep run on
   a fork that carries the real decision.

What the deep path reads from the router — the whole list, and the reason the
keep rule is one comparison (`_stale_reason`): `agents/tool_gate.py` offers
`find_files` only on a `file_lookup` turn while FILE_SEARCH_ENABLED is on. That
is the only reader of `decision_from` inside the deep path (the dispatcher and
the fast path are the others, and they run in this root, on the real session).
The speculative fork carries FALLBACK_DECISION in `state["router_decision"]`
— without it the fork would hold the PREVIOUS turn's decision, which session
state keeps across turns — so the speculative executor is offered exactly what
a turn with an unreadable decision is offered. When the real decision would
offer the tool, the speculation is not the run the dispatcher would have made.

The executor's request is the same in every path of this root, and it differs
from SPECULATIVE_DEEP=0 in one content. A fork taken before the router has
answered cannot hold the router's event, so the speculative executor never
sees ADK's rendering of it ("For context: … [router] said: {json}"); the
restarted and the fast-path-declined runs are forked the same way, with this
turn's router events left out, so they do not see it either. The critic's
request loses the same content. The alternative — pruning it from the requests
in both modes — would edit the executor's AND the critic's callbacks and change
SPECULATIVE_DEEP=0, which has to stay the exact rollback. Nothing reads that
content: the tool gate reads the decision from state, the executor and critic
prompts never mention the router, and both agents predate it (blend 04: "the
deep path is the pipeline as it was"). Earlier turns' router events are in
the fork like every other earlier event.

Why the fast path never runs BESIDE a speculation. The per-pass stores —
`round_limiter._ROUNDS`, `circuit_breaker._BUDGETS`, the evidence index and
its brief/draft/feedback stores — are in-process dicts keyed on the invocation
id, and the speculation shares the turn's id (so its log lines, its model_call
lines and its events all carry the one id the turn is known by). The fast path
resets the evidence index when it starts (`fast_path.py`, "The fast path owns
this turn's index") and records into it; a deep run beside it would lose its
evidence to that reset and have the fast path's rows mixed into its own. So a
FAST decision stops a speculation that is still running, and a fast path that
then declines is followed by a restarted deep run. One that has already
finished — the router took longer than the whole deep path — writes nothing
any more: it waits out the fast path and, on a decline, is the deep run (no
FAST intent is one the tool gate treats apart). A discarded speculation's
entries in those stores are cleared before the next branch runs
(`_Speculation.discard`), which is what the executor's own per-pass reset does
(`agent.py:_before_agent`).

Exceptions. A router that RAISES costs the deep path, not the turn: the
decision is FALLBACK_DECISION, which is what an unreadable one already costs
(`dispatcher.py`: "A broken router costs latency only") — and with the deep
path already running it costs nothing at all. A speculative run that raises
holds its exception: a kept run relays its events and then raises it, where
SPECULATIVE_DEEP=0 would have raised it; a cancelled one drops it (logged as
`error=<type>`), because the branch that ran instead never needed it — the
held-exception rule of the critic's speculation (`critic.py`).

Timestamps. A held event keeps its object, but the copy the Runner is handed
is stamped when it is relayed (monotonic, one microsecond apart at least).
Relayed as created, the kept events would be timestamped BEFORE the router's
event that precedes them in the session, and a session store that orders by
timestamp — the Vertex one appends each event with its own — would read the
turn back in another order than it was streamed.

One line per turn, next to `dispatcher: intent` (`dispatcher.log_speculation`):

    speculation: outcome=<kept|cancelled:<branch>|restarted:<reason>|off>
      lead_ms=<n|-> buffered=<n> finished=<0|1> error=<type|-> inv=… tenant=…

`lead_ms` is how far the deep run had got when the decision landed (its whole
run, when it had already finished), `buffered` how many events it held then.
`kept` + `lead_ms` is the time saved, bounded by the router's duration;
`cancelled:*` and `restarted:*` are what the speculation cost.
`cancelled:aborted` is a turn that ended before any decision (the caller went
away). `off` is logged by the plain `Dispatcher`: SPECULATIVE_DEEP=0, or a
resumable invocation, which this root runs serially (forking one would fork
its agent states, and nothing deployed is resumable).
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from types import SimpleNamespace
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.sessions import BaseSessionService, Session
from typing_extensions import override

from .. import config
from ..schemas.router import FALLBACK_DECISION, RouterDecision
from ..tenant import label_of
from . import dispatcher as dp
from .circuit_breaker import reset_tool_budget
from .evidence_index import reset_evidence_index
from .round_limiter import reset_rounds
from .router import ROUTER_STATE_KEY, decision_from
from .tool_gate import GATED_INTENT

logger = logging.getLogger(__name__)

# On by default. Off is the ROLLBACK, and it is exact: the root is then
# [sandbox_echo, router, dispatcher], today's tree, built from the same objects
# (agent.py:build_root). Read at import — a change needs a redeploy. Read here
# rather than in config.py so this change stays in one module.
SPECULATIVE_DEEP: bool = os.environ.get("SPECULATIVE_DEEP", "1").lower() in ("1", "true", "yes")

# Relayed events are stamped at least this far apart (seconds): one
# microsecond, the resolution a session store's datetime keeps.
_TICK = 1e-6


# ── what the deep path reads from the router ─────────────────────────────────


def _tool_gate_offers(decision: RouterDecision) -> bool:
    """Whether `tool_gate` (agents/tool_gate.py) offers `find_files` on a turn
    with this decision — the one thing the deep path reads from the router.
    Read off the config MODULE at call time, as the gate itself reads it."""
    return bool(config.FILE_SEARCH_ENABLED) and decision.intent == GATED_INTENT


def _stale_reason(decision: RouterDecision) -> str | None:
    """Why a speculation run on FALLBACK_DECISION is not the deep run the
    dispatcher would have made on `decision` — or None when it is."""
    if _tool_gate_offers(decision) != _tool_gate_offers(FALLBACK_DECISION):
        return GATED_INTENT
    return None


def _wrote_decision(ctx: InvocationContext) -> bool:
    """Did anything in THIS invocation commit a router decision to state?"""
    for event in ctx.session.events:
        if event.invocation_id != ctx.invocation_id:
            continue
        delta = event.actions.state_delta if event.actions else None
        if ROUTER_STATE_KEY in (delta or {}):
            return True
    return False


def _decision_of_this_turn(ctx: InvocationContext) -> RouterDecision:
    """`decision_from(ctx)`, unless this turn's router committed nothing.

    `state["router_decision"]` outlives the turn — it is ordinary session
    state, written through the router's output_key. A router that answered
    with no text writes nothing, and state then still holds the PREVIOUS
    turn's decision, which `decision_from` would read first: the previous
    question's branch, entity and all. The event scan behind it is already
    confined to this invocation for exactly that reason (router.py:
    `_from_events`), so it is asked alone, and a turn with nothing readable
    gets FALLBACK_DECISION — the deep path — as blend 04's edge table wants.
    """
    if _wrote_decision(ctx):
        return decision_from(ctx)
    this_turn_only = SimpleNamespace(
        invocation_id=ctx.invocation_id,
        session=SimpleNamespace(state={}, events=ctx.session.events),
    )
    return decision_from(this_turn_only)


# ── the fork ─────────────────────────────────────────────────────────────────


class _ForkSessionService(BaseSessionService):
    """The session service of a fork: `append_event` is BaseSessionService's —
    temp: state applied then trimmed, the state_delta applied, the event
    appended — which is the part of the Runner's append that the agents of one
    invocation can observe. Nothing outside the fork can be reached through it:
    the store-level methods refuse. (Outside live mode, no agent code calls the
    session service at all; the Runner does. This exists so a future one that
    does writes to the fork, never to the real store.)"""

    async def create_session(self, **_: Any) -> Session:
        raise NotImplementedError("a speculative fork has no store")

    async def get_session(self, **_: Any) -> Session | None:
        raise NotImplementedError("a speculative fork has no store")

    async def list_sessions(self, **_: Any) -> Any:
        raise NotImplementedError("a speculative fork has no store")

    async def delete_session(self, **_: Any) -> None:
        raise NotImplementedError("a speculative fork has no store")


def _fork_session(ctx: InvocationContext, *, decision: RouterDecision, router_name: str) -> Session:
    """A copy of the session as the deep path should find it.

    State is deep-copied (it is small, and a nested write must not reach the
    real one) and carries `decision` as the router decision. Events are the
    same objects in a new list, minus this turn's router events: past events
    are never mutated by the agents that read them (context_pruning.py copies
    on write, ADK copies contents for each request), and deep-copying a
    200-turn thread's tool payloads on every turn would be the cost this
    change exists to remove."""
    state = copy.deepcopy(dict(ctx.session.state))
    state[ROUTER_STATE_KEY] = decision.model_dump(exclude_none=True)
    events = [
        event
        for event in ctx.session.events
        if not (event.invocation_id == ctx.invocation_id and event.author == router_name)
    ]
    return ctx.session.model_copy(update={"state": state, "events": events})


def _delta_of(event: Event) -> dict | None:
    """The event's state_delta as yielded — BEFORE the fork's append trims its
    temp: keys in place — so the copy the Runner gets carries all of it."""
    delta = event.actions.state_delta if event.actions else None
    return dict(delta) if delta else None


def _for_caller(event: Event, delta: dict | None, timestamp: float) -> Event:
    """The copy of a fork event that the Runner appends to the real session.

    A copy, because the fork's event object stays in the fork's history and
    the Runner's append rewrites fields in place (the temp: trim). Same id,
    content and author; the state_delta as it was yielded; stamped now."""
    update: dict[str, Any] = {"timestamp": timestamp}
    if delta is not None and event.actions is not None:
        update["actions"] = event.actions.model_copy(update={"state_delta": delta})
    return event.model_copy(update=update)


_END = object()  # queue marker: the speculative run has ended


class _Speculation:
    """One deep run on a fork, pumped by its own task into a queue.

    The pump plays the Runner's part inside the fork — append each non-partial
    event before asking for the next — and never waits for the real caller,
    so the deep path runs on while its events wait. `relay()` hands them to
    the caller in order: first those held, then the rest as they come."""

    def __init__(
        self,
        agent: BaseAgent,
        ctx: InvocationContext,
        *,
        decision: RouterDecision,
        router_name: str,
    ) -> None:
        self.session = _fork_session(ctx, decision=decision, router_name=router_name)
        self.ctx = ctx.model_copy(
            update={
                "session": self.session,
                "session_service": _ForkSessionService(),
                # ADK keeps per-agent run state in these dicts and every
                # context copy shares them; the fork gets its own.
                "agent_states": copy.deepcopy(ctx.agent_states),
                "end_of_agents": dict(ctx.end_of_agents),
            }
        )
        self.produced = 0
        self.error: Exception | None = None
        self.discarded = False
        self._cleared = False
        self.started = time.monotonic()
        self.ended: float | None = None
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._task = asyncio.create_task(
            self._pump(agent), name=f"speculative-deep {ctx.invocation_id}"
        )

    async def _pump(self, agent: BaseAgent) -> None:
        try:
            async with aclosing(agent.run_async(self.ctx)) as events:
                async for event in events:
                    delta = _delta_of(event)
                    if not event.partial:
                        await self.ctx.session_service.append_event(self.session, event)
                    self.produced += 1
                    self._queue.put_nowait((event, delta))
        except Exception as exc:  # noqa: BLE001 — held for relay(); see the module docstring
            self.error = exc
        finally:
            self.ended = time.monotonic()
            self._queue.put_nowait(_END)

    def progress(self) -> dict[str, Any]:
        """The `speculation:` line's fields, as of now."""
        until = self.ended if self.ended is not None else time.monotonic()
        return {
            "lead_ms": round((until - self.started) * 1000),
            "buffered": self.produced,
            "finished": self.ended is not None,
            "error": type(self.error).__name__ if self.error is not None else None,
        }

    @property
    def done(self) -> bool:
        """The run has stopped: no callback of it will run again."""
        return self._task.done()

    @property
    def reusable(self) -> bool:
        """Ran to its end, raised nothing, and was not thrown away."""
        return self.done and not self.discarded and not self._task.cancelled() and not self.error

    async def relay(self) -> AsyncGenerator[Event, None]:
        """Every event of the run, in order, each as a fresh copy for the
        Runner; then the run's exception, if it raised one."""
        last = 0.0
        while True:
            item = await self._queue.get()
            if item is _END:
                if self.error is not None:
                    raise self.error
                return
            event, delta = item
            last = max(time.time(), last + _TICK)
            yield _for_caller(event, delta, last)

    async def close(self) -> None:
        """Cancel the run if it is still going, and wait until it has stopped:
        its model and tool calls are cancelled with it, and no callback of it
        can run after this returns."""
        if not self._task.done():
            self._task.cancel()
        await asyncio.wait({self._task})

    def clear(self, ctx: InvocationContext) -> None:
        """Clear what the stopped run left in the invocation-keyed stores —
        the executor's own per-pass reset (`agent.py:_before_agent`). Once: a
        second clear would wipe what the branch that ran since has written."""
        if self._cleared:
            return
        self._cleared = True
        reset_rounds(ctx)
        reset_tool_budget(ctx)
        reset_evidence_index(ctx)

    async def discard(self, ctx: InvocationContext) -> None:
        """Throw the run away before another branch of this turn runs: stop
        it, then clear its store entries."""
        await self.close()
        self.discarded = True
        self.clear(ctx)


# ── the root ─────────────────────────────────────────────────────────────────


class SpeculativeDispatch(BaseAgent):
    """The router and the dispatcher, with the deep path started beside the
    router (module docstring). `sub_agents` is `[router, dispatcher]`; the
    dispatcher's own sub-agents are the fast path and the deep agent."""

    def _parts(self) -> tuple[BaseAgent, dp.Dispatcher]:
        router = next((a for a in self.sub_agents if not isinstance(a, dp.Dispatcher)), None)
        dispatcher = next((a for a in self.sub_agents if isinstance(a, dp.Dispatcher)), None)
        if router is None or dispatcher is None:
            raise ValueError(f"{self.name}: sub_agents must be [router, dispatcher]")
        return router, dispatcher

    async def _serial(
        self, ctx: InvocationContext, router: BaseAgent, dispatcher: dp.Dispatcher
    ) -> AsyncGenerator[Event, None]:
        """Exactly the SPECULATIVE_DEEP=0 order; the dispatcher logs `off`."""
        async with aclosing(router.run_async(ctx)) as events:
            async for event in events:
                yield event
        async with aclosing(dispatcher.run_async(ctx)) as events:
            async for event in events:
                yield event

    @override
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        router, dispatcher = self._parts()
        deep = dispatcher._deep_agent()
        run: _Speculation | None = None
        if not ctx.is_resumable:
            try:
                run = _Speculation(deep, ctx, decision=FALLBACK_DECISION, router_name=router.name)
            except Exception:  # noqa: BLE001 — no speculation must never cost the turn
                logger.exception(
                    "speculation: could not fork the session (inv=%s) tenant=%s — serial",
                    ctx.invocation_id,
                    label_of(ctx),
                )
        if run is None:
            async with aclosing(self._serial(ctx, router, dispatcher)) as events:
                async for event in events:
                    yield event
            return

        logged = False
        try:
            # ── the router, live ──
            router_error: Exception | None = None
            events = router.run_async(ctx)
            try:
                while True:
                    try:
                        event = await anext(events)
                    except StopAsyncIteration:
                        break
                    except Exception as exc:  # noqa: BLE001 — see the module docstring
                        router_error = exc
                        break
                    yield event
            finally:
                await events.aclose()

            if router_error is not None:
                logger.warning(
                    "router: raised %s (inv=%s) tenant=%s — deep path",
                    type(router_error).__name__,
                    ctx.invocation_id,
                    label_of(ctx),
                    exc_info=router_error,
                )
                decision = FALLBACK_DECISION
            else:
                decision = _decision_of_this_turn(ctx)
            branch = dp.choose(decision)
            dp.log_decision(ctx, decision, branch)
            at_decision = run.progress()

            # ── the deep branch: keep, or restart ──
            if branch == dp.DEEP:
                reason = _stale_reason(decision)
                if reason is None:
                    dp.log_speculation(ctx, "kept", **at_decision)
                    logged = True
                    async with aclosing(run.relay()) as events:
                        async for event in events:
                            yield event
                    return
                await run.discard(ctx)
                dp.log_speculation(ctx, f"restarted:{reason}", **at_decision)
                logged = True
                run = _Speculation(deep, ctx, decision=decision, router_name=router.name)
                async with aclosing(run.relay()) as events:
                    async for event in events:
                        yield event
                return

            # ── every other branch runs with nothing beside it ──
            payload = dp.no_data_payload(ctx, decision, branch)
            if payload is not None:
                await run.discard(ctx)
                dp.log_speculation(ctx, f"cancelled:{branch}", **at_decision)
                logged = True
                yield payload
                return

            # The fast path, as the dispatcher runs it. A run still going is
            # stopped first (module docstring). One that has already finished
            # will never read the stores again, so only its entries go; its
            # events wait out the fast path, and a decline uses them — a FAST
            # intent is never one the tool gate treats apart.
            if run.done:
                run.clear(ctx)
            else:
                await run.discard(ctx)
            declined = False
            try:
                async with aclosing(dp.fast_path.run_async(ctx)) as events:
                    async for event in events:
                        yield event
                declined = dp.outcome(ctx.invocation_id) != "answered"
            finally:
                reuse = declined and run.reusable and _stale_reason(decision) is None
                if reuse:
                    result = "kept"
                else:
                    result = "restarted:fast_declined" if declined else f"cancelled:{dp.FAST}"
                dp.log_speculation(ctx, result, **at_decision)
                logged = True
            if not declined:
                return
            dp.log_fast_path_declined(ctx)
            if not reuse:
                await run.discard(ctx)
                run = _Speculation(deep, ctx, decision=decision, router_name=router.name)
            async with aclosing(run.relay()) as events:
                async for event in events:
                    yield event
        finally:
            # A caller that went away, an exception anywhere above, or a run
            # that ended: nothing speculative outlives the turn.
            unlogged = None if logged else run.progress()
            await run.close()
            if unlogged is not None:
                dp.log_speculation(ctx, "cancelled:aborted", **unlogged)
