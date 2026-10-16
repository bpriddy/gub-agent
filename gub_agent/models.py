"""
models.py — one model object per agent, more than one vendor behind it.

The sandbox swaps the model per call by writing `llm_request.model`
(sandbox.py). ADK's `Gemini` honours that field for any Gemini id, so a single
Gemini instance covers the whole Gemini allowlist. A Claude id needs a different
LLM class (ADK `Claude` — Anthropic on Vertex), and an LlmAgent's model is fixed
at construction. `VendorRouter` is that fixed object: it reads the id the
sandbox wrote and hands the request to the right vendor client.

What the router changes for a Claude request, and why:

  thinking     Claude has no named thinking levels. The sandbox validator
               already forces DYNAMIC for any model outside
               SANDBOX_THINKING_LEVEL_MODELS, and DYNAMIC is
               `thinking_budget=-1`, which ADK maps to Anthropic adaptive
               thinking — nothing to translate. A named level reaching here is
               a bug upstream; it is replaced by adaptive and logged rather than
               raised, because an exception inside the engine is an empty 200
               stream to the caller (gub-agent README, "How a failed sandbox
               run looks").
  JSON output  The critic runs with `output_schema` and no tools, which ADK
               realises as Gemini's native JSON mode (`response_schema` on the
               request config). The Anthropic path ignores that field, so the
               router states the schema in the system instruction and trims
               anything around the JSON object in the reply; ADK's
               `validate_schema` then parses it exactly as it parses Gemini's.
  temperature  ADK drops sampling parameters when thinking is on for Claude
               (Opus 4.6+ reject them). A sandbox `temperature` on a Claude arm
               is therefore inert — logged per request so nobody reads a
               difference into it.

Everything else — tools and tool results, the system prompt, streaming,
thought parts (`Part.thought=True`) — ADK's Claude class maps itself.

Prod is untouched: with SANDBOX_ENABLED=0 the sandbox never writes
`llm_request.model`, so the router only ever sees the deployed Gemini id and
forwards the request object unchanged.

Every call through the router — either vendor — ends in ONE `model_call:` log
line (`_ModelCall`): time to first chunk, time to last chunk, the token counts
and the outcome. It is the $0 measurement for latency work: the engine logs
already carry it, so no eval run is needed to see where a turn's time went.

The router hedge (ROUTER_HEDGE_AFTER_MS, config.py). The router agent's first
token sometimes waits 10-67 s inside Vertex, in day-dependent episodes, and a
replay of stalled production requests did not stall: the condition is the
server's, not the request's. So a ROUTER call (ADK's `adk_agent_name` label,
the one the `model_call:` line names) that has sent no first chunk after that
many milliseconds is sent a second time — a copy of the same request object,
taken before the first send, so the same model, contents and config, a sandbox
run's per-call overrides included — and the stream that yields a first chunk
first is relayed; the other is cancelled (`_hedged`). The answer is one of two
identical requests' answers: only when it arrives changes.

- Each request runs in a task of its own (`_Leg`), which reads its stream one
  chunk per ask — nothing is read ahead of ADK, so the clocks still stop where
  ADK reads — and closes it in that task. Each keeps its own `model_call:`
  line: the loser's says `status=error:cancelled`, and the second request's
  ends in ` hedge=1`, after `tenant=` like the speculation's ` spec=1`, so
  `NOT textPayload:"hedge=1"` still counts one router line per call. When the
  second one wins, the first one's line is cancelled at its first chunk: its
  `dur_ms` is the wait the turn saw.
- A failure before the hedge fires is today's failure. After it has fired, a
  failed request leaves the other to answer; when both fail, the first one's
  error is raised, as it would have been. A request genai is already
  retrying (a 429/5xx answered it; `retries=`) is not hedged. Once the hedge
  has fired, though, genai may retry either request on its own budget: each
  line's `retries=` says so.
- The caller's cancellation — the turn aborted, the bot gone — cancels both,
  and nothing outlives the call.
- One line when it fires, once the race is decided:

    router_hedge: fired=1 winner=<primary|hedge|-> primary_ttft_ms=<n|->
      hedge_ttft_ms=<n|-> inv=… tenant=…

  Each first-token time is counted from its own request's start (`-`: none
  arrived); `winner=-`: neither answered — both failed, or the caller left
  first. Against `dispatcher: intent`
  with the same `tenant=` clause it is the share of turns that paid for a
  second router call (~$0.004 for a short session's 2.6k-token router prompt,
  ~$0.03 for the ~20k-token uncached prompt of a long one).

The executor and formatter hedges (EXECUTOR_HEDGE_AFTER_MS,
FORMATTER_HEDGE_AFTER_MS). The same mechanism, keyed on the same label, for
the two agents whose stalls a reader waits through on a deep turn: an executor
ReAct round's first token has waited 102.7 s with the request healthy (no
retry), and a formatter's 62 s. Each agent has its own deadline
(`agent_hedge_after_ms`), because a healthy first chunk takes very different
times — the executor thinks before its first chunk, the router and the
formatter do not. Everything above holds for them as for the router: one copy
of the request taken before the first send, one stream relayed whole, the
loser cancelled and logged, no second send while genai retries.

- Streamed calls only. "No first chunk after the deadline" means a late
  first TOKEN only when the call streams (the run's RunConfig is SSE, as the
  Chat bots run it): a call without streaming has exactly one chunk, the
  whole reply, so its deadline would bound the whole generation and fire on
  every long, healthy executor round or formatter brief — a sandbox /batch
  run, a caller with no run_config. Those take the unhedged path. The
  router's reply is short enough that its hedge keeps covering both.
- Nothing a model call does has a side effect: the executor's tools run in
  ADK AFTER the model's reply has been read, from the relayed (winning) stream
  only, so a duplicated request never duplicates a GUB call.
- The router keeps its `router_hedge:` line, byte for byte. A fired executor
  or formatter hedge logs its own line instead (added, never a renamed one):

    model_hedge: agent=<name> fired=1 winner=<primary|hedge|->
      primary_ttft_ms=<n|-> hedge_ttft_ms=<n|-> inv=… tenant=…

  Against the agent's `model_call:` lines without ` hedge=1`, it is the share
  of that agent's calls that paid for a second send — at the executor's
  prompt sizes (20-65k tokens, often uncached) ~$0.03-0.10 per fire, up to
  ~$0.11 for a 75k formatter brief. Input only: the cancelled request may
  also have spent thinking tokens, and its `model_call:` line reports none of
  its usage (no usage arrives before its first chunk), so a token sum over
  `model_call:` lines does not include the second sends.

The critic hedge (CRITIC_HEDGE_AFTER_MS). The same mechanism for the critic,
whose streamed first token stalls the same way and which a pass waits for: its
own deadline in `agent_hedge_after_ms`, its own `model_hedge: agent=critic …`
line. A speculative critic call that the pass has already settled in code is
cancelled by the pass (critic.py, `settle_speculation`): the cancellation lands
in `_hedged` like any caller's and every request that is out is stopped, each
writing its `model_call:` line as `status=error:cancelled` — both once the spare
has been sent (and then one `model_hedge:` line with `winner=-`), only the first
when the deadline has not yet passed (and then none).

The hedge cap. Each fired hedge is an extra request, so one invocation may fire
only so many (`_HEDGE_LIMIT`): two for the executor and the formatter together,
one for the critic. The router is outside it: one call per invocation, at most
one spare.

- The budget belongs to the invocation id. Neither a new pass, nor a discarded
  speculation, nor the speculative deep run's restart resets it: it bounds the
  money actually spent, and the fast path, a discarded deep run and the kept one
  share one invocation id.
- It is spent only when a hedge fires. A call that would exceed it takes the
  unhedged path — it waits on its own first request and relays it whole — and
  logs once, at the decision point,

    hedge_capped: agent=<name> after_ms=<n> used=<n> limit=<n> inv=… tenant=…

  never a `model_hedge:` line, since no second request was sent.
- A call with no bound invocation (`inv=-`) is never capped, the store keeps
  the `_MAX_TRACKED` newest invocations (an evicted invocation starts again with
  a full budget) and any failure inside the budget code lets the hedge fire: a
  cap must never cost a call.

The answer round's deadline (EXECUTOR_SYNTHESIS_HEDGE_AFTER_MS). The executor's
round past the ReAct budget (round_limiter.py) is the one that writes the
answer, and its first chunk arrives after ALL of its thinking: in a local run two
rounds of 4.3k and 6.5k thought tokens answered at 26.5 s and 28.9 s with
nothing wrong with the request. A hedge at the tool rounds' deadline fires on
those and loses to its own first request, one prompt of input spent and one of
the invocation's two executor-or-formatter hedges used up. So that round has a
deadline of its own, `synthesis_hedge_after_ms`; the tool rounds keep
`agent_hedge_after_ms`.

- Only the executor has such a round. `round_limit` marks the call
  (`mark_synthesis_round`) after `bind_model_call` named the caller, so the mark
  lasts for that one model call: the next call binds a fresh, unmarked caller.
- 0 (the default) means the round has no deadline of its own: every executor
  round takes the executor's, as before. The executor's own deadline at 0 still
  means no executor hedge at all, whatever this one says.
- Added, for a marked call, to the line of a hedge that fired (`model_hedge:`)
  and to the line of one the cap withheld (`hedge_capped:`), before `inv=`:
  `round=synthesis`. On an executor line its absence means a tool round, so the
  two are told apart by `textPayload:"round=synthesis"`; it is written whatever
  this deadline is, so the baseline can be read from the same lines.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from collections import Counter, OrderedDict
from collections.abc import AsyncGenerator
from contextlib import aclosing
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
from pydantic import Field, PrivateAttr
from typing_extensions import override

from .config import AGENT_NAME, CRITIC_AGENT_NAME, FORMATTER_AGENT_NAME
from .tenant import label_of

logger = logging.getLogger("gub_agent.models")

CLAUDE_PREFIX = "claude-"

#: The agent whose calls are hedged: agents/router.py's ROUTER_NAME, which ADK
#: puts in every request's `adk_agent_name` label. Not imported from there —
#: that module imports this one; a test pins the two together. Its deadline is
#: `router_hedge_after_ms` and its line `router_hedge:`; every other agent's is
#: `agent_hedge_after_ms` and `model_hedge:`.
HEDGED_AGENT = "router"


def is_claude(model_id: str | None) -> bool:
    """True for an Anthropic id as Vertex names it (`claude-sonnet-5`, `claude-haiku-4-5@…`)."""
    return bool(model_id) and str(model_id).startswith(CLAUDE_PREFIX)


class VendorRouter(BaseLlm):
    """A BaseLlm that dispatches on `llm_request.model`: Gemini ids to one shared
    Gemini client, Claude ids to a per-id ADK `Claude` (Vertex) client created on
    first use. `model` (BaseLlm's field) is the deployed default — what runs when
    the sandbox wrote nothing."""

    #: The Gemini client every non-Claude request goes to (typed as BaseLlm so a
    #: test can stand in a recorder; production passes ADK's Gemini).
    gemini: BaseLlm
    #: Vertex location for Anthropic models. Claude on Vertex serves from the
    #: `global` endpoint; overridable per deploy (CLAUDE_VERTEX_LOCATION).
    claude_location: str = "global"
    claude_max_tokens: int = 8192
    #: Milliseconds a router call may go without a first chunk before its
    #: request is sent a second time (module docstring, "The router hedge");
    #: 0 never does. config.build_model passes ROUTER_HEDGE_AFTER_MS.
    router_hedge_after_ms: int = 0
    #: The same deadline for other agents, by the name the `model_call:` line
    #: uses (module docstring, "The executor and formatter hedges"); an agent
    #: absent here, or at 0, never hedges. config.build_model passes the
    #: executor's, the formatter's and the critic's (EXECUTOR_/FORMATTER_/
    #: CRITIC_HEDGE_AFTER_MS).
    agent_hedge_after_ms: dict[str, int] = Field(default_factory=dict)
    #: The executor's deadline for its round past the ReAct budget (module
    #: docstring, "The answer round's deadline"); 0 = the executor's own, as for
    #: every other round. config.build_model passes
    #: EXECUTOR_SYNTHESIS_HEDGE_AFTER_MS.
    synthesis_hedge_after_ms: int = 0

    _claude: dict[str, BaseLlm] = PrivateAttr(default_factory=dict)

    @classmethod
    @override
    def supported_models(cls) -> list[str]:
        # Never registered by name — agents receive an instance — but honest.
        return [r".*"]

    def _claude_for(self, model_id: str) -> BaseLlm:
        llm = self._claude.get(model_id)
        if llm is not None:
            return llm
        try:
            from google.adk.models.anthropic_llm import Claude  # noqa: PLC0415 — optional vendor
        except ImportError as exc:
            raise RuntimeError(
                "sandbox: a Claude model was requested but the `anthropic[vertex]` package "
                "is not in this build — add it to gub_agent/requirements.txt and redeploy."
            ) from exc
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        # The Claude class parses project/location out of a full resource path for
        # its Vertex client and still calls the API with the bare id from
        # llm_request.model — so this pins where the model is served without
        # touching the genai client's GOOGLE_CLOUD_LOCATION pin (config.py).
        resource = (
            f"projects/{project}/locations/{self.claude_location}/publishers/anthropic/models/{model_id}"
            if project
            else model_id
        )
        llm = Claude(model=resource, max_tokens=self.claude_max_tokens)
        self._claude[model_id] = llm
        logger.info("sandbox: Claude client created for %s (%s)", model_id, self.claude_location)
        return llm

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        agent = _agent_of(llm_request)
        after_ms = self._hedge_after_ms(agent, stream)
        if after_ms > 0:
            spare = _spare_of(llm_request, agent)
            if spare is not None:
                async with aclosing(
                    self._hedged(llm_request, spare, stream, after_ms=after_ms, agent=agent)
                ) as responses:
                    async for response in responses:
                        yield response
                return
        # The same responses in the same order, only timed on the way through:
        # the measurement sits here because this is the one object that sees
        # every chunk of every agent's call, whichever vendor serves it.
        call = _ModelCall.begin(self, llm_request, stream)
        try:
            async for response in self._generate(llm_request, stream):
                call.saw(response)
                yield response
        except GeneratorExit:
            call.status = "error:closed"  # the consumer stopped reading
            raise
        except asyncio.CancelledError:
            call.status = "error:cancelled"  # e.g. the caller's stream deadline
            raise
        except Exception as exc:
            call.status = f"error:{_error_code(exc)}"
            raise
        finally:
            call.log()

    def _hedge_after_ms(self, agent: str, stream: bool) -> int:
        """The hedge deadline for this agent's calls, in ms; 0 = never hedged.
        Every agent but the router is hedged only on a streamed call (module
        docstring, "Streamed calls only")."""
        if agent == HEDGED_AGENT:
            return self.router_hedge_after_ms
        if not stream:
            return 0
        after_ms = self.agent_hedge_after_ms.get(agent, 0)
        if after_ms > 0 and self.synthesis_hedge_after_ms > 0 and agent == AGENT_NAME:
            # The executor's answer round, marked by round_limit for this call.
            caller = _CALLER.get()
            if caller is not None and caller.agent == agent and caller.synthesis:
                return self.synthesis_hedge_after_ms
        return after_ms

    async def _generate(
        self, llm_request: LlmRequest, stream: bool
    ) -> AsyncGenerator[LlmResponse, None]:
        if not is_claude(llm_request.model):
            async for response in self.gemini.generate_content_async(llm_request, stream=stream):
                yield response
            return

        llm = self._claude_for(str(llm_request.model))
        request, wants_json = adapt_request_for_claude(llm_request)
        async for response in llm.generate_content_async(request, stream=stream):
            yield trim_json_reply(response) if wants_json and not response.partial else response

    async def _hedged(
        self,
        llm_request: LlmRequest,
        spare: LlmRequest,
        stream: bool,
        *,
        after_ms: int,
        agent: str,
    ) -> AsyncGenerator[LlmResponse, None]:
        """A call whose request is sent a second time, as `spare`, when its
        first chunk is `after_ms` late (module docstring, "The router hedge" —
        the same for every hedged agent but for its line)."""
        primary = _Leg(self, llm_request, stream, hedge=False)
        legs = [primary]
        stopping = "error:cancelled"  # what a request still going at the end logs
        reported = False  # the `router_hedge:` / `model_hedge:` line, once per fired hedge
        try:
            asked = primary.ask()
            await asyncio.wait({asked}, timeout=after_ms / 1000)
            if asked.done() or primary.call.retries:
                # In time — a failure too, raised below exactly as it is
                # unhedged — or genai is already retrying it: no second send.
                winner, outcome = primary, await asked
            elif not (take := _take_hedge(agent, primary.call.caller))[0]:
                # This invocation has fired all the hedges its budget allows
                # (module docstring, "The hedge cap"): the call waits on its own
                # request, exactly as unhedged. The check and the spend in
                # _take_hedge have no await between them.
                _log_capped(agent, after_ms, take, primary.call.caller)
                winner, outcome = primary, await asked
            else:
                hedge = _Leg(self, spare, stream, hedge=True)
                legs.append(hedge)
                winner, outcome = await _race(primary, asked, hedge, hedge.ask())
                reported = True
                _log_hedge(agent, primary, hedge, winner)
                for leg in legs:
                    if leg is not winner:
                        leg.stop("error:cancelled")
                if winner is None:  # both failed: the primary's error, as unhedged
                    raise outcome.error
            while outcome is not _END:
                if isinstance(outcome, _Failed):
                    raise outcome.error
                yield outcome
                outcome = await winner.ask()
        except GeneratorExit:
            stopping = "error:closed"  # the consumer stopped reading
            raise
        finally:
            # A caller that went away, a winner that failed, or a stream that
            # ended: both requests are stopped and have ended when this returns.
            for leg in legs:
                leg.stop(stopping)
            await asyncio.wait({leg.task for leg in legs})
            if len(legs) > 1 and not reported:
                # Fired, and the caller left before either answered: the
                # second send was still paid for, so it is still counted.
                _log_hedge(agent, primary, legs[1], None)


# ── The per-call line ─────────────────────────────────────────────────────────
#
#   model_call: agent=… model=… stream=0|1 ttft_ms=<n>|- dur_ms=… prompt=… cached=…
#     thoughts=… out=… status=ok|error:<code> retries=<n>|- inv=… tenant=…
#
# One line per call, written when the call ends — never per streamed chunk.
# `ttft_ms` is the wait for the first chunk (a non-streamed call has one, so
# there it equals `dur_ms`), and `-` when none arrived: a call cancelled or
# failed before its first chunk has no first-token time, and its duration in
# that field would read as a first-token stall. `dur_ms` runs to the LAST
# chunk (to the end of the call, when none arrived), not to the end of the
# generator: after a function-call reply ADK runs the tools while this
# generator is suspended, and tool time is not model time. Both are measured
# where ADK reads them, so a streamed call's gaps include ADK's own handling of
# each chunk. Token counts are the last usage_metadata the call reported (0
# when it reported none, as a call with no chunk never does): `cached` is the
# part of `prompt` served from the context cache, `thoughts` are billed as
# output beside `out`. `prompt=` is a COUNT: no request or reply text is ever
# logged here.
#
# Count calls with `textPayload:"model_call: agent="`. `tenant=` goes last, as
# on the dispatcher line — numerator and denominator need the same clause.


@dataclass(frozen=True)
class _Caller:
    """Who is about to call the model, as its before_model_callback saw it."""

    agent: str
    invocation_id: str
    tenant: str
    #: The executor's round past the ReAct budget, the one that writes the
    #: answer (`mark_synthesis_round`); False until round_limit says otherwise.
    synthesis: bool = False


# Set by `bind_model_call`, read by the call that follows it. A ContextVar and
# not a module global because ParallelAgent runs its branches as asyncio tasks
# (parallel_agent.py), each with its own copy of the context: the format gate's
# formatter and the speculative critic bind and call side by side without
# seeing each other's caller.
_CALLER: ContextVar[_Caller | None] = ContextVar("gub_model_caller", default=None)
_IN_FLIGHT: ContextVar[_ModelCall | None] = ContextVar("gub_model_call", default=None)


def bind_model_call(callback_context: Any) -> None:
    """First step of every model agent's before_model_callback: record the
    agent, invocation and tenant for the `model_call:` line of the call that
    follows. The model object never sees the invocation, so the one place that
    does hands it over; the request is not touched.

    Every LlmAgent in the tree calls this (pinned by
    tests/unit/test_model_call_log.py). A call it did not precede still logs,
    with `inv=- tenant=-` rather than a neighbour's."""
    _CALLER.set(
        _Caller(
            agent=getattr(callback_context, "agent_name", None) or "-",
            invocation_id=getattr(callback_context, "invocation_id", None) or "-",
            tenant=label_of(callback_context),
        )
    )


def mark_synthesis_round() -> None:
    """Called by round_limit for the executor's round past the ReAct budget, after
    `bind_model_call` bound the caller: the model call that follows is the answer
    round (module docstring, "The answer round's deadline"). The caller is
    replaced, never edited — it is frozen and a copy of it may sit in a call
    already in flight — and the next `bind_model_call` replaces it again, so the
    mark covers this one call. With no bound caller there is nothing to mark."""
    caller = _CALLER.get()
    if caller is not None:
        _CALLER.set(replace(caller, synthesis=True))


# genai retries 429/408/5xx inside the request (HttpRetryOptions, config.py)
# and says so only by logging "Retrying …" through tenacity's before_sleep hook
# at INFO on this logger — there is no callback. A logging filter reads those
# records without touching genai: it counts the retry against the call in
# flight in the same task and always lets the record through. The logger name
# and the message are genai's, not an API; if either changes the count reads 0,
# never wrong in the other direction. And a record below the logger's level is
# never created, so where INFO is off for genai the line says `retries=-`.
_GENAI_LOGGER = logging.getLogger("google_genai._api_client")


class _CountRetries(logging.Filter):
    gub_retry_counter = True

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if str(record.msg).startswith("Retrying"):
                call = _IN_FLIGHT.get()
                if call is not None:
                    call.retries += 1
        except Exception:  # noqa: BLE001 — a log filter must never fail genai's request
            pass
        return True


if not any(getattr(f, "gub_retry_counter", False) for f in _GENAI_LOGGER.filters):
    _GENAI_LOGGER.addFilter(_CountRetries())


def _error_code(exc: BaseException) -> str:
    """The HTTP status of a failed call when the SDK carries one (genai's
    APIError.code, Anthropic's status_code), else the exception's type."""
    for attr in ("code", "status_code"):
        code = getattr(exc, attr, None)
        if isinstance(code, (int, str)) and str(code).strip():
            return _token(code)
    return type(exc).__name__


def _token(value: object) -> str:
    """One space-free token, so the line splits on spaces."""
    return "_".join(str(value).split()) or "-"


def _count(usage: Any, field_name: str) -> int:
    value = getattr(usage, field_name, None) if usage is not None else None
    return int(value) if isinstance(value, (int, float)) else 0


class _ModelCall:
    """One model call's clock, token counts and outcome, logged once."""

    def __init__(self, agent: str, model: str, stream: bool, caller: _Caller | None, genai: bool):
        self.agent = agent
        self.model = model
        self.stream = stream
        # The caller counts only if it bound for THIS agent; anything else is a
        # leftover from an earlier call in the same task.
        self.caller = caller if caller is not None and caller.agent == agent else None
        self.genai = genai
        self.started = time.monotonic()
        self.first: float | None = None
        self.last: float | None = None
        self.usage: Any = None
        self.status = "ok"
        self.retries = 0
        self.hedge = False  # the second request of a hedged call

    @classmethod
    def begin(cls, router: VendorRouter, llm_request: LlmRequest, stream: bool) -> _ModelCall:
        caller = _CALLER.get()
        agent = _agent_of(llm_request)
        model = str(llm_request.model or router.model)
        call = cls(agent, model, stream, caller, genai=not is_claude(model))
        _IN_FLIGHT.set(call)
        return call

    def saw(self, response: LlmResponse) -> None:
        now = time.monotonic()
        if self.first is None:
            self.first = now
        self.last = now
        if response.usage_metadata is not None:
            self.usage = response.usage_metadata
        if response.error_code:
            # A reply with no usable content (a block, a malformed call).
            self.status = f"error:{_token(response.error_code)}"

    @property
    def ttft_ms(self) -> int | str:
        """Milliseconds to the first chunk, `-` when none has arrived."""
        return "-" if self.first is None else round((self.first - self.started) * 1000)

    def log(self) -> None:
        end = time.monotonic()
        last = self.last if self.last is not None else end
        observable = self.genai and _GENAI_LOGGER.isEnabledFor(logging.INFO)
        (logger.info if self.status == "ok" else logger.warning)(
            "model_call: agent=%s model=%s stream=%d ttft_ms=%s dur_ms=%d prompt=%d "
            "cached=%d thoughts=%d out=%d status=%s retries=%s inv=%s tenant=%s"
            + (HEDGE_MARK if self.hedge else ""),
            self.agent,
            self.model,
            1 if self.stream else 0,
            self.ttft_ms,
            round((last - self.started) * 1000),
            _count(self.usage, "prompt_token_count"),
            _count(self.usage, "cached_content_token_count"),
            _count(self.usage, "thoughts_token_count"),
            _count(self.usage, "candidates_token_count"),
            self.status,
            self.retries if observable else "-",
            self.caller.invocation_id if self.caller else "-",
            self.caller.tenant if self.caller else "-",
        )


def _agent_of(llm_request: LlmRequest) -> str:
    """The calling agent, as the `model_call:` line names it."""
    config = getattr(llm_request, "config", None)
    labels = getattr(config, "labels", None) or {}
    caller = _CALLER.get()
    # ADK labels every request with the calling agent's name (base_llm_flow:
    # _ADK_AGENT_NAME_LABEL_KEY) just before it calls the model.
    return labels.get("adk_agent_name") or (caller.agent if caller else "-")


# ── The router hedge (module docstring) ───────────────────────────────────────

# Appended to the second request's `model_call:` line, after `tenant=`.
HEDGE_MARK = " hedge=1"

_END = object()  # a leg's outcome: its stream ended


@dataclass(frozen=True)
class _Failed:
    """A leg's outcome: its request raised. Handed to the reader as a value,
    never left as the task's exception: the reader decides which failure the
    caller sees, and no task exception goes unretrieved."""

    error: Exception


def _spare_of(llm_request: LlmRequest, agent: str = HEDGED_AGENT) -> LlmRequest | None:
    """The request a hedge sends: a deep copy taken NOW, before the first
    send. The vendor client edits the request it is handed — ADK's Gemini
    appends a user turn after a model one and merges its tracking headers
    into `config.http_options` — so a copy taken later would not be what the
    first request sent, and two sends must never share one object. About
    1 ms for a 20k-token router request, a few ms for an executor round with
    its tool results. None if it cannot be copied: the call then runs
    unhedged, exactly as with the flag off."""
    try:
        return llm_request.model_copy(deep=True)
    except Exception as exc:  # noqa: BLE001 — a hedge must never cost the call
        if agent == HEDGED_AGENT:
            logger.warning("router_hedge: request not copyable (%s) — unhedged", type(exc).__name__)
        else:
            caller = _CALLER.get()
            logger.warning(
                "model_hedge: agent=%s request not copyable (%s) — unhedged inv=%s tenant=%s",
                agent,
                type(exc).__name__,
                caller.invocation_id if caller else "-",
                caller.tenant if caller else "-",
            )
        return None


def _settle(asked: asyncio.Future[Any] | None, outcome: Any) -> None:
    if asked is not None and not asked.done():
        asked.set_result(outcome)


class _Leg:
    """One request of a hedged call, in a task of its own.

    The task owns the vendor's generator from its first step to its close and
    advances it one chunk per `ask()`: the stream is read when ADK reads it,
    never ahead (so `dur_ms` still stops at the chunk ADK read last), and
    never from another task — aiohttp's timeouts and OpenTelemetry's context
    are bound to the task that entered them. The request's `model_call:` line
    is written by the task when the request ends, however it ends."""

    def __init__(
        self, router: VendorRouter, llm_request: LlmRequest, stream: bool, *, hedge: bool
    ) -> None:
        self.role = "hedge" if hedge else "primary"
        # Begun here, in the caller's context, so the task's copy of that
        # context holds THIS call as the one in flight: genai's retries inside
        # the task count on this request's line, never on the other's.
        self.call = _ModelCall.begin(router, llm_request, stream)
        self.call.hedge = hedge
        self._asked: asyncio.Future[Any] | None = None
        self._wake = asyncio.Event()
        self._started = False
        self._stop_status: str | None = None
        self.task = asyncio.create_task(
            self._run(router._generate(llm_request, stream)),
            name=f"{self.call.agent}-{self.role}",
        )

    def ask(self) -> asyncio.Future[Any]:
        """The next outcome — a response, `_END` or `_Failed` — as a future
        the task fulfils. One at a time: the next ask follows its answer."""
        self._asked = asyncio.get_running_loop().create_future()
        self._wake.set()
        return self._asked

    def stop(self, status: str) -> None:
        """Cancel the request if it is still going; its line says `status`
        (the first one given)."""
        if self.task.done():
            return
        if self._stop_status is None:
            self._stop_status = status
        if not self._started:
            # Cancelled before its first step, the task never runs its body:
            # nothing was sent, and the line is written here instead.
            self.call.status = self._stop_status
            self.call.log()
        self.task.cancel()

    async def _run(self, responses: AsyncGenerator[LlmResponse, None]) -> None:
        self._started = True
        call = self.call
        asked: asyncio.Future[Any] | None = None
        try:
            while True:
                await self._wake.wait()
                self._wake.clear()
                asked = self._asked
                try:
                    response = await anext(responses)
                except StopAsyncIteration:
                    _settle(asked, _END)
                    return
                call.saw(response)
                _settle(asked, response)
        except asyncio.CancelledError as exc:
            call.status = self._stop_status or "error:cancelled"
            # Not always our stop(): the vendor can raise one itself (a
            # library-internal cancel). The reader hears of it as of any
            # failure (unhedged, it reaches the caller at once) and never
            # waits on it. After a stop() nobody reads this future any more.
            _settle(asked, _Failed(exc))
            raise
        except Exception as exc:  # noqa: BLE001 — handed to the reader as _Failed
            call.status = f"error:{_error_code(exc)}"
            _settle(asked, _Failed(exc))
        finally:
            try:
                # Closed here, in its own task. A stream that will not close
                # cleanly is abandoned either way; its error must not become
                # the task's.
                with contextlib.suppress(Exception):
                    await responses.aclose()
            finally:
                call.log()


async def _race(
    primary: _Leg, first_p: asyncio.Future[Any], hedge: _Leg, first_h: asyncio.Future[Any]
) -> tuple[_Leg | None, Any]:
    """The leg whose first outcome is a response (or the end of its stream),
    and that outcome. A leg that failed leaves the race to the other; when
    both have failed there is no winner, and the outcome is the primary's
    failure."""
    waiting = {first_p: primary, first_h: hedge}
    failed: dict[str, _Failed] = {}
    while waiting:
        done, _ = await asyncio.wait(set(waiting), return_when=asyncio.FIRST_COMPLETED)
        # Both in the same step: the primary first — a tie never demotes it.
        for asked in sorted(done, key=lambda future: waiting[future] is not primary):
            leg = waiting.pop(asked)
            outcome = asked.result()
            if not isinstance(outcome, _Failed):
                return leg, outcome
            failed[leg.role] = outcome
    return None, failed[primary.role]


def _log_hedge(agent: str, primary: _Leg, hedge: _Leg, winner: _Leg | None) -> None:
    """The fired hedge's one line: the router's `router_hedge:` as it always
    was, every other agent's `model_hedge: agent=…` (module docstring)."""
    caller = primary.call.caller
    where = (
        winner.role if winner is not None else "-",
        primary.call.ttft_ms,
        hedge.call.ttft_ms,
        caller.invocation_id if caller else "-",
        caller.tenant if caller else "-",
    )
    if agent == HEDGED_AGENT:
        logger.info(
            "router_hedge: fired=1 winner=%s primary_ttft_ms=%s hedge_ttft_ms=%s inv=%s tenant=%s",
            *where,
        )
    else:
        # `round=synthesis` only for the executor's answer round (module
        # docstring), before `inv=`, so `tenant=` stays last.
        answer_round = " round=synthesis" if caller is not None and caller.synthesis else ""
        logger.info(
            "model_hedge: agent=%s fired=1 winner=%s primary_ttft_ms=%s hedge_ttft_ms=%s"
            + answer_round
            + " inv=%s tenant=%s",
            agent,
            *where,
        )


# ── The hedge cap (module docstring) ──────────────────────────────────────────

#: The budget an agent's hedges are drawn from. The router is in none.
_HEDGE_GROUP = {
    AGENT_NAME: "exec_fmt",
    FORMATTER_AGENT_NAME: "exec_fmt",
    CRITIC_AGENT_NAME: "critic",
}
#: How many hedges one invocation may fire, per budget.
_HEDGE_LIMIT = {"exec_fmt": 2, "critic": 1}
#: Hedges fired so far, by invocation id and budget. Unlocked module state, like
#: round_limiter._ROUNDS: an invocation runs on one event loop, and its check and
#: spend (`_take_hedge`) have no await between them.
_HEDGES: OrderedDict[str, Counter[str]] = OrderedDict()
_MAX_TRACKED = 256  # the bound and the eviction round_limiter._ROUNDS has


def _take_hedge(agent: str, caller: _Caller | None) -> tuple[bool, int, int]:
    """Spend one of the invocation's hedges for this agent's budget:
    `(allowed, used, limit)`, with `used` counting the one just spent when it
    is allowed. The router, an agent in no budget and a call with no bound
    invocation are never capped; a failure here lets the hedge fire."""
    group = _HEDGE_GROUP.get(agent)
    inv = caller.invocation_id if caller else "-"
    if group is None or inv == "-":
        return True, 0, 0
    limit = _HEDGE_LIMIT[group]
    try:
        spent = _HEDGES.get(inv)
        if spent is None:
            spent = _HEDGES[inv] = Counter()
            while len(_HEDGES) > _MAX_TRACKED:
                _HEDGES.popitem(last=False)
        else:
            _HEDGES.move_to_end(inv)
        if spent[group] >= limit:
            return False, spent[group], limit
        spent[group] += 1
        return True, spent[group], limit
    except Exception:  # noqa: BLE001 — a cap must never cost the call: fail open
        return True, 0, limit


def _log_capped(
    agent: str, after_ms: int, take: tuple[bool, int, int], caller: _Caller | None
) -> None:
    """The one line of a hedge the cap withheld (module docstring)."""
    _, used, limit = take
    # `round=synthesis` for the executor's answer round, as on the fired line
    # (module docstring), before `inv=`: with the deadlines equal, `after_ms=`
    # alone would not tell a withheld answer-round hedge from a tool round's.
    answer_round = " round=synthesis" if caller is not None and caller.synthesis else ""
    logger.info(
        "hedge_capped: agent=%s after_ms=%d used=%d limit=%d" + answer_round + " inv=%s tenant=%s",
        agent,
        after_ms,
        used,
        limit,
        caller.invocation_id if caller else "-",
        caller.tenant if caller else "-",
    )


# ── Request adaptation ────────────────────────────────────────────────────────

_JSON_INSTRUCTION = (
    "OUTPUT FORMAT: reply with exactly one JSON object that conforms to the JSON "
    "schema below — no prose before or after it, no markdown fences, no comments.\n"
    "JSON schema:\n{schema}"
)


def _schema_text(schema: Any) -> str:
    """The output schema as JSON Schema text, whatever ADK put on the request."""
    if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
        return json.dumps(schema.model_json_schema(), indent=2)
    if hasattr(schema, "model_dump"):
        return json.dumps(schema.model_dump(exclude_none=True), indent=2, default=str)
    if isinstance(schema, (dict, list)):
        return json.dumps(schema, indent=2, default=str)
    return str(schema)


def adapt_request_for_claude(llm_request: LlmRequest) -> tuple[LlmRequest, bool]:
    """A copy of the request the Anthropic path can serve, plus whether the
    caller wanted structured JSON (then the reply is trimmed to the object).
    The original request is not mutated: a shared config object leaking a
    Claude adaptation into the next Gemini call is exactly the class of bug
    sandbox.py's `_thinking_config` docstring warns about."""
    request = llm_request.model_copy()
    request.config = llm_request.config.model_copy(deep=True)
    cfg = request.config

    # Thinking: only adaptive (-1) or an explicit budget are meaningful here.
    tc = cfg.thinking_config
    if tc is not None and tc.thinking_budget is None:
        logger.warning(
            "sandbox: model %s got a named thinking level (%s); Claude takes none — "
            "using adaptive thinking",
            llm_request.model,
            tc.thinking_level,
        )
        cfg.thinking_config = genai_types.ThinkingConfig(
            thinking_budget=-1,
            include_thoughts=tc.include_thoughts,
        )
    thinking_on = (
        cfg.thinking_config is not None and (cfg.thinking_config.thinking_budget or 0) != 0
    )
    if thinking_on and cfg.temperature is not None:
        logger.warning(
            "sandbox: temperature=%s is ignored for %s — Anthropic rejects sampling "
            "parameters while thinking is on",
            cfg.temperature,
            llm_request.model,
        )

    # Structured output: say the schema, then let validate_schema parse the text.
    wants_json = (
        cfg.response_schema is not None or getattr(cfg, "response_json_schema", None) is not None
    )
    if wants_json:
        schema = (
            cfg.response_schema if cfg.response_schema is not None else cfg.response_json_schema
        )
        request.append_instructions([_JSON_INSTRUCTION.format(schema=_schema_text(schema))])
        cfg.response_schema = None
        cfg.response_mime_type = None
        if hasattr(cfg, "response_json_schema"):
            cfg.response_json_schema = None
    return request, wants_json


def trim_json_reply(response: LlmResponse) -> LlmResponse:
    """Keep only the outermost JSON object of the reply's text parts (thought
    parts untouched). `validate_schema` already strips ```json fences; this
    covers a stray sentence before or after the object."""
    content = response.content
    if content is None or not content.parts:
        return response
    text = "".join(p.text for p in content.parts if p.text and not p.thought)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return response
    trimmed = text[start : end + 1]
    if trimmed == text.strip():
        return response
    parts = [p for p in content.parts if not (p.text and not p.thought)]
    parts.append(genai_types.Part.from_text(text=trimmed))
    response.content = genai_types.Content(role=content.role, parts=parts)
    return response
