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
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
from pydantic import PrivateAttr
from typing_extensions import override

from .tenant import label_of

logger = logging.getLogger("gub_agent.models")

CLAUDE_PREFIX = "claude-"


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


# ── The per-call line ─────────────────────────────────────────────────────────
#
#   model_call: agent=… model=… stream=0|1 ttft_ms=… dur_ms=… prompt=… cached=…
#     thoughts=… out=… status=ok|error:<code> retries=<n>|- inv=… tenant=…
#
# One line per call, written when the call ends — never per streamed chunk.
# `ttft_ms` is the wait for the first chunk (a non-streamed call has one, so
# there it equals `dur_ms`); `dur_ms` runs to the LAST chunk, not to the end of
# the generator: after a function-call reply ADK runs the tools while this
# generator is suspended, and tool time is not model time. Both are measured
# where ADK reads them, so a streamed call's gaps include ADK's own handling of
# each chunk. Token counts are the last usage_metadata the call reported:
# `cached` is the part of `prompt` served from the context cache, `thoughts`
# are billed as output beside `out`. `prompt=` is a COUNT: no request or reply
# text is ever logged here.
#
# Count calls with `textPayload:"model_call: agent="`. `tenant=` goes last, as
# on the dispatcher line — numerator and denominator need the same clause.


@dataclass(frozen=True)
class _Caller:
    """Who is about to call the model, as its before_model_callback saw it."""

    agent: str
    invocation_id: str
    tenant: str


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

    @classmethod
    def begin(cls, router: VendorRouter, llm_request: LlmRequest, stream: bool) -> _ModelCall:
        caller = _CALLER.get()
        config = getattr(llm_request, "config", None)
        labels = getattr(config, "labels", None) or {}
        # ADK labels every request with the calling agent's name (base_llm_flow:
        # _ADK_AGENT_NAME_LABEL_KEY) just before it calls the model.
        agent = labels.get("adk_agent_name") or (caller.agent if caller else "-")
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

    def log(self) -> None:
        end = time.monotonic()
        first = self.first if self.first is not None else end
        last = self.last if self.last is not None else end
        observable = self.genai and _GENAI_LOGGER.isEnabledFor(logging.INFO)
        (logger.info if self.status == "ok" else logger.warning)(
            "model_call: agent=%s model=%s stream=%d ttft_ms=%d dur_ms=%d prompt=%d "
            "cached=%d thoughts=%d out=%d status=%s retries=%s inv=%s tenant=%s",
            self.agent,
            self.model,
            1 if self.stream else 0,
            round((first - self.started) * 1000),
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
