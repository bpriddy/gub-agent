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
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
from pydantic import PrivateAttr
from typing_extensions import override

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
        if not is_claude(llm_request.model):
            async for response in self.gemini.generate_content_async(llm_request, stream=stream):
                yield response
            return

        llm = self._claude_for(str(llm_request.model))
        request, wants_json = adapt_request_for_claude(llm_request)
        async for response in llm.generate_content_async(request, stream=stream):
            yield trim_json_reply(response) if wants_json and not response.partial else response


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
