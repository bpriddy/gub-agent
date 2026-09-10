"""
models.VendorRouter — one model object, Gemini and Claude behind it.

The load-bearing case is again the NEGATIVE one: a request whose model is a
Gemini id (every prod request, every sandbox run that did not pick Claude) must
reach the Gemini client as the SAME object, untouched — the router adds nothing
to the path production takes.

The positive cases pin what the router does for a Claude id and only then:
lazy per-id Claude client with the project/location resource path, a copied
request (never the original), a named thinking level replaced by adaptive,
the critic's output_schema restated as an instruction with the native JSON
fields cleared, and the reply trimmed to its JSON object. No live model calls.
"""

from __future__ import annotations

import logging

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types
from pydantic import BaseModel

from gub_agent import config
from gub_agent.models import VendorRouter, adapt_request_for_claude, is_claude, trim_json_reply


class _Verdict(BaseModel):
    sufficient: bool
    reason: str


class _Recorder(BaseLlm):
    """A BaseLlm that records the request it was given and yields one reply."""

    seen: list = []
    reply_text: str = "ok"

    @classmethod
    def supported_models(cls):
        return [r".*"]

    async def generate_content_async(self, llm_request, stream=False):
        self.seen.append(llm_request)
        yield LlmResponse(
            content=genai_types.Content(
                role="model", parts=[genai_types.Part.from_text(text=self.reply_text)]
            ),
            partial=False,
        )


def _router(**kw) -> tuple[VendorRouter, _Recorder]:
    gemini = _Recorder(model="gemini-3.5-flash", seen=[])
    router = VendorRouter(model="gemini-3.5-flash", gemini=gemini, **kw)  # type: ignore[arg-type]
    return router, gemini


def _request(model: str, *, thinking: genai_types.ThinkingConfig | None, schema=None) -> LlmRequest:
    cfg = genai_types.GenerateContentConfig(thinking_config=thinking, system_instruction="BASE")
    req = LlmRequest(model=model, config=cfg)
    if schema is not None:
        req.set_output_schema(schema)
    return req


async def _collect(agen):
    return [r async for r in agen]


# ── 1. Gemini path is a pass-through ──────────────────────────────────────────


def test_is_claude():
    assert is_claude("claude-sonnet-5")
    assert is_claude("claude-haiku-4-5@20251001")
    assert not is_claude("gemini-3.5-flash")
    assert not is_claude(None)


async def test_gemini_request_reaches_gemini_untouched():
    """The prod path: same request object, no copy, no config change."""
    router, gemini = _router()
    req = _request("gemini-2.5-pro", thinking=genai_types.ThinkingConfig(thinking_level="MEDIUM"))
    out = await _collect(router.generate_content_async(req))
    assert gemini.seen == [req] and gemini.seen[0] is req
    assert req.config.thinking_config.thinking_level == "MEDIUM"
    assert out[0].content.parts[0].text == "ok"


async def test_no_claude_client_until_a_claude_id_arrives():
    router, _ = _router()
    await _collect(router.generate_content_async(_request("gemini-3.5-flash", thinking=None)))
    assert router._claude == {}


# ── 2. Claude path ────────────────────────────────────────────────────────────


async def test_claude_id_routes_to_a_lazy_per_id_client(monkeypatch):
    router, gemini = _router(claude_location="global")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-1")
    created: list[str] = []

    class FakeClaude(_Recorder):
        pass

    def fake_claude(model: str, max_tokens: int):
        created.append(model)
        return FakeClaude(model=model, seen=[])

    import google.adk.models.anthropic_llm as anthropic_llm

    monkeypatch.setattr(anthropic_llm, "Claude", fake_claude)

    req = _request("claude-sonnet-5", thinking=genai_types.ThinkingConfig(thinking_budget=-1))
    await _collect(router.generate_content_async(req))
    await _collect(router.generate_content_async(req))

    assert gemini.seen == []
    # One client per id, built with the full Vertex resource path (project + location).
    assert created == [
        "projects/proj-1/locations/global/publishers/anthropic/models/claude-sonnet-5"
    ]
    claude = router._claude["claude-sonnet-5"]
    assert len(claude.seen) == 2
    # The Claude client sees a COPY: the bare id stays in `model` (the API is
    # called with it), the original request is not mutated.
    assert claude.seen[0] is not req
    assert claude.seen[0].model == "claude-sonnet-5"


def test_adapt_replaces_named_thinking_level_with_adaptive(caplog):
    req = _request(
        "claude-sonnet-5",
        thinking=genai_types.ThinkingConfig(thinking_level="MEDIUM", include_thoughts=True),
    )
    with caplog.at_level(logging.WARNING, logger="gub_agent.models"):
        adapted, wants_json = adapt_request_for_claude(req)
    assert wants_json is False
    assert adapted.config.thinking_config.thinking_budget == -1
    assert adapted.config.thinking_config.include_thoughts is True
    # original untouched
    assert req.config.thinking_config.thinking_level == "MEDIUM"
    assert "named thinking level" in caplog.text


def test_adapt_keeps_dynamic_thinking_as_is():
    req = _request("claude-haiku-4-5", thinking=genai_types.ThinkingConfig(thinking_budget=-1))
    adapted, _ = adapt_request_for_claude(req)
    assert adapted.config.thinking_config.thinking_budget == -1


def test_adapt_restates_output_schema_as_instruction_and_clears_native_json():
    req = _request(
        "claude-sonnet-5", thinking=genai_types.ThinkingConfig(thinking_budget=-1), schema=_Verdict
    )
    assert (
        req.config.response_schema is _Verdict
        and req.config.response_mime_type == "application/json"
    )
    adapted, wants_json = adapt_request_for_claude(req)
    assert wants_json is True
    assert adapted.config.response_schema is None and adapted.config.response_mime_type is None
    instruction = adapted.config.system_instruction
    text = instruction if isinstance(instruction, str) else str(instruction)
    assert "exactly one JSON object" in text and '"sufficient"' in text and '"reason"' in text
    assert "BASE" in text  # the agent's own instruction is kept
    # original untouched
    assert req.config.response_schema is _Verdict


def test_adapt_warns_that_temperature_is_inert_with_thinking(caplog):
    req = _request("claude-sonnet-5", thinking=genai_types.ThinkingConfig(thinking_budget=-1))
    req.config.temperature = 0.2
    with caplog.at_level(logging.WARNING, logger="gub_agent.models"):
        adapt_request_for_claude(req)
    assert "temperature=0.2 is ignored" in caplog.text


# ── 3. Reply trimming ─────────────────────────────────────────────────────────


def _reply(*texts: str, thought: str | None = None) -> LlmResponse:
    parts = [genai_types.Part.from_text(text=t) for t in texts]
    if thought:
        parts.insert(0, genai_types.Part(text=thought, thought=True))
    return LlmResponse(content=genai_types.Content(role="model", parts=parts), partial=False)


def test_trim_extracts_the_json_object_and_keeps_thoughts():
    r = trim_json_reply(
        _reply(
            "Here is my verdict:\n",
            '{"sufficient": true, "reason": "ok"}',
            "\nHope that helps.",
            thought="hmm",
        )
    )
    texts = [p.text for p in r.content.parts if not p.thought]
    assert texts == ['{"sufficient": true, "reason": "ok"}']
    assert [p.text for p in r.content.parts if p.thought] == ["hmm"]


def test_trim_leaves_clean_json_and_non_json_alone():
    clean = _reply('{"sufficient": false, "reason": "x"}')
    assert trim_json_reply(clean) is clean
    prose = _reply("no braces here")
    assert trim_json_reply(prose) is prose


# ── 4. Wiring ─────────────────────────────────────────────────────────────────


def test_build_model_is_a_router_over_gemini():
    llm = config.build_model()
    assert isinstance(llm, VendorRouter)
    assert llm.model == config.GEMINI_MODEL
    assert llm.gemini.model == config.GEMINI_MODEL
    assert llm.claude_location == config.CLAUDE_VERTEX_LOCATION


def test_validator_message_names_adaptive_thinking_for_claude(monkeypatch):
    from gub_agent.sandbox import read_overrides

    monkeypatch.setattr(config, "SANDBOX_ENABLED", True)
    monkeypatch.setattr(config, "SANDBOX_MODEL_ALLOWLIST", ("gemini-3.5-flash", "claude-sonnet-5"))
    monkeypatch.setattr(config, "SANDBOX_THINKING_LEVEL_MODELS", ("gemini-3.5-flash",))
    with pytest.raises(ValueError, match="Claude thinks adaptively"):
        read_overrides({"sandbox": {"model": "claude-sonnet-5"}})
    # DYNAMIC on all three roles is accepted (the formatter, blend 03, always
    # runs and carries its own named-level baseline).
    ov = read_overrides(
        {
            "sandbox": {
                "model": "claude-sonnet-5",
                "thinking_level": "DYNAMIC",
                "critic_thinking_level": "DYNAMIC",
                "formatter_thinking_level": "DYNAMIC",
            }
        }
    )
    assert ov.model == "claude-sonnet-5"
