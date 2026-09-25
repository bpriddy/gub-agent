"""
The `model_call:` line — models.VendorRouter, one INFO line per model call.

The line is the $0 measurement for latency work (engine logs, no eval run), so
what is pinned here is that it can be trusted as one:

- exactly one line per call, written when the call ends — never per streamed
  chunk — with every field present and numeric: ttft_ms, dur_ms and the four
  token counts;
- a failed call still logs, as `status=error:<code>`, and the error still
  propagates unchanged;
- the agent, invocation and tenant are the CALLER's: bound by the agent's own
  before_model_callback, kept apart across the ParallelAgent's branch tasks,
  and never borrowed from a neighbour's call;
- genai's HttpRetryOptions retries are counted through genai's own retry path,
  and read `-` where that path is not observable;
- no prompt or reply text reaches the log;
- every LlmAgent in the deployed tree binds its caller (a real run of each,
  over a stub model).

No model calls: every model here is a stub BaseLlm.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest
import tenacity
from google.adk.agents import LlmAgent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import _api_client as genai_api_client
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from gub_agent.config import AGENT_NAME, GEMINI_MODEL, build_model
from gub_agent.models import VendorRouter, bind_model_call

GENAI_LOGGER = "google_genai._api_client"
FIELDS = [
    "agent",
    "model",
    "stream",
    "ttft_ms",
    "dur_ms",
    "prompt",
    "cached",
    "thoughts",
    "out",
    "status",
    "retries",
    "inv",
    "tenant",
]
NUMERIC = ["stream", "ttft_ms", "dur_ms", "prompt", "cached", "thoughts", "out"]
USAGE = genai_types.GenerateContentResponseUsageMetadata(
    prompt_token_count=1200,
    cached_content_token_count=800,
    thoughts_token_count=55,
    candidates_token_count=40,
)


def _text(text: str, *, partial: bool | None = None, usage=None) -> LlmResponse:
    return LlmResponse(
        content=genai_types.Content(role="model", parts=[genai_types.Part.from_text(text=text)]),
        partial=partial,
        usage_metadata=usage,
    )


class _Stub(BaseLlm):
    """A BaseLlm that replays a script: a response, a pause (float seconds) or
    an exception to raise, in order."""

    script: list = []
    calls: int = 0

    @classmethod
    def supported_models(cls):
        return [r".*"]

    async def generate_content_async(self, llm_request, stream=False):
        self.calls += 1
        for step in self.script:
            if isinstance(step, BaseException):
                raise step
            if isinstance(step, (int, float)):
                await asyncio.sleep(step)
                continue
            yield step


def _router(*script) -> tuple[VendorRouter, _Stub]:
    gemini = _Stub(model=GEMINI_MODEL, script=list(script), calls=0)
    return VendorRouter(model=GEMINI_MODEL, gemini=gemini), gemini  # type: ignore[arg-type]


def _request(agent: str = "critic", *, text: str = "how is chevy?", model=GEMINI_MODEL):
    config = genai_types.GenerateContentConfig(
        system_instruction="You are the critic.", labels={"adk_agent_name": agent}
    )
    return LlmRequest(
        model=model,
        contents=[genai_types.Content(role="user", parts=[genai_types.Part(text=text)])],
        config=config,
    )


def _bind(agent: str = "critic", inv: str = "inv-1", state: dict | None = None) -> None:
    bind_model_call(SimpleNamespace(agent_name=agent, invocation_id=inv, state=state or {}))


def _lines(caplog) -> list[dict]:
    out = []
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith("model_call: "):
            fields = dict(pair.split("=", 1) for pair in message[len("model_call: ") :].split())
            fields["_level"] = record.levelname
            out.append(fields)
    return out


async def _drain(agen) -> list:
    return [response async for response in agen]


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="gub_agent.models")
    return caplog


# ── one line per call ────────────────────────────────────────────────────────


async def test_a_non_streamed_call_logs_one_line_with_every_field_numeric(logs):
    router, _ = _router(_text('{"sufficient": true}', usage=USAGE))
    _bind(state={"tenant": "chevy"})

    out = await _drain(router.generate_content_async(_request()))

    assert [r.content.parts[0].text for r in out] == ['{"sufficient": true}']
    [line] = _lines(logs)
    assert list(line)[:-1] == FIELDS  # the order, as documented in models.py
    assert all(line[f].isdigit() for f in NUMERIC)
    assert line["agent"] == "critic" and line["model"] == GEMINI_MODEL
    assert line["stream"] == "0" and line["ttft_ms"] == line["dur_ms"]
    assert (line["prompt"], line["cached"], line["thoughts"], line["out"]) == (
        "1200",
        "800",
        "55",
        "40",
    )
    assert line["status"] == "ok" and line["_level"] == "INFO"
    assert (line["inv"], line["tenant"]) == ("inv-1", "chevy")


async def test_a_streamed_call_logs_once_when_it_ends_not_per_chunk(logs):
    router, _ = _router(
        0.02,
        _text("The ", partial=True),
        0.02,
        _text("answer", partial=True),
        _text("The answer", partial=False, usage=USAGE),
    )
    _bind()

    stream = router.generate_content_async(_request(), stream=True)
    first = await anext(stream)
    assert first.partial and _lines(logs) == []  # mid-stream: nothing yet
    rest = [response async for response in stream]

    assert len(rest) == 2
    [line] = _lines(logs)
    assert line["stream"] == "1" and line["status"] == "ok"
    assert all(line[f].isdigit() for f in NUMERIC)
    assert 15 <= int(line["ttft_ms"]) < int(line["dur_ms"])
    assert line["out"] == "40"


async def test_the_clock_stops_at_the_last_chunk_not_when_adk_resumes(logs):
    """After a function-call reply ADK runs the tools while this generator is
    suspended on its last yield (base_llm_flow: _postprocess inside the model
    loop), so the call only ENDS once they are done. That is tool time, and
    `dur_ms` must not count it."""
    router, _ = _router(_text("call org_query", usage=USAGE))
    _bind()

    stream = router.generate_content_async(_request())
    await anext(stream)
    await asyncio.sleep(0.05)  # the tools run
    assert [r async for r in stream] == []

    [line] = _lines(logs)
    assert int(line["dur_ms"]) < 30


async def test_a_call_that_reports_no_usage_logs_zeros(logs):
    router, _ = _router(_text("ok"))
    _bind()

    await _drain(router.generate_content_async(_request()))

    [line] = _lines(logs)
    assert (line["prompt"], line["cached"], line["thoughts"], line["out"]) == ("0",) * 4


# ── failures ─────────────────────────────────────────────────────────────────


async def test_a_failed_call_logs_its_status_and_the_error_propagates(logs):
    boom = genai_errors.APIError(503, {"error": {"code": 503, "message": "unavailable"}})
    router, _ = _router(0.01, boom)
    _bind()

    with pytest.raises(genai_errors.APIError) as raised:
        await _drain(router.generate_content_async(_request()))

    assert raised.value is boom
    [line] = _lines(logs)
    assert line["status"] == "error:503" and line["_level"] == "WARNING"
    assert all(line[f].isdigit() for f in NUMERIC)
    assert line["ttft_ms"] == line["dur_ms"]  # no chunk arrived: both run to the failure


async def test_a_stream_that_dies_mid_way_keeps_its_first_chunk_time(logs):
    router, _ = _router(_text("par", partial=True), 0.02, RuntimeError("stream reset"))
    _bind()

    with pytest.raises(RuntimeError, match="stream reset"):
        await _drain(router.generate_content_async(_request(), stream=True))

    [line] = _lines(logs)
    assert line["status"] == "error:RuntimeError"
    assert int(line["ttft_ms"]) < 15  # the chunk came at once; the failure later


async def test_a_reply_without_usable_content_logs_its_error_code(logs):
    router, _ = _router(LlmResponse(error_code="MALFORMED_FUNCTION_CALL", error_message="x"))
    _bind()

    await _drain(router.generate_content_async(_request()))

    [line] = _lines(logs)
    assert line["status"] == "error:MALFORMED_FUNCTION_CALL"


async def test_a_cancelled_call_logs_once(logs):
    router, _ = _router(5.0, _text("never"))
    _bind()

    task = asyncio.create_task(_drain(router.generate_content_async(_request())))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    [line] = _lines(logs)
    assert line["status"] == "error:cancelled"


# ── whose call it is ─────────────────────────────────────────────────────────


async def test_a_call_its_agent_did_not_bind_borrows_nothing(logs):
    """The router bound, then a critic call arrives without binding: the line
    names the critic (ADK's label) and admits it does not know the rest."""
    router, _ = _router(_text("ok"))
    _bind(agent="router", inv="inv-router", state={"tenant": "chevy"})

    await _drain(router.generate_content_async(_request("critic")))

    [line] = _lines(logs)
    assert (line["agent"], line["inv"], line["tenant"]) == ("critic", "-", "-")


async def test_parallel_branches_keep_their_own_caller(logs):
    """ParallelAgent runs each branch as a task with its own context copy
    (parallel_agent.py). Two branches bound and calling at the same time must
    each log their own invocation — here the calls overlap on purpose."""
    _bind(agent=AGENT_NAME, inv="inv-parent")

    async def branch(agent: str, inv: str) -> None:
        router, _ = _router(0.02, _text("ok"))
        _bind(agent=agent, inv=inv, state={"tenant": "chevy"})
        await asyncio.sleep(0.01)  # both have bound before either calls
        await _drain(router.generate_content_async(_request(agent)))

    await asyncio.gather(
        asyncio.create_task(branch("formatter", "inv-f")),
        asyncio.create_task(branch("critic", "inv-c")),
    )

    seen = {line["agent"]: (line["inv"], line["tenant"]) for line in _lines(logs)}
    assert seen == {"formatter": ("inv-f", "chevy"), "critic": ("inv-c", "chevy")}


# ── retries ──────────────────────────────────────────────────────────────────


class _RetryingStub(BaseLlm):
    """A Gemini stand-in whose request goes through genai's OWN retry path:
    tenacity built by genai's `retry_args` from our HttpRetryOptions — the
    before_sleep hook included — with only the backoff sleep zeroed."""

    failures: int = 1

    @classmethod
    def supported_models(cls):
        return [r".*"]

    async def generate_content_async(self, llm_request, stream=False):
        args = dict(genai_api_client.retry_args(build_model().gemini.retry_options))
        args["wait"] = tenacity.wait_fixed(0)
        attempts = {"n": 0}

        async def request():
            attempts["n"] += 1
            if attempts["n"] <= self.failures:
                raise genai_errors.APIError(429, {"error": {"code": 429, "message": "busy"}})
            return _text("ok", usage=USAGE)

        yield await tenacity.AsyncRetrying(**args)(request)


async def test_genai_retries_are_counted_on_the_call(logs):
    logs.set_level(logging.INFO, logger=GENAI_LOGGER)
    router = VendorRouter(model=GEMINI_MODEL, gemini=_RetryingStub(model=GEMINI_MODEL, failures=2))
    _bind()

    await _drain(router.generate_content_async(_request()))

    [line] = _lines(logs)
    assert line["retries"] == "2" and line["status"] == "ok"
    # Counted, not swallowed: genai's own record still reaches the log.
    assert sum("Retrying" in r.getMessage() for r in logs.records) == 2


async def test_a_call_without_retries_counts_zero(logs):
    logs.set_level(logging.INFO, logger=GENAI_LOGGER)
    router, _ = _router(_text("ok"))
    _bind()

    await _drain(router.generate_content_async(_request()))

    [line] = _lines(logs)
    assert line["retries"] == "0"


async def test_retries_read_unknown_where_genai_does_not_log_them(logs):
    # The logger's own level, not caplog.set_level: that would also raise the
    # capture handler's level and hide the model_call line itself.
    genai_logger = logging.getLogger(GENAI_LOGGER)
    previous = genai_logger.level
    genai_logger.setLevel(logging.WARNING)
    router = VendorRouter(model=GEMINI_MODEL, gemini=_RetryingStub(model=GEMINI_MODEL, failures=1))
    _bind()
    try:
        await _drain(router.generate_content_async(_request()))
    finally:
        genai_logger.setLevel(previous)

    [line] = _lines(logs)
    assert line["retries"] == "-" and line["status"] == "ok"


# ── content never logged, Claude path measured too ───────────────────────────


async def test_no_prompt_or_reply_text_reaches_the_log(caplog):
    caplog.set_level(logging.DEBUG, logger="gub_agent.models")
    router, _ = _router(_text("SECRET-ANSWER", usage=USAGE))
    _bind()
    request = _request(text="SECRET-QUESTION")
    request.config.system_instruction = "SECRET-INSTRUCTION"

    await _drain(router.generate_content_async(request))

    assert len(_lines(caplog)) == 1
    assert "SECRET" not in caplog.text


async def test_the_claude_path_is_measured_and_still_trimmed(logs, monkeypatch):
    import google.adk.models.anthropic_llm as anthropic_llm

    class FakeClaude(_Stub):
        pass

    monkeypatch.setattr(
        anthropic_llm,
        "Claude",
        lambda model, max_tokens: FakeClaude(
            model=model, script=[_text('Verdict: {"sufficient": true}', usage=USAGE)], calls=0
        ),
    )
    router, gemini = _router()
    _bind()
    request = _request(model="claude-sonnet-5")
    request.config.response_schema = {"type": "object"}

    out = await _drain(router.generate_content_async(request))

    assert gemini.calls == 0
    assert out[0].content.parts[-1].text == '{"sufficient": true}'  # trim_json_reply ran
    [line] = _lines(logs)
    assert line["model"] == "claude-sonnet-5" and line["status"] == "ok"
    assert line["retries"] == "-"  # the Anthropic SDK's retries are not genai's
    assert line["prompt"] == "1200" and line["inv"] == "inv-1"


# ── the deployed tree ────────────────────────────────────────────────────────


class _PerAgentStub(BaseLlm):
    """Answers each agent in the shape its output_schema needs."""

    @classmethod
    def supported_models(cls):
        return [r".*"]

    async def generate_content_async(self, llm_request, stream=False):
        agent = llm_request.config.labels.get("adk_agent_name")
        reply = {
            "router": {"intent": "exploratory", "confidence": 0.9},
            "critic": {
                "info_sufficient": True,
                "answer_satisfies": True,
                "sufficient": True,
                "reason": "covered",
                "feedback": "",
            },
            "formatter": {"kind": "abstain", "headline": "NO_COMPANY_RECORDS"},
        }.get(agent)
        yield _text(json.dumps(reply) if reply else "12 live campaigns.", usage=USAGE)


def _model_agents(agent) -> list[LlmAgent]:
    found = [agent] if isinstance(agent, LlmAgent) else []
    for sub in agent.sub_agents:
        found += _model_agents(sub)
    return found


async def test_every_model_agent_in_the_tree_logs_its_own_calls(logs):
    """Each LlmAgent of root_agent, run for real (its own callbacks, ADK's
    flow) over a stub model: its line names it, this invocation and the
    turn's tenant. An agent added without `bind_model_call` fails here."""
    from gub_agent.agent import root_agent

    agents = _model_agents(root_agent)
    assert sorted(a.name for a in agents) == sorted([AGENT_NAME, "critic", "formatter", "router"])
    for agent in agents:
        clone = agent.clone(
            update={"model": VendorRouter(model=GEMINI_MODEL, gemini=_PerAgentStub(model="stub"))}
        )
        runner = InMemoryRunner(agent=clone, app_name="gub")
        session = await runner.session_service.create_session(
            app_name="gub", user_id="u", state={"tenant": "chevy"}
        )
        message = genai_types.Content(role="user", parts=[genai_types.Part(text="how is chevy?")])
        logs.clear()
        events = [
            e
            async for e in runner.run_async(user_id="u", session_id=session.id, new_message=message)
        ]

        [line] = _lines(logs)
        assert line["agent"] == agent.name
        assert line["inv"] == events[-1].invocation_id
        assert line["tenant"] == "chevy" and line["status"] == "ok"
