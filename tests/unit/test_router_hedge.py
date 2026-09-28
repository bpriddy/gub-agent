"""
The router hedge — models.VendorRouter under ROUTER_HEDGE_AFTER_MS.

A router call that has had no first chunk by the deadline is sent a second
time, and the stream that answers first is relayed. Pinned here:

- a stalled primary loses to the hedge, streamed or not; a healthy one never
  fires it; a primary that answers after the hedge fired still wins;
- the second request is the first one's: the same bytes, a sandbox run's
  per-call overrides included, and not the object the first vendor call edits;
- the loser is cancelled and closed, and no task, unretrieved exception or
  un-awaited coroutine outlives the call;
- both requests keep their `model_call:` line — the loser's says
  `status=error:cancelled`, the second one's ends in ` hedge=1` — and a fired
  hedge logs one `router_hedge:` line, `tenant=` last;
- errors: before the hedge fires, exactly today's; after it, the survivor
  answers; when both fail, the primary's error is raised;
- a CancelledError the vendor raises itself fails the request like any
  error: no second send before the deadline, no call left hanging;
- a call genai is already retrying is not hedged, and each request counts its
  own retries;
- the caller's cancellation, and a consumer that stops reading, stop both;
- ROUTER_HEDGE_AFTER_MS=0, any agent but the router, and a router call that
  answers in time all take the unhedged path's bytes: the same chunks, one
  call, the same log line;
- the flag's default, and the deployed router's model carrying it.

No model calls: every model here is a stub BaseLlm.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import tenacity
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import _api_client as genai_api_client
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from gub_agent import config
from gub_agent import models as models_module
from gub_agent.agents.router import ROUTER_NAME
from gub_agent.config import GEMINI_MODEL, build_model
from gub_agent.models import HEDGED_AGENT, VendorRouter, bind_model_call
from gub_agent.sandbox import sandbox_before_model

REPO = Path(__file__).resolve().parents[2]
GENAI_LOGGER = "google_genai._api_client"
AFTER_MS = 50  # the deadline in these tests
STALL = 5.0  # a first token that does not come within a test
USAGE = genai_types.GenerateContentResponseUsageMetadata(
    prompt_token_count=1200,
    cached_content_token_count=800,
    thoughts_token_count=55,
    candidates_token_count=40,
)
HEDGE_LINE = re.compile(
    r"^router_hedge: fired=1 winner=(primary|hedge|-) primary_ttft_ms=(\d+|-) "
    r"hedge_ttft_ms=(\d+|-) inv=(\S+) tenant=(\S+)$"
)


def _text(text: str, *, partial: bool | None = None, usage=None) -> LlmResponse:
    return LlmResponse(
        content=genai_types.Content(role="model", parts=[genai_types.Part.from_text(text=text)]),
        partial=partial,
        usage_metadata=usage,
    )


def _answer(who: str, *, stream: bool) -> list:
    """A router reply marked with who sent it, as the stream delivers it."""
    whole = json.dumps({"intent": "smalltalk", "confidence": 0.95, "by": who})
    if not stream:
        return [_text(whole, usage=USAGE)]
    return [
        _text(whole[:12], partial=True),
        _text(whole[12:], partial=True),
        _text(whole, usage=USAGE),
    ]


def _503(message: str = "unavailable") -> genai_errors.APIError:
    return genai_errors.APIError(503, {"error": {"code": 503, "message": message}})


# Set by a test that fixes the clock: then a script's pause moves it instead of
# sleeping, and models.py reads it (`_fixed_clock`).
_CLOCK: list[float] = []


class _Stub(BaseLlm):
    """Call n replays scripts[n] (the last one repeats): a response, a pause
    (float seconds) or an exception to raise, in order. Remembers every
    request as it was handed over, and how each call ended."""

    scripts: list = []
    edits: bool = False  # edit the request it is handed, as ADK's Gemini does
    seen: list = []  # each call's request, serialised on arrival
    handed: list = []  # each call's request object
    pulled: list = []  # per call: how many responses it has produced
    cancelled: list = []  # per call: it was cancelled
    closed: list = []  # per call: its generator has exited, however

    @classmethod
    def supported_models(cls):
        return [r".*"]

    async def generate_content_async(self, llm_request, stream=False):
        n = len(self.seen)
        self.seen.append(llm_request.model_dump_json(exclude_none=True))
        self.handed.append(llm_request)
        self.pulled.append(0)
        self.cancelled.append(False)
        self.closed.append(False)
        if self.edits:
            llm_request.contents.append(
                genai_types.Content(role="user", parts=[genai_types.Part(text="Continue.")])
            )
            llm_request.config.labels["edited"] = "yes"
        try:
            for step in self.scripts[min(n, len(self.scripts) - 1)]:
                if isinstance(step, BaseException):
                    raise step
                if isinstance(step, (int, float)):
                    if _CLOCK:
                        _CLOCK[0] += step
                        await asyncio.sleep(0)
                    else:
                        await asyncio.sleep(step)
                    continue
                self.pulled[n] += 1
                yield step
        except asyncio.CancelledError:
            self.cancelled[n] = True
            raise
        finally:
            self.closed[n] = True


def _router(*scripts, after_ms: int = AFTER_MS, edits: bool = False) -> tuple[VendorRouter, _Stub]:
    stub = _Stub(
        model=GEMINI_MODEL,
        scripts=[list(s) for s in scripts],
        edits=edits,
        seen=[],
        handed=[],
        pulled=[],
        cancelled=[],
        closed=[],
    )
    router = VendorRouter(model=GEMINI_MODEL, gemini=stub, router_hedge_after_ms=after_ms)  # type: ignore[arg-type]
    return router, stub


def _request(agent: str = "router", *, model: str = GEMINI_MODEL) -> LlmRequest:
    config_ = genai_types.GenerateContentConfig(
        system_instruction="Route the question.", labels={"adk_agent_name": agent}
    )
    return LlmRequest(
        model=model,
        contents=[genai_types.Content(role="user", parts=[genai_types.Part(text="hi there")])],
        config=config_,
    )


def _bind(agent: str = "router", inv: str = "inv-1", state: dict | None = None) -> None:
    state = {"tenant": "chevy"} if state is None else state
    bind_model_call(SimpleNamespace(agent_name=agent, invocation_id=inv, state=state))


def _calls(caplog) -> list[dict]:
    """Every `model_call:` line as fields, plus `_hedge` (its ` hedge=1`)."""
    out = []
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith("model_call: "):
            words = message[len("model_call: ") :].split()
            fields = dict(word.split("=", 1) for word in words)
            fields["_last"] = words[-1].split("=", 1)[0]
            fields["_hedge"] = message.endswith(" hedge=1")
            out.append(fields)
    return out


def _by_role(caplog) -> tuple[dict, dict]:
    """The primary's and the hedge's `model_call:` lines."""
    lines = _calls(caplog)
    assert len(lines) == 2, lines
    [primary] = [line for line in lines if not line["_hedge"]]
    [hedge] = [line for line in lines if line["_hedge"]]
    return primary, hedge


def _hedge_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("router_hedge:")]


def _by(responses: list[LlmResponse]) -> set[str]:
    """Who sent the complete replies relayed."""
    return {json.loads(r.content.parts[0].text)["by"] for r in responses if not r.partial}


def _pending() -> set[asyncio.Task]:
    return {t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()}


async def _drain(agen) -> list:
    return [response async for response in agen]


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="gub_agent.models")
    caplog.set_level(logging.INFO, logger=GENAI_LOGGER)
    return caplog


# ── the race ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "no-stream"])
async def test_a_stalled_primary_loses_to_the_hedge(logs, stream):
    router, stub = _router(
        [STALL, *_answer("primary", stream=stream)], [0.01, *_answer("hedge", stream=stream)]
    )
    _bind()

    started = time.monotonic()
    out = await _drain(router.generate_content_async(_request(), stream=stream))
    wall = time.monotonic() - started

    assert _by(out) == {"hedge"}
    assert [bool(r.partial) for r in out] == ([True, True, False] if stream else [False])
    assert wall < 1.0  # the deadline plus the hedge, not the stall
    assert len(stub.seen) == 2
    assert stub.cancelled == [True, False] and stub.closed == [True, True]
    primary, hedge = _by_role(logs)
    assert primary["status"] == "error:cancelled" and primary["ttft_ms"] == "-"
    assert hedge["status"] == "ok" and hedge["ttft_ms"].isdigit()
    assert hedge["out"] == "40" and hedge["stream"] == ("1" if stream else "0")
    [line] = _hedge_lines(logs)
    assert HEDGE_LINE.match(line).group(1, 2, 4, 5) == ("hedge", "-", "inv-1", "chevy")


async def test_a_healthy_primary_never_fires_the_hedge(logs):
    router, stub = _router([0.01, *_answer("primary", stream=True)], after_ms=500)
    _bind()

    out = await _drain(router.generate_content_async(_request(), stream=True))

    assert _by(out) == {"primary"} and len(stub.seen) == 1
    assert _hedge_lines(logs) == []
    [line] = _calls(logs)
    assert line["status"] == "ok" and not line["_hedge"]


async def test_a_primary_that_answers_after_the_hedge_fired_still_wins(logs):
    router, stub = _router(
        [0.15, *_answer("primary", stream=True)], [STALL, *_answer("hedge", stream=True)]
    )
    _bind()

    out = await _drain(router.generate_content_async(_request(), stream=True))

    assert _by(out) == {"primary"} and len(stub.seen) == 2
    assert stub.cancelled == [False, True] and stub.closed == [True, True]
    primary, hedge = _by_role(logs)
    assert primary["status"] == "ok" and int(primary["ttft_ms"]) >= 150
    assert hedge["status"] == "error:cancelled" and hedge["ttft_ms"] == "-"
    [line] = _hedge_lines(logs)
    winner, primary_ttft, hedge_ttft, _, _ = HEDGE_LINE.match(line).groups()
    assert (winner, hedge_ttft) == ("primary", "-") and int(primary_ttft) >= 150


async def test_the_hedge_line_has_one_format_and_ends_with_the_tenant(logs):
    router, _ = _router(
        [STALL, *_answer("primary", stream=True)], [0.02, *_answer("hedge", stream=True)]
    )
    _bind(inv="e-123", state={"tenant": "chevy"})

    await _drain(router.generate_content_async(_request(), stream=True))

    [line] = _hedge_lines(logs)
    match = HEDGE_LINE.match(line)
    assert match, line
    assert line.split()[-1] == "tenant=chevy"
    assert int(match.group(3)) >= 20  # the hedge's own first-token time, from ITS start
    # The router_hedge line and both model_call lines carry the same inv.
    assert {c["inv"] for c in _calls(logs)} == {"e-123"} and match.group(4) == "e-123"
    # ` hedge=1` goes after tenant=, like the speculation's ` spec=1`.
    _, hedge = _by_role(logs)
    assert hedge["_last"] == "hedge" and hedge["tenant"] == "chevy"


# ── the same request ─────────────────────────────────────────────────────────


async def test_the_hedge_sends_the_first_requests_bytes_not_its_edited_object(logs):
    """ADK's Gemini edits the request it is handed (a user turn after a model
    one, tracking headers). The hedge is sent the request as it was BEFORE
    the first send; the primary gets the caller's own object, as unhedged."""
    router, stub = _router(
        [STALL, *_answer("primary", stream=True)],
        [0.01, *_answer("hedge", stream=True)],
        edits=True,
    )
    _bind()
    request = _request()
    before = request.model_dump_json(exclude_none=True)

    await _drain(router.generate_content_async(request, stream=True))

    assert stub.seen == [before, before]
    assert stub.handed[0] is request and stub.handed[1] is not request


async def test_a_sandbox_runs_overrides_reach_both_requests(logs, monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_ENABLED", True)
    state = {"tenant": "chevy", "sandbox": {"router_thinking_level": "LOW", "temperature": 0.3}}
    ctx = SimpleNamespace(agent_name="router", invocation_id="inv-1", state=state)
    request = _request()
    bind_model_call(ctx)
    sandbox_before_model(ctx, request, role="router")  # as the router's callback does
    router, stub = _router(
        [STALL, *_answer("primary", stream=True)], [0.01, *_answer("hedge", stream=True)]
    )

    await _drain(router.generate_content_async(request, stream=True))

    assert len(stub.seen) == 2 and stub.seen[0] == stub.seen[1]
    sent = [json.loads(seen)["config"] for seen in stub.seen]
    for config_ in sent:
        assert config_["temperature"] == 0.3
        assert config_["thinking_config"]["thinking_level"] == "LOW"


async def test_a_request_that_cannot_be_copied_runs_unhedged(logs, monkeypatch):
    def refuse(self, *, update=None, deep=False):
        raise RuntimeError("not copyable")

    monkeypatch.setattr(LlmRequest, "model_copy", refuse)
    router, stub = _router([0.15, *_answer("primary", stream=True)])
    _bind()

    out = await _drain(router.generate_content_async(_request(), stream=True))

    assert _by(out) == {"primary"} and len(stub.seen) == 1
    assert "router_hedge: request not copyable (RuntimeError) — unhedged" in logs.text
    [line] = _calls(logs)
    assert line["status"] == "ok" and not line["_hedge"]


# ── nothing outlives the call ────────────────────────────────────────────────


async def test_the_loser_is_cancelled_and_nothing_outlives_the_call(logs, recwarn):
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    reported: list[str] = []
    loop.set_exception_handler(lambda _loop, context: reported.append(context["message"]))
    pending = _pending()
    try:
        router, stub = _router(
            [STALL, *_answer("primary", stream=True)], [0.01, *_answer("hedge", stream=True)]
        )
        _bind()
        await _drain(router.generate_content_async(_request(), stream=True))
        gc.collect()
        await asyncio.sleep(0.05)  # any stray callback or task would run now
        gc.collect()
    finally:
        loop.set_exception_handler(previous)

    assert stub.cancelled[0] and stub.closed == [True, True]
    assert _pending() <= pending
    assert reported == []  # no "Task was destroyed but it is pending", no unretrieved exception
    assert [str(w.message) for w in recwarn if "never awaited" in str(w.message)] == []


async def test_the_callers_cancellation_cancels_both_requests(logs):
    router, stub = _router(
        [STALL, *_answer("primary", stream=True)], [STALL, *_answer("hedge", stream=True)]
    )
    _bind()
    pending = _pending()

    task = asyncio.create_task(_drain(router.generate_content_async(_request(), stream=True)))
    for _ in range(200):
        if len(stub.seen) == 2:
            break
        await asyncio.sleep(0.005)
    assert len(stub.seen) == 2  # the hedge has fired
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stub.cancelled == [True, True] and stub.closed == [True, True]
    assert _pending() <= pending
    primary, hedge = _by_role(logs)
    assert primary["status"] == hedge["status"] == "error:cancelled"
    # Fired, and nobody answered: the line still counts the second send.
    [line] = _hedge_lines(logs)
    assert HEDGE_LINE.match(line).group(1, 2, 3) == ("-", "-", "-")


async def test_a_consumer_that_stops_reading_closes_the_winner(logs):
    router, stub = _router(
        [STALL, *_answer("primary", stream=True)], [0.01, *_answer("hedge", stream=True)]
    )
    _bind()
    pending = _pending()

    stream = router.generate_content_async(_request(), stream=True)
    first = await anext(stream)
    await stream.aclose()

    assert first.partial
    assert stub.closed == [True, True] and _pending() <= pending
    primary, hedge = _by_role(logs)
    assert primary["status"] == "error:cancelled"
    assert hedge["status"] == "error:closed"  # as an unhedged call a consumer closes


async def test_the_winners_stream_is_read_only_as_it_is_asked_for(logs):
    """No read-ahead: the stream advances when ADK asks, so `dur_ms` still
    stops at the chunk ADK read last and never at one read early."""
    router, stub = _router(
        [STALL, *_answer("primary", stream=True)], [0.01, *_answer("hedge", stream=True)]
    )
    _bind()

    stream = router.generate_content_async(_request(), stream=True)
    await anext(stream)
    await asyncio.sleep(0.05)
    assert stub.pulled[1] == 1  # the hedge's first chunk only
    rest = [response async for response in stream]
    assert len(rest) == 2 and stub.pulled[1] == 3


async def test_the_clock_stops_at_the_last_chunk_not_when_adk_resumes(logs):
    """The hedged twin of the model_call test: ADK runs tools while this
    generator waits on its last yield, and that is not model time."""
    router, _ = _router([STALL, _text("x")], [0.01, _text("call org_query", usage=USAGE)])
    _bind()

    stream = router.generate_content_async(_request())
    await anext(stream)
    await asyncio.sleep(0.1)  # the tools run
    assert [r async for r in stream] == []

    _, hedge = _by_role(logs)
    assert int(hedge["dur_ms"]) < 60


# ── errors ───────────────────────────────────────────────────────────────────


async def test_a_primary_that_fails_before_the_deadline_fails_as_today(logs):
    boom = _503()
    router, stub = _router([0.01, boom], after_ms=500)
    _bind()

    with pytest.raises(genai_errors.APIError) as raised:
        await _drain(router.generate_content_async(_request(), stream=True))

    assert raised.value is boom and len(stub.seen) == 1
    assert _hedge_lines(logs) == []
    [line] = _calls(logs)
    assert line["status"] == "error:503" and line["ttft_ms"] == "-"


async def test_a_primary_that_fails_after_the_hedge_fired_leaves_the_hedge_to_answer(logs):
    router, _ = _router([0.1, _503()], [0.2, *_answer("hedge", stream=True)])
    _bind()

    out = await _drain(router.generate_content_async(_request(), stream=True))

    assert _by(out) == {"hedge"}
    primary, hedge = _by_role(logs)
    assert primary["status"] == "error:503" and hedge["status"] == "ok"
    [line] = _hedge_lines(logs)
    assert HEDGE_LINE.match(line).group(1) == "hedge"


async def test_a_hedge_that_fails_leaves_the_primary_to_answer(logs):
    router, _ = _router([0.2, *_answer("primary", stream=True)], [0.01, _503()])
    _bind()

    out = await _drain(router.generate_content_async(_request(), stream=True))

    assert _by(out) == {"primary"}
    primary, hedge = _by_role(logs)
    assert primary["status"] == "ok" and hedge["status"] == "error:503"
    [line] = _hedge_lines(logs)
    assert HEDGE_LINE.match(line).group(1) == "primary"


async def test_when_both_fail_the_primarys_error_is_raised(logs):
    first, second = _503("primary"), _503("hedge")
    router, _ = _router([0.1, first], [0.01, second])
    _bind()

    with pytest.raises(genai_errors.APIError) as raised:
        await _drain(router.generate_content_async(_request(), stream=True))

    assert raised.value is first
    primary, hedge = _by_role(logs)
    assert primary["status"] == hedge["status"] == "error:503"
    [line] = _hedge_lines(logs)
    assert HEDGE_LINE.match(line).group(1) == "-"


async def test_a_winner_that_fails_mid_stream_fails_the_call(logs):
    """After its first chunk the winner IS the call: its later failure is
    raised, as an unhedged call's is — the other stream is not spliced in."""
    router, _ = _router(
        [STALL, *_answer("primary", stream=True)],
        [0.01, _text("par", partial=True), RuntimeError("stream reset")],
    )
    _bind()

    got = []
    with pytest.raises(RuntimeError, match="stream reset"):
        async for response in router.generate_content_async(_request(), stream=True):
            got.append(response)

    assert len(got) == 1
    primary, hedge = _by_role(logs)
    assert primary["status"] == "error:cancelled" and hedge["status"] == "error:RuntimeError"


# ── retries ──────────────────────────────────────────────────────────────────


class _Retrying(BaseLlm):
    """Call n waits `stall[n]`, then goes through genai's OWN retry path
    (tenacity built from our HttpRetryOptions, before_sleep hook included),
    failing `failures[n]` times with a 429 and waiting `wait` in between."""

    stall: list = []
    failures: list = []
    wait: float = 0.0
    calls: int = 0

    @classmethod
    def supported_models(cls):
        return [r".*"]

    async def generate_content_async(self, llm_request, stream=False):
        n = self.calls
        self.calls += 1
        await asyncio.sleep(self.stall[n])
        args = dict(genai_api_client.retry_args(build_model().gemini.retry_options))
        args["wait"] = tenacity.wait_fixed(self.wait)
        attempts = {"n": 0}

        async def request():
            attempts["n"] += 1
            if attempts["n"] <= self.failures[n]:
                raise genai_errors.APIError(429, {"error": {"code": 429, "message": "busy"}})
            return _text(json.dumps({"by": f"call{n}"}), usage=USAGE)

        yield await tenacity.AsyncRetrying(**args)(request)


async def test_a_call_genai_is_retrying_is_not_hedged(logs):
    """A 429 answered it and genai is backing off: a second send would stack
    a second retry budget on a congested pool."""
    retrying = _Retrying(model=GEMINI_MODEL, stall=[0.0], failures=[1], wait=0.15)
    router = VendorRouter(model=GEMINI_MODEL, gemini=retrying, router_hedge_after_ms=AFTER_MS)
    _bind()

    out = await _drain(router.generate_content_async(_request()))

    assert _by(out) == {"call0"} and retrying.calls == 1
    assert _hedge_lines(logs) == []
    [line] = _calls(logs)
    assert line["retries"] == "1" and line["status"] == "ok"


async def test_each_request_counts_its_own_retries(logs):
    retrying = _Retrying(model=GEMINI_MODEL, stall=[STALL, 0.0], failures=[0, 1], wait=0.0)
    router = VendorRouter(model=GEMINI_MODEL, gemini=retrying, router_hedge_after_ms=AFTER_MS)
    _bind()

    out = await _drain(router.generate_content_async(_request()))

    assert _by(out) == {"call1"} and retrying.calls == 2
    primary, hedge = _by_role(logs)
    assert (primary["retries"], primary["status"]) == ("0", "error:cancelled")
    assert (hedge["retries"], hedge["status"]) == ("1", "ok")


# ── the unhedged path, byte for byte ─────────────────────────────────────────


@pytest.fixture
def _fixed_clock(monkeypatch):
    """models.py's clock, moved only by the stub's pauses."""
    _CLOCK[:] = [100.0]
    monkeypatch.setattr(models_module, "time", SimpleNamespace(monotonic=lambda: _CLOCK[0]))
    yield
    _CLOCK.clear()


UNHEDGED_LINE = (
    "model_call: agent={agent} model=gemini-3.5-flash stream=1 ttft_ms=120 dur_ms=250 "
    "prompt=1200 cached=800 thoughts=55 out=40 status=ok retries=0 inv=inv-1 tenant=chevy"
)


@pytest.mark.parametrize(
    "agent,after_ms",
    [("router", 0), ("critic", AFTER_MS), ("router", 60_000)],
    ids=["flag-0", "not-the-router", "answered-in-time"],
)
async def test_the_unhedged_path_is_todays_byte_for_byte(logs, _fixed_clock, agent, after_ms):
    """The same chunks, one call, and the `model_call:` line of today — for
    the rollback (0), for every agent but the router, and for a router call
    that answers before its deadline. The same test passes on the code before
    the hedge."""
    script = [
        0.12,
        _text("The ", partial=True),
        0.13,
        _text("The answer", usage=USAGE),
    ]
    router, stub = _router(script, after_ms=after_ms)
    _bind(agent=agent)

    out = await _drain(router.generate_content_async(_request(agent), stream=True))

    assert out == [script[1], script[3]] and len(stub.seen) == 1
    assert [r.getMessage() for r in logs.records if r.name == "gub_agent.models"] == [
        UNHEDGED_LINE.format(agent=agent)
    ]


async def test_a_non_router_agent_never_hedges(logs):
    router, stub = _router([0.2, _text("verdict", usage=USAGE)], [0.01, _text("never")])
    _bind(agent="critic")

    started = time.monotonic()
    out = await _drain(router.generate_content_async(_request("critic")))

    assert [r.content.parts[0].text for r in out] == ["verdict"]
    assert len(stub.seen) == 1 and time.monotonic() - started >= 0.2
    assert _hedge_lines(logs) == []


async def test_a_router_without_the_field_never_hedges(logs):
    """VendorRouter's default is off: only build_model turns it on."""
    stub = _Stub(
        model=GEMINI_MODEL,
        scripts=[[0.1, _text("late")], [_text("never")]],
        seen=[],
        handed=[],
        pulled=[],
        cancelled=[],
        closed=[],
    )
    router = VendorRouter(model=GEMINI_MODEL, gemini=stub)  # type: ignore[arg-type]
    _bind()

    await _drain(router.generate_content_async(_request()))

    assert router.router_hedge_after_ms == 0 and len(stub.seen) == 1


# ── the flag ─────────────────────────────────────────────────────────────────


def test_the_hedged_agent_is_the_router():
    assert HEDGED_AGENT == ROUTER_NAME


_READ_FLAG = """
from gub_agent import config
from gub_agent.agents.router import router_agent
from gub_agent.agent import executor_agent
print(config.ROUTER_HEDGE_AFTER_MS, router_agent.model.router_hedge_after_ms,
      executor_agent.model.router_hedge_after_ms)
"""


@pytest.mark.parametrize("value,expected", [(None, 4000), ("0", 0), ("2500", 2500)])
def test_the_flag_reaches_the_deployed_routers_model(value, expected):
    """Read at import, so each value gets a fresh interpreter."""
    env = {k: v for k, v in os.environ.items() if k != "ROUTER_HEDGE_AFTER_MS"}
    if value is not None:
        env["ROUTER_HEDGE_AFTER_MS"] = value
    env["PYTHONPATH"] = str(REPO)
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _READ_FLAG],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert out.stdout.split()[-3:] == [str(expected)] * 3


@pytest.mark.parametrize("name", ["deploy-prod.env", "deploy-sandbox.env"])
def test_both_deploy_env_files_state_the_flag(name):
    values = [
        line.partition("=")[2].strip()
        for line in (REPO / name).read_text().splitlines()
        if line.startswith("ROUTER_HEDGE_AFTER_MS=")
    ]
    assert values == ["4000"]


# ── a CancelledError the vendor raises itself ────────────────────────────────
#
# A library-internal cancel (an aiohttp disconnect surfacing as one) that no
# stop() asked for: the reader must hear of it, never wait on it — no second
# send for a request that has already died, and no call left hanging.


async def test_a_vendor_raised_cancel_before_the_deadline_fails_as_today(logs):
    router, stub = _router([asyncio.CancelledError()], after_ms=AFTER_MS)
    _bind()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(_drain(router.generate_content_async(_request(), stream=True)), 2)

    assert len(stub.seen) == 1
    assert _hedge_lines(logs) == []
    [line] = _calls(logs)
    assert line["status"] == "error:cancelled"


async def test_a_vendor_raised_cancel_after_the_hedge_fired_leaves_the_hedge_to_answer(logs):
    router, stub = _router([0.1, asyncio.CancelledError()], [0.2, *_answer("hedge", stream=True)])
    _bind()

    out = await asyncio.wait_for(_drain(router.generate_content_async(_request(), stream=True)), 2)

    assert _by(out) == {"hedge"} and len(stub.seen) == 2
    primary, hedge = _by_role(logs)
    assert primary["status"] == "error:cancelled" and hedge["status"] == "ok"
    [line] = _hedge_lines(logs)
    assert HEDGE_LINE.match(line).group(1) == "hedge"


async def test_both_vendor_raised_cancels_fail_the_call_without_hanging(logs):
    first = asyncio.CancelledError("primary")
    router, _ = _router([0.1, first], [0.01, asyncio.CancelledError("hedge")])
    _bind()

    with pytest.raises(asyncio.CancelledError) as raised:
        await asyncio.wait_for(_drain(router.generate_content_async(_request(), stream=True)), 2)

    assert raised.value.args == ("primary",)
    [line] = _hedge_lines(logs)
    assert HEDGE_LINE.match(line).group(1) == "-"
    assert _pending() == set()
