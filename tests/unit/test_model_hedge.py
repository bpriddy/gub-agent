"""
The executor and formatter hedges — models.VendorRouter under
EXECUTOR_HEDGE_AFTER_MS / FORMATTER_HEDGE_AFTER_MS.

The router hedge's mechanism (tests/unit/test_router_hedge.py pins it whole),
switched on per agent by `agent_hedge_after_ms`. Pinned here, for what is new:

- a stalled executor or formatter call loses to its hedge, streamed; a healthy
  one never fires it; each agent fires on its OWN deadline and the router's
  deadline never applies to them (nor theirs to the router);
- a call that does not stream is never hedged: its one chunk is the whole
  reply, so a deadline there would bound the generation, not the first token
  (the router's own hedge still covers both);
- a reply that is a function call is relayed from ONE request only, the
  winner's: the tools ADK then runs are run once;
- a fired hedge logs one `model_hedge: agent=…` line (`tenant=` last) and no
  `router_hedge:` line; the two `model_call:` lines are the router hedge's;
- the caller's cancellation stops both requests, and nothing outlives the call;
- with the flags at 0 (the default), every call takes today's path, byte for
  byte;
- the flags reach the deployed executor's and formatter's model, the
  formatter's name is the one its deadline is keyed on, and both deploy env
  files state both flags with the same value.

No model calls: every model here is a stub BaseLlm.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types as genai_types

from gub_agent import config
from gub_agent import models as models_module
from gub_agent.agents.formatter import FORMATTER_NAME
from gub_agent.config import AGENT_NAME, GEMINI_MODEL
from gub_agent.models import VendorRouter

from .test_router_hedge import (
    _CLOCK,
    AFTER_MS,
    GENAI_LOGGER,
    REPO,
    STALL,
    USAGE,
    _bind,
    _by_role,
    _calls,
    _drain,
    _pending,
    _request,
    _Stub,
    _text,
)

EXECUTOR = AGENT_NAME
FORMATTER = config.FORMATTER_AGENT_NAME
MODEL_HEDGE_LINE = re.compile(
    r"^model_hedge: agent=(\S+) fired=1 winner=(primary|hedge|-) primary_ttft_ms=(\d+|-) "
    r"hedge_ttft_ms=(\d+|-) inv=(\S+) tenant=(\S+)$"
)


def _vendor(
    *scripts, agent_ms: dict[str, int] | None = None, router_ms: int = 0
) -> tuple[VendorRouter, _Stub]:
    stub = _Stub(
        model=GEMINI_MODEL,
        scripts=[list(s) for s in scripts],
        edits=False,
        seen=[],
        handed=[],
        pulled=[],
        cancelled=[],
        closed=[],
    )
    router = VendorRouter(
        model=GEMINI_MODEL,
        gemini=stub,  # type: ignore[arg-type]
        router_hedge_after_ms=router_ms,
        agent_hedge_after_ms=agent_ms or {},
    )
    return router, stub


def _answer(who: str) -> list:
    """An executor's prose reply as SSE delivers it: deltas, then the whole."""
    whole = f"Chevy has three live campaigns (by {who})."
    return [
        _text(whole[:10], partial=True),
        _text(whole[10:], partial=True),
        _text(whole, usage=USAGE),
    ]


def _who(responses: list[LlmResponse]) -> set[str]:
    return {
        r.content.parts[0].text.rsplit("by ", 1)[1].rstrip(").")
        for r in responses
        if not r.partial and r.content and r.content.parts and r.content.parts[0].text
    }


def _function_call(who: str) -> list:
    """An executor round whose reply is a tool call, tagged with who sent it."""
    call = genai_types.Part(
        function_call=genai_types.FunctionCall(
            id=f"call-{who}", name="get_account_overview", args={"account": "chevy"}
        )
    )
    return [
        LlmResponse(content=genai_types.Content(role="model", parts=[call]), usage_metadata=USAGE)
    ]


def _model_hedge_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("model_hedge:")]


def _router_hedge_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("router_hedge:")]


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="gub_agent.models")
    caplog.set_level(logging.INFO, logger=GENAI_LOGGER)
    return caplog


# ── the race, per agent ──────────────────────────────────────────────────────


@pytest.mark.parametrize("agent", [EXECUTOR, FORMATTER], ids=["executor", "formatter"])
async def test_a_stalled_call_loses_to_its_hedge(logs, agent):
    router, stub = _vendor(
        [STALL, *_answer("primary")], [0.01, *_answer("hedge")], agent_ms={agent: AFTER_MS}
    )
    _bind(agent=agent, inv="e-42")

    started = time.monotonic()
    out = await _drain(router.generate_content_async(_request(agent), stream=True))
    wall = time.monotonic() - started

    assert _who(out) == {"hedge"}
    assert [bool(r.partial) for r in out] == [True, True, False]
    assert wall < 1.0  # the deadline plus the hedge, not the stall
    assert len(stub.seen) == 2 and stub.cancelled == [True, False]
    primary, hedge = _by_role(logs)
    assert (primary["agent"], primary["status"], primary["ttft_ms"]) == (
        agent,
        "error:cancelled",
        "-",
    )
    assert (hedge["agent"], hedge["status"]) == (agent, "ok") and hedge["_last"] == "hedge"
    [line] = _model_hedge_lines(logs)
    match = MODEL_HEDGE_LINE.match(line)
    assert match, line
    assert match.group(1, 2, 3, 5, 6) == (agent, "hedge", "-", "e-42", "chevy")
    assert line.split()[-1] == "tenant=chevy"
    assert _router_hedge_lines(logs) == []  # the router's line is the router's alone


async def test_a_healthy_executor_round_never_fires_the_hedge(logs):
    router, stub = _vendor([0.01, *_answer("primary")], agent_ms={EXECUTOR: 500})
    _bind(agent=EXECUTOR)

    out = await _drain(router.generate_content_async(_request(EXECUTOR), stream=True))

    assert _who(out) == {"primary"} and len(stub.seen) == 1
    assert _model_hedge_lines(logs) == []
    [line] = _calls(logs)
    assert line["status"] == "ok" and not line["_hedge"]


async def test_each_agent_fires_on_its_own_deadline(logs):
    """The executor waits for its own (long) deadline; the formatter's short
    one never applies to it."""
    agent_ms = {EXECUTOR: 400, FORMATTER: AFTER_MS}
    router, stub = _vendor(
        [0.15, *_answer("primary")], [0.01, *_answer("hedge")], agent_ms=agent_ms
    )
    _bind(agent=EXECUTOR)

    out = await _drain(router.generate_content_async(_request(EXECUTOR), stream=True))

    assert _who(out) == {"primary"} and len(stub.seen) == 1  # 150 ms < 400 ms: no hedge
    assert _model_hedge_lines(logs) == []


async def test_each_agent_fires_at_its_own_deadline_not_at_anothers(logs):
    """A positive case: the executor's hedge goes out at ITS deadline (400 ms),
    not at the formatter's (50 ms)."""
    agent_ms = {EXECUTOR: 400, FORMATTER: AFTER_MS}
    router, stub = _vendor(
        [STALL, *_answer("primary")], [0.01, *_answer("hedge")], agent_ms=agent_ms
    )
    _bind(agent=EXECUTOR)

    started = time.monotonic()
    sent: list[float] = []
    original = stub.generate_content_async

    async def timed(llm_request, stream=False):
        sent.append(time.monotonic() - started)
        async for response in original(llm_request, stream=stream):
            yield response

    object.__setattr__(stub, "generate_content_async", timed)
    out = await _drain(router.generate_content_async(_request(EXECUTOR), stream=True))

    assert _who(out) == {"hedge"} and len(sent) == 2
    assert sent[1] >= 0.39, sent  # the executor's deadline, not the formatter's 50 ms


@pytest.mark.parametrize("agent", [EXECUTOR, FORMATTER], ids=["executor", "formatter"])
async def test_a_call_that_does_not_stream_is_never_hedged(logs, agent):
    """A run with no SSE run_config (a sandbox /batch run, a caller that sends
    none): the one chunk is the whole reply. A healthy 150 ms generation past a
    50 ms deadline must not be sent twice."""
    router, stub = _vendor(
        [0.15, _text("A whole reply (by primary).", usage=USAGE)],
        [0.01, _text("never (by hedge).")],
        agent_ms={agent: AFTER_MS},
    )
    _bind(agent=agent)

    out = await _drain(router.generate_content_async(_request(agent), stream=False))

    assert _who(out) == {"primary"} and len(stub.seen) == 1
    assert _model_hedge_lines(logs) == []
    [line] = _calls(logs)
    assert line["stream"] == "0" and not line["_hedge"]


async def test_the_router_deadline_never_hedges_the_executor_nor_the_reverse(logs):
    router, stub = _vendor(
        [0.15, *_answer("primary")], [0.01, *_answer("hedge")], router_ms=AFTER_MS
    )
    _bind(agent=EXECUTOR)
    out = await _drain(router.generate_content_async(_request(EXECUTOR), stream=True))
    assert _who(out) == {"primary"} and len(stub.seen) == 1

    router, stub = _vendor(
        [0.15, _text('{"intent":"smalltalk"}', usage=USAGE)], agent_ms={EXECUTOR: AFTER_MS}
    )
    _bind(agent="router")
    await _drain(router.generate_content_async(_request("router"), stream=True))
    assert len(stub.seen) == 1
    assert _model_hedge_lines(logs) == [] and _router_hedge_lines(logs) == []


async def test_a_tool_call_reply_is_relayed_from_the_winner_alone(logs):
    """The executor's tools run in ADK after the reply, from what this call
    relays. Two requests went out; one function call comes back."""
    router, stub = _vendor(
        [STALL, *_function_call("primary")],
        [0.01, *_function_call("hedge")],
        agent_ms={EXECUTOR: AFTER_MS},
    )
    _bind(agent=EXECUTOR)

    out = await _drain(router.generate_content_async(_request(EXECUTOR), stream=True))

    calls = [p.function_call for r in out for p in (r.content.parts or []) if p.function_call]
    assert [c.id for c in calls] == ["call-hedge"]
    assert stub.pulled == [0, 1]  # the stalled request never produced its call


async def test_the_callers_cancellation_stops_both_requests(logs):
    router, stub = _vendor(
        [STALL, *_answer("primary")], [STALL, *_answer("hedge")], agent_ms={EXECUTOR: AFTER_MS}
    )
    _bind(agent=EXECUTOR)

    task = asyncio.create_task(
        _drain(router.generate_content_async(_request(EXECUTOR), stream=True))
    )
    await asyncio.sleep(0.2)  # past the deadline: both requests are out
    assert len(stub.seen) == 2
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stub.cancelled == [True, True] and stub.closed == [True, True]
    assert _pending() == set()
    [line] = _model_hedge_lines(logs)
    assert MODEL_HEDGE_LINE.match(line).group(2) == "-"  # neither answered


async def test_a_request_that_cannot_be_copied_runs_unhedged(logs, monkeypatch):
    router, stub = _vendor([0.1, *_answer("primary")], agent_ms={FORMATTER: AFTER_MS})
    _bind(agent=FORMATTER)

    def refuse(self, **kwargs):
        raise TypeError("not copyable")

    monkeypatch.setattr(type(_request(FORMATTER)), "model_copy", refuse)
    out = await _drain(router.generate_content_async(_request(FORMATTER), stream=True))

    assert _who(out) == {"primary"} and len(stub.seen) == 1
    warnings = [r.getMessage() for r in logs.records if r.levelno == logging.WARNING]
    assert (
        "model_hedge: agent=formatter request not copyable (TypeError) — unhedged "
        "inv=inv-1 tenant=chevy"
    ) in warnings


# ── the flags off: today's path, byte for byte ───────────────────────────────


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
    "agent,agent_ms",
    [
        (EXECUTOR, {}),
        (FORMATTER, {}),
        (EXECUTOR, {EXECUTOR: 0}),
        (FORMATTER, {EXECUTOR: AFTER_MS}),
        (EXECUTOR, {EXECUTOR: 60_000}),
    ],
    ids=["executor-off", "formatter-off", "executor-0", "other-agent-on", "answered-in-time"],
)
async def test_the_unhedged_path_is_todays_byte_for_byte(logs, _fixed_clock, agent, agent_ms):
    script = [0.12, _text("The ", partial=True), 0.13, _text("The answer", usage=USAGE)]
    router, stub = _vendor(script, agent_ms=agent_ms)
    _bind(agent=agent)

    out = await _drain(router.generate_content_async(_request(agent), stream=True))

    assert out == [script[1], script[3]] and len(stub.seen) == 1
    assert [r.getMessage() for r in logs.records if r.name == "gub_agent.models"] == [
        UNHEDGED_LINE.format(agent=agent)
    ]


# ── the flags ────────────────────────────────────────────────────────────────


def test_the_formatter_deadline_is_keyed_on_the_formatters_name():
    assert config.FORMATTER_AGENT_NAME == FORMATTER_NAME


_READ_FLAGS = """
import json
from gub_agent import config
from gub_agent.agent import executor_agent
from gub_agent.agents.formatter import formatter_agent
from gub_agent.agents.router import router_agent
print(json.dumps([
    executor_agent.model.agent_hedge_after_ms,
    formatter_agent.model.agent_hedge_after_ms,
    router_agent.model.router_hedge_after_ms,
]))
"""


@pytest.mark.parametrize(
    "executor,formatter,expected",
    [
        (None, None, {}),
        ("0", "0", {}),
        ("25000", None, {"gub_agent": 25000}),
        ("25000", "20000", {"gub_agent": 25000, "formatter": 20000}),
    ],
    ids=["default", "both-0", "executor-only", "both"],
)
def test_the_flags_reach_the_deployed_models(executor, formatter, expected):
    """Read at import, so each setting gets a fresh interpreter. The router's
    own deadline is untouched by either."""
    names = (
        "EXECUTOR_HEDGE_AFTER_MS",
        "FORMATTER_HEDGE_AFTER_MS",
        "ROUTER_HEDGE_AFTER_MS",
        "AGENT_NAME",
    )
    env = {k: v for k, v in os.environ.items() if k not in names}
    for key, value in (
        ("EXECUTOR_HEDGE_AFTER_MS", executor),
        ("FORMATTER_HEDGE_AFTER_MS", formatter),
    ):
        if value is not None:
            env[key] = value
    env["PYTHONPATH"] = str(REPO)
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _READ_FLAGS],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    executor_map, formatter_map, router_ms = json.loads(out.stdout.strip().splitlines()[-1])
    assert executor_map == expected and formatter_map == expected
    assert router_ms == 4000


def _stated(name: str, key: str) -> list[str]:
    return [
        line.partition("=")[2].strip()
        for line in (REPO / name).read_text().splitlines()
        if line.startswith(f"{key}=")
    ]


@pytest.mark.parametrize("key", ["EXECUTOR_HEDGE_AFTER_MS", "FORMATTER_HEDGE_AFTER_MS"])
def test_both_deploy_env_files_state_the_flag_with_one_value(key):
    """Stated, not defaulted — the rollback is a line already in front of the
    operator — and the same in both, so a sandbox run measures what
    production runs."""
    prod, sandbox = _stated("deploy-prod.env", key), _stated("deploy-sandbox.env", key)
    assert len(prod) == 1 and len(sandbox) == 1, (prod, sandbox)
    assert prod == sandbox
    assert int(prod[0]) >= 0
