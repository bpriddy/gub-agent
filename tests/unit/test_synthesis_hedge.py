"""
The executor's answer-round hedge deadline — models.VendorRouter under
EXECUTOR_SYNTHESIS_HEDGE_AFTER_MS (latency-05 wave 3).

The executor's round past the ReAct budget writes the answer, and its first
chunk arrives after ALL of its thinking, so a hedge at the tool rounds' deadline
fires on a healthy call and loses. That round is marked (`round_limit` calls
`models.mark_synthesis_round`) and waits for a deadline of its own. Pinned here:

- a marked executor call that stalls past the executor's deadline but answers
  before the answer round's is not hedged; one that stalls past the answer
  round's is, at that deadline, and its `model_hedge:` line says
  `round=synthesis` (before `inv=`, `tenant=` still last);
- an unmarked executor call still hedges at the executor's deadline and its
  line is the old one, without the key;
- with the answer round's deadline at 0 a marked call hedges at the executor's
  (today's behaviour) and its line still says `round=synthesis`; with the
  executor's own deadline at 0 nothing is hedged, marked or not;
- the deadline is the executor's alone: the formatter's, the critic's and the
  router's keep theirs, and a caller bound for another agent is ignored; a call
  that does not stream is never hedged;
- the mark covers one model call: the next `bind_model_call` replaces the
  caller, an unbound call is left alone, the caller is replaced and not edited,
  and concurrent tasks mark only their own;
- the hedge cap counts a hedge fired on the answer round, and a capped answer
  round's `hedge_capped:` line names its deadline and says `round=synthesis`
  (before `inv=`); a capped tool round's line is the old one;
- wiring: `round_limit` marks exactly the call past MAX_ROUNDS, and the
  executor's real before_model chain does so on a real ADK session, ending in a
  request the router holds to the later deadline;
- the flag: `build_model()` hands it to the router, its code default is 0, and
  both deploy env files state the same value.

No model calls: every model here is a stub BaseLlm.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

import pytest
from google.adk.agents import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai import types as genai_types

from gub_agent import agent as agent_module
from gub_agent import config
from gub_agent import models as models_module
from gub_agent.agents.round_limiter import MAX_ROUNDS, reset_rounds, round_limit
from gub_agent.config import AGENT_NAME, GEMINI_MODEL
from gub_agent.models import bind_model_call
from tests.helpers import invocation_ctx

from .test_model_hedge import (
    CAPPED_LINE,
    CRITIC,
    EXECUTOR,
    FORMATTER,
    MODEL_HEDGE_LINE,
    _answer,
    _capped_lines,
    _model_hedge_lines,
    _router_hedge_lines,
    _stated,
    _vendor,
    _who,
)
from .test_router_hedge import AFTER_MS, GENAI_LOGGER, REPO, STALL, _bind, _drain, _request

E_MS = AFTER_MS  # the executor's deadline in these tests: 50 ms
S_MS = 300  # the answer round's: later, still short enough for a test
MID_S = 0.18  # past the executor's deadline, inside the answer round's: 120 ms to spare each way
SYNTHESIS_LINE = re.compile(
    r"^model_hedge: agent=(\S+) fired=1 winner=(primary|hedge|-) primary_ttft_ms=(\d+|-) "
    r"hedge_ttft_ms=(\d+|-) round=synthesis inv=(\S+) tenant=(\S+)$"
)
CAPPED_SYNTHESIS_LINE = re.compile(
    r"^hedge_capped: agent=(\S+) after_ms=(\d+) used=(\d+) limit=(\d+) round=synthesis "
    r"inv=(\S+) tenant=(\S+)$"
)
FLAG = "EXECUTOR_SYNTHESIS_HEDGE_AFTER_MS"


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="gub_agent.models")
    caplog.set_level(logging.INFO, logger=GENAI_LOGGER)
    return caplog


@pytest.fixture(autouse=True)
def _no_bound_caller():
    """No test inherits a caller (or its mark) from another."""
    token = models_module._CALLER.set(None)
    yield
    models_module._CALLER.reset(token)


def _mark() -> None:
    mark = getattr(models_module, "mark_synthesis_round", None)
    assert mark is not None, "models.mark_synthesis_round does not exist"
    mark()


def _marked() -> bool:
    caller = models_module._CALLER.get()
    assert caller is not None, "no caller is bound"
    return bool(getattr(caller, "synthesis", False))


def _router(
    *scripts,
    synthesis_ms: int = S_MS,
    agent_ms: dict[str, int] | None = None,
    router_ms: int = 0,
):
    """A VendorRouter over a stub, the executor's deadline E_MS unless told
    otherwise, and the answer round's deadline in place (checked: a router that
    has not got the field is not under test)."""
    router, stub = _vendor(
        *scripts,
        agent_ms={EXECUTOR: E_MS} if agent_ms is None else agent_ms,
        router_ms=router_ms,
        synthesis_ms=synthesis_ms,
    )
    assert getattr(router, "synthesis_hedge_after_ms", None) == synthesis_ms
    return router, stub


def _timed(stub) -> list[float]:
    """Seconds after this call at which each request reached the stub."""
    started = time.monotonic()
    sent: list[float] = []
    original = stub.generate_content_async

    async def timed(llm_request, stream=False):
        sent.append(time.monotonic() - started)
        async for response in original(llm_request, stream=stream):
            yield response

    object.__setattr__(stub, "generate_content_async", timed)
    return sent


async def _call(router, agent: str = EXECUTOR, *, stream: bool = True):
    return await _drain(router.generate_content_async(_request(agent), stream=stream))


# ── the deadline itself ──────────────────────────────────────────────────────


async def test_a_marked_call_that_answers_before_the_answer_round_deadline_is_not_hedged(logs):
    """180 ms is past the executor's 50 ms and inside the answer round's 300 ms."""
    router, stub = _router([MID_S, *_answer("primary")], [0.01, *_answer("hedge")])
    _bind(agent=EXECUTOR, inv="s-1")
    _mark()

    out = await _call(router)

    assert _who(out) == {"primary"} and len(stub.seen) == 1
    assert _model_hedge_lines(logs) == []


async def test_a_marked_call_that_stalls_past_the_answer_round_deadline_is_hedged_there(logs):
    router, stub = _router([STALL, *_answer("primary")], [0.01, *_answer("hedge")])
    sent = _timed(stub)
    _bind(agent=EXECUTOR, inv="s-2")
    _mark()

    started = time.monotonic()
    out = await _call(router)
    wall = time.monotonic() - started

    assert _who(out) == {"hedge"} and len(stub.seen) == 2 and wall < 1.0
    assert sent[1] >= S_MS / 1000 - 0.01, sent  # the answer round's deadline, not 50 ms
    [line] = _model_hedge_lines(logs)
    match = SYNTHESIS_LINE.match(line)
    assert match, line
    assert match.group(1, 2, 3, 5, 6) == (EXECUTOR, "hedge", "-", "s-2", "chevy")
    assert line.split()[-1] == "tenant=chevy"
    assert line.index(" round=synthesis ") < line.index(" inv=")
    assert _router_hedge_lines(logs) == []


async def test_an_unmarked_call_hedges_at_the_executor_deadline_with_the_old_line(logs):
    """The answer round's deadline is configured and a marked call came first;
    the next call of the same invocation is a tool round again."""
    router, stub = _router(
        [MID_S, *_answer("answer-round")],
        [STALL, *_answer("tool-round")],
        [0.01, *_answer("tool-round-hedge")],
    )
    sent = _timed(stub)
    _bind(agent=EXECUTOR, inv="s-3")
    _mark()
    assert _who(await _call(router)) == {"answer-round"} and len(stub.seen) == 1

    _bind(agent=EXECUTOR, inv="s-3")  # the next model call: bound afresh
    assert not _marked()
    out = await _call(router)

    assert _who(out) == {"tool-round-hedge"} and len(stub.seen) == 3
    assert sent[2] - sent[1] < S_MS / 1000 - 0.05, sent  # at 50 ms, not at 200 ms
    [line] = _model_hedge_lines(logs)
    match = MODEL_HEDGE_LINE.match(line)  # the old line, whole
    assert match and "round=" not in line, line
    assert match.group(1, 2, 5, 6) == (EXECUTOR, "hedge", "s-3", "chevy")


async def test_without_an_answer_round_deadline_a_marked_call_hedges_at_the_executors(logs):
    """0 is the rollback: today's deadline for every round — and the line still
    says which round it was, so the baseline can be read off the same lines."""
    router, stub = _router([STALL, *_answer("primary")], [0.01, *_answer("hedge")], synthesis_ms=0)
    sent = _timed(stub)
    _bind(agent=EXECUTOR, inv="s-4")
    _mark()

    out = await _call(router)

    assert _who(out) == {"hedge"} and len(stub.seen) == 2
    assert sent[1] < S_MS / 1000 - 0.05, sent  # at the executor's 50 ms
    [line] = _model_hedge_lines(logs)
    match = SYNTHESIS_LINE.match(line)
    assert match, line
    assert match.group(1, 2, 5, 6) == (EXECUTOR, "hedge", "s-4", "chevy")


@pytest.mark.parametrize("agent_ms", [{}, {EXECUTOR: 0}], ids=["absent", "zero"])
async def test_a_marked_call_is_never_hedged_when_the_executor_hedge_is_off(logs, agent_ms):
    """The answer round's deadline replaces the executor's; it does not switch
    the executor's hedge on. 350 ms is past both deadlines."""
    router, stub = _router(
        [0.35, *_answer("primary")], [0.01, *_answer("hedge")], agent_ms=agent_ms
    )
    _bind(agent=EXECUTOR, inv="s-5")
    _mark()

    out = await _call(router)

    assert _who(out) == {"primary"} and len(stub.seen) == 1
    assert _model_hedge_lines(logs) == []


@pytest.mark.parametrize(
    "caller_agent,call_agent",
    [
        (FORMATTER, FORMATTER),
        (CRITIC, CRITIC),
        ("router", "router"),
        (FORMATTER, EXECUTOR),
        (EXECUTOR, FORMATTER),
    ],
    ids=[
        "formatter",
        "critic",
        "router",
        "executor-call-formatter-caller",
        "formatter-call-executor-caller",
    ],
)
async def test_the_answer_round_deadline_is_the_executors_alone(logs, caller_agent, call_agent):
    """A marked caller never moves another agent's deadline, and a caller bound
    for one agent never moves the deadline of a call labelled for another. Every
    deadline here is 50 ms; the call stalls past it and answers well inside the
    answer round's 300 ms, so a deadline that wrongly moved there would not
    hedge."""
    router, stub = _router(
        [MID_S, *_answer("primary")],
        [0.01, *_answer("hedge")],
        agent_ms={EXECUTOR: E_MS, FORMATTER: E_MS, CRITIC: E_MS},
        router_ms=E_MS,
    )
    _bind(agent=caller_agent, inv="s-6")
    _mark()
    assert _marked()

    out = await _call(router, call_agent)

    assert _who(out) == {"hedge"} and len(stub.seen) == 2


async def test_a_call_that_does_not_stream_is_never_hedged_on_the_answer_round(logs):
    """The answer round's deadline must not turn a non-streamed call into a
    hedged one: its one chunk is the whole reply. 200 ms is past both."""
    router, stub = _router(
        [0.2, _text_whole("primary")], [0.01, _text_whole("hedge")], synthesis_ms=100
    )
    _bind(agent=EXECUTOR, inv="s-7")
    _mark()

    out = await _call(router, stream=False)

    assert _who(out) == {"primary"} and len(stub.seen) == 1
    assert _model_hedge_lines(logs) == []


def _text_whole(who: str):
    return _answer(who)[-1]


# ── what the mark covers ─────────────────────────────────────────────────────


async def test_the_mark_covers_one_call_and_the_caller_is_replaced_not_edited():
    models_module._CALLER.set(None)
    _bind(agent=EXECUTOR, inv="m-1")
    before = models_module._CALLER.get()
    assert not _marked()

    _mark()
    after = models_module._CALLER.get()

    assert after is not before  # replaced: a call in flight may hold the old one
    assert before.synthesis is False and after.synthesis is True
    assert (after.agent, after.invocation_id, after.tenant) == (EXECUTOR, "m-1", "chevy")
    with pytest.raises(dataclasses.FrozenInstanceError):
        after.synthesis = False  # type: ignore[misc]

    _bind(agent=EXECUTOR, inv="m-1")  # the next model call
    assert not _marked()
    assert models_module._CALLER.get() is not after and after.synthesis is True


async def test_marking_with_no_bound_caller_binds_nothing():
    models_module._CALLER.set(None)

    _mark()  # does not raise

    assert models_module._CALLER.get() is None


async def test_concurrent_tasks_mark_only_their_own_caller():
    gate = asyncio.Event()
    seen: dict[str, object] = {}

    async def marker():
        _bind(agent=EXECUTOR, inv="task-a")
        await asyncio.sleep(0)
        _mark()
        gate.set()
        await asyncio.sleep(0)
        seen["a"] = models_module._CALLER.get()

    async def bystander():
        _bind(agent=EXECUTOR, inv="task-b")
        await gate.wait()  # the other task has marked its own by now
        await asyncio.sleep(0)
        seen["b"] = models_module._CALLER.get()

    await asyncio.gather(marker(), bystander())

    assert (seen["a"].invocation_id, getattr(seen["a"], "synthesis", None)) == ("task-a", True)
    assert (seen["b"].invocation_id, getattr(seen["b"], "synthesis", None)) == ("task-b", False)
    assert models_module._CALLER.get() is None  # nor did either reach this task


async def test_two_executor_calls_at_once_are_held_to_their_own_deadlines(logs):
    """One router, one marked call and one tool round in flight together: the
    marked one waits for the answer round's deadline, the other is hedged at the
    executor's. (The stub serves its scripts in arrival order: the marked
    call's first.)"""
    router, stub = _router(
        [MID_S, *_answer("answer-round")],
        [STALL, *_answer("tool-round")],
        [0.01, *_answer("tool-round-hedge")],
    )

    async def answer_round():
        _bind(agent=EXECUTOR, inv="c-1")
        _mark()
        return await _call(router)

    async def tool_round():
        _bind(agent=EXECUTOR, inv="c-2")
        return await _call(router)

    answer, tool = await asyncio.gather(answer_round(), tool_round())

    assert _who(answer) == {"answer-round"} and _who(tool) == {"tool-round-hedge"}
    assert len(stub.seen) == 3
    [line] = _model_hedge_lines(logs)
    assert MODEL_HEDGE_LINE.match(line) and "round=" not in line and "inv=c-2 " in line


# ── the cap ──────────────────────────────────────────────────────────────────


async def test_a_hedge_of_the_answer_round_spends_the_invocations_budget(logs):
    """The pair's budget is two: a formatter hedge, the answer round's, and then
    a third marked stall is capped — and its line names the ANSWER ROUND's
    deadline, the one that call waited out."""
    answer_ms = 150
    slow = _answer("third")
    router, stub = _router(
        [STALL, *_answer("fmt")],
        [0.01, *_answer("fmt-hedge")],
        [STALL, *_answer("synth")],
        [0.01, *_answer("synth-hedge")],
        [0.28, *slow],
        agent_ms={EXECUTOR: E_MS, FORMATTER: E_MS},
        synthesis_ms=answer_ms,
    )
    _bind(agent=FORMATTER, inv="k-1")
    assert _who(await _call(router, FORMATTER)) == {"fmt-hedge"}
    _bind(agent=EXECUTOR, inv="k-1")
    _mark()
    assert _who(await _call(router)) == {"synth-hedge"}  # spent the second hedge
    _bind(agent=EXECUTOR, inv="k-1")
    _mark()

    third = await _call(router)

    assert third == slow  # the primary's reply, relayed whole
    assert len(stub.seen) == 5  # no second request for the third call
    lines = _model_hedge_lines(logs)
    assert len(lines) == 2 and SYNTHESIS_LINE.match(lines[1]) and "round=" not in lines[0]
    [line] = _capped_lines(logs)
    match = CAPPED_SYNTHESIS_LINE.match(line)  # the key sits before inv=, tenant= last
    assert match, line
    assert match.groups() == (EXECUTOR, str(answer_ms), "2", "2", "k-1", "chevy")


async def test_a_capped_tool_round_keeps_the_old_line(logs):
    """The key is the answer round's alone: a withheld hedge of an unmarked
    executor call writes the line it always did."""
    router, stub = _router(
        [STALL, *_answer("fmt")],
        [0.01, *_answer("fmt-hedge")],
        [STALL, *_answer("tool")],
        [0.01, *_answer("tool-hedge")],
        [MID_S, *_answer("third")],
        agent_ms={EXECUTOR: E_MS, FORMATTER: E_MS},
    )
    _bind(agent=FORMATTER, inv="k-2")
    assert _who(await _call(router, FORMATTER)) == {"fmt-hedge"}
    _bind(agent=EXECUTOR, inv="k-2")
    assert _who(await _call(router)) == {"tool-hedge"}  # spent the second hedge
    _bind(agent=EXECUTOR, inv="k-2")

    assert _who(await _call(router)) == {"third"} and len(stub.seen) == 5

    [line] = _capped_lines(logs)
    match = CAPPED_LINE.match(line)
    assert match and "round=" not in line, line
    assert match.groups() == (EXECUTOR, str(E_MS), "2", "2", "k-2", "chevy")


# ── wiring ───────────────────────────────────────────────────────────────────


def _req() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(tools=["tool"]), tools_dict={"tool": 1}, contents=[]
    )


async def test_round_limit_marks_exactly_the_calls_past_the_budget():
    inv = f"wire-{uuid.uuid4().hex[:8]}"
    ctx = SimpleNamespace(agent_name=EXECUTOR, invocation_id=inv, state={"tenant": "chevy"})
    reset_rounds(ctx)

    marks = []
    for _ in range(MAX_ROUNDS + 2):
        bind_model_call(ctx)  # the chain binds, then ticks
        round_limit(ctx, _req())
        marks.append(_marked())
    assert marks == [False] * MAX_ROUNDS + [True, True]

    reset_rounds(ctx)  # a critic-requested pass starts over
    bind_model_call(ctx)
    round_limit(ctx, _req())
    assert not _marked()

    # No bound caller: the limiter still strips the tools and nothing is marked.
    models_module._CALLER.set(None)
    reset_rounds(ctx)
    request = None
    for _ in range(MAX_ROUNDS + 1):
        request = _req()
        round_limit(ctx, request)
    assert request.config.tools == [] and models_module._CALLER.get() is None
    assert callable(getattr(models_module, "mark_synthesis_round", None))


def _executor_request() -> LlmRequest:
    return LlmRequest(
        model=GEMINI_MODEL,
        contents=[genai_types.Content(role="user", parts=[genai_types.Part(text="hi")])],
        config=genai_types.GenerateContentConfig(
            system_instruction="Answer from the data.",
            labels={"adk_agent_name": AGENT_NAME},
            tools=[
                genai_types.Tool(
                    function_declarations=[
                        genai_types.FunctionDeclaration(name="org_query", description="d")
                    ]
                )
            ],
        ),
    )


async def test_the_real_chain_holds_the_answer_round_to_the_later_deadline(logs):
    """agent._before_model on a real session and a real CallbackContext, then the
    request it leaves goes through the router. Round 4 is the marked one and
    waits; the first round of the next pass is a tool round and is hedged at the
    executor's deadline."""
    inv = f"wire-{uuid.uuid4().hex[:8]}"
    ctx = await invocation_ctx(state={"tenant": "chevy"}, invocation_id=inv, user_text="hi")
    ctx = ctx.model_copy(update={"agent": BaseAgent(name=AGENT_NAME)})
    callback_context = CallbackContext(ctx)
    reset_rounds(callback_context)
    router, stub = _router(
        [MID_S, *_answer("answer-round")],
        [STALL, *_answer("tool-round")],
        [0.01, *_answer("tool-round-hedge")],
    )

    for _ in range(MAX_ROUNDS):  # the tool rounds: never marked
        agent_module._before_model(callback_context, _executor_request())
        assert not _marked()
    request = _executor_request()
    agent_module._before_model(callback_context, request)
    assert _marked() and request.config.tools == []  # tools gone, call marked
    answer = await _drain(router.generate_content_async(request, stream=True))

    assert _who(answer) == {"answer-round"} and len(stub.seen) == 1
    assert _model_hedge_lines(logs) == []

    reset_rounds(callback_context)  # the critic sent it back: a new pass
    request = _executor_request()
    agent_module._before_model(callback_context, request)
    assert not _marked()
    again = await _drain(router.generate_content_async(request, stream=True))

    assert _who(again) == {"tool-round-hedge"} and len(stub.seen) == 3
    [line] = _model_hedge_lines(logs)
    assert MODEL_HEDGE_LINE.match(line) and "round=" not in line, line


# ── the flag ─────────────────────────────────────────────────────────────────


def test_build_model_hands_the_flag_to_the_router(monkeypatch):
    monkeypatch.setattr(config, FLAG, 31000, raising=False)

    router = config.build_model()

    assert getattr(router, "synthesis_hedge_after_ms", None) == 31000
    monkeypatch.setattr(config, FLAG, 0, raising=False)
    assert getattr(config.build_model(), "synthesis_hedge_after_ms", None) == 0


_READ_FLAG = """
import json
from gub_agent import config
from gub_agent.agent import executor_agent
print(json.dumps([
    getattr(config, "EXECUTOR_SYNTHESIS_HEDGE_AFTER_MS", "missing"),
    getattr(executor_agent.model, "synthesis_hedge_after_ms", "missing"),
]))
"""


@pytest.mark.parametrize("value,expected", [(None, 0), ("0", 0), ("35000", 35000)])
def test_the_flag_reaches_the_deployed_executors_model(value, expected):
    """Read at import, so each value gets a fresh interpreter; unset is 0."""
    env = {k: v for k, v in os.environ.items() if k != FLAG}
    if value is not None:
        env[FLAG] = value
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
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [expected, expected]


def test_both_deploy_env_files_state_the_answer_round_deadline_once_and_alike():
    prod, sandbox = _stated("deploy-prod.env", FLAG), _stated("deploy-sandbox.env", FLAG)
    assert prod == sandbox == ["35000"]
