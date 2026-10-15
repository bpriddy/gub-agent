"""
The executor, formatter and critic hedges — models.VendorRouter under
EXECUTOR_HEDGE_AFTER_MS / FORMATTER_HEDGE_AFTER_MS / CRITIC_HEDGE_AFTER_MS, and
the per-invocation hedge cap.

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
- the flags reach the deployed executor's, formatter's and critic's model, the
  formatter's and the critic's names are the ones their deadlines are keyed on,
  and both deploy env files state every flag with the same value;
- the hedge cap: one invocation fires at most two executor-or-formatter hedges
  and one critic hedge. A call past its budget waits on its own request, relays
  it whole and logs one `hedge_capped:` line; the router, a call with no bound
  invocation and a hedge that never fired are outside the budget; two
  concurrent calls of one invocation share it; the store is bounded and a
  failure in it lets the hedge fire.

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
from gub_agent.agents.critic import critic_agent
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
    _by,
    _by_role,
    _calls,
    _drain,
    _pending,
    _request,
    _Retrying,
    _Stub,
    _text,
)

EXECUTOR = AGENT_NAME
FORMATTER = config.FORMATTER_AGENT_NAME
CRITIC = config.CRITIC_AGENT_NAME
MODEL_HEDGE_LINE = re.compile(
    r"^model_hedge: agent=(\S+) fired=1 winner=(primary|hedge|-) primary_ttft_ms=(\d+|-) "
    r"hedge_ttft_ms=(\d+|-) inv=(\S+) tenant=(\S+)$"
)
CAPPED_LINE = re.compile(
    r"^hedge_capped: agent=(\S+) after_ms=(\d+) used=(\d+) limit=(\d+) inv=(\S+) tenant=(\S+)$"
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


@pytest.mark.parametrize(
    "agent", [EXECUTOR, FORMATTER, CRITIC], ids=["executor", "formatter", "critic"]
)
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


@pytest.mark.parametrize(
    "agent", [EXECUTOR, FORMATTER, CRITIC], ids=["executor", "formatter", "critic"]
)
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


# ── the hedge cap ────────────────────────────────────────────────────────────


def _stalled_then_hedged(tag: str) -> list[list]:
    """The scripts of one call that stalls and is rescued: the stalled primary,
    then the hedge's quick answer."""
    return [[STALL, *_answer(tag)], [0.01, *_answer(f"{tag}-hedge")]]


def _capped_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("hedge_capped:")]


async def _call(router: VendorRouter, agent: str, inv: str = "e-42") -> list[LlmResponse]:
    """One streamed call of `agent` in invocation `inv`, drained."""
    _bind(agent=agent, inv=inv)
    return await _drain(router.generate_content_async(_request(agent), stream=True))


def _caller(agent: str, inv: str = "e-1"):
    return models_module._Caller(agent=agent, invocation_id=inv, tenant="chevy")


async def test_the_third_executor_or_formatter_stall_of_an_invocation_is_not_hedged(logs):
    """Two fires are the pair's budget: the third stalled call waits on its own
    request, relays it whole, and writes one line that says why."""
    slow = _answer("third")
    router, stub = _vendor(
        *_stalled_then_hedged("first"),
        *_stalled_then_hedged("second"),
        [0.2, *slow],
        agent_ms={EXECUTOR: AFTER_MS, FORMATTER: AFTER_MS},
    )

    first = await _call(router, EXECUTOR)
    second = await _call(router, FORMATTER)
    third = await _call(router, EXECUTOR)

    assert _who(first) == {"first-hedge"} and _who(second) == {"second-hedge"}
    assert third == slow  # the primary's reply, relayed whole
    assert len(stub.seen) == 5  # two fires, and the third call sent once
    assert len(_model_hedge_lines(logs)) == 2
    [line] = _capped_lines(logs)
    match = CAPPED_LINE.match(line)
    assert match, line
    assert match.groups() == (EXECUTOR, str(AFTER_MS), "2", "2", "e-42", "chevy")
    assert line.split()[-1] == "tenant=chevy"
    assert not line.startswith(("model_hedge:", "router_hedge:")) and "hedge=1" not in line
    lines = _calls(logs)
    assert len(lines) == 5
    assert (lines[-1]["agent"], lines[-1]["status"], lines[-1]["_hedge"]) == (EXECUTOR, "ok", False)


async def test_the_second_critic_stall_of_an_invocation_is_not_hedged(logs):
    slow = _answer("second")
    router, stub = _vendor(
        *_stalled_then_hedged("first"), [0.2, *slow], agent_ms={CRITIC: AFTER_MS}
    )

    first = await _call(router, CRITIC)
    second = await _call(router, CRITIC)

    assert _who(first) == {"first-hedge"} and second == slow
    assert len(stub.seen) == 3 and len(_model_hedge_lines(logs)) == 1
    [line] = _capped_lines(logs)
    assert CAPPED_LINE.match(line).group(1, 3, 4) == (CRITIC, "1", "1")


@pytest.mark.parametrize(
    "order",
    [(EXECUTOR, FORMATTER, CRITIC), (CRITIC, EXECUTOR, FORMATTER)],
    ids=["pair-then-critic", "critic-then-pair"],
)
async def test_the_critic_and_the_executor_formatter_pair_spend_separate_budgets(logs, order):
    """Three fires in one invocation, one more than a single shared budget of
    two would allow."""
    router, stub = _vendor(
        *[s for i, _ in enumerate(order) for s in _stalled_then_hedged(f"call{i}")],
        agent_ms={EXECUTOR: AFTER_MS, FORMATTER: AFTER_MS, CRITIC: AFTER_MS},
    )

    outs = [await _call(router, agent) for agent in order]

    assert [_who(out) for out in outs] == [{f"call{i}-hedge"} for i in range(3)]
    assert len(_model_hedge_lines(logs)) == 3 and _capped_lines(logs) == []


async def test_another_invocation_has_its_own_budget(logs):
    slow = _answer("third")
    router, stub = _vendor(
        *_stalled_then_hedged("a1"),
        *_stalled_then_hedged("a2"),
        [0.2, *slow],
        *_stalled_then_hedged("b1"),
        agent_ms={EXECUTOR: AFTER_MS},
    )

    a1 = await _call(router, EXECUTOR, "e-1")
    a2 = await _call(router, EXECUTOR, "e-1")
    a3 = await _call(router, EXECUTOR, "e-1")  # past e-1's budget
    b1 = await _call(router, EXECUTOR, "e-2")  # e-2's own

    assert (_who(a1), _who(a2), _who(b1)) == ({"a1-hedge"}, {"a2-hedge"}, {"b1-hedge"})
    assert a3 == slow
    assert len(_model_hedge_lines(logs)) == 3
    [line] = _capped_lines(logs)
    assert CAPPED_LINE.match(line).group(5) == "e-1"


async def test_the_router_is_outside_the_cap_and_spends_nothing(logs):
    router, stub = _vendor(
        *[s for i in range(3) for s in _stalled_then_hedged(f"r{i}")],
        *[s for i in range(2) for s in _stalled_then_hedged(f"x{i}")],
        router_ms=AFTER_MS,
        agent_ms={EXECUTOR: AFTER_MS},
    )

    for i in range(3):  # more than either budget allows
        assert _who(await _call(router, "router")) == {f"r{i}-hedge"}
    for i in range(2):  # and the executor still has both of its own
        assert _who(await _call(router, EXECUTOR)) == {f"x{i}-hedge"}

    assert len(_router_hedge_lines(logs)) == 3 and len(_model_hedge_lines(logs)) == 2
    assert _capped_lines(logs) == []


async def test_a_call_with_no_bound_invocation_is_never_capped(logs):
    router, stub = _vendor(
        *[s for i in range(4) for s in _stalled_then_hedged(f"c{i}")],
        agent_ms={EXECUTOR: AFTER_MS},
    )
    models_module._CALLER.set(None)

    for i in range(4):
        out = await _drain(router.generate_content_async(_request(EXECUTOR), stream=True))
        assert _who(out) == {f"c{i}-hedge"}

    lines = _model_hedge_lines(logs)
    assert len(lines) == 4
    assert all(MODEL_HEDGE_LINE.match(line).group(5, 6) == ("-", "-") for line in lines)
    assert _capped_lines(logs) == [] and models_module._HEDGES == {}


async def test_a_hedge_that_never_fired_spends_none_of_the_budget(logs):
    """Three healthy formatter calls in the same budget (a deadline they never
    reach), then the executor still fires both of its hedges and only the third
    stall is capped — the line says `used=2`."""
    slow = _answer("third")
    router, stub = _vendor(
        *[[0.01, *_answer(f"fast{i}")] for i in range(3)],
        *_stalled_then_hedged("s1"),
        *_stalled_then_hedged("s2"),
        [0.2, *slow],
        agent_ms={EXECUTOR: AFTER_MS, FORMATTER: 2000},
    )

    for i in range(3):
        assert _who(await _call(router, FORMATTER)) == {f"fast{i}"}
    assert _who(await _call(router, EXECUTOR)) == {"s1-hedge"}
    assert _who(await _call(router, EXECUTOR)) == {"s2-hedge"}
    assert await _call(router, EXECUTOR) == slow

    assert len(_model_hedge_lines(logs)) == 2
    [line] = _capped_lines(logs)
    assert CAPPED_LINE.match(line).group(1, 3, 4) == (EXECUTOR, "2", "2")


async def test_two_concurrent_calls_of_one_invocation_share_one_budget(logs):
    """The critic's budget is one: of two calls stalled at once, exactly one
    sends a spare and the other waits on its own request."""
    router, stub = _vendor(
        [0.3, *_answer("A")],
        [0.3, *_answer("B")],
        [0.01, *_answer("H")],
        agent_ms={CRITIC: AFTER_MS},
    )
    _bind(agent=CRITIC, inv="e-42")

    async def one():
        return await _drain(router.generate_content_async(_request(CRITIC), stream=True))

    outs = await asyncio.gather(one(), one())

    assert len(stub.seen) == 3  # both primaries and ONE spare
    assert len(_model_hedge_lines(logs)) == 1 and len(_capped_lines(logs)) == 1
    names = sorted(next(iter(_who(out))) for out in outs)
    assert names[1] == "H" and names[0] in ("A", "B")


async def test_a_failing_budget_lets_the_hedge_fire(logs, monkeypatch):
    class _Broken:
        def get(self, *args):
            raise RuntimeError("the budget store is unavailable")

    monkeypatch.setattr(models_module, "_HEDGES", _Broken())
    router, stub = _vendor(*_stalled_then_hedged("only"), agent_ms={EXECUTOR: AFTER_MS})

    out = await _call(router, EXECUTOR)

    assert _who(out) == {"only-hedge"} and len(stub.seen) == 2
    assert len(_model_hedge_lines(logs)) == 1 and _capped_lines(logs) == []


def test_take_hedge_spends_per_budget_and_never_caps_the_uncapped():
    take = models_module._take_hedge
    assert [take(EXECUTOR, _caller(EXECUTOR)) for _ in range(3)] == [
        (True, 1, 2),
        (True, 2, 2),
        (False, 2, 2),
    ]
    assert take(FORMATTER, _caller(FORMATTER)) == (False, 2, 2)  # the pair shares a budget
    assert [take(CRITIC, _caller(CRITIC)) for _ in range(2)] == [(True, 1, 1), (False, 1, 1)]
    assert take(EXECUTOR, _caller(EXECUTOR, "e-2")) == (True, 1, 2)
    for agent in ("router", "some-other-agent"):
        assert take(agent, _caller(agent)) == (True, 0, 0)
    assert take(EXECUTOR, None) == (True, 0, 0)
    assert take(EXECUTOR, _caller(EXECUTOR, "-")) == (True, 0, 0)
    assert set(models_module._HEDGES) == {"e-1", "e-2"}  # the uncapped never enter the store


def test_the_budget_store_is_bounded_and_forgets_the_least_recent_invocation_first():
    take = models_module._take_hedge
    bound = models_module._MAX_TRACKED
    for i in range(bound):
        assert take(EXECUTOR, _caller(EXECUTOR, f"e-{i}"))[0]
    assert len(models_module._HEDGES) == bound
    take(EXECUTOR, _caller(EXECUTOR, "e-0"))  # touched: now the most recent
    take(EXECUTOR, _caller(EXECUTOR, f"e-{bound}"))  # one more than the bound
    assert len(models_module._HEDGES) == bound
    assert "e-0" in models_module._HEDGES and "e-1" not in models_module._HEDGES
    # An invocation that was dropped starts again with a full budget: the cap fails open.
    assert take(EXECUTOR, _caller(EXECUTOR, "e-1")) == (True, 1, 2)


@pytest.mark.parametrize("agent,limit", [(EXECUTOR, 2), (CRITIC, 1)], ids=["executor", "critic"])
async def test_a_call_genai_is_retrying_spends_none_of_the_budget(logs, agent, limit):
    """A call genai is already retrying (a 429 answered it) is not hedged, so it
    must not spend the hedge either: `limit` retried calls in one invocation, and
    a genuine stall after them still gets its hedge."""
    gemini = _Retrying(
        model=GEMINI_MODEL,
        stall=[0.0] * limit + [STALL, 0.0],
        failures=[1] * limit + [0, 0],
        wait=0.15,
    )
    router = VendorRouter(
        model=GEMINI_MODEL,
        gemini=gemini,  # type: ignore[arg-type]
        agent_hedge_after_ms={agent: AFTER_MS},
    )

    for i in range(limit):
        assert _by(await _call(router, agent)) == {f"call{i}"}
    assert [line["retries"] for line in _calls(logs)] == ["1"] * limit  # the retry path was taken
    assert _model_hedge_lines(logs) == [] and models_module._HEDGES == {}

    assert _by(await _call(router, agent)) == {f"call{limit + 1}"}  # the spare answered the stall

    assert gemini.calls == limit + 2
    [fired] = _model_hedge_lines(logs)
    assert MODEL_HEDGE_LINE.match(fired).group(1, 2) == (agent, "hedge")
    assert _capped_lines(logs) == []


async def test_the_lines_of_a_speculative_run_end_in_the_spec_mark(logs):
    """The speculative deep run marks every line it makes with ` spec=1`
    (agents/speculation.py), after tenant=: the cap's and the hedge's included."""
    from gub_agent.agents import speculation

    router, stub = _vendor(
        *_stalled_then_hedged("first"), [0.2, *_answer("second")], agent_ms={CRITIC: AFTER_MS}
    )
    token = speculation._SPECULATIVE.set(True)
    try:
        await _call(router, CRITIC)
        await _call(router, CRITIC)
    finally:
        speculation._SPECULATIVE.reset(token)

    [capped] = _capped_lines(logs)
    assert capped.startswith("hedge_capped: agent=critic ") and capped.endswith(
        " tenant=chevy spec=1"
    )
    [fired] = _model_hedge_lines(logs)
    assert fired.endswith(" tenant=chevy spec=1")


def test_the_per_pass_resets_do_not_return_a_spent_budget():
    """The executor's own per-pass reset (agent.py `_before_agent`) and a
    speculative run's `clear` call the same three resets. The hedge budget
    bounds the money spent over the WHOLE invocation, so none of them may touch
    it: a reset that did would turn "one critic hedge per invocation" into one
    per pass."""
    from gub_agent.agent import _before_agent
    from gub_agent.agents.circuit_breaker import reset_tool_budget
    from gub_agent.agents.evidence_index import reset_evidence_index
    from gub_agent.agents.round_limiter import reset_rounds

    take = models_module._take_hedge
    ctx = SimpleNamespace(invocation_id="e-1", state={"tenant": "chevy"})
    assert [take(CRITIC, _caller(CRITIC)) for _ in range(2)] == [(True, 1, 1), (False, 1, 1)]
    assert take(EXECUTOR, _caller(EXECUTOR)) == (True, 1, 2)

    _before_agent(ctx)
    for reset in (reset_rounds, reset_tool_budget, reset_evidence_index):
        reset(ctx)

    assert take(CRITIC, _caller(CRITIC)) == (False, 1, 1)
    assert take(EXECUTOR, _caller(EXECUTOR)) == (True, 2, 2)


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


def test_the_critic_deadline_is_keyed_on_the_critics_name():
    assert config.CRITIC_AGENT_NAME == critic_agent.name


_READ_FLAGS = """
import json
from gub_agent import config
from gub_agent.agent import executor_agent
from gub_agent.agents.critic import critic_agent
from gub_agent.agents.formatter import formatter_agent
from gub_agent.agents.router import router_agent
print(json.dumps([
    executor_agent.model.agent_hedge_after_ms,
    formatter_agent.model.agent_hedge_after_ms,
    critic_agent.model.agent_hedge_after_ms,
    router_agent.model.router_hedge_after_ms,
]))
"""


@pytest.mark.parametrize(
    "executor,formatter,critic,expected",
    [
        (None, None, None, {}),
        ("0", "0", "0", {}),
        ("25000", None, None, {"gub_agent": 25000}),
        ("25000", "20000", None, {"gub_agent": 25000, "formatter": 20000}),
        (None, None, "15000", {"critic": 15000}),
        ("25000", "20000", "15000", {"gub_agent": 25000, "formatter": 20000, "critic": 15000}),
    ],
    ids=["default", "all-0", "executor-only", "executor-formatter", "critic-only", "all"],
)
def test_the_flags_reach_the_deployed_models(executor, formatter, critic, expected):
    """Read at import, so each setting gets a fresh interpreter. The router's
    own deadline is untouched by any of them."""
    names = (
        "EXECUTOR_HEDGE_AFTER_MS",
        "FORMATTER_HEDGE_AFTER_MS",
        "CRITIC_HEDGE_AFTER_MS",
        "ROUTER_HEDGE_AFTER_MS",
        "AGENT_NAME",
    )
    env = {k: v for k, v in os.environ.items() if k not in names}
    for key, value in (
        ("EXECUTOR_HEDGE_AFTER_MS", executor),
        ("FORMATTER_HEDGE_AFTER_MS", formatter),
        ("CRITIC_HEDGE_AFTER_MS", critic),
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
    executor_map, formatter_map, critic_map, router_ms = json.loads(
        out.stdout.strip().splitlines()[-1]
    )
    assert executor_map == formatter_map == critic_map == expected
    assert router_ms == 4000


def _stated(name: str, key: str) -> list[str]:
    return [
        line.partition("=")[2].strip()
        for line in (REPO / name).read_text().splitlines()
        if line.startswith(f"{key}=")
    ]


@pytest.mark.parametrize(
    "key", ["EXECUTOR_HEDGE_AFTER_MS", "FORMATTER_HEDGE_AFTER_MS", "CRITIC_HEDGE_AFTER_MS"]
)
def test_both_deploy_env_files_state_the_flag_with_one_value(key):
    """Stated, not defaulted — the rollback is a line already in front of the
    operator — and the same in both, so a sandbox run measures what
    production runs."""
    prod, sandbox = _stated("deploy-prod.env", key), _stated("deploy-sandbox.env", key)
    assert len(prod) == 1 and len(sandbox) == 1, (prod, sandbox)
    assert prod == sandbox
    assert int(prod[0]) >= 0
