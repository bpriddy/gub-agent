"""
circuit_breaker — per-executor-pass tool budget + loop dedup, and the reset.

Same retry-iteration concern as round_limiter: the budget is keyed on
invocation_id (reused across LoopAgent iterations), so reset_tool_budget() runs
per executor pass to keep a critic-requested retry from inheriting the first
pass's count. Fake ADK tool/context objects; logic is synchronous.
"""

from __future__ import annotations

from types import SimpleNamespace

from gub_agent.agents.circuit_breaker import (
    MAX_TOOL_CALLS,
    circuit_breaker,
    reset_tool_budget,
)


def _tool(name: str = "org_query") -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _ctx(inv: str) -> SimpleNamespace:
    return SimpleNamespace(invocation_id=inv)


async def test_allows_a_call_within_budget():
    reset_tool_budget(_ctx("a"))
    assert circuit_breaker(_tool(), {"q": 1}, _ctx("a")) is None


async def test_dedupes_an_identical_repeat():
    reset_tool_budget(_ctx("b"))
    ctx = _ctx("b")
    assert circuit_breaker(_tool(), {"q": 1}, ctx) is None  # first — allowed
    blocked = circuit_breaker(_tool(), {"q": 1}, ctx)  # exact repeat
    assert blocked is not None and blocked["error"] is True


async def test_caps_at_the_tool_budget():
    reset_tool_budget(_ctx("c"))
    ctx = _ctx("c")
    for i in range(MAX_TOOL_CALLS):  # distinct args — dedup doesn't fire
        assert circuit_breaker(_tool(), {"q": i}, ctx) is None
    over = circuit_breaker(_tool(), {"q": 999}, ctx)  # one past the cap
    assert over is not None and over["error"] is True


async def test_reset_gives_a_retry_a_fresh_budget():
    inv = "d"
    reset_tool_budget(_ctx(inv))
    ctx = _ctx(inv)
    for i in range(MAX_TOOL_CALLS):
        circuit_breaker(_tool(), {"q": i}, ctx)  # exhaust
    assert circuit_breaker(_tool(), {"q": 999}, ctx)["error"] is True  # over budget

    reset_tool_budget(_ctx(inv))  # critic-requested retry pass
    assert circuit_breaker(_tool(), {"q": 1000}, ctx) is None  # fresh budget
