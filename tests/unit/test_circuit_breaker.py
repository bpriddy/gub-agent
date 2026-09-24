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


# ── Query weakening (find_files) ──────────────────────────────────────────────
# The failure these pin was measured on the sandbox engine 2026-09-24: asked for
# "the final OnStar pitch pre-read doc" the endpoint correctly found nothing, so
# the model dropped words and retried until "pre-read" matched an unrelated
# file, then listed five of them under a "not found" headline.


def _find_files(query: str, **extra) -> dict:
    return {"query": query, **extra}


async def test_refuses_a_query_that_drops_words_from_an_earlier_one():
    reset_tool_budget(_ctx("w1"))
    tool = _tool("find_files")
    assert circuit_breaker(tool, _find_files("OnStar pitch pre-read"), _ctx("w1")) is None
    # Every step the model actually took, each a subset of the first.
    for weaker in ("OnStar pitch", "pitch pre-read", "pre-read"):
        blocked = circuit_breaker(tool, _find_files(weaker), _ctx("w1"))
        assert blocked is not None, weaker
        assert blocked["error"] is True
        assert "fewer words" in blocked["message"]


async def test_allows_a_query_that_adds_words():
    reset_tool_budget(_ctx("w2"))
    tool = _tool("find_files")
    assert circuit_breaker(tool, _find_files("BHAC teaser"), _ctx("w2")) is None
    # More description is more specific, not less — the direction that can only
    # improve an answer, so it stays allowed.
    broader = _find_files("BHAC Hits The Road 30s teaser")
    assert circuit_breaker(tool, broader, _ctx("w2")) is None


async def test_allows_an_unrelated_second_search():
    reset_tool_budget(_ctx("w3"))
    tool = _tool("find_files")
    assert circuit_breaker(tool, _find_files("BHAC teaser"), _ctx("w3")) is None
    other = _find_files("Silverado HD strategic brief")
    assert circuit_breaker(tool, other, _ctx("w3")) is None


async def test_refuses_a_pure_reordering_which_carries_no_new_information():
    reset_tool_budget(_ctx("w4"))
    tool = _tool("find_files")
    assert circuit_breaker(tool, _find_files("OnStar pitch"), _ctx("w4")) is None
    # Same words, different order and punctuation: the args hash differs, so
    # loop detection misses it, but nothing new is being asked.
    assert circuit_breaker(tool, _find_files("pitch, OnStar"), _ctx("w4")) is not None


async def test_narrowing_by_account_does_not_excuse_a_weaker_query():
    reset_tool_budget(_ctx("w5"))
    tool = _tool("find_files")
    assert circuit_breaker(tool, _find_files("OnStar pitch pre-read"), _ctx("w5")) is None
    # The model reached for account_id on its third and fourth attempts; scoping
    # the search does not make a shorter description a better one.
    weaker = _find_files("pre-read", account_id="acc-1")
    assert circuit_breaker(tool, weaker, _ctx("w5")) is not None


async def test_the_guard_is_find_files_only():
    reset_tool_budget(_ctx("w6"))
    assert circuit_breaker(_tool("org_query"), _find_files("a b c"), _ctx("w6")) is None
    assert circuit_breaker(_tool("org_query"), _find_files("a b"), _ctx("w6")) is None


async def test_a_retry_pass_may_search_again():
    reset_tool_budget(_ctx("w7"))
    tool = _tool("find_files")
    assert circuit_breaker(tool, _find_files("OnStar pitch pre-read"), _ctx("w7")) is None
    # The critic asked for another pass; the budget resets, so the executor is
    # not stuck with the previous pass's history.
    reset_tool_budget(_ctx("w7"))
    assert circuit_breaker(tool, _find_files("pre-read"), _ctx("w7")) is None
