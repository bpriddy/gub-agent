"""
round_limiter — the ReAct round cap and its per-executor-pass reset.

The reset is the load-bearing part: ADK's LoopAgent reuses ONE invocation_id
across retry iterations, so without reset_rounds() a critic-requested retry
would inherit the first pass's round count and open with tools already stripped —
unable to make the call the critic asked for. Driven with fake ADK callback
objects; the round logic is synchronous (tests are async to match the suite).
"""

from __future__ import annotations

from types import SimpleNamespace

from gub_agent.agents.round_limiter import MAX_ROUNDS, reset_rounds, round_limit


def _ctx(inv: str) -> SimpleNamespace:
    return SimpleNamespace(invocation_id=inv)


def _req() -> SimpleNamespace:
    """A fresh llm_request stand-in with tools present."""
    return SimpleNamespace(
        config=SimpleNamespace(tools=["tool"]), tools_dict={"tool": 1}, contents=[]
    )


async def test_within_budget_keeps_tools():
    reset_rounds(_ctx("a"))
    req = _req()
    round_limit(_ctx("a"), req)  # round 1 — under the cap
    assert req.config.tools == ["tool"]
    assert req.contents == []  # no synthesize directive yet


async def test_past_budget_strips_tools_and_appends_directive():
    reset_rounds(_ctx("b"))
    req = None
    for _ in range(MAX_ROUNDS + 1):  # one past the cap
        req = _req()
        round_limit(_ctx("b"), req)
    assert req.config.tools == []  # tools removed → model must synthesize
    assert req.tools_dict == {}
    assert len(req.contents) == 1  # "answer now" directive appended


async def test_reset_gives_a_retry_a_fresh_budget():
    inv = "c"
    reset_rounds(_ctx(inv))
    for _ in range(MAX_ROUNDS + 1):  # exhaust — last call strips tools
        round_limit(_ctx(inv), _req())

    # A critic-requested retry: before_agent_callback resets the pass budget.
    reset_rounds(_ctx(inv))
    req = _req()
    round_limit(_ctx(inv), req)  # first round of the retry pass

    assert req.config.tools == ["tool"]  # NOT stripped — the fix
    assert req.contents == []
