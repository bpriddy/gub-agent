"""
The critic beside the format gate (CRITIC_PARALLEL) — agent.py:build_deep_agent.

The deep loop used to run [executor, format_gate, critic_gate, escalator] one
after another, although the critic judges the executor's tool coverage and
never reads the formatter's payload. The parallel wiring runs the critic LLM
while the formatter renders, holds its events, and lets a resolver named
`critic_gate` make CriticGate's decisions after the join.

What has to stay true, and is pinned here on a REAL ADK runner (InMemoryRunner
over the real LoopAgent / ParallelAgent / CriticResolver / escalator, with
stand-ins only for the three model-backed agents):

- the two really overlap, and the critic's branch still sees the executor's
  work (and not the formatter's — ADK's branch isolation);
- the loop exits on the escalator's escalate, and an insufficient verdict still
  buys the executor a second pass that can read the feedback;
- the abstain pass and the from-memory re-query are decided on THIS pass's
  payload — never on a stale one left in state by an earlier turn — and a held
  verdict they overrule never reaches the stream, the state or the bot;
- the stream keeps its order and its authors: the last payload event is the
  format gate's, the verdict follows it;
- CRITIC_PARALLEL=0 builds the serial tree it replaced, and makes the same
  decisions as the parallel one on the same turns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from google.adk.agents import BaseAgent, LoopAgent, ParallelAgent
from google.adk.events import Event, EventActions
from google.adk.flows.llm_flows.contents import _get_contents
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from gub_agent import config
from gub_agent.agent import build_deep_agent
from gub_agent.agents.critic import (
    CriticGate,
    CriticResolver,
    EscalateIfSufficient,
    SpeculativeCritic,
)
from gub_agent.config import AGENT_NAME

REPO = Path(__file__).resolve().parents[2]

ANSWER = {"kind": "answer", "headline": "chevy is live", "citations": ["org_query:a1"]}
ABSTAIN = {"kind": "abstain", "headline": "NO_COMPANY_RECORDS"}
SUFFICIENT = {
    "info_sufficient": True,
    "answer_satisfies": True,
    "sufficient": True,
    "reason": "LLM: covered",
    "feedback": "",
}
INSUFFICIENT = {
    "info_sufficient": False,
    "answer_satisfies": False,
    "sufficient": False,
    "reason": "LLM: the Q3 campaign was never read",
    "feedback": "call get_campaign for the Q3 push",
}


# ── stand-ins for the model-backed agents ────────────────────────────────────


def _text_event(ctx, author: str, text: str, *, partial: bool = False) -> Event:
    return Event(
        invocation_id=ctx.invocation_id,
        author=author,
        branch=ctx.branch,
        partial=partial or None,
        content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]),
    )


class _Executor(BaseAgent):
    """Per pass: an optional tool call + its result, then a text answer."""

    passes: list = []  # [(made_tool_call, text)], one per pass
    runs: int = 0

    async def _run_async_impl(self, ctx):
        tool_call, text = self.passes[min(self.runs, len(self.passes) - 1)]
        self.runs += 1
        if tool_call:
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            function_call=genai_types.FunctionCall(
                                id="call-1", name="org_query", args={}
                            )
                        )
                    ],
                ),
            )
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                content=genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                id="call-1", name="org_query", response={"total": 12}
                            )
                        )
                    ],
                ),
            )
        if text:
            yield _text_event(ctx, self.name, text)


class _Formatter(BaseAgent):
    """The format gate's formatter: its payload event carries the branch ADK
    gave it, exactly as an LlmAgent's does."""

    payloads: list = []  # one per pass; None = the gate emits nothing
    runs: int = 0
    started: asyncio.Event | None = None  # set on entry
    done: asyncio.Event | None = None  # set once its payload is committed
    wait_for: asyncio.Event | None = None
    timeout: float = 2.0

    async def _run_async_impl(self, ctx):
        payload = self.payloads[min(self.runs, len(self.payloads) - 1)]
        self.runs += 1
        if self.started is not None:
            self.started.set()
        if self.wait_for is not None:
            await asyncio.wait_for(self.wait_for.wait(), timeout=self.timeout)
        if payload is not None:
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=genai_types.Content(
                    role="model", parts=[genai_types.Part(text=json.dumps(payload))]
                ),
                actions=EventActions(state_delta={"answer_payload": payload}),
            )
        # Resumed only after the runner consumed (and appended) the event.
        if self.done is not None:
            self.done.set()


class _Gate(BaseAgent):
    """format_gate's shape: runs its formatter, relays what it emits."""

    async def _run_async_impl(self, ctx):
        async for event in self.sub_agents[0].run_async(ctx):
            yield event


class _Critic(BaseAgent):
    """The critic LLM: a streamed partial, then the verdict as text + state,
    the way an LlmAgent with output_key emits it. Records what its branch saw."""

    verdicts: list = []  # one per call
    ran: list = []
    saw: list = []  # the rendered request contents, per call
    session_authors: list = []  # who was in the session when it looked
    raises: Exception | None = None
    started: asyncio.Event | None = None
    wait_for: asyncio.Event | None = None
    timeout: float = 2.0

    async def _run_async_impl(self, ctx):
        self.ran.append(ctx.invocation_id)
        if self.started is not None:
            self.started.set()
        if self.wait_for is not None:
            await asyncio.wait_for(self.wait_for.wait(), timeout=self.timeout)
        self.session_authors.append([e.author for e in ctx.session.events])
        contents = _get_contents(ctx.branch, ctx.session.events, self.name)
        self.saw.append(
            " ".join(
                str(part.text or part.function_call or part.function_response or "")
                for content in contents
                for part in content.parts or []
            )
        )
        if self.raises is not None:
            raise self.raises
        verdict = self.verdicts[min(len(self.ran) - 1, len(self.verdicts) - 1)]
        yield _text_event(ctx, self.name, '{"info_suff', partial=True)
        event = _text_event(ctx, self.name, json.dumps(verdict))
        event.actions = EventActions(state_delta={"critic_verdict": verdict})
        yield event


def _pipeline(
    *,
    parallel: bool,
    executor_passes: list,
    payloads: list,
    verdicts: list | None = None,
    critic_raises: Exception | None = None,
) -> tuple[LoopAgent, _Executor, _Formatter, _Critic]:
    executor = _Executor(name=AGENT_NAME, passes=executor_passes, runs=0)
    formatter = _Formatter(name="formatter", payloads=payloads, runs=0)
    gate = _Gate(name="format_gate", sub_agents=[formatter])
    critic = _Critic(
        name="critic",
        verdicts=verdicts or [SUFFICIENT],
        ran=[],
        saw=[],
        session_authors=[],
        raises=critic_raises,
    )
    escalator = EscalateIfSufficient(name="loop_escalator")
    loop = build_deep_agent(executor, gate, critic, escalator, parallel=parallel)
    return loop, executor, formatter, critic


async def _run(
    loop: LoopAgent, *, question: str = "how is chevy?", state=None, prior=None
) -> tuple[list[Event], dict, list[Event]]:
    """One turn through a real runner: (streamed events, final state, session events)."""
    runner = InMemoryRunner(agent=loop, app_name="gub")
    session = await runner.session_service.create_session(
        app_name="gub", user_id="u", state=state or {}
    )
    for event in prior or []:
        await runner.session_service.append_event(session, event)
    message = genai_types.Content(role="user", parts=[genai_types.Part(text=question)])
    streamed = [
        event
        async for event in runner.run_async(user_id="u", session_id=session.id, new_message=message)
    ]
    final = await runner.session_service.get_session(
        app_name="gub", user_id="u", session_id=session.id
    )
    return streamed, dict(final.state), final.events


def _complete(events: list[Event]) -> list[Event]:
    return [event for event in events if not event.partial]


def _authors(events: list[Event]) -> list[str]:
    return [event.author for event in _complete(events)]


# ── the tree ─────────────────────────────────────────────────────────────────


def test_the_default_tree_runs_the_critic_beside_the_format_gate():
    from gub_agent.agent import deep_agent, root_agent
    from gub_agent.agents.critic import critic_agent
    from gub_agent.agents.format_gate import format_gate

    if not config.CRITIC_PARALLEL:
        pytest.skip("CRITIC_PARALLEL=0 in this environment — the serial tree is pinned below")
    names = [agent.name for agent in deep_agent.sub_agents]
    assert names == [AGENT_NAME, "format_and_critic", "critic_gate", "loop_escalator"]
    fork = deep_agent.sub_agents[1]
    assert isinstance(fork, ParallelAgent)
    assert fork.sub_agents[0] is format_gate
    assert isinstance(fork.sub_agents[1], SpeculativeCritic)
    assert fork.sub_agents[1].sub_agents == [critic_agent]
    resolver = deep_agent.sub_agents[2]
    assert isinstance(resolver, CriticResolver)
    assert resolver.payload_authors == ("format_gate", "formatter")
    assert deep_agent.max_iterations == 2
    # The critic is IN the tree: the runner resolves every event author with
    # find_sub_agent, and an author it cannot find is a warning per event per
    # turn (agents/_agent_router.py).
    assert root_agent.find_agent("critic") is critic_agent


_SERIAL_TREE = [
    "LoopAgent",
    "gub_pipeline",
    [
        ["LlmAgent", AGENT_NAME, []],
        ["FormatGate", "format_gate", [["LlmAgent", "formatter", []]]],
        ["CriticGate", "critic_gate", [["LlmAgent", "critic", []]]],
        ["EscalateIfSufficient", "loop_escalator", []],
    ],
]

_DUMP = """
import json
from gub_agent import agent
from gub_agent.agents import critic
from gub_agent.agents.format_gate import format_gate

def tree(a):
    return [type(a).__name__, a.name, [tree(s) for s in a.sub_agents]]

deep = agent.deep_agent
print(json.dumps({
    "root": [a.name for a in agent.root_agent.sub_agents],
    "deep": tree(deep),
    "max_iterations": deep.max_iterations,
    "same_objects": [
        deep.sub_agents[0] is agent.executor_agent,
        deep.sub_agents[1] is format_gate,
        deep.sub_agents[2].sub_agents[0] is critic.critic_agent,
        deep.sub_agents[3] is critic.escalator_agent,
    ],
}))
"""


def test_critic_parallel_0_builds_the_serial_tree_it_replaced():
    """The rollback. Read at import, so it is checked in a fresh interpreter:
    the tree must be today's serial one, built from the same module objects."""
    env = {**os.environ, "CRITIC_PARALLEL": "0", "PYTHONPATH": str(REPO)}
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _DUMP],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    dumped = json.loads(out.stdout.strip().splitlines()[-1])
    assert dumped["root"] == ["sandbox_echo", "router", "dispatcher"]
    assert dumped["deep"] == _SERIAL_TREE
    assert dumped["max_iterations"] == 2
    assert dumped["same_objects"] == [True, True, True, True]


def test_the_builder_makes_both_shapes():
    parallel, *_ = _pipeline(parallel=True, executor_passes=[(True, "x")], payloads=[ANSWER])
    assert [type(a).__name__ for a in parallel.sub_agents] == [
        "_Executor",
        "ParallelAgent",
        "CriticResolver",
        "EscalateIfSufficient",
    ]
    assert [a.name for a in parallel.sub_agents[1].sub_agents] == [
        "format_gate",
        "critic_speculation",
    ]

    serial, *_ = _pipeline(parallel=False, executor_passes=[(True, "x")], payloads=[ANSWER])
    assert [type(a).__name__ for a in serial.sub_agents] == [
        "_Executor",
        "_Gate",
        "CriticGate",
        "EscalateIfSufficient",
    ]
    assert [a.name for a in serial.sub_agents] == [
        AGENT_NAME,
        "format_gate",
        "critic_gate",
        "loop_escalator",
    ]
    gate = serial.sub_agents[2]
    assert type(gate) is CriticGate  # the gate itself, not the resolver
    assert [a.name for a in gate.sub_agents] == ["critic"]


# ── concurrency and branch isolation ─────────────────────────────────────────


async def test_the_critic_runs_while_the_formatter_does():
    """Each waits for the other to have STARTED. Run one after the other, the
    first would time out waiting; overlapped, both finish."""
    loop, _, formatter, critic = _pipeline(
        parallel=True, executor_passes=[(True, "12 live campaigns.")], payloads=[ANSWER]
    )
    formatter.started, critic.started = asyncio.Event(), asyncio.Event()
    formatter.wait_for, critic.wait_for = critic.started, formatter.started

    streamed, state, _ = await _run(loop)

    assert critic.ran and formatter.runs == 1
    assert state["critic_verdict"]["sufficient"] is True
    assert streamed[-1].author == "loop_escalator" and streamed[-1].actions.escalate


async def test_the_serial_tree_does_not_overlap_them():
    """Control for the test above: the same barrier deadlocks the serial tree,
    so the overlap is the wiring's doing and not the stand-ins'."""
    loop, _, formatter, critic = _pipeline(
        parallel=False, executor_passes=[(True, "12 live campaigns.")], payloads=[ANSWER]
    )
    formatter.started, critic.started = asyncio.Event(), asyncio.Event()
    formatter.wait_for, critic.wait_for = critic.started, formatter.started
    formatter.timeout = critic.timeout = 0.3

    with pytest.raises(TimeoutError):
        await _run(loop)


async def test_the_critics_branch_sees_the_executor_but_not_the_formatter():
    """ParallelAgent isolates history per branch and shares state. The critic
    must still see the executor's call, result and text (the evidence it
    judges); the formatter's JSON, on the sibling branch, it does not see —
    and never needed (prompts/critic.py calls it pipeline plumbing)."""
    loop, _, formatter, critic = _pipeline(
        parallel=True, executor_passes=[(True, "There are 12 live campaigns.")], payloads=[ANSWER]
    )
    # The critic looks only after the formatter's payload has been committed.
    formatter.done = critic.wait_for = asyncio.Event()

    await _run(loop)

    [seen] = critic.saw
    assert "There are 12 live campaigns." in seen
    assert "org_query" in seen  # the call and its result
    assert "chevy is live" not in seen  # the formatter's payload…
    assert "formatter" in critic.session_authors[0]  # …which WAS in the session


# ── loop control ─────────────────────────────────────────────────────────────


async def test_a_sufficient_verdict_ends_the_loop_after_one_pass():
    loop, executor, _, critic = _pipeline(
        parallel=True, executor_passes=[(True, "12 live campaigns.")], payloads=[ANSWER]
    )

    streamed, state, _ = await _run(loop)

    assert executor.runs == 1 and len(critic.ran) == 1
    assert state["critic_verdict"] == SUFFICIENT
    assert [e.author for e in streamed if e.actions.escalate] == ["loop_escalator"]


async def test_an_insufficient_verdict_buys_a_second_pass_that_reads_the_feedback():
    loop, executor, formatter, critic = _pipeline(
        parallel=True,
        executor_passes=[(True, "12 live campaigns."), (True, "Q3 push is live.")],
        payloads=[ANSWER, ANSWER],
        verdicts=[INSUFFICIENT, SUFFICIENT],
    )

    streamed, state, session_events = await _run(loop)

    assert executor.runs == 2 and formatter.runs == 2
    # The retry reads the feedback from "[critic] said: …" — nothing injects
    # state into its prompt — so the held verdict must be in its history.
    verdict_at = next(i for i, e in enumerate(session_events) if e.author == "critic")
    assert session_events[verdict_at + 1].author == AGENT_NAME  # pass 2 starts
    contents = _get_contents(None, session_events[: verdict_at + 1], AGENT_NAME)
    rendered = " ".join(p.text or "" for c in contents for p in c.parts or [])
    assert INSUFFICIENT["feedback"] in rendered
    assert streamed[-1].author == "loop_escalator"
    assert state["critic_verdict"]["sufficient"] is True


# ── the decisions after the join ─────────────────────────────────────────────


async def test_an_abstain_after_a_tool_call_passes_and_the_held_verdict_is_dropped(caplog):
    """The speculative critic said "insufficient"; CriticGate's rule for an
    abstain payload after a tool call is a pass without the LLM. The dropped
    verdict must reach neither the stream (the bot would count it and arm a
    pass restart) nor the state (the escalator would read it as a retry)."""
    loop, executor, _, critic = _pipeline(
        parallel=True,
        executor_passes=[(True, "GUB has no account by that name.")],
        payloads=[ABSTAIN],
        verdicts=[INSUFFICIENT],
    )

    with caplog.at_level(logging.INFO, logger="gub_agent.agents.critic"):
        streamed, state, session_events = await _run(loop)

    assert len(critic.ran) == 1  # it ran — speculatively
    assert "critic" not in [e.author for e in streamed]
    assert "critic" not in [e.author for e in session_events]
    assert state["critic_verdict"]["reason"] == (
        "Deterministic pass: abstain AnswerPayload (no critic LLM run)."
    )
    assert executor.runs == 1
    assert "critic_speculation: verdict discarded" in caplog.text
    assert "tenant=anomaly" in caplog.text


async def test_a_from_memory_abstain_is_sent_back_after_the_join():
    """No tool call and an abstain payload: the draft was written from memory
    and is sent back once, in code, authored as the critic."""
    loop, executor, _, critic = _pipeline(
        parallel=True,
        executor_passes=[(False, "Here is what's new: 3 hires."), (True, "3 hires, 12 live.")],
        payloads=[ABSTAIN, ANSWER],
        verdicts=[SUFFICIENT],
    )

    streamed, state, _ = await _run(loop, question="whats new")

    assert executor.runs == 2
    send_back = [e for e in _complete(streamed) if e.author == "critic"][0]
    verdict = json.loads(send_back.content.parts[0].text)
    assert verdict["sufficient"] is False and "NO tool call" in verdict["feedback"]
    assert streamed[-1].author == "loop_escalator"


async def test_a_stale_payload_left_in_state_by_an_earlier_turn_is_not_read():
    """The gate emits nothing when the executor produced no text — on purpose.
    State then still holds the previous turn's abstain payload, and a gate that
    read state would settle THIS turn on it (here: a from-memory re-query).
    The resolver reads this pass's payload, finds none, and asks the critic."""
    earlier = [
        Event(
            invocation_id="inv-earlier",
            author="format_gate",
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=json.dumps(ABSTAIN))]
            ),
            actions=EventActions(state_delta={"answer_payload": ABSTAIN}),
        )
    ]
    loop, executor, _, critic = _pipeline(
        parallel=True,
        executor_passes=[(False, "Thanks, noted.")],
        payloads=[None],
        verdicts=[SUFFICIENT],
    )

    streamed, state, _ = await _run(loop, state={"answer_payload": ABSTAIN}, prior=earlier)

    assert state["answer_payload"] == ABSTAIN  # still there, still stale
    assert len(critic.ran) == 1
    assert state["critic_verdict"] == SUFFICIENT  # the critic's, not a re-query
    assert executor.runs == 1


async def test_the_first_passes_payload_is_not_the_seconds():
    """Same edge inside one turn: pass 1 abstained from memory and was sent
    back; pass 2's executor called a tool but wrote nothing, so its gate
    emitted nothing. Pass 1's abstain payload is this invocation's, and still
    not this PASS's — the critic decides pass 2, not a second abstain pass."""
    loop, executor, _, critic = _pipeline(
        parallel=True,
        executor_passes=[(False, "Copied from memory."), (True, "")],
        payloads=[ABSTAIN, None],
        verdicts=[SUFFICIENT],
    )

    _, state, _ = await _run(loop, question="whats new")

    assert executor.runs == 2
    assert state["critic_verdict"] == SUFFICIENT


async def test_the_bare_marker_makes_no_critic_call():
    loop, _, _, critic = _pipeline(
        parallel=True, executor_passes=[(False, "NO_COMPANY_RECORDS")], payloads=[ABSTAIN]
    )

    streamed, state, _ = await _run(loop)

    assert critic.ran == []  # known before the fork: no speculative call
    assert state["critic_verdict"]["sufficient"] is True
    assert "critic_gate" in _authors(streamed)


async def test_a_disabled_critic_makes_no_critic_call(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_ENABLED", True)
    loop, _, _, critic = _pipeline(
        parallel=True, executor_passes=[(True, "12 live campaigns.")], payloads=[ANSWER]
    )

    _, state, _ = await _run(loop, state={"sandbox": {"critic_enabled": False}})

    assert critic.ran == []
    assert state["critic_verdict"]["reason"] == "sandbox: critic disabled"


async def test_a_held_critic_error_fails_only_a_turn_that_needs_the_verdict():
    boom = RuntimeError("critic model failed")

    settled, *_ = _pipeline(
        parallel=True,
        executor_passes=[(True, "GUB has no account by that name.")],
        payloads=[ABSTAIN],
        critic_raises=boom,
    )
    _, state, _ = await _run(settled)  # the abstain pass never needed it
    assert state["critic_verdict"]["sufficient"] is True

    needed, *_ = _pipeline(
        parallel=True,
        executor_passes=[(True, "12 live campaigns.")],
        payloads=[ANSWER],
        critic_raises=boom,
    )
    with pytest.raises(RuntimeError, match="critic model failed"):
        await _run(needed)


async def test_a_pass_with_nothing_held_falls_back_to_the_serial_order(monkeypatch, caplog):
    """The branch holds a result whenever it had a call to make. If the
    resolver finds none anyway (an evicted entry, a resumed invocation), it
    runs the critic itself, after the payload — never a verdict nobody made."""
    from gub_agent.agents import critic as critic_module

    monkeypatch.setattr(critic_module, "_hold", lambda invocation_id, run: None)
    loop, _, _, critic = _pipeline(
        parallel=True, executor_passes=[(True, "12 live campaigns.")], payloads=[ANSWER]
    )

    with caplog.at_level(logging.WARNING, logger="gub_agent.agents.critic"):
        streamed, state, _ = await _run(loop)

    assert len(critic.ran) == 2  # the dropped speculation, then the fallback
    assert state["critic_verdict"] == SUFFICIENT
    assert _authors(streamed)[-2:] == ["critic", "loop_escalator"]
    assert "critic_speculation: nothing held" in caplog.text


# ── the stream the bot reads ─────────────────────────────────────────────────


async def test_the_stream_keeps_its_order_and_its_authors():
    """The bot takes the LAST formatter/format_gate payload and ignores the
    critic's; it restarts its pass on a critic `sufficient: false`. Held and
    released after the join, the critic's events land where they always did:
    after the payload, before the escalator — partial included."""
    loop, *_ = _pipeline(
        parallel=True, executor_passes=[(True, "12 live campaigns.")], payloads=[ANSWER]
    )

    streamed, _, _ = await _run(loop)

    authors = [e.author for e in streamed]
    assert authors == [
        AGENT_NAME,  # call
        AGENT_NAME,  # result
        AGENT_NAME,  # answer
        "formatter",  # the payload
        "critic",  # its streamed partial
        "critic",  # the verdict
        "loop_escalator",
    ]
    payload_events = [e for e in streamed if e.author in ("formatter", "format_gate")]
    assert json.loads(payload_events[-1].content.parts[0].text) == ANSWER
    assert streamed[4].partial and not streamed[5].partial


@pytest.mark.parametrize(
    "executor_passes,payloads,verdicts",
    [
        pytest.param([(True, "12 live.")], [ANSWER], [SUFFICIENT], id="answer-sufficient"),
        pytest.param(
            [(True, "12 live."), (True, "Q3 live.")],
            [ANSWER, ANSWER],
            [INSUFFICIENT, SUFFICIENT],
            id="retry",
        ),
        pytest.param([(True, "No such account.")], [ABSTAIN], [INSUFFICIENT], id="abstain"),
        pytest.param(
            [(False, "Copied."), (True, "12 live.")],
            [ABSTAIN, ANSWER],
            [SUFFICIENT],
            id="from-memory",
        ),
        pytest.param([(False, "NO_COMPANY_RECORDS")], [ABSTAIN], [SUFFICIENT], id="marker"),
    ],
)
async def test_both_wirings_reach_the_same_decisions(executor_passes, payloads, verdicts):
    """Same turn, both trees: same passes, same verdict in state, and the same
    complete events in the same order. What differs is only WHEN the critic ran
    and, on an abstain, that the parallel one ran it for nothing."""
    results = []
    for parallel in (False, True):
        loop, executor, _, _ = _pipeline(
            parallel=parallel,
            executor_passes=executor_passes,
            payloads=payloads,
            verdicts=verdicts,
        )
        streamed, state, _ = await _run(loop)
        results.append(
            (
                executor.runs,
                state["critic_verdict"],
                _authors(streamed),
                [json.dumps(e.actions.state_delta, sort_keys=True) for e in _complete(streamed)],
            )
        )
    assert results[0] == results[1]
