"""
The deep path started beside the router (SPECULATIVE_DEEP) — agents/speculation.py.

Every turn used to wait for the router's one model call before anything else
could start. The speculative root starts the deep path at the same moment on a
fork of the session, holds its events, and keeps them only if the dispatcher's
decision picks the deep path; otherwise it cancels the run and nothing of it
is seen.

Pinned here on a REAL ADK runner (InMemoryRunner, SSE streaming, the real
LoopAgent / ParallelAgent / FormatGate / CriticResolver / FastPath and the
executor's real callback chain), with only the MODELS stubbed — scripted
BaseLlm objects that sleep, answer and remember what they were asked — and two
stub tools. No model is ever called.

- kept: the wall time is the longer of router and deep path, not their sum;
  the stream, the session and the state are the ones SPECULATIVE_DEEP=0
  produces; the executor and the critic are sent the same requests as on
  every other SPECULATIVE_DEEP=1 path, and the SPECULATIVE_DEEP=0 ones minus
  this turn's router event;
- cancelled (smalltalk, workspace_personal, clarify, a fast-path answer): the
  speculative call is cancelled, no event or state of it reaches anything, no
  task outlives the turn, the per-invocation stores are clear, and the branch
  answers exactly as with SPECULATIVE_DEEP=0;
- restarted: a `file_lookup` turn (the one input the deep path reads from the
  router) and a fast path that declines while the speculation still runs each
  get a fresh deep run; a speculation that finished before the router answered
  waits out the fast path and serves its decline;
- a router that raises, writes nothing or writes garbage costs the deep path,
  and the speculation is kept;
- an exception inside the speculative run fails only a turn that keeps it;
- a caller that goes away cancels whatever is running;
- the per-pass round and tool budgets, the sandbox overrides, the tenant label
  and the conversation window reach the fork.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types

from gub_agent import agent as agent_module
from gub_agent import config
from gub_agent.agent import build_deep_agent, build_root
from gub_agent.agents import circuit_breaker as circuit_breaker_module
from gub_agent.agents import critic as critic_module
from gub_agent.agents import dispatcher as dp
from gub_agent.agents import fast_path as fp
from gub_agent.agents import formatter as formatter_module
from gub_agent.agents import round_limiter as round_limiter_module
from gub_agent.agents import router as router_module
from gub_agent.agents import speculation
from gub_agent.agents.answers import not_found_payload
from gub_agent.agents.circuit_breaker import circuit_breaker
from gub_agent.agents.critic import CriticVerdict, EscalateIfSufficient
from gub_agent.agents.evidence_index import evidence_index, record_evidence
from gub_agent.agents.format_gate import FormatGate
from gub_agent.config import AGENT_NAME
from gub_agent.sandbox import SandboxEcho
from gub_agent.schemas import AnswerPayload, RouterDecision
from gub_agent.tenant import tenant_instruction

REPO = Path(__file__).resolve().parents[2]

ROUTER_DELAY = 0.15
EXEC_DELAY = 0.03  # per executor model call
TOOL_DELAY = 0.02
SIDE_DELAY = 0.02  # critic, formatter

ANSWER = {
    "kind": "answer",
    "headline": "chevy is live",
    "blocks": [{"kind": "text", "text": "the chevy account is live."}],
    "citations": ["org_query:a1:status"],
    "facts": [
        {
            "evidence_id": "org_query:a1:status",
            "entity_id": "a1",
            "field": "status",
            "value": "live",
        }
    ],
}
SUFFICIENT = {
    "info_sufficient": True,
    "answer_satisfies": True,
    "sufficient": True,
    "reason": "LLM: covered",
    "feedback": "",
}
DRAFT = "The chevy account is live."
CALL = {"name": "org_query", "args": {"entity": "chevy"}}
REACT = "react"  # the executor's default script: look up once per turn, then answer


def _decision(intent: str = "exploratory", confidence: float = 0.9, **over) -> str:
    return json.dumps({"intent": intent, "confidence": confidence, "language": "en", **over})


# ── the stubs ────────────────────────────────────────────────────────────────


def _part_shape(part: genai_types.Part) -> str:
    if part.function_call:
        return f"call:{part.function_call.name}"
    if part.function_response:
        response = json.dumps(part.function_response.response, sort_keys=True)
        return f"result:{part.function_response.name}:{response}"
    if part.text is not None:
        return ("thought:" if part.thought else "text:") + part.text
    return "?"


def _rendered(llm_request) -> dict:
    """What a model was asked, minus the ids ADK makes up per run."""
    cfg = llm_request.config
    return {
        "contents": [
            (content.role, tuple(_part_shape(p) for p in content.parts or []))
            for content in llm_request.contents or []
        ],
        "tools": sorted(llm_request.tools_dict),
        "system": str(getattr(cfg, "system_instruction", "") or ""),
        "temperature": getattr(cfg, "temperature", None),
    }


class _Scripted(BaseLlm):
    """A model that answers from a script after a delay, and remembers what it
    was asked. A reply is text (streamed as two partial chunks and then the
    whole, when the call streams), a dict (one function call), or an exception
    to raise. The last reply repeats."""

    replies: list = []
    delay: float = 0.0
    delays: list = []  # per call, overriding `delay` for the first len(delays) calls
    until: Any = None  # reply only once this zero-argument callable is true
    requests: list = []
    calls: int = 0
    in_flight: int = 0
    cancelled: int = 0

    @classmethod
    def supported_models(cls):
        return [r".*"]

    async def generate_content_async(self, llm_request, stream=False):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        if reply == REACT:
            reply = _react(llm_request)
        self.calls += 1
        self.requests.append(_rendered(llm_request))
        delay = self.delays[self.calls - 1] if self.calls <= len(self.delays) else self.delay
        self.in_flight += 1
        try:
            for _ in range(1000):
                if self.until is None or self.until():
                    break
                await asyncio.sleep(0.005)
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.in_flight -= 1
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, dict):
            call = genai_types.FunctionCall(name=reply["name"], args=reply.get("args", {}))
            yield LlmResponse(
                content=genai_types.Content(
                    role="model", parts=[genai_types.Part(function_call=call)]
                )
            )
            return
        if stream and len(reply) > 1:
            half = len(reply) // 2
            for chunk in (reply[:half], reply[half:]):
                yield LlmResponse(
                    content=genai_types.Content(role="model", parts=[genai_types.Part(text=chunk)]),
                    partial=True,
                )
        yield LlmResponse(
            content=genai_types.Content(role="model", parts=[genai_types.Part(text=reply)])
        )


def _react(llm_request) -> dict | str:
    """CALL until this turn has a tool result, then DRAFT — so a run that was
    cancelled and restarted asks for what a fresh run asks for."""
    for content in reversed(llm_request.contents or []):
        parts = content.parts or []
        if any(part.function_response for part in parts):
            return DRAFT
        text = parts[0].text if parts and parts[0].text else ""
        if content.role == "user" and text and not text.startswith("For context:"):
            return CALL
    return CALL


def _model(replies: list, delay: float, delays: list | None = None) -> _Scripted:
    return _Scripted(
        model="stub",
        replies=replies,
        delay=delay,
        delays=delays or [],
        requests=[],
        calls=0,
        in_flight=0,
        cancelled=0,
    )


def _speculating() -> bool:
    return any(t.get_name().startswith("speculative-deep") for t in asyncio.all_tasks())


# When the router answers, relative to the speculation — so a test says which
# race it is about instead of hoping a delay decides it.


def mid_call(models: SimpleNamespace) -> bool:
    """The speculative executor call is in flight."""
    return models.executor.in_flight > 0


def ran_to_end(models: SimpleNamespace) -> bool:
    """The speculative run has ended (answered or raised)."""
    return models.executor.calls > 0 and not _speculating()


async def org_query(entity: str, tool_context: ToolContext) -> dict:
    """Query org records for an entity."""
    # temp: state lives for the invocation and is trimmed before it is stored.
    tool_context.state["temp:looked_up"] = entity
    await asyncio.sleep(TOOL_DELAY)
    return {"results": [{"id": "a1", "name": "chevy", "status": "live"}], "total": 1}


async def find_files(query: str) -> dict:
    """Find Drive files by name."""
    return {"hits": [{"id": "f1", "name": "chevy brief"}]}


def _build(
    *,
    speculative: bool,
    router: list,
    executor: list | None = None,
    router_delay: float = ROUTER_DELAY,
    exec_delay: float = EXEC_DELAY,
    exec_delays: list | None = None,
    router_until=None,
) -> SimpleNamespace:
    """The whole root around stub models: [sandbox_echo, router, dispatcher]
    or its speculative shape, built by agent.py's own builders."""
    models = SimpleNamespace(
        router=_model(router, router_delay),
        executor=_model(executor or [REACT], exec_delay, exec_delays),
        critic=_model([json.dumps(SUFFICIENT)], SIDE_DELAY),
        formatter=_model([json.dumps(ANSWER)], SIDE_DELAY),
    )
    if router_until is not None:
        models.router.until = lambda: router_until(models)
    router_agent = LlmAgent(
        name="router",
        model=models.router,
        instruction="Route the question.",
        output_schema=RouterDecision,
        output_key=router_module.ROUTER_STATE_KEY,
        before_model_callback=router_module._router_before_model,
    )
    executor_agent = LlmAgent(
        name=AGENT_NAME,
        model=models.executor,
        # The executor's own tenancy wrapper: a tenant turn's scope rides state.
        instruction=tenant_instruction(lambda _ctx: "Answer from GUB."),
        tools=[org_query, find_files],
        before_agent_callback=agent_module._before_agent,
        before_model_callback=agent_module._before_model,
        before_tool_callback=circuit_breaker,
        after_tool_callback=record_evidence,
    )
    formatter_agent = LlmAgent(
        name="formatter",
        model=models.formatter,
        instruction="Render the answer.",
        output_schema=AnswerPayload,
        output_key="answer_payload",
        include_contents="none",
        before_model_callback=formatter_module._formatter_before_model,
    )
    critic_agent = LlmAgent(
        name="critic",
        model=models.critic,
        instruction=critic_module._critic_instruction,
        output_schema=CriticVerdict,
        output_key="critic_verdict",
        before_model_callback=critic_module._critic_before_model,
    )
    deep = build_deep_agent(
        executor_agent,
        FormatGate(name="format_gate", sub_agents=[formatter_agent]),
        critic_agent,
        EscalateIfSufficient(name="loop_escalator"),
        parallel=True,
    )
    fast = fp.FastPath(name="fast_path")
    dispatcher = dp.Dispatcher(name="dispatcher", sub_agents=[fast, deep])
    root = build_root(
        SandboxEcho(name="sandbox_echo"), router_agent, dispatcher, speculative=speculative
    )
    return SimpleNamespace(root=root, models=models, fast=fast)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """File search and the sandbox off, unless a test turns them on."""
    monkeypatch.setattr(config, "FILE_SEARCH_ENABLED", False)
    monkeypatch.setattr(config, "SANDBOX_ENABLED", False)
    yield


def _use_fast_path(monkeypatch, built: SimpleNamespace, lookup) -> None:
    """Route the dispatcher module's fast path to this tree's, with `lookup`
    as the campaign_status lookup."""
    monkeypatch.setattr(dp, "fast_path", built.fast)
    monkeypatch.setitem(fp._LOOKUPS, "campaign_status", lookup)


class _Session:
    """One conversation on one runner, turn after turn."""

    def __init__(self, root: BaseAgent) -> None:
        self.runner = InMemoryRunner(agent=root, app_name="gub")
        self.id: str | None = None

    async def open(self, state: dict | None = None) -> _Session:
        session = await self.runner.session_service.create_session(
            app_name="gub", user_id="u", state=state or {}
        )
        self.id = session.id
        return self

    def stream(self, question: str):
        message = genai_types.Content(role="user", parts=[genai_types.Part(text=question)])
        return self.runner.run_async(
            user_id="u",
            session_id=self.id,
            new_message=message,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )

    async def turn(self, question: str = "how is chevy doing?") -> SimpleNamespace:
        started = time.monotonic()
        streamed = [event async for event in self.stream(question)]
        wall = time.monotonic() - started
        final = await self.runner.session_service.get_session(
            app_name="gub", user_id="u", session_id=self.id
        )
        return SimpleNamespace(
            streamed=streamed,
            events=final.events,
            state=dict(final.state),
            wall=wall,
            inv=final.events[-1].invocation_id,
        )


async def _one_turn(built: SimpleNamespace, question: str = "how is chevy doing?", state=None):
    session = await _Session(built.root).open(state)
    return await session.turn(question)


def _shape(events: list[Event]) -> list[tuple]:
    """An event list without ids and timestamps: author, partial, parts,
    state keys."""
    return [
        (
            event.author,
            bool(event.partial),
            tuple(_part_shape(p) for p in (event.content.parts if event.content else None) or []),
            tuple(sorted((event.actions.state_delta or {}) if event.actions else {})),
        )
        for event in events
    ]


def _payloads(events: list[Event]) -> list[dict]:
    """What the bot's answer channel reads: every complete formatter /
    format_gate text, last one wins."""
    return [
        json.loads(event.content.parts[0].text)
        for event in events
        if not event.partial and event.author in ("formatter", "format_gate") and event.content
    ]


def _is_router_context(content: tuple) -> bool:
    return any(part.startswith("text:[router] said:") for part in content[1])


def _minus_this_turns_router(requests: list[dict]) -> list[dict]:
    """The requests with ADK's rendering of THIS turn's router event taken
    out — the last one in each request; an earlier turn's stays."""
    trimmed = []
    for request in requests:
        contents = list(request["contents"])
        at = max(i for i, content in enumerate(contents) if _is_router_context(content))
        trimmed.append({**request, "contents": contents[:at] + contents[at + 1 :]})
    return trimmed


def _lines(caplog, prefix: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


def _pending_tasks() -> set[asyncio.Task]:
    return {t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()}


# ── the tree ─────────────────────────────────────────────────────────────────


def test_the_default_root_runs_router_and_dispatcher_inside_the_speculative_dispatch():
    from gub_agent.agent import dispatcher, root_agent
    from gub_agent.agents.router import router_agent

    if not speculation.SPECULATIVE_DEEP:
        pytest.skip("SPECULATIVE_DEEP=0 in this environment — the serial root is pinned below")
    assert [a.name for a in root_agent.sub_agents] == ["sandbox_echo", "speculative_dispatch"]
    wrapper = root_agent.sub_agents[1]
    assert isinstance(wrapper, speculation.SpeculativeDispatch)
    assert wrapper.sub_agents == [router_agent, dispatcher]
    # Every author a turn can emit is still found in the tree.
    for name in ("router", "fast_path", AGENT_NAME, "formatter", "critic", "loop_escalator"):
        assert root_agent.find_agent(name) is not None


_DUMP = """
import json
from gub_agent import agent
from gub_agent.agents.router import router_agent
print(json.dumps({
    "root": [type(a).__name__ + ":" + a.name for a in agent.root_agent.sub_agents],
    "same_objects": [
        agent.root_agent.sub_agents[0] is agent.sandbox_echo,
        agent.root_agent.sub_agents[1] is router_agent,
        agent.root_agent.sub_agents[2] is agent.dispatcher,
    ],
}))
"""


def test_speculative_deep_0_builds_todays_root():
    """The rollback. Read at import, so checked in a fresh interpreter."""
    env = {**os.environ, "SPECULATIVE_DEEP": "0", "PYTHONPATH": str(REPO)}
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
    assert dumped["root"] == [
        "SandboxEcho:sandbox_echo",
        "LlmAgent:router",
        "Dispatcher:dispatcher",
    ]
    assert dumped["same_objects"] == [True, True, True]


def test_the_builder_makes_both_shapes():
    serial = _build(speculative=False, router=[_decision()]).root
    assert [a.name for a in serial.sub_agents] == ["sandbox_echo", "router", "dispatcher"]
    speculative = _build(speculative=True, router=[_decision()]).root
    assert [a.name for a in speculative.sub_agents] == ["sandbox_echo", "speculative_dispatch"]
    assert [a.name for a in speculative.sub_agents[1].sub_agents] == ["router", "dispatcher"]


# ── kept ─────────────────────────────────────────────────────────────────────


async def test_a_kept_turn_takes_the_longer_of_router_and_deep_path_not_their_sum(caplog):
    # ADK's first run in a process pays ~2 s of one-off setup; neither
    # measured turn may be the one that absorbs it.
    await _one_turn(_build(speculative=False, router=[_decision()], router_delay=0, exec_delay=0))
    router_delay, exec_delay = 0.4, 0.15
    serial = _build(
        speculative=False, router=[_decision()], router_delay=router_delay, exec_delay=exec_delay
    )
    speculative = _build(
        speculative=True, router=[_decision()], router_delay=router_delay, exec_delay=exec_delay
    )

    before = await _one_turn(serial)
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
        after = await _one_turn(speculative)

    deep_min = 2 * exec_delay + TOOL_DELAY + SIDE_DELAY
    assert before.wall >= router_delay + deep_min  # the serial order: the sum
    assert after.wall >= max(router_delay, deep_min)  # speculative: the longer one…
    assert after.wall < router_delay + deep_min - 0.15  # …and nowhere near the sum
    assert after.wall < before.wall - 0.25
    [line] = _lines(caplog, "speculation:")
    assert line.startswith("speculation: outcome=kept lead_ms=")
    lead = int(line.split("lead_ms=")[1].split()[0])
    assert lead >= min(router_delay, deep_min) * 1000 * 0.8
    assert f"inv={after.inv} tenant=anomaly" in line
    assert len(_lines(caplog, "dispatcher: intent=exploratory")) == 1


async def test_a_kept_turn_streams_and_stores_what_the_serial_root_does():
    """Same stream (authors, partial deltas then the complete event, payload
    last), same session events, same state — the speculation is invisible."""
    before = await _one_turn(_build(speculative=False, router=[_decision()]))
    after = await _one_turn(_build(speculative=True, router=[_decision()]))

    assert _shape(after.streamed) == _shape(before.streamed)
    assert _shape(after.events) == _shape(before.events)
    assert after.state == before.state
    assert _payloads(after.streamed)[-1] == _payloads(before.streamed)[-1] == ANSWER
    # The executor's prose streamed as deltas, then once complete.
    prose = [(bool(e.partial), e.content.parts[0].text) for e in after.streamed if _is_prose(e)]
    half = len(DRAFT) // 2
    assert prose == [(True, DRAFT[:half]), (True, DRAFT[half:]), (False, DRAFT)]
    # One invocation, in stream order, timestamps rising.
    assert {e.invocation_id for e in after.events[1:]} == {after.inv}
    stamps = [e.timestamp for e in after.events]
    assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)


def _is_prose(event: Event) -> bool:
    parts = event.content.parts if event.content and event.content.parts else []
    return event.author == AGENT_NAME and bool(parts) and parts[0].text is not None


async def test_the_deep_path_is_asked_the_same_on_every_path_and_never_sees_this_turns_router(
    monkeypatch,
):
    """The executor's and the critic's requests: the same whether the run was
    kept or restarted (here after a declining fast path), and the
    SPECULATIVE_DEEP=0 requests with exactly one content fewer — ADK's
    rendering of this turn's router event. The previous turn's router event
    stays, as every earlier event does."""

    async def declined(decision, shim, question):
        return None

    async def two_turns(built, second_decision):
        _use_fast_path(monkeypatch, built, declined)
        built.models.router.replies[:] = [_decision(), second_decision]
        session = await _Session(built.root).open({"context_turn_window": 5})
        await session.turn("hi, what can you do?")
        start = len(built.models.executor.requests), len(built.models.critic.requests)
        await session.turn()
        # The last pass of the turn: a restarted turn's speculation came first.
        return (
            built.models.executor.requests[start[0] :][-2:],
            built.models.critic.requests[start[1] :][-1:],
        )

    fast = _decision("campaign_status", 0.95, entity_surface="chevy")
    serial_exec, serial_critic = await two_turns(
        _build(speculative=False, router=[], exec_delay=0.01), _decision()
    )
    kept_exec, kept_critic = await two_turns(
        _build(speculative=True, router=[], exec_delay=0.01), _decision()
    )
    # Restarted: the second turn's speculative call is still in flight when
    # the router picks the fast path, so the decline gets a fresh deep run.
    restarted_exec, restarted_critic = await two_turns(
        _build(
            speculative=True,
            router=[],
            exec_delay=0.01,
            exec_delays=[0.01, 0.01, 5.0],
            router_until=lambda m: m.router.calls < 2 or m.executor.in_flight > 0,
        ),
        fast,
    )

    assert len(kept_exec) == 2 and len(kept_critic) == 1
    assert kept_exec == restarted_exec and kept_critic == restarted_critic
    assert kept_exec == _minus_this_turns_router(serial_exec)
    assert kept_critic == _minus_this_turns_router(serial_critic)
    # One content apart: the serial request holds both turns' router events,
    # the speculative one only the previous turn's.
    for serial, kept in [(serial_exec[0], kept_exec[0]), (serial_critic[0], kept_critic[0])]:
        assert sum(map(_is_router_context, serial["contents"])) == 2
        assert sum(map(_is_router_context, kept["contents"])) == 1
        assert len(serial["contents"]) == len(kept["contents"]) + 1


# ── cancelled ────────────────────────────────────────────────────────────────


CANCELLED = [
    pytest.param(_decision("smalltalk", 0.97), "smalltalk", id="smalltalk"),
    pytest.param(_decision("workspace_personal", 0.95), "abstain", id="workspace-personal"),
    pytest.param(
        _decision("campaign_facts", 0.5, entity_surface="Silverado"), "clarify", id="clarify"
    ),
]


@pytest.mark.parametrize("decision,branch", CANCELLED)
async def test_a_no_data_branch_cancels_the_speculation_and_leaks_nothing(decision, branch, caplog):
    serial = _build(speculative=False, router=[decision], exec_delay=5.0)
    speculative = _build(speculative=True, router=[decision], exec_delay=5.0, router_until=mid_call)

    before = await _one_turn(serial)
    pending = _pending_tasks()
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
        after = await _one_turn(speculative)

    # The speculative executor call was made, and cancelled mid-flight.
    assert speculative.models.executor.calls == 1
    assert speculative.models.executor.cancelled == 1
    assert _pending_tasks() <= pending  # nothing outlives the turn
    # Not one event or key of it anywhere.
    assert _shape(after.streamed) == _shape(before.streamed)
    assert _shape(after.events) == _shape(before.events)
    assert [e.author for e in after.events] == ["user", "router", "format_gate"]
    assert after.state == before.state
    assert "critic_verdict" not in after.state
    assert _payloads(after.streamed) == _payloads(before.streamed)
    # The per-invocation stores it wrote into are clear again.
    assert after.inv not in round_limiter_module._ROUNDS
    assert after.inv not in circuit_breaker_module._BUDGETS
    assert evidence_index(after.inv) == {}
    [line] = _lines(caplog, "speculation:")
    assert line.startswith(f"speculation: outcome=cancelled:{branch} lead_ms=")
    assert "finished=0 error=-" in line


class _Probe(BaseAgent):
    """Runs after the root and records the session AS THE RUNNER HOLDS IT —
    the in-memory object every agent of the turn reads, not the stored copy."""

    seen: list = []

    async def _run_async_impl(self, ctx):
        self.seen.append((dict(ctx.session.state), [e.author for e in ctx.session.events]))
        return
        yield


async def test_a_finished_speculation_leaves_nothing_in_the_live_session():
    """The speculation ran to its end before the router answered — payload,
    verdict and all — and the turn was a greeting. The session the rest of
    the invocation reads must hold the greeting and nothing of the run."""
    built = _build(speculative=True, router=[_decision("smalltalk", 0.97)], router_until=ran_to_end)
    probe = _Probe(name="probe", seen=[])
    outer = SequentialAgent(name="outer", sub_agents=[built.root, probe])

    await _one_turn(SimpleNamespace(root=outer))

    assert built.models.formatter.calls == 1 and built.models.critic.calls == 1  # it finished
    [(state, authors)] = probe.seen
    assert authors == ["user", "router", "format_gate"]
    assert set(state) == {"router_decision", "answer_payload"}
    assert state["router_decision"]["intent"] == "smalltalk"
    assert state["answer_payload"]["headline"] != ANSWER["headline"]


async def test_a_kept_speculation_leaves_the_live_session_as_the_serial_root_does():
    """The session the runner holds in memory — what any later agent of the
    invocation reads — including temp: state, which is applied there and
    trimmed from what is stored."""
    seen = []
    for speculative in (False, True):
        built = _build(speculative=speculative, router=[_decision()])
        probe = _Probe(name="probe", seen=[])
        outer = SequentialAgent(name="outer", sub_agents=[built.root, probe])
        after = await _one_turn(SimpleNamespace(root=outer))
        seen.append((probe.seen, after))

    ([(before_state, before_authors)], before), ([(after_state, after_authors)], after) = seen
    assert after_authors == before_authors
    assert after_state == before_state
    assert after_state["temp:looked_up"] == "chevy"
    assert "temp:looked_up" not in after.state  # the stored session never has it


async def test_a_fast_path_answer_cancels_the_speculation(monkeypatch, caplog):
    decision = _decision("campaign_status", 0.95, entity_surface="Silverado")
    seen = []

    async def answered(decision, shim, question):
        seen.append(None)
        return fp.Lookup(payload=not_found_payload("en", "Silverado"), tool="find")

    results = []
    for speculative in (False, True):
        until = mid_call if speculative else None
        built = _build(
            speculative=speculative, router=[decision], exec_delay=5.0, router_until=until
        )
        _use_fast_path(monkeypatch, built, answered)
        with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
            results.append((built, await _one_turn(built)))

    (_, before), (built, after) = results
    assert built.models.executor.cancelled == 1
    assert _shape(after.streamed) == _shape(before.streamed)
    assert _shape(after.events) == _shape(before.events)
    assert after.state == before.state
    assert [e.author for e in after.streamed if not e.partial][-1] == "format_gate"
    assert any(e.author == "fast_path" and e.partial for e in after.streamed)  # tool activity
    assert _lines(caplog, "speculation:")[-1].startswith("speculation: outcome=cancelled:fast ")


# ── restarted ────────────────────────────────────────────────────────────────


async def test_a_file_lookup_turn_restarts_the_deep_path_with_the_tool_it_earns(
    monkeypatch, caplog
):
    """The one thing the deep path reads from the router: `find_files` is
    offered on a file_lookup turn. The speculation ran without it, so it is
    not the run the dispatcher would have made."""
    monkeypatch.setattr(config, "FILE_SEARCH_ENABLED", True)
    decision = _decision("file_lookup", 0.9)
    before_built = _build(speculative=False, router=[decision])
    before = await _one_turn(before_built)
    built = _build(speculative=True, router=[decision], exec_delays=[5.0], router_until=mid_call)
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
        after = await _one_turn(built)

    speculative_call, *restarted_calls = built.models.executor.requests
    assert "find_files" not in speculative_call["tools"]  # what FALLBACK is offered
    assert built.models.executor.cancelled == 1
    assert all("find_files" in call["tools"] for call in restarted_calls)
    assert [c["tools"] for c in restarted_calls] == [
        c["tools"] for c in before_built.models.executor.requests
    ]
    assert _shape(after.streamed) == _shape(before.streamed)
    assert _shape(after.events) == _shape(before.events)
    assert after.state == before.state
    [line] = _lines(caplog, "speculation:")
    assert line.startswith("speculation: outcome=restarted:file_lookup ")


async def test_a_file_lookup_turn_keeps_the_speculation_while_file_search_is_off(caplog):
    """Flag off, the gate hides the tool on every intent — nothing differs."""
    built = _build(speculative=True, router=[_decision("file_lookup", 0.9)])
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
        await _one_turn(built)
    assert built.models.executor.cancelled == 0
    assert _lines(caplog, "speculation:")[0].startswith("speculation: outcome=kept ")


async def test_the_speculation_never_reads_the_previous_turns_decision(monkeypatch, caplog):
    """Session state keeps `router_decision` across turns. A fork that
    inherited it would run the speculation on the LAST question's decision —
    here a file_lookup, offering `find_files` to a turn that does not earn it."""
    monkeypatch.setattr(config, "FILE_SEARCH_ENABLED", True)
    router = [_decision("file_lookup", 0.9), _decision()]
    runs = []
    for speculative in (False, True):
        built = _build(speculative=speculative, router=router, exec_delay=0.01)
        session = await _Session(built.root).open()
        await session.turn("find the chevy brief")
        start = len(built.models.executor.requests)
        with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
            await session.turn()
        runs.append(built.models.executor.requests[start:])

    serial, kept = runs
    assert [r["tools"] for r in kept] == [r["tools"] for r in serial] == [["org_query"]] * 2
    assert _lines(caplog, "speculation:")[-1].startswith("speculation: outcome=kept ")


async def test_a_declining_fast_path_runs_alone_and_then_a_fresh_deep_path(monkeypatch, caplog):
    """The fast path resets and fills the invocation's evidence index, so it
    never runs beside a speculation: the speculation is stopped first, and a
    decline gets a deep run of its own — the one SPECULATIVE_DEEP=0 makes."""
    decision = _decision("campaign_status", 0.95, entity_surface="Silverado")
    in_flight_at_lookup = []

    def declines(built):
        async def lookup(decision, shim, question):
            in_flight_at_lookup.append(built.models.executor.in_flight)
            return None  # ambiguous entity: not the fast path's business

        return lookup

    results = []
    for speculative in (False, True):
        if speculative:  # the speculative call is still in flight at the decision
            built = _build(
                speculative=True, router=[decision], exec_delays=[5.0], router_until=mid_call
            )
        else:
            built = _build(speculative=False, router=[decision])
        _use_fast_path(monkeypatch, built, declines(built))
        with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
            results.append((built, await _one_turn(built)))

    (_, before), (built, after) = results
    assert in_flight_at_lookup == [0, 0]  # nothing ran beside the lookup
    assert built.models.executor.cancelled == 1
    assert built.models.executor.calls == 3  # the cancelled one, then the deep run's two
    assert _shape(after.streamed) == _shape(before.streamed)
    assert _shape(after.events) == _shape(before.events)
    assert after.state == before.state
    assert _lines(caplog, "speculation:")[-1].startswith(
        "speculation: outcome=restarted:fast_declined "
    )
    assert _lines(caplog, "dispatcher: fast path declined")


async def test_a_speculation_that_finished_before_the_router_serves_a_fast_path_decline(
    monkeypatch, caplog
):
    """The router took longer than the whole deep path. A finished run writes
    nothing any more, so it waits out the fast path instead of being thrown
    away, and a decline gets it instead of a second deep run."""
    decision = _decision("campaign_status", 0.95, entity_surface="Silverado")

    async def declines(decision, shim, question):
        return None

    results = []
    for speculative in (False, True):
        until = ran_to_end if speculative else None
        built = _build(speculative=speculative, router=[decision], router_until=until)
        _use_fast_path(monkeypatch, built, declines)
        with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
            results.append((built, await _one_turn(built)))

    (serial, before), (built, after) = results
    assert built.models.executor.calls == serial.models.executor.calls == 2  # one deep run
    assert built.models.executor.cancelled == 0
    assert _shape(after.streamed) == _shape(before.streamed)
    assert _shape(after.events) == _shape(before.events)
    assert after.state == before.state
    [line] = _lines(caplog, "speculation: outcome=kept")
    assert "finished=1 error=-" in line


async def test_a_fast_path_answer_after_a_finished_speculation_leaves_nothing_of_it(monkeypatch):
    decision = _decision("campaign_status", 0.95, entity_surface="Silverado")

    async def answered(decision, shim, question):
        return fp.Lookup(payload=not_found_payload("en", "Silverado"), tool="find")

    built = _build(speculative=True, router=[decision], router_until=ran_to_end)
    _use_fast_path(monkeypatch, built, answered)
    after = await _one_turn(built)

    assert built.models.formatter.calls == 1  # the speculation ran to its end…
    assert [e.author for e in after.events] == ["user", "router", "format_gate"]
    assert "critic_verdict" not in after.state  # …and none of it was kept
    assert after.inv not in round_limiter_module._ROUNDS


# ── a broken router ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "router_reply",
    [
        pytest.param(RuntimeError("router model failed"), id="raises"),
        pytest.param("not json at all", id="unparseable"),
        pytest.param(json.dumps({"intent": "vibes", "confidence": 0.99}), id="schema-invalid"),
    ],
)
async def test_a_broken_router_costs_the_deep_path_and_the_speculation_is_kept(
    router_reply, caplog
):
    built = _build(speculative=True, router=[router_reply])
    with caplog.at_level(logging.INFO):
        after = await _one_turn(built)

    assert built.models.executor.cancelled == 0
    assert _payloads(after.streamed)[-1]["headline"] == ANSWER["headline"]
    assert after.state["critic_verdict"]["sufficient"] is True
    assert _lines(caplog, "dispatcher: intent=exploratory confidence=0.00 branch=deep")
    assert _lines(caplog, "speculation:")[0].startswith("speculation: outcome=kept ")


async def test_a_router_that_writes_nothing_does_not_route_on_the_last_turns_decision(caplog):
    """State outlives the turn. A router that answered with no text leaves the
    PREVIOUS turn's decision there (here: smalltalk); this turn must not be
    answered as a greeting — it is an unreadable decision, so the deep path."""
    built = _build(speculative=True, router=[""])
    stale = json.loads(_decision("smalltalk", 0.97))
    with caplog.at_level(logging.INFO):
        after = await _one_turn(built, state={"router_decision": stale})

    assert _payloads(after.streamed)[-1]["headline"] == ANSWER["headline"]
    assert _lines(caplog, "dispatcher: intent=exploratory confidence=0.00 branch=deep")
    assert _lines(caplog, "speculation:")[0].startswith("speculation: outcome=kept ")


# ── exceptions and disconnects ───────────────────────────────────────────────


async def test_a_kept_speculation_that_raised_fails_the_turn_as_the_serial_root_does(caplog):
    boom = RuntimeError("executor model failed")
    serial = _build(speculative=False, router=[_decision()], executor=[boom])
    speculative = _build(
        speculative=True, router=[_decision()], executor=[boom], router_until=ran_to_end
    )

    with pytest.raises(RuntimeError, match="executor model failed"):
        await _one_turn(serial)
    pending = _pending_tasks()
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
        with pytest.raises(RuntimeError, match="executor model failed"):
            await _one_turn(speculative)

    assert _pending_tasks() <= pending
    [line] = _lines(caplog, "speculation:")
    assert line.startswith("speculation: outcome=kept ")
    assert "finished=1 error=RuntimeError" in line


async def test_a_cancelled_speculation_that_raised_costs_nothing(caplog):
    built = _build(
        speculative=True,
        router=[_decision("smalltalk", 0.97)],
        executor=[RuntimeError("executor model failed")],
        router_until=ran_to_end,
    )
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
        after = await _one_turn(built)

    assert [e.author for e in after.events] == ["user", "router", "format_gate"]
    assert _payloads(after.streamed)[-1]["kind"] == "answer"  # the greeting
    [line] = _lines(caplog, "speculation:")
    assert "outcome=cancelled:smalltalk" in line and "error=RuntimeError" in line


async def test_a_caller_that_goes_away_before_the_decision_cancels_the_speculation(caplog):
    built = _build(speculative=True, router=[_decision()], router_delay=5.0, exec_delay=5.0)
    session = await _Session(built.root).open()
    pending = _pending_tasks()

    stream = session.stream("how is chevy doing?")
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
        # Nothing is streamed before the router answers. Wait until the
        # speculative executor call is in flight, then cancel the read, as a
        # bot deadline does, and close the stream.
        read = asyncio.ensure_future(anext(stream))
        for _ in range(500):
            if built.models.executor.in_flight:
                break
            await asyncio.sleep(0.01)
        read.cancel()
        with pytest.raises(asyncio.CancelledError):
            await read
        await stream.aclose()

    assert built.models.executor.calls == 1
    assert built.models.executor.cancelled == 1
    assert built.models.router.cancelled == 1
    assert _pending_tasks() <= pending
    [line] = _lines(caplog, "speculation:")
    assert line.startswith("speculation: outcome=cancelled:aborted ")


async def test_a_caller_that_goes_away_mid_relay_cancels_the_rest_of_the_run():
    built = _build(speculative=True, router=[_decision()], router_delay=0.05, exec_delay=0.3)
    session = await _Session(built.root).open()
    pending = _pending_tasks()

    stream = session.stream("how is chevy doing?")
    async for event in stream:
        if event.author == AGENT_NAME:  # the relay has started
            break
    await stream.aclose()

    assert _pending_tasks() <= pending
    # The executor was in its second call (after the tool) or about to be.
    assert built.models.formatter.calls == 0
    assert built.models.critic.calls == 0


# ── the per-invocation guards, and what the fork carries ─────────────────────


async def test_the_round_and_tool_budgets_hold_on_a_kept_speculation():
    """A repeated call is refused by the circuit breaker and the fourth round
    is sent without tools by the round limiter — the same way on both roots:
    the budgets are the invocation's, and only one deep run ever holds them."""
    script = [CALL, CALL, {"name": "org_query", "args": {"entity": "gmc"}}, DRAFT]
    runs = []
    for speculative in (False, True):
        built = _build(
            speculative=speculative, router=[_decision()], executor=script, exec_delay=0.02
        )
        turn = await _one_turn(built)
        runs.append((built.models.executor.requests, turn))

    (serial_requests, before), (kept_requests, after) = runs
    assert kept_requests == _minus_this_turns_router(serial_requests)
    assert [r["tools"] for r in kept_requests] == [["org_query"]] * 3 + [[]]
    refused = [
        part
        for e in after.events
        for part in (e.content.parts if e.content else [])
        if part.function_response and part.function_response.response.get("error")
    ]
    assert (
        len(refused) == 1 and "already called" in refused[0].function_response.response["message"]
    )
    assert _shape(after.events) == _shape(before.events)


async def test_sandbox_overrides_and_the_tenant_reach_the_fork(monkeypatch, caplog):
    monkeypatch.setattr(config, "SANDBOX_ENABLED", True)
    state = {"sandbox": {"temperature": 0.3}, "tenant": "chevy"}
    built = _build(speculative=True, router=[_decision()])
    with caplog.at_level(logging.INFO):
        after = await _one_turn(built, state=state)

    assert {r["temperature"] for r in built.models.executor.requests} == {0.3}
    assert {r["temperature"] for r in built.models.router.requests} == {0.3}
    assert all("## Tenant surface" in r["system"] for r in built.models.executor.requests)
    assert after.events[1].author == "sandbox_echo"  # provenance, before any work
    [line] = _lines(caplog, "speculation:")
    assert line.startswith("speculation: outcome=kept") and line.endswith("tenant=chevy")
    assert all(line.endswith("tenant=chevy") for line in _lines(caplog, "turn_window: agent="))


async def test_the_conversation_window_cuts_the_fork_where_it_cuts_the_session():
    """state["context_turn_window"] is the session's: the speculative executor
    is windowed to the same turns as the serial one."""

    async def third_turn(built):
        session = await _Session(built.root).open({"context_turn_window": 1})
        await session.turn("first question")
        await session.turn("second question")
        start = len(built.models.executor.requests)
        await session.turn("third question")
        return built.models.executor.requests[start:]

    router = [_decision()] * 3
    serial = await third_turn(_build(speculative=False, router=router, exec_delay=0.01))
    kept = await third_turn(_build(speculative=True, router=router, exec_delay=0.01))

    assert kept == _minus_this_turns_router(serial)
    texts = [p for c in kept[0]["contents"] for p in c[1]]
    assert "text:third question" in texts
    assert not any("first question" in p or "second question" in p for p in texts)


# ── the pieces ───────────────────────────────────────────────────────────────


def test_the_keep_rule_is_the_tool_gates_rule(monkeypatch):
    """`_stale_reason` restates one condition of `tool_gate`; drive both over
    every intent with the flag both ways, so the two cannot drift apart."""
    from typing import get_args

    from gub_agent.agents.tool_gate import GATED_TOOL, tool_gate
    from gub_agent.schemas.router import FALLBACK_DECISION, Intent

    def offered(decision: RouterDecision) -> bool:
        ctx = SimpleNamespace(
            invocation_id="inv",
            session=SimpleNamespace(state={"router_decision": decision.model_dump()}, events=[]),
        )
        decl = genai_types.FunctionDeclaration(name=GATED_TOOL)
        request = SimpleNamespace(
            config=genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(function_declarations=[decl])]
            ),
            tools_dict={GATED_TOOL: object()},
        )
        tool_gate(ctx, request)
        return GATED_TOOL in request.tools_dict

    for enabled in (False, True):
        monkeypatch.setattr(config, "FILE_SEARCH_ENABLED", enabled)
        for intent in get_args(Intent):
            decision = RouterDecision(intent=intent, confidence=0.9)
            stale = offered(decision) != offered(FALLBACK_DECISION)
            assert (speculation._stale_reason(decision) is not None) is stale, (enabled, intent)


def test_the_log_lines_keep_their_bytes(caplog):
    ctx = SimpleNamespace(invocation_id="e-1", session=SimpleNamespace(state={"tenant": "chevy"}))
    decision = RouterDecision(intent="assessment", confidence=0.8123)
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
        dp.log_decision(ctx, decision, dp.DEEP)
        dp.log_speculation(ctx, "off")
        dp.log_speculation(ctx, "kept", lead_ms=5812, buffered=7, finished=False)
        dp.log_fast_path_declined(ctx)
    assert [r.getMessage() for r in caplog.records] == [
        "dispatcher: intent=assessment confidence=0.81 branch=deep (inv=e-1) tenant=chevy",
        "speculation: outcome=off lead_ms=- buffered=0 finished=0 error=- inv=e-1 tenant=chevy",
        "speculation: outcome=kept lead_ms=5812 buffered=7 finished=0 error=- inv=e-1 tenant=chevy",
        "dispatcher: fast path declined (inv=e-1) — deep path",
    ]


async def test_the_serial_root_logs_off(caplog):
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
        after = await _one_turn(_build(speculative=False, router=[_decision()], exec_delay=0.01))
    assert _lines(caplog, "speculation:") == [
        f"speculation: outcome=off lead_ms=- buffered=0 finished=0 error=- inv={after.inv} "
        "tenant=anomaly"
    ]


# ── how the deployed engine drives it ────────────────────────────────────────


def _sync_stream(runner: InMemoryRunner, session_id: str, question: str = "how is chevy doing?"):
    """AdkApp.stream_query's path on Agent Engine: the SYNC Runner.run, which
    runs run_async on a thread of its own, in an event loop of its own, and
    cancels that invocation's task when the caller closes the stream early."""
    message = genai_types.Content(role="user", parts=[genai_types.Part(text=question)])
    return runner.run(
        user_id="u",
        session_id=session_id,
        new_message=message,
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    )


async def test_the_agent_engine_path_keeps_and_streams_the_same(caplog):
    results = []
    for speculative in (False, True):
        built = _build(speculative=speculative, router=[_decision()], exec_delay=0.02)
        runner = InMemoryRunner(agent=built.root, app_name="gub")
        session = await runner.session_service.create_session(app_name="gub", user_id="u")
        with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
            streamed = await asyncio.to_thread(lambda: list(_sync_stream(runner, session.id)))
        results.append(streamed)

    before, after = results
    assert _shape(after) == _shape(before)
    assert _lines(caplog, "speculation:")[-1].startswith("speculation: outcome=kept ")


async def test_the_agent_engine_path_cancels_the_run_when_the_caller_closes_early():
    built = _build(speculative=True, router=[_decision()], router_delay=0.05, exec_delay=5.0)
    runner = InMemoryRunner(agent=built.root, app_name="gub")
    session = await runner.session_service.create_session(app_name="gub", user_id="u")

    def read_until_the_decision_then_close() -> float:
        stream = _sync_stream(runner, session.id)
        for event in stream:
            if event.author == "router" and not event.partial:
                break
        started = time.monotonic()
        stream.close()  # cancels the invocation's task and joins its thread
        return time.monotonic() - started

    closed_in = await asyncio.to_thread(read_until_the_decision_then_close)

    assert closed_in < 1.0  # not the 5 s the executor call would have taken
    assert built.models.executor.calls == 1 and built.models.executor.cancelled == 1
    final = await runner.session_service.get_session(
        app_name="gub", user_id="u", session_id=session.id
    )
    assert [e.author for e in final.events] == ["user", "router"]


async def test_a_resumable_invocation_is_run_serially(caplog):
    """Nothing deployed is resumable; a resumable app gets today's order."""
    from google.adk.apps import App, ResumabilityConfig
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    built = _build(speculative=True, router=[_decision()], exec_delay=0.01)
    runner = Runner(
        app=App(
            name="gub",
            root_agent=built.root,
            resumability_config=ResumabilityConfig(is_resumable=True),
        ),
        session_service=InMemorySessionService(),
    )
    session = await runner.session_service.create_session(app_name="gub", user_id="u")
    message = genai_types.Content(role="user", parts=[genai_types.Part(text="how is chevy?")])
    with caplog.at_level(logging.INFO, logger="gub_agent.agents.dispatcher"):
        streamed = [
            e
            async for e in runner.run_async(user_id="u", session_id=session.id, new_message=message)
        ]

    assert _payloads(streamed)[-1] == ANSWER
    assert _lines(caplog, "speculation:")[0].startswith("speculation: outcome=off ")
