"""
SYNTHESIS_ROUND_THINKING_LEVEL — the thinking level of the executor's answer
round (latency-05 wave 3).

The round past the ReAct budget (agents/round_limiter.py) writes the answer, and
its first chunk waits for all of its thinking. With the switch at LOW (env, or
the sandbox override `synthesis_round_thinking_level`), that one round asks for
LOW; every other round, pass, invocation and agent is untouched.

Driven through the executor's REAL before_model chain (`agent._before_model`) on
a real ADK session and a real CallbackContext, with the planner's own
ThinkingConfig object in the request as ADK's planner hands it over. Pinned:

- switch AUTO: rounds 1..5 carry the planner's object and write no line;
- switch LOW: rounds 1..3 untouched, rounds 4 and 5 LOW with
  include_thoughts = EMIT_THINKING, a FRESH config (the planner's object keeps
  MEDIUM for the next round, pass and invocation), thinking_config the only
  field written;
- the `synthesis_thinking:` line: exact fields, `tenant=` last, `was=` for
  MEDIUM / HIGH / DYNAMIC / no config, `pass=2` on a retry pass;
- it never raises thinking: LOW, MINIMAL, OFF and a positive budget are left
  alone and write no line;
- sandbox: an explicit override wins over the env either way (AUTO included),
  None follows the env, and with the sandbox off the override is ignored;
- model guards: Claude and a model outside SANDBOX_THINKING_LEVEL_MODELS are
  untouched, a missing model id means the default model, a missing config
  does not raise;
- ordering: the sandbox's executor thinking_level applies to rounds 1..3 and
  round 4 without the switch; with it, round 4 only is LOW;
- provenance and validation: resolved_config (AUTO for a model the switch
  skips, whatever the env says), overridden_keys, a named level refused with a
  model that takes none, a bad value refused;
- the sandbox reader is touched on the answer round only, and "takes a named
  level" is one function for the switch and for the provenance;
- the env parse (only LOW after strip + upper is LOW) in a clean interpreter,
  the two vocabularies kept together, both deploy env files stating AUTO;
- the critic, formatter and router chains never write that line.

Every "nothing happens" test ends with a positive control (the same chain with
the switch on does act), so a probe that cannot fire cannot pass.

The tests reach the new switch only through names that exist on the parent
(`config` attributes set with raising=False, `agent._before_model`,
`read_overrides`) or through getattr, so on the parent they fail on an
assertion, not on an import. No model calls: nothing here sends a request.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import typing
import uuid
from types import SimpleNamespace

import pydantic
import pytest
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event, EventActions
from google.adk.flows.llm_flows.contents import _get_contents
from google.adk.models.llm_request import LlmRequest
from google.genai import types as genai_types

from gub_agent import agent as agent_module
from gub_agent import config, sandbox
from gub_agent.agents import context_pruning, round_limiter
from gub_agent.agents.critic import _critic_before_model
from gub_agent.agents.formatter import _formatter_before_model
from gub_agent.agents.round_limiter import reset_rounds
from gub_agent.agents.router import _router_before_model
from gub_agent.config import AGENT_NAME, GEMINI_MODEL
from gub_agent.sandbox import SandboxOverrides, read_overrides, resolved_config
from tests.helpers import invocation_ctx

from .test_model_hedge import REPO, _stated

FLAG = "SYNTHESIS_ROUND_THINKING_LEVEL"
KEY = "synthesis_round_thinking_level"
LINE = re.compile(
    r"^synthesis_thinking: level=(\S+) was=(\S+) round=(\d+) pass=(\d+) inv=(\S+) tenant=(\S+)$"
)
CLAUDE = "claude-sonnet-4-5"
OTHER_GEMINI = "gemini-2.5-pro"
LOGGER = "gub_agent.agents.round_limiter"
PLANNER = object()  # "the request carries the planner's own ThinkingConfig"


# ── fixtures and helpers ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _pinned(monkeypatch):
    """What a dev's env must not change: a window wide enough for the history
    under test, the sandbox off, one thinking-level model, a known allowlist,
    thought summaries off, and the switch at AUTO."""
    monkeypatch.setattr(context_pruning, "CONTEXT_TURN_WINDOW", 20)
    monkeypatch.setattr(config, "SANDBOX_ENABLED", False)
    monkeypatch.setattr(config, "SANDBOX_THINKING_LEVEL_MODELS", (GEMINI_MODEL,))
    monkeypatch.setattr(config, "SANDBOX_MODEL_ALLOWLIST", (GEMINI_MODEL, OTHER_GEMINI, CLAUDE))
    monkeypatch.setattr(config, "EMIT_THINKING", False)
    monkeypatch.setattr(config, FLAG, "AUTO", raising=False)


@pytest.fixture
def flag(monkeypatch):
    """Set the env flag as a module attribute. raising=False: on the parent the
    attribute does not exist, and the test must still reach its assertion."""

    def _set(value: str) -> None:
        monkeypatch.setattr(config, FLAG, value, raising=False)

    return _set


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER)
    return caplog


@pytest.fixture
def sandbox_on(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_ENABLED", True)


def _lines(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.getMessage().startswith("synthesis_thinking:")
    ]


def _inv() -> str:
    """A fresh invocation id per call: the round counter is process-wide."""
    return f"w3t2-{uuid.uuid4().hex[:8]}"


def _planner_cfg() -> genai_types.ThinkingConfig:
    """The executor planner's own long-lived ThinkingConfig."""
    return agent_module.executor_agent.planner.thinking_config


def _level(request) -> str:
    """What a request's thinking asks for, as one word (the line's vocabulary)."""
    thinking = request.config.thinking_config
    if thinking is None:
        return "-"
    if thinking.thinking_level is not None:
        return thinking.thinking_level.name
    return {0: "OFF", -1: "DYNAMIC"}.get(
        thinking.thinking_budget, f"budget={thinking.thinking_budget}"
    )


def _text(author: str, role: str, text: str, inv: str) -> Event:
    return Event(
        invocation_id=inv,
        author=author,
        content=genai_types.Content(role=role, parts=[genai_types.Part(text=text)]),
    )


def _verdict(inv: str) -> Event:
    """The critic gate's verdict event: a pass was judged and sent back."""
    return Event(
        invocation_id=inv,
        author="critic",
        content=genai_types.Content(
            role="model", parts=[genai_types.Part(text='{"sufficient": false}')]
        ),
        actions=EventActions(state_delta={"critic_verdict": {"sufficient": False}}),
    )


def _declared() -> genai_types.Tool:
    return genai_types.Tool(
        function_declarations=[
            genai_types.FunctionDeclaration(name="get_account_overview", description="d"),
            genai_types.FunctionDeclaration(name="org_query", description="d"),
        ]
    )


async def _session(
    inv: str, *, state: dict | None = None, judged: bool = False
) -> tuple[CallbackContext, list[Event]]:
    """A real session holding this turn's question (and, with `judged`, a first
    pass's answer and the critic's verdict on it), and the CallbackContext."""
    events = [_text("user", "user", "whats new?", inv)]
    if judged:
        events += [_text(AGENT_NAME, "model", "Earlier answer.", inv), _verdict(inv)]
    return CallbackContext(await invocation_ctx(state=state, invocation_id=inv, events=events)), (
        events
    )


def _request(events: list[Event], *, model: str | None = GEMINI_MODEL, thinking=PLANNER):
    """The executor's request as ADK builds it each round: the planner's own
    ThinkingConfig object in `config.thinking_config` unless `thinking` says
    otherwise."""
    return LlmRequest(
        model=model,
        contents=_get_contents(None, events, AGENT_NAME),
        config=genai_types.GenerateContentConfig(
            system_instruction="Answer from the data.",
            labels={"adk_agent_name": AGENT_NAME},
            tools=[_declared()],
            thinking_config=_planner_cfg() if thinking is PLANNER else thinking,
        ),
    )


def _run(cb, events, rounds: int, **kwargs) -> list[LlmRequest]:
    """`rounds` model calls of one invocation through the executor's chain, a
    fresh request each (as ADK builds one per round)."""
    requests = []
    for _ in range(rounds):
        request = _request(events, **kwargs)
        agent_module._before_model(cb, request)
        requests.append(request)
    return requests


async def _control(flag, logs) -> None:
    """The positive control: with the switch at LOW the same chain acts on its
    round 4. A "nothing happens" assertion is only worth anything next to it."""
    flag("LOW")
    logs.clear()
    cb, events = await _session(_inv())
    requests = _run(cb, events, 4)
    assert [_level(r) for r in requests] == ["MEDIUM", "MEDIUM", "MEDIUM", "LOW"]
    assert len(_lines(logs)) == 1


def _synthesis(callback_context, llm_request):
    """synthesis_round_thinking as a module attribute, so a missing function is
    an assertion failure and not an import error."""
    fn = getattr(round_limiter, "synthesis_round_thinking", None)
    assert fn is not None, "round_limiter.synthesis_round_thinking does not exist"
    return fn(callback_context, llm_request)


# ── the switch at AUTO changes nothing ───────────────────────────────────────


async def test_the_switch_at_auto_leaves_every_round_on_the_planners_config(flag, logs):
    flag("AUTO")
    inv = _inv()
    cb, events = await _session(inv)
    planner = _planner_cfg()
    before = planner.model_dump()

    requests = _run(cb, events, 5)

    for rnd, request in enumerate(requests, start=1):
        assert request.config.thinking_config is planner, rnd
        assert _level(request) == "MEDIUM", rnd
    assert planner.model_dump() == before
    assert _lines(logs) == []
    await _control(flag, logs)


# ── the switch at LOW: which rounds, what is written ─────────────────────────


@pytest.mark.parametrize("emit", [False, True])
async def test_low_lowers_the_answer_rounds_only_and_never_touches_the_planners_config(
    flag, logs, monkeypatch, emit
):
    """The planner was built with include_thoughts False; with EMIT_THINKING on
    the new config must carry True, so the value comes from the call and not
    from the planner's object."""
    monkeypatch.setattr(config, "EMIT_THINKING", emit)
    flag("LOW")
    inv = _inv()
    cb, events = await _session(inv)
    planner = _planner_cfg()
    snapshot = planner.model_dump()

    requests = _run(cb, events, 5)

    for rnd in (1, 2, 3):
        request = requests[rnd - 1]
        assert request.config.thinking_config is planner, rnd
        assert _level(request) == "MEDIUM", rnd
        assert request.config.tools, rnd  # the tool rounds keep their tools
    for rnd in (4, 5):
        request = requests[rnd - 1]
        assert _level(request) == "LOW", rnd
        assert request.config.thinking_config is not planner, rnd
        assert request.config.thinking_config.include_thoughts is emit, rnd
        assert request.config.thinking_config.thinking_budget is None, rnd
        assert not request.config.tools, rnd  # round_limit still last writer of tools
    # The planner's object is what it was: later calls start from MEDIUM again.
    assert planner.model_dump() == snapshot
    assert planner.thinking_level.name == "MEDIUM"
    assert planner.include_thoughts is False

    # The next invocation's round 1 is the planner's own again.
    cb2, events2 = await _session(_inv())
    [first] = _run(cb2, events2, 1)
    assert first.config.thinking_config is planner
    assert _level(first) == "MEDIUM"


async def test_a_new_pass_starts_from_the_planners_level_again(flag, logs):
    flag("LOW")
    inv = _inv()
    cb, events = await _session(inv)
    planner = _planner_cfg()

    first_pass = _run(cb, events, 4)
    assert [_level(r) for r in first_pass] == ["MEDIUM", "MEDIUM", "MEDIUM", "LOW"]

    reset_rounds(SimpleNamespace(invocation_id=inv))  # the executor's before_agent_callback
    second_pass = _run(cb, events, 4)

    assert [_level(r) for r in second_pass] == ["MEDIUM", "MEDIUM", "MEDIUM", "LOW"]
    for request in second_pass[:3]:
        assert request.config.thinking_config is planner


async def test_the_switch_writes_thinking_config_and_nothing_else(flag):
    """Round 4 on two invocations, the switch at AUTO and at LOW: the requests
    differ in `config.thinking_config` alone."""
    exclude = {"config": {"thinking_config"}}
    dumps, levels = [], []
    for value in ("AUTO", "LOW"):
        flag(value)
        cb, events = await _session(_inv())
        request = _run(cb, events, 4)[3]
        dumps.append(request.model_dump_json(exclude=exclude))
        levels.append(_level(request))

    assert levels == ["MEDIUM", "LOW"]
    assert dumps[0] == dumps[1]


# ── the log line ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("state", "label"),
    [({}, "anomaly"), ({"tenant": "tenant-b"}, "tenant-b")],
    ids=["default-tenant", "named-tenant"],
)
async def test_the_line_names_level_was_round_pass_inv_and_tenant_last(flag, logs, state, label):
    flag("LOW")
    inv = _inv()
    cb, events = await _session(inv, state=state)

    for rnd in (1, 2, 3):
        _run(cb, events, 1)
        assert _lines(logs) == [], rnd
    _run(cb, events, 2)  # rounds 4 and 5

    lines = _lines(logs)
    assert len(lines) == 2
    for line, rnd in zip(lines, ("4", "5"), strict=True):
        match = LINE.match(line)
        assert match, line
        assert match.groups() == ("LOW", "MEDIUM", rnd, "1", inv, label)
        assert line.split()[-1] == f"tenant={label}"
    records = [r for r in logs.records if r.getMessage().startswith("synthesis_thinking:")]
    assert {(r.name, r.levelno) for r in records} == {(LOGGER, logging.INFO)}


@pytest.mark.parametrize(
    ("was", "executor_level"),
    [("HIGH", "HIGH"), ("DYNAMIC", "DYNAMIC")],
)
async def test_was_names_the_level_the_request_carried(flag, logs, sandbox_on, was, executor_level):
    """The executor's level raised (or unbounded) by a sandbox override: the
    line says what the answer round would have thought at."""
    flag("LOW")
    inv = _inv()
    cb, events = await _session(inv, state={"sandbox": {"thinking_level": executor_level}})

    requests = _run(cb, events, 4)

    assert [_level(r) for r in requests] == [was, was, was, "LOW"]
    [line] = _lines(logs)
    assert LINE.match(line).groups() == ("LOW", was, "4", "1", inv, "anomaly")
    # DYNAMIC carried a budget; the replacement carries a level, not both.
    assert requests[3].config.thinking_config.thinking_budget is None


async def test_was_is_a_dash_for_a_request_with_no_thinking_config(flag, logs):
    flag("LOW")
    inv = _inv()
    cb, events = await _session(inv)

    requests = _run(cb, events, 4, thinking=None)

    assert [_level(r) for r in requests] == ["-", "-", "-", "LOW"]
    [line] = _lines(logs)
    assert LINE.match(line).groups() == ("LOW", "-", "4", "1", inv, "anomaly")


async def test_the_line_names_the_pass_it_was_written_in(flag, logs):
    flag("LOW")
    inv = _inv()
    cb, events = await _session(inv)
    _run(cb, events, 4)

    reset_rounds(SimpleNamespace(invocation_id=inv))
    cb2, events2 = await _session(inv, judged=True)  # the first pass carries a verdict now
    _run(cb2, events2, 4)

    lines = _lines(logs)
    assert len(lines) == 2, lines
    first, second = lines
    assert LINE.match(first).groups() == ("LOW", "MEDIUM", "4", "1", inv, "anomaly")
    assert LINE.match(second).groups() == ("LOW", "MEDIUM", "4", "2", inv, "anomaly")


# ── never raises thinking ────────────────────────────────────────────────────


@pytest.mark.parametrize("level", ["LOW", "MINIMAL", "OFF"])
async def test_a_request_at_or_below_low_is_left_alone(flag, logs, sandbox_on, level):
    flag("LOW")
    inv = _inv()
    cb, events = await _session(inv, state={"sandbox": {"thinking_level": level}})

    requests = _run(cb, events, 5)

    expected = requests[2].config.thinking_config.model_dump()
    for rnd in (4, 5):
        assert requests[rnd - 1].config.thinking_config.model_dump() == expected
    assert {_level(r) for r in requests} == {level}
    assert _lines(logs) == []
    await _control(flag, logs)


async def test_a_request_with_a_positive_budget_is_left_alone(flag, logs):
    flag("LOW")
    theirs = genai_types.ThinkingConfig(thinking_budget=1024, include_thoughts=False)
    cb, events = await _session(_inv())

    requests = _run(cb, events, 5, thinking=theirs)

    for request in requests:
        assert request.config.thinking_config is theirs
    assert _level(requests[3]) == "budget=1024"
    assert _lines(logs) == []
    await _control(flag, logs)


# ── sandbox override ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("env", "override", "acts"),
    [
        ("AUTO", "LOW", True),
        ("AUTO", None, False),
        ("LOW", "AUTO", False),
        ("LOW", None, True),
        ("LOW", "LOW", True),
        ("AUTO", "AUTO", False),
    ],
)
async def test_an_explicit_override_wins_over_the_env_and_none_follows_it(
    flag, logs, sandbox_on, env, override, acts
):
    flag(env)
    given = {KEY: override} if override is not None else {"label": "arm"}
    inv = _inv()
    cb, events = await _session(inv, state={"sandbox": given})

    requests = _run(cb, events, 5)

    assert [_level(r) for r in requests] == ["MEDIUM"] * 3 + (
        ["LOW"] * 2 if acts else ["MEDIUM"] * 2
    )
    assert len(_lines(logs)) == (2 if acts else 0)
    if not acts:
        await _control(flag, logs)


@pytest.mark.parametrize(
    ("env", "override", "acts"),
    [("AUTO", "LOW", False), ("LOW", "AUTO", True)],
    ids=["low-override-ignored", "auto-override-ignored"],
)
async def test_the_override_is_inert_unless_the_sandbox_is_enabled(
    flag, logs, monkeypatch, env, override, acts
):
    monkeypatch.setattr(config, "SANDBOX_ENABLED", False)
    flag(env)
    cb, events = await _session(_inv(), state={"sandbox": {KEY: override}})

    requests = _run(cb, events, 4)

    assert _level(requests[3]) == ("LOW" if acts else "MEDIUM")
    assert len(_lines(logs)) == (1 if acts else 0)
    if not acts:
        await _control(flag, logs)


# ── model guards ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model", [CLAUDE, OTHER_GEMINI], ids=["claude", "other-gemini"])
async def test_a_model_that_takes_no_named_level_is_left_alone(flag, logs, model):
    flag("LOW")
    cb, events = await _session(_inv())
    planner = _planner_cfg()

    requests = _run(cb, events, 5, model=model)

    for request in requests:
        assert request.model == model
        assert request.config.thinking_config is planner
    assert _lines(logs) == []
    await _control(flag, logs)


async def test_the_allowlist_is_read_at_call_time(flag, logs, monkeypatch):
    """A model added to SANDBOX_THINKING_LEVEL_MODELS takes the lowered level."""
    flag("LOW")
    monkeypatch.setattr(config, "SANDBOX_THINKING_LEVEL_MODELS", (GEMINI_MODEL, OTHER_GEMINI))
    cb, events = await _session(_inv())

    requests = _run(cb, events, 4, model=OTHER_GEMINI)

    assert _level(requests[3]) == "LOW"
    assert len(_lines(logs)) == 1


async def test_a_missing_model_id_means_the_default_model(flag, logs):
    flag("LOW")
    cb, events = await _session(_inv())

    requests = _run(cb, events, 4, model=None)

    assert [_level(r) for r in requests] == ["MEDIUM", "MEDIUM", "MEDIUM", "LOW"]
    assert len(_lines(logs)) == 1


async def test_a_missing_default_model_outside_the_list_is_left_alone(flag, logs, monkeypatch):
    """model None falls back to GEMINI_MODEL, and that name is checked too."""
    flag("LOW")
    monkeypatch.setattr(config, "SANDBOX_THINKING_LEVEL_MODELS", (OTHER_GEMINI,))
    cb, events = await _session(_inv())

    requests = _run(cb, events, 4, model=None)

    assert {_level(r) for r in requests} == {"MEDIUM"}
    assert _lines(logs) == []
    # the same switch acts once the default model is on the list
    monkeypatch.setattr(config, "SANDBOX_THINKING_LEVEL_MODELS", (GEMINI_MODEL,))
    cb, events = await _session(_inv())
    assert _level(_run(cb, events, 4, model=None)[3]) == "LOW"


async def test_a_request_with_no_config_does_not_raise(flag, logs):
    flag("LOW")
    inv = _inv()
    cb, _ = await _session(inv)
    for _ in range(4):
        round_limiter._tick(inv)
    no_config = SimpleNamespace(model=GEMINI_MODEL, config=None, contents=[])

    assert _synthesis(cb, no_config) is None
    assert no_config.config is None
    assert _lines(logs) == []
    # and the same call on a request that has one acts
    request = _request([_text("user", "user", "whats new?", inv)])
    _synthesis(cb, request)
    assert _level(request) == "LOW"


# ── ordering against the sandbox's executor level ────────────────────────────


async def test_the_sandbox_executor_level_applies_to_every_round_and_low_takes_round_four(
    flag, logs, sandbox_on
):
    state = {"sandbox": {"thinking_level": "HIGH"}}

    flag("AUTO")
    cb, events = await _session(_inv(), state=state)
    requests = _run(cb, events, 4)
    assert [_level(r) for r in requests] == ["HIGH"] * 4
    assert _lines(logs) == []

    flag("LOW")
    inv = _inv()
    cb, events = await _session(inv, state=state)
    requests = _run(cb, events, 4)
    assert [_level(r) for r in requests] == ["HIGH", "HIGH", "HIGH", "LOW"]
    [line] = _lines(logs)
    assert LINE.match(line).groups() == ("LOW", "HIGH", "4", "1", inv, "anomaly")


def test_the_switch_runs_after_the_round_limit_and_the_sandbox():
    """A pure source-order pin: it reads the counter round_limit ticked, and the
    sandbox's thinking override is applied before it, so that an explicit
    synthesis level is the more specific of the two."""
    source = open(agent_module.__file__).read()
    body = source.split("def _before_model", 1)[1].split("executor_agent = Agent(", 1)[0]
    assert "synthesis_round_thinking(callback_context" in body
    switch = body.rindex("synthesis_round_thinking(callback_context")
    assert body.rindex("round_limit(callback_context") < switch
    assert body.index("sandbox_before_model(callback_context") < switch


# ── provenance and validation ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("env", "override", "effective"),
    [("AUTO", None, "AUTO"), ("LOW", None, "LOW"), ("AUTO", "LOW", "LOW"), ("LOW", "AUTO", "AUTO")],
)
def test_resolved_config_reports_the_effective_value(sandbox_on, flag, env, override, effective):
    flag(env)
    given = {KEY: override} if override is not None else {"label": "arm"}

    resolved = resolved_config(read_overrides({"sandbox": given}))

    assert resolved.get(KEY) == effective


@pytest.mark.parametrize("given", ["LOW", "AUTO"])
def test_an_explicit_value_is_reported_as_overridden_even_when_it_equals_the_env(
    sandbox_on, flag, given
):
    flag(given)
    overrides = read_overrides({"sandbox": {KEY: given}})

    assert getattr(overrides, KEY, None) == given
    assert KEY in overrides.overridden_keys()
    assert KEY in resolved_config(overrides)["overridden_keys"]
    assert not overrides.is_empty()  # SandboxEcho will emit sandbox_resolved


def test_a_none_override_is_not_reported_and_keeps_the_provenance_key(sandbox_on, flag):
    flag("LOW")
    overrides = read_overrides({"sandbox": {"label": "arm"}})
    field = SandboxOverrides.model_fields.get(KEY)

    assert field is not None and field.default is None
    assert KEY not in overrides.overridden_keys()
    resolved = resolved_config(overrides)
    assert KEY in resolved and KEY not in resolved["overridden_keys"]


def _every_role_dynamic(model: str, **extra) -> dict:
    """The override a model that takes no named level needs (tests/unit/
    test_sandbox.py): every other role's level DYNAMIC."""
    return {
        "model": model,
        "thinking_level": "DYNAMIC",
        "critic_thinking_level": "DYNAMIC",
        "formatter_thinking_level": "DYNAMIC",
        "router_thinking_level": "DYNAMIC",
        **extra,
    }


@pytest.mark.parametrize("model", [CLAUDE, OTHER_GEMINI], ids=["claude", "other-gemini"])
def test_a_named_answer_round_level_is_refused_for_a_model_that_takes_none(sandbox_on, model):
    """Every other level is DYNAMIC here, so the answer round's is the only
    offender, and the message names it."""
    with pytest.raises(ValueError, match=f"{KEY}=LOW"):
        read_overrides({"sandbox": _every_role_dynamic(model, **{KEY: "LOW"})})


@pytest.mark.parametrize("value", ["AUTO", None], ids=["auto", "none"])
@pytest.mark.parametrize("model", [CLAUDE, OTHER_GEMINI], ids=["claude", "other-gemini"])
def test_auto_and_none_are_accepted_for_a_model_that_takes_no_named_level(sandbox_on, model, value):
    given = _every_role_dynamic(model, **({KEY: value} if value is not None else {}))
    overrides = read_overrides({"sandbox": given})

    assert overrides.model == model
    assert getattr(overrides, KEY, "<missing>") == value


def test_a_named_answer_round_level_is_accepted_for_a_model_that_takes_named_levels(sandbox_on):
    overrides = read_overrides({"sandbox": {"model": GEMINI_MODEL, KEY: "LOW"}})

    assert getattr(overrides, KEY, "<missing>") == "LOW"


@pytest.mark.parametrize("model", [CLAUDE, OTHER_GEMINI], ids=["claude", "other-gemini"])
def test_resolved_config_reports_auto_for_a_model_the_switch_skips(sandbox_on, flag, model):
    """The env says LOW, but the answer round is left alone for this model (a
    named level is a 400 there): the provenance must not label the run LOW."""
    flag("LOW")

    resolved = resolved_config(read_overrides({"sandbox": _every_role_dynamic(model)}))

    assert resolved.get(KEY) == "AUTO"


def test_resolved_config_reports_low_for_a_model_added_to_the_list(sandbox_on, flag, monkeypatch):
    flag("LOW")
    monkeypatch.setattr(config, "SANDBOX_THINKING_LEVEL_MODELS", (GEMINI_MODEL, OTHER_GEMINI))

    resolved = resolved_config(read_overrides({"sandbox": {"model": OTHER_GEMINI}}))

    assert resolved.get(KEY) == "LOW"


def test_takes_a_named_level_is_one_rule(monkeypatch):
    takes = getattr(sandbox, "takes_named_thinking_level", None)
    assert takes is not None, "sandbox.takes_named_thinking_level does not exist"

    assert takes(None) and takes(GEMINI_MODEL)  # None is the deployed default
    assert not takes(OTHER_GEMINI) and not takes(CLAUDE)
    monkeypatch.setattr(
        config, "SANDBOX_THINKING_LEVEL_MODELS", (GEMINI_MODEL, OTHER_GEMINI, CLAUDE)
    )
    assert takes(OTHER_GEMINI)
    assert not takes(CLAUDE)  # Anthropic has no named levels, whatever the list says


async def test_the_sandbox_reader_is_touched_on_the_answer_round_only(flag, monkeypatch):
    """The round gate comes first and is state only: rounds 1..3 never reach the
    sandbox reader (which warns about what it ignores) once more; the answer
    round reads it once."""
    flag("LOW")
    real = round_limiter.read_overrides
    calls: list[int] = []

    def counting(state):
        calls.append(1)
        return real(state)

    monkeypatch.setattr(round_limiter, "read_overrides", counting, raising=False)
    cb, events = await _session(_inv())

    _run(cb, events, 3)
    assert calls == []

    _run(cb, events, 1)  # round 4
    assert calls == [1]


@pytest.mark.parametrize("bad", ["HIGH", "MEDIUM", "low", "auto", "", 1, True])
def test_a_bad_value_is_refused_not_run_as_baseline(sandbox_on, bad):
    with pytest.raises(ValueError, match=KEY):
        read_overrides({"sandbox": {KEY: bad}})


def test_a_bad_value_is_a_validation_error_at_the_model_level():
    with pytest.raises(pydantic.ValidationError):
        SandboxOverrides(**{KEY: "HIGH"})


async def test_a_bad_override_fails_the_call_in_the_chain(sandbox_on, flag):
    flag("AUTO")
    cb, events = await _session(_inv(), state={"sandbox": {KEY: "HIGH"}})

    with pytest.raises(ValueError, match=KEY):
        _run(cb, events, 1)


# ── env, vocabulary, deploy files ────────────────────────────────────────────


_ENV_CASES = {
    None: "AUTO",
    "AUTO": "AUTO",
    "": "AUTO",
    "MEDIUM": "AUTO",
    "HIGH": "AUTO",
    "garbage": "AUTO",
    "LOWER": "AUTO",
    "LOW": "LOW",
    "low": "LOW",
    " Low ": "LOW",
}

_READ_ENV = """
import importlib, json, os
from gub_agent import config
out = {}
for case in json.loads(os.environ["W3_CASES"]):
    if case is None:
        os.environ.pop(%(flag)r, None)
    else:
        os.environ[%(flag)r] = case
    importlib.reload(config)
    out[json.dumps(case)] = getattr(config, %(flag)r, "<missing>")
print(json.dumps(out))
"""


def test_the_code_default_is_auto_and_only_low_reads_as_low():
    """Read at import, so each value is read by a reload of config in a clean
    interpreter (one interpreter for all of them: an import is seconds)."""
    env = {k: v for k, v in os.environ.items() if k != FLAG}
    env["PYTHONPATH"] = str(REPO)
    env["W3_CASES"] = json.dumps(list(_ENV_CASES))
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _READ_ENV % {"flag": FLAG}],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got == {json.dumps(case): expected for case, expected in _ENV_CASES.items()}


def test_the_env_vocabulary_is_the_sandbox_vocabulary():
    levels = getattr(config, "SYNTHESIS_ROUND_THINKING_LEVELS", None)
    literal = getattr(sandbox, "SynthesisRoundThinkingLevel", None)

    assert literal is not None, "sandbox.SynthesisRoundThinkingLevel does not exist"
    assert levels == typing.get_args(literal) == ("AUTO", "LOW")


def test_both_deploy_env_files_state_the_flag_off_with_one_value():
    """Stated, not defaulted, the same in both, and OFF: a paid replay of the
    round on both levels decides whether it is turned on."""
    prod, sandbox_env = _stated("deploy-prod.env", FLAG), _stated("deploy-sandbox.env", FLAG)
    assert len(prod) == 1 and len(sandbox_env) == 1, (prod, sandbox_env)
    assert prod == sandbox_env == ["AUTO"]


# ── other agents are unaffected ──────────────────────────────────────────────


async def test_the_critic_formatter_and_router_chains_never_lower_thinking(flag, logs):
    """Each of their before_model chains, on an invocation whose round counter is
    already past MAX_ROUNDS and with the switch at LOW, over a request that
    thinks at MEDIUM (a lowerable state): no line, the config still theirs. The
    executor's chain on the same invocation then does act."""
    flag("LOW")
    inv = _inv()
    cb, events = await _session(inv)
    for _ in range(round_limiter.MAX_ROUNDS + 2):
        round_limiter._tick(inv)

    for chain in (_critic_before_model, _formatter_before_model, _router_before_model):
        theirs = genai_types.ThinkingConfig(thinking_level="MEDIUM", include_thoughts=False)
        request = LlmRequest(
            model=GEMINI_MODEL,
            contents=_get_contents(None, events, AGENT_NAME),
            config=genai_types.GenerateContentConfig(thinking_config=theirs),
        )

        chain(cb, request)

        assert request.config.thinking_config is theirs, chain.__name__
        assert _level(request) == "MEDIUM", chain.__name__
    assert _lines(logs) == []

    request = _request(events)
    agent_module._before_model(cb, request)
    assert _level(request) == "LOW"
    assert len(_lines(logs)) == 1
