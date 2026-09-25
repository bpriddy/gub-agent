"""
Thinking off for the router and the formatter (ROUTER_THINKING_OFF /
FORMATTER_THINKING_OFF, config.py).

Neither deliberates — one classifies, the other renders given text into a given
schema — and at thinking_level=LOW both spent thought tokens on the critical
path of every turn. With the flags on (the default) both planners send
thinking_budget=0; at 0 they send LOW, as before. What is pinned here:

- the planner shapes: a level, a budget, or the dynamic default — each a FRESH
  ThinkingConfig, never one shared object;
- the request each real module agent sends through its real callbacks and the
  real VendorRouter, over a stub model: budget 0 with the flags on, LOW with
  them off — in a fresh interpreter per value, because the flags are read at
  import;
- the sandbox still wins: a router_thinking_level / formatter_thinking_level
  override replaces the config for that call only, a model swap outside
  SANDBOX_THINKING_LEVEL_MODELS still needs DYNAMIC on every role, and the
  provenance names the level that actually runs ("OFF").

No model calls: every model here is a stub BaseLlm.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from gub_agent import config, sandbox
from gub_agent.agents.formatter import formatter_agent
from gub_agent.agents.router import router_agent
from gub_agent.config import GEMINI_MODEL, build_thinking_planner
from gub_agent.models import VendorRouter
from gub_agent.sandbox import (
    THINKING_OFF,
    SandboxEcho,
    baseline_planner,
    read_overrides,
    resolved_config,
)
from tests.helpers import invocation_ctx

REPO = Path(__file__).resolve().parents[2]

# What each agent's output_schema accepts, so the run completes.
REPLIES = {
    "router": {"intent": "exploratory", "confidence": 0.9},
    "formatter": {"kind": "abstain", "headline": "NO_COMPANY_RECORDS"},
}


def _level(cfg: genai_types.ThinkingConfig) -> str | None:
    level = cfg.thinking_level
    return None if level is None else str(getattr(level, "value", level))


# ── the planner shapes ───────────────────────────────────────────────────────


def test_a_budget_is_its_own_shape():
    off = build_thinking_planner(thinking_budget=0).thinking_config
    assert off.thinking_budget == 0 and off.thinking_level is None
    assert off.include_thoughts is False

    low = build_thinking_planner(thinking_level="LOW").thinking_config
    assert _level(low) == "LOW" and low.thinking_budget is None

    dynamic = build_thinking_planner().thinking_config
    assert dynamic.thinking_budget == -1 and dynamic.thinking_level is None


def test_thinking_off_never_asks_for_thoughts(monkeypatch):
    """The sandbox engine runs EMIT_THINKING=1. With thinking off there is
    nothing to summarise, and include_thoughts=true beside budget 0 is a
    pairing production has never sent, so neither path may produce it."""
    from gub_agent import sandbox

    monkeypatch.setattr(config, "EMIT_THINKING", True)
    assert build_thinking_planner(thinking_budget=0).thinking_config.include_thoughts is False
    assert sandbox._thinking_config(sandbox.THINKING_OFF).include_thoughts is False
    assert build_thinking_planner(thinking_level="LOW").thinking_config.include_thoughts is True
    assert sandbox._thinking_config("LOW").include_thoughts is True


def test_a_level_and_a_budget_together_are_refused():
    with pytest.raises(ValueError, match="not both"):
        build_thinking_planner(thinking_level="LOW", thinking_budget=0)


def test_every_planner_carries_its_own_config():
    """The planner hands its object to every request; a shared one would let a
    mutation in one agent's request reach another agent's."""
    first, second = (
        build_thinking_planner(thinking_budget=0),
        build_thinking_planner(thinking_budget=0),
    )
    assert first.thinking_config is not second.thinking_config
    assert baseline_planner(THINKING_OFF).thinking_config is not (
        baseline_planner(THINKING_OFF).thinking_config
    )


@pytest.mark.parametrize(
    "level,budget,named",
    [(THINKING_OFF, 0, None), ("DYNAMIC", -1, None), ("LOW", None, "LOW")],
)
def test_the_baseline_vocabulary_maps_to_the_planner(level, budget, named):
    cfg = baseline_planner(level).thinking_config
    assert cfg.thinking_budget == budget and _level(cfg) == named


# ── what the deployed agents send, per flag value ───────────────────────────

_DUMP = """
import asyncio, json
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from gub_agent import sandbox
from gub_agent.agents.formatter import formatter_agent
from gub_agent.agents.router import router_agent
from gub_agent.config import GEMINI_MODEL
from gub_agent.models import VendorRouter

REPLIES = %(replies)s
SEEN = {}

class Stub(BaseLlm):
    @classmethod
    def supported_models(cls):
        return [r".*"]

    async def generate_content_async(self, llm_request, stream=False):
        agent = llm_request.config.labels.get("adk_agent_name")
        cfg = llm_request.config.thinking_config
        level = cfg.thinking_level
        SEEN[agent] = {
            "budget": cfg.thinking_budget,
            "level": None if level is None else str(getattr(level, "value", level)),
        }
        text = json.dumps(REPLIES[agent])
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))

async def main():
    for agent in (router_agent, formatter_agent):
        clone = agent.clone(
            update={"model": VendorRouter(model=GEMINI_MODEL, gemini=Stub(model="stub"))}
        )
        runner = InMemoryRunner(agent=clone, app_name="gub")
        session = await runner.session_service.create_session(app_name="gub", user_id="u")
        message = types.Content(role="user", parts=[types.Part(text="how is chevy?")])
        async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=message):
            pass
    resolved = sandbox.resolved_config(sandbox.SandboxOverrides(label="probe"))
    print(json.dumps({
        "seen": SEEN,
        "provenance": {
            "router": resolved["router_thinking_level"],
            "formatter": resolved["formatter_thinking_level"],
        },
    }))

asyncio.run(main())
"""


@pytest.mark.parametrize(
    "flag,budget,level,reported",
    [("1", 0, None, "OFF"), ("0", None, "LOW", "LOW")],
    ids=["thinking-off", "rollback"],
)
def test_the_router_and_the_formatter_send_what_the_flags_say(flag, budget, level, reported):
    """Read at import, so each value gets a fresh interpreter. The request is
    the one the stub model receives: the planner's config after every
    request processor and the agent's own before_model_callback."""
    env = {
        **os.environ,
        "ROUTER_THINKING_OFF": flag,
        "FORMATTER_THINKING_OFF": flag,
        "SANDBOX_ENABLED": "0",
        "PYTHONPATH": str(REPO),
    }
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _DUMP % {"replies": repr(REPLIES)}],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    dumped = json.loads(out.stdout.strip().splitlines()[-1])
    for agent in ("router", "formatter"):
        assert dumped["seen"][agent] == {"budget": budget, "level": level}, agent
        assert dumped["provenance"][agent] == reported, agent


# ── the sandbox still wins ───────────────────────────────────────────────────


class _Recorder(BaseLlm):
    """The model behind the real VendorRouter: records each request's model and
    thinking config, and answers in the agent's output_schema."""

    seen: list = []

    @classmethod
    def supported_models(cls):
        return [r".*"]

    async def generate_content_async(self, llm_request, stream=False):
        agent = llm_request.config.labels.get("adk_agent_name")
        self.seen.append((llm_request.model, llm_request.config.thinking_config))
        text = json.dumps(REPLIES[agent])
        yield LlmResponse(
            content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)])
        )


async def _ask(agent, state: dict, *, planner=None) -> tuple[str, genai_types.ThinkingConfig]:
    """One run of a module agent — its planner (or `planner`, standing in for
    another flag value), its callbacks — over a stub."""
    recorder = _Recorder(model="stub", seen=[])
    update = {"model": VendorRouter(model=GEMINI_MODEL, gemini=recorder)}
    if planner is not None:
        update["planner"] = planner
    clone = agent.clone(update=update)
    runner = InMemoryRunner(agent=clone, app_name="gub")
    session = await runner.session_service.create_session(app_name="gub", user_id="u", state=state)
    message = genai_types.Content(role="user", parts=[genai_types.Part(text="how is chevy?")])
    async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=message):
        pass
    [seen] = recorder.seen
    return seen


@pytest.fixture
def sandbox_on(monkeypatch):
    monkeypatch.setattr(config, "SANDBOX_ENABLED", True)
    monkeypatch.setattr(config, "SANDBOX_MODEL_ALLOWLIST", ("gemini-3.5-flash", "gemini-2.5-pro"))
    monkeypatch.setattr(config, "SANDBOX_THINKING_LEVEL_MODELS", ("gemini-3.5-flash",))


@pytest.mark.parametrize(
    "agent,key",
    [(router_agent, "router"), (formatter_agent, "formatter")],
    ids=["router", "formatter"],
)
async def test_an_override_replaces_the_baseline_for_that_call_only(sandbox_on, agent, key):
    """LOW asked for, LOW sent — whatever the flag made the baseline. The
    planner's own config is not touched: the next ordinary call sends the
    baseline again, the same object the planner holds."""
    baseline = agent.planner.thinking_config

    _, overridden = await _ask(agent, {"sandbox": {f"{key}_thinking_level": "LOW"}})
    assert _level(overridden) == "LOW" and overridden.thinking_budget is None
    assert overridden is not baseline

    _, plain = await _ask(agent, {})
    assert plain is baseline  # a shallow clone shares the module agent's planner


@pytest.mark.parametrize(
    "agent,key",
    [(router_agent, "router"), (formatter_agent, "formatter")],
    ids=["router", "formatter"],
)
async def test_off_is_an_override_too(sandbox_on, agent, key):
    """The provenance says OFF, so a run can ask for OFF — here on an agent
    whose baseline is LOW, as on an engine deployed with the flag at 0 — and
    gets thinking_budget=0."""
    low = baseline_planner("LOW")
    _, cfg = await _ask(agent, {"sandbox": {f"{key}_thinking_level": "OFF"}}, planner=low)
    assert cfg.thinking_budget == 0 and cfg.thinking_level is None
    _, plain = await _ask(agent, {}, planner=low)
    assert _level(plain) == "LOW"  # the control: without the override, LOW


async def test_a_swapped_model_outside_the_named_level_set_still_needs_dynamic(
    sandbox_on, monkeypatch
):
    """gemini-2.5-pro takes neither a named level nor thinking off. With the
    OFF baseline the run is refused up front, naming the level that would
    apply; with DYNAMIC on every role it runs, and the router and formatter
    send the unbounded budget to the swapped model — never budget 0."""
    monkeypatch.setattr(sandbox, "ROUTER_THINKING_LEVEL", THINKING_OFF)
    monkeypatch.setattr(sandbox, "FORMATTER_THINKING_LEVEL", THINKING_OFF)
    with pytest.raises(ValueError, match="DYNAMIC") as exc:
        read_overrides({"sandbox": {"model": "gemini-2.5-pro"}})
    assert "router_thinking_level=OFF" in str(exc.value)
    assert "formatter_thinking_level=OFF" in str(exc.value)

    swapped = {
        "sandbox": {
            "model": "gemini-2.5-pro",
            "thinking_level": "DYNAMIC",
            "critic_thinking_level": "DYNAMIC",
            "formatter_thinking_level": "DYNAMIC",
            "router_thinking_level": "DYNAMIC",
        }
    }
    for agent in (router_agent, formatter_agent):
        model, cfg = await _ask(agent, swapped)
        assert model == "gemini-2.5-pro"
        assert cfg.thinking_budget == -1 and cfg.thinking_level is None


async def test_the_provenance_reports_the_level_that_runs(sandbox_on, monkeypatch):
    """The echo reports OFF for an untouched router and formatter when that is
    the baseline, and the override when there is one."""
    monkeypatch.setattr(sandbox, "ROUTER_THINKING_LEVEL", THINKING_OFF)
    monkeypatch.setattr(sandbox, "FORMATTER_THINKING_LEVEL", THINKING_OFF)

    untouched = resolved_config(read_overrides({"sandbox": {"label": "probe"}}))
    assert untouched["router_thinking_level"] == "OFF"
    assert untouched["formatter_thinking_level"] == "OFF"

    ctx = await invocation_ctx(state={"sandbox": {"router_thinking_level": "LOW"}})
    [event] = [e async for e in SandboxEcho(name="sandbox_echo").run_async(ctx)]
    echoed = event.actions.state_delta["sandbox_resolved"]
    assert echoed["router_thinking_level"] == "LOW"
    assert echoed["formatter_thinking_level"] == "OFF"
