"""
sandbox — per-call overrides for prompt / model / thinking / temperature.

The load-bearing part is the NEGATIVE case: with no `sandbox` key in session
state (or with SANDBOX_ENABLED off) every override point must be a no-op —
same model, same thinking config object, same prompt text, and no extra event
in the trace. The whole sandbox epic rests on that invariant, and the prod
engine runs this code path on every single turn, so it is pinned by tests
rather than by inspection (cases 1 and 2).

The positive cases pin the three ADK mechanisms the design rests on: the model
swap through `llm_request.model`, DYNAMIC thinking as `thinking_budget=-1`, and
a state-borne prompt containing `{campaign}` surviving the instruction
processor (that last one is verified against real ADK objects, including a
demonstration that the un-bypassed path would have crashed).

Fake ADK request objects via SimpleNamespace, real InvocationContext/Event for
the agent-level cases. No live model calls anywhere.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.adk.utils import instructions_utils

from gub_agent import config
from gub_agent.agents.critic import CriticGate, CriticVerdict
from gub_agent.sandbox import (
    SandboxEcho,
    read_overrides,
    resolved_config,
    sandbox_before_model,
    sandbox_instruction,
)

ALLOWLIST = ("gemini-3.5-flash", "gemini-2.5-pro")
THINKING_LEVEL_MODELS = ("gemini-3.5-flash",)

# Identity sentinel: the planner assigns its own long-lived ThinkingConfig into
# the request, so "unchanged" means the SAME object, not an equal one.
BASE_THINKING = SimpleNamespace(thinking_level="MEDIUM", include_thoughts=False)


@pytest.fixture
def sandbox_on(monkeypatch):
    """Sandbox engine: flag on, allowlist pinned so a dev's env can't change
    what the error messages are expected to say."""
    monkeypatch.setattr(config, "SANDBOX_ENABLED", True)
    monkeypatch.setattr(config, "SANDBOX_MODEL_ALLOWLIST", ALLOWLIST)
    monkeypatch.setattr(config, "SANDBOX_THINKING_LEVEL_MODELS", THINKING_LEVEL_MODELS)


@pytest.fixture
def sandbox_off(monkeypatch):
    """Prod engine."""
    monkeypatch.setattr(config, "SANDBOX_ENABLED", False)
    monkeypatch.setattr(config, "SANDBOX_MODEL_ALLOWLIST", ALLOWLIST)


def _req() -> SimpleNamespace:
    """An llm_request as it looks when before_model_callback fires: the basic
    processor has filled `model`, the planner has filled `thinking_config`."""
    return SimpleNamespace(
        model="gemini-3.5-flash",
        config=SimpleNamespace(thinking_config=BASE_THINKING, temperature=None, tools=["tool"]),
        tools_dict={"tool": 1},
        contents=[],
    )


def _ctx_for(state: dict) -> SimpleNamespace:
    """A callback/instruction context stand-in carrying session state."""
    return SimpleNamespace(state=state, invocation_id="inv-1")


async def _real_ctx(state: dict) -> InvocationContext:
    """A real ADK InvocationContext (needed for agent-level cases)."""
    service = InMemorySessionService()
    session = await service.create_session(app_name="gub", user_id="u", state=state)
    agent = BaseAgent(name="host")
    return InvocationContext(
        session_service=service,
        invocation_id="inv-1",
        agent=agent,
        session=session,
    )


def _base_provider(text: str = "BASE PROMPT"):
    return lambda _ctx: text


# ── 1. empty state → nothing changes ──────────────────────────────────────────


async def test_empty_state_changes_nothing(sandbox_on, caplog):
    """No `sandbox` key: model, thinking config, prompt and trace all untouched.
    This is the path every prod turn takes."""
    req = _req()
    with caplog.at_level(logging.DEBUG, logger="gub_agent.sandbox"):
        sandbox_before_model(_ctx_for({}), req, role="executor")

    assert req.model == "gemini-3.5-flash"
    assert req.config.thinking_config is BASE_THINKING  # same object, not just equal
    assert req.config.temperature is None
    assert req.config.tools == ["tool"]  # sandbox never touches tools

    provider = sandbox_instruction(_base_provider(), role="executor")
    assert provider(_ctx_for({})) == "BASE PROMPT"

    # No provenance event either — an extra event would change the trace shape.
    ctx = await _real_ctx({})
    assert [event async for event in SandboxEcho(name="sandbox_echo").run_async(ctx)] == []
    assert caplog.records == []


# ── 2. flag off + non-empty state → nothing changes, nothing logged ───────────


async def test_disabled_ignores_a_populated_sandbox_key(sandbox_off, caplog):
    """Prod engine with someone's sandbox payload in state: full ignore, and
    SILENT — prod must not be noisy about a key meant for a sandbox run. The
    payload is deliberately invalid (bad model, bad temperature, unknown key):
    even that must not raise here."""
    state = {
        "sandbox": {
            "model": "gemini-9-ultra",
            "temperature": 42.0,
            "thinking_level": "HIGH",
            "executor_instruction": "OVERRIDE",
            "critic_enabled": False,
            "nonsense_key": 1,
        }
    }
    req = _req()
    with caplog.at_level(logging.DEBUG, logger="gub_agent.sandbox"):
        sandbox_before_model(_ctx_for(state), req, role="executor")
        provider = sandbox_instruction(_base_provider(), role="executor")
        assert provider(_ctx_for(state)) == "BASE PROMPT"
        assert read_overrides(state).is_empty()

    assert req.model == "gemini-3.5-flash"
    assert req.config.thinking_config is BASE_THINKING
    assert req.config.temperature is None
    assert caplog.records == []  # not a warning, not an error, nothing

    ctx = await _real_ctx(state)
    assert [event async for event in SandboxEcho(name="sandbox_echo").run_async(ctx)] == []


# ── 3. model override ─────────────────────────────────────────────────────────


async def test_model_override_reaches_the_request(sandbox_on):
    """The swap point: google_llm issues the call with `model=llm_request.model`,
    so overwriting that field in before_model_callback changes which model
    answers — no redeploy."""
    req = _req()
    sandbox_before_model(_ctx_for({"sandbox": {"model": "gemini-3.5-flash"}}), req, role="executor")
    assert req.model == "gemini-3.5-flash"
    assert req.config.thinking_config is BASE_THINKING  # untouched knobs stay untouched


async def test_model_without_named_thinking_support_needs_dynamic(sandbox_on):
    """Live-verified trap: gemini-2.5-pro rejects `thinking_level` with 400, and
    inside the engine that 400 is an EMPTY 200 stream for the caller. The
    baseline planners carry named levels, so `model` alone is not a valid
    override for such a model — refuse up front, with the fix in the message."""
    with pytest.raises(ValueError, match="DYNAMIC") as exc:
        read_overrides({"sandbox": {"model": "gemini-2.5-pro"}})
    assert "thinking_level=MEDIUM" in str(exc.value)  # the baseline that would apply
    assert "critic_thinking_level=LOW" in str(exc.value)
    assert "formatter_thinking_level=LOW" in str(exc.value)
    assert "router_thinking_level=LOW" in str(exc.value)

    # One role fixed is not enough while the critic still runs.
    with pytest.raises(ValueError, match="critic_thinking_level=LOW"):
        read_overrides({"sandbox": {"model": "gemini-2.5-pro", "thinking_level": "DYNAMIC"}})

    # The formatter always runs — its level must be DYNAMIC for such a model too.
    with pytest.raises(ValueError, match="formatter_thinking_level=LOW"):
        read_overrides(
            {
                "sandbox": {
                    "model": "gemini-2.5-pro",
                    "thinking_level": "DYNAMIC",
                    "critic_thinking_level": "DYNAMIC",
                }
            }
        )

    # And so does the router (blend 04) — it runs BEFORE any retrieval, so its
    # 400 would empty the stream before the turn had done anything at all.
    with pytest.raises(ValueError, match="router_thinking_level=LOW"):
        read_overrides(
            {
                "sandbox": {
                    "model": "gemini-2.5-pro",
                    "thinking_level": "DYNAMIC",
                    "critic_thinking_level": "DYNAMIC",
                    "formatter_thinking_level": "DYNAMIC",
                }
            }
        )


async def test_dynamic_on_all_roles_unlocks_such_a_model(sandbox_on):
    all_roles = {
        "model": "gemini-2.5-pro",
        "thinking_level": "DYNAMIC",
        "critic_thinking_level": "DYNAMIC",
        "formatter_thinking_level": "DYNAMIC",
        "router_thinking_level": "DYNAMIC",
    }
    assert read_overrides({"sandbox": all_roles}).model == "gemini-2.5-pro"
    # With the critic off, its level is irrelevant.
    no_critic = {
        "model": "gemini-2.5-pro",
        "thinking_level": "DYNAMIC",
        "formatter_thinking_level": "DYNAMIC",
        "router_thinking_level": "DYNAMIC",
        "critic_enabled": False,
    }
    assert read_overrides({"sandbox": no_critic}).critic_enabled is False


# ── 4. DYNAMIC thinking ───────────────────────────────────────────────────────


async def test_dynamic_thinking_is_an_unbounded_budget(sandbox_on):
    """'DYNAMIC' is not a genai thinking_level — it means an unbounded budget,
    the same shape build_thinking_planner(None) produces."""
    req = _req()
    sandbox_before_model(_ctx_for({"sandbox": {"thinking_level": "DYNAMIC"}}), req, role="executor")
    cfg = req.config.thinking_config
    assert cfg is not BASE_THINKING  # a FRESH object: mutating the planner's
    assert cfg.thinking_budget == -1  # own config would leak into later calls
    assert cfg.thinking_level is None


async def test_critic_thinking_is_separate_from_the_executors(sandbox_on):
    """The critic reads critic_thinking_level, not thinking_level — one run can
    move the executor without disturbing the judge."""
    state = {"sandbox": {"thinking_level": "HIGH", "critic_thinking_level": "MINIMAL"}}
    critic_req, executor_req = _req(), _req()
    sandbox_before_model(_ctx_for(state), critic_req, role="critic")
    sandbox_before_model(_ctx_for(state), executor_req, role="executor")
    assert critic_req.config.thinking_config.thinking_level == "MINIMAL"
    assert executor_req.config.thinking_config.thinking_level == "HIGH"


async def test_formatter_thinking_is_its_own_knob(sandbox_on):
    """The formatter reads formatter_thinking_level only — and with no
    formatter key at all, a formatter request is untouched even when the other
    roles are overridden (the no-op invariant, per role)."""
    state = {"sandbox": {"thinking_level": "HIGH", "formatter_thinking_level": "MINIMAL"}}
    formatter_req, executor_req = _req(), _req()
    sandbox_before_model(_ctx_for(state), formatter_req, role="formatter")
    sandbox_before_model(_ctx_for(state), executor_req, role="executor")
    assert formatter_req.config.thinking_config.thinking_level == "MINIMAL"
    assert executor_req.config.thinking_config.thinking_level == "HIGH"

    untouched = _req()
    sandbox_before_model(
        _ctx_for({"sandbox": {"critic_thinking_level": "HIGH"}}), untouched, role="formatter"
    )
    assert untouched.config.thinking_config is BASE_THINKING


async def test_router_thinking_is_its_own_knob(sandbox_on):
    """Blend 04's role: the router reads router_thinking_level only, and with
    no router key a router request is untouched even when every other role is
    overridden (the per-role no-op invariant)."""
    state = {"sandbox": {"thinking_level": "HIGH", "router_thinking_level": "MINIMAL"}}
    router_req, executor_req = _req(), _req()
    sandbox_before_model(_ctx_for(state), router_req, role="router")
    sandbox_before_model(_ctx_for(state), executor_req, role="executor")
    assert router_req.config.thinking_config.thinking_level == "MINIMAL"
    assert executor_req.config.thinking_config.thinking_level == "HIGH"

    untouched = _req()
    sandbox_before_model(
        _ctx_for({"sandbox": {"formatter_thinking_level": "HIGH"}}), untouched, role="router"
    )
    assert untouched.config.thinking_config is BASE_THINKING


async def test_the_router_prompt_is_tunable_per_call(sandbox_on):
    """The point of the role: the misroute rate is driven down by editing this
    prompt from the sandbox UI, not by a redeploy."""
    provider = sandbox_instruction(_base_provider(), role="router")
    text = provider(_ctx_for({"sandbox": {"router_instruction": "ROUTE LIKE THIS"}}))
    assert text.startswith("ROUTE LIKE THIS")
    assert "Current date" in text  # the date block is not part of the experiment
    # A router override leaves the executor's prompt alone.
    executor = sandbox_instruction(_base_provider(), role="executor")
    assert executor(_ctx_for({"sandbox": {"router_instruction": "ROUTE LIKE THIS"}})) == (
        _base_provider()(_ctx_for({}))
    )


async def test_the_router_has_a_baseline_variant(sandbox_on):
    """Registry role wiring: `router_variant` resolves, and an unknown name
    still refuses to fall back to the baseline."""
    assert read_overrides({"sandbox": {"router_variant": "baseline"}}).router_variant == "baseline"
    with pytest.raises(ValueError, match="router prompt variant"):
        read_overrides({"sandbox": {"router_variant": "v9_nope"}})


# ── 5. a prompt with literal braces ───────────────────────────────────────────


async def test_prompt_with_braces_survives(sandbox_on):
    """A prompt from state may contain literal `{campaign}`.

    `canonical_instruction` returns bypass_state_injection=True for callable
    instructions, so ADK's state-substitution pass never runs over our text.
    Both agents keep callable instructions for exactly this reason — the second
    half of the test shows what the un-bypassed path would have done.
    """
    text = "Answer about {campaign} using {tool_result}."
    state = {"sandbox": {"executor_instruction": text}}
    provider = sandbox_instruction(_base_provider(), role="executor")

    agent = LlmAgent(name="probe", model="gemini-3.5-flash", instruction=provider)
    ctx = await _real_ctx(state)
    instruction, bypass = await agent.canonical_instruction(ReadonlyContext(ctx))

    assert bypass is True
    assert text in instruction  # braces intact, no substitution attempted
    assert "## Current date" in instruction  # date note still appended

    with pytest.raises(KeyError):  # what a plain-string instruction would hit
        await instructions_utils.inject_session_state(text, ReadonlyContext(ctx))


# ── 6. model outside the allowlist ────────────────────────────────────────────


async def test_model_outside_the_allowlist_is_a_legible_error(sandbox_on):
    """Hard error, and the message says what IS allowed — a silent fallback to
    the default model would make an A/B compare the baseline with itself."""
    with pytest.raises(ValueError) as exc:
        read_overrides({"sandbox": {"model": "claude-opus-5"}})

    message = str(exc.value)
    assert "claude-opus-5" in message
    for allowed in ALLOWLIST:
        assert allowed in message


async def test_temperature_out_of_range_is_not_clamped(sandbox_on):
    """Out of range fails; clamping would silently distort the experiment."""
    with pytest.raises(ValueError, match="0.0..2.0"):
        read_overrides({"sandbox": {"temperature": 2.5}})


async def test_unknown_variant_does_not_fall_back_to_baseline(sandbox_on):
    """A name the registry doesn't hold (or a build with no registry at all)
    fails the run rather than quietly running the baseline prompt. Validated
    eagerly, at read time — not on the fourth model call of the answer."""
    with pytest.raises(ValueError, match="variant"):
        read_overrides({"sandbox": {"executor_variant": "v9_does_not_exist"}})


async def test_registered_variant_resolves(sandbox_on):
    """The other half of the above: a real registry name passes validation, so
    the no-fallback rule can't be satisfied by rejecting everything."""
    overrides = read_overrides({"sandbox": {"executor_variant": "v2_concise"}})
    assert overrides.executor_variant == "v2_concise"


# ── 7. unknown key ────────────────────────────────────────────────────────────


async def test_unknown_key_warns_and_the_run_continues(sandbox_on, caplog):
    """Callers evolve faster than the engine redeploys: a key this build doesn't
    know must not kill the run — but must not pass unnoticed either."""
    state = {
        "sandbox": {
            "model": "gemini-2.5-pro",
            "thinking_level": "DYNAMIC",
            "critic_thinking_level": "DYNAMIC",
            "formatter_thinking_level": "DYNAMIC",
            "router_thinking_level": "DYNAMIC",
            "top_k": 7,
        }
    }
    req = _req()
    with caplog.at_level(logging.WARNING, logger="gub_agent.sandbox"):
        sandbox_before_model(_ctx_for(state), req, role="executor")

    assert req.model == "gemini-2.5-pro"  # the known key still applied
    assert any("top_k" in record.getMessage() for record in caplog.records)


# ── 8. critic_enabled=False ───────────────────────────────────────────────────


class _RecordingCritic(BaseAgent):
    """Stand-in for the critic LLM — records whether it was run at all."""

    ran: list[str] = []

    async def _run_async_impl(self, ctx):
        self.ran.append(ctx.invocation_id)
        yield Event(invocation_id=ctx.invocation_id, author=self.name)


async def test_critic_disabled_skips_the_llm_but_still_writes_a_verdict(sandbox_on):
    """`critic_enabled=false` removes the critic call from the run. A verdict is
    still written: the escalator reads `sufficient` off it to exit the loop, so
    without one the executor would run a second, pointless pass."""
    recorder = _RecordingCritic(name="critic")
    gate = CriticGate(name="critic_gate", sub_agents=[recorder])
    ctx = await _real_ctx({"sandbox": {"critic_enabled": False}})

    events = [event async for event in gate.run_async(ctx)]

    assert recorder.ran == []  # the critic LLM never ran
    assert len(events) == 1
    verdict = events[0].actions.state_delta["critic_verdict"]
    assert verdict["sufficient"] is True
    assert verdict["reason"] == "sandbox: critic disabled"
    # Same shape the critic itself emits — the escalator and the debug client
    # both read these field names.
    assert set(verdict) == set(CriticVerdict.model_fields)


async def test_critic_runs_when_not_disabled(sandbox_on):
    """Control for the case above: with no override the gate runs the critic, so
    the skip is caused by the flag and not by the fake context."""
    recorder = _RecordingCritic(name="critic")
    gate = CriticGate(name="critic_gate", sub_agents=[recorder])
    ctx = await _real_ctx({})

    events = [event async for event in gate.run_async(ctx)]

    assert recorder.ran == ["inv-1"]
    assert [event.author for event in events] == ["critic"]


# ── provenance ────────────────────────────────────────────────────────────────


async def test_resolved_config_reports_what_actually_ran(sandbox_on):
    """The provenance payload: effective values (not just the requested ones),
    flat and JSON-safe, so the batch runner can write one row per run."""
    state = {
        "sandbox": {
            "model": "gemini-3.5-flash",  # a model that accepts the named level below
            "thinking_level": "LOW",
            "temperature": 0.2,
            "executor_instruction": "OVERRIDE",
            "label": "concise-v2",
        }
    }
    resolved = resolved_config(read_overrides(state))

    assert resolved["model"] == "gemini-3.5-flash"
    assert resolved["thinking_level"] == "LOW"
    assert resolved["critic_thinking_level"] == "LOW"  # untouched baseline
    assert resolved["temperature"] == 0.2
    assert resolved["label"] == "concise-v2"
    assert resolved["executor_prompt_source"] == "inline"
    assert len(resolved["executor_prompt_sha256"]) == 12
    assert resolved["critic_prompt_source"] == "baseline"
    assert resolved["critic_prompt_sha256"] is None
    assert resolved["formatter_thinking_level"] == "LOW"  # untouched baseline
    assert resolved["formatter_prompt_source"] == "baseline"
    assert resolved["formatter_prompt_sha256"] is None
    assert resolved["router_thinking_level"] == "LOW"  # untouched baseline
    assert resolved["router_prompt_source"] == "baseline"
    assert resolved["router_prompt_sha256"] is None
    assert resolved["overridden_keys"] == [
        "executor_instruction",
        "label",
        "model",
        "temperature",
        "thinking_level",
    ]


async def test_echo_emits_the_provenance_event(sandbox_on):
    """Provenance travels as an event state_delta, not a `callback_context.state`
    write: flat writes there don't survive between calls (which is why
    round_limiter keeps its own dict), and the event stream is what the batch
    runner reads."""
    ctx = await _real_ctx(
        {
            "sandbox": {
                "model": "gemini-2.5-pro",
                "thinking_level": "DYNAMIC",
                "critic_thinking_level": "DYNAMIC",
                "formatter_thinking_level": "DYNAMIC",
                "router_thinking_level": "DYNAMIC",
                "label": "run-7",
            }
        }
    )

    events = [event async for event in SandboxEcho(name="sandbox_echo").run_async(ctx)]

    assert len(events) == 1
    resolved = events[0].actions.state_delta["sandbox_resolved"]
    assert resolved["model"] == "gemini-2.5-pro"
    assert resolved["label"] == "run-7"
