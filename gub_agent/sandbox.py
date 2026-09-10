"""
sandbox.py — per-call overrides for prompt, model, thinking level and temperature.

The executor and the critic are built once, at module import (`agent.py`,
`agents/critic.py`). Changing a prompt, a model or a thinking level therefore
means editing code or env and running `adk deploy agent_engine`: 10-15 minutes
per iteration, into the same engine that serves the live Chat bot. This module
moves those four knobs into ONE session-state key, read fresh on every call:

    create_session(state={"sandbox": {"model": "gemini-2.5-pro",
                                      "thinking_level": "LOW",
                                      "executor_instruction": "<text>"}})

Three ADK facts make it work without a rebuild (verified against ADK 2.6.1):

1. `instruction` on both agents is a CALLABLE, and `canonical_instruction`
   returns `bypass_state_injection=True` for callables
   (`agents/llm_agent.py:692-698`), so the instruction processor skips
   `inject_session_state` (`flows/llm_flows/instructions.py:51-59`). A prompt
   arriving from state may contain literal `{...}` without the KeyError crash
   that the warning in `prompts/__init__.py:11-15` describes — that warning
   applies to plain-STRING instructions only. Keep both instructions callables
   and this stays true.
2. `before_model_callback` fires after every request processor
   (`base_llm_flow.py:973`/`:1054` versus `:1391`), the basic processor has
   already copied the agent's model into `llm_request.model`
   (`flows/llm_flows/basic.py:71`), and the Gemini client issues the call with
   `model=llm_request.model` (`models/google_llm.py:251`). Overwriting that
   field in the callback changes which model answers.
3. Provenance is emitted as an EVENT with a `state_delta`, not written through
   `callback_context.state`: flat writes there do not survive between calls,
   which is exactly why `agents/round_limiter.py:11-13` keeps an in-process
   dict keyed on `invocation_id`. `SandboxEcho` mirrors how `CriticGate` writes
   its verdict (`agents/critic.py:186-203`).

Two invariants the whole sandbox epic rests on:

- With no `sandbox` key (or with `SANDBOX_ENABLED` off) EVERY function here is a
  no-op: nothing touched, nothing logged, no extra event in the trace. Pinned by
  tests, not by inspection.
- An invalid value of a KNOWN key is a hard error, never a silent fallback. A
  model outside the allowlist, a temperature out of range, an unknown variant
  name — all fail loudly, because an A/B that silently ran the baseline in both
  arms is worse than an A/B that did not run.

Boundary: `llm_request.model` is swapped inside the same genai client (Vertex,
global endpoint, pinned in `config.py:22`). A cross-provider model needs a
different `BaseLlm` class and is out of scope.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event, EventActions
from google.genai import types as genai_types
from pydantic import BaseModel, ConfigDict, ValidationError

from . import config
from .instruction_utils import current_date_note

logger = logging.getLogger(__name__)

# The single state key the caller writes, and the one this module writes back.
STATE_KEY = "sandbox"
RESOLVED_STATE_KEY = "sandbox_resolved"

# Prompts above this go through the variant registry (`prompts/variants/`)
# instead of the wire: a session-state payload is not a document store.
MAX_PROMPT_BYTES = 64 * 1024

Role = Literal["executor", "critic", "formatter"]
ThinkingLevel = Literal["MINIMAL", "LOW", "MEDIUM", "HIGH", "DYNAMIC"]

# Baseline thinking levels, reported in the provenance when nothing overrides
# them. Imported by the construction sites (agent.py, agents/critic.py,
# agents/formatter.py) so the provenance can never drift from what actually
# runs.
EXECUTOR_THINKING_LEVEL = "MEDIUM"
CRITIC_THINKING_LEVEL = "LOW"
FORMATTER_THINKING_LEVEL = "LOW"


class SandboxOverrides(BaseModel):
    """The `state["sandbox"]` contract.

    `extra="ignore"` plus an explicit unknown-key warn in `read_overrides`:
    a typo'd key must not fail the run (callers evolve faster than the engine
    redeploys) but must not pass unnoticed either.
    """

    model_config = ConfigDict(extra="ignore")

    executor_instruction: str | None = None
    executor_variant: str | None = None
    critic_instruction: str | None = None
    critic_variant: str | None = None
    formatter_instruction: str | None = None
    formatter_variant: str | None = None
    model: str | None = None
    thinking_level: ThinkingLevel | None = None
    critic_thinking_level: ThinkingLevel | None = None
    formatter_thinking_level: ThinkingLevel | None = None
    temperature: float | None = None
    critic_enabled: bool = True
    label: str | None = None

    def overridden_keys(self) -> list[str]:
        """Keys whose value actually differs from the baseline — value-based, so
        an explicit `critic_enabled: true` is not reported as an override."""
        return sorted(self.model_dump(exclude_defaults=True))

    def is_empty(self) -> bool:
        """True when this object changes nothing about the run."""
        return not self.overridden_keys()


_EMPTY = SandboxOverrides()


def _state_of(ctx: Any) -> Any:
    """Session state from either context flavour: `ReadonlyContext.state`
    (instruction providers) or `CallbackContext.state` (model callbacks), with
    `ctx.session.state` as the fallback for fakes and older shapes."""
    state = getattr(ctx, "state", None)
    if state is not None:
        return state
    session = getattr(ctx, "session", None)
    return getattr(session, "state", None)


def _explain(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", ())) or "sandbox"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


def _resolve_variant(name: str, role: Role) -> str:
    """Text for a registry variant name (`prompts/variants/`).

    An unknown name — or a build with no registry at all — is an ERROR, never a
    fall back to baseline: otherwise an A/B would compare the baseline against
    itself and report a tie.

    The import is local and the ImportError branch is kept even though the
    registry now ships in-tree: `adk deploy agent_engine` bundles the package,
    and a registry missing from the deployed image is precisely the failure this
    must not paper over — the run would look fine and compare two identical
    prompts.
    """
    try:
        from .prompts.variants import resolve_variant  # noqa: PLC0415 — see docstring
    except ImportError as exc:
        raise ValueError(
            f"sandbox: {role}_variant={name!r} requested, but the prompt-variant "
            "registry (gub_agent/prompts/variants) is not present in this build "
            f"— pass {role}_instruction inline instead. Refusing to fall back to "
            "the baseline prompt."
        ) from exc
    return resolve_variant(name, role)


def _check_prompt_size(text: str, field: str) -> None:
    size = len(text.encode("utf-8"))
    if size > MAX_PROMPT_BYTES:
        raise ValueError(
            f"sandbox: {field} is {size} bytes, over the {MAX_PROMPT_BYTES} byte "
            "cap — register the text as a prompt variant and pass the variant "
            "name instead of inline text."
        )


def _validate(overrides: SandboxOverrides) -> None:
    """Semantic checks that pydantic can't express legibly. Raises ValueError."""
    if overrides.model is not None and overrides.model not in config.SANDBOX_MODEL_ALLOWLIST:
        allowed = ", ".join(config.SANDBOX_MODEL_ALLOWLIST) or "(empty allowlist)"
        raise ValueError(
            f"sandbox: model {overrides.model!r} is not allowed. Allowed models: "
            f"{allowed} (set SANDBOX_MODEL_ALLOWLIST to widen)."
        )
    if overrides.model is not None and overrides.model not in config.SANDBOX_THINKING_LEVEL_MODELS:
        # A named thinking_level is a 3-series knob; other models return 400 for
        # it — and the baseline planners carry a named level, so "model only" is
        # not a valid override for such a model. Verified live (gemini-2.5-pro).
        executor_level = overrides.thinking_level or EXECUTOR_THINKING_LEVEL
        critic_level = overrides.critic_thinking_level or CRITIC_THINKING_LEVEL
        formatter_level = overrides.formatter_thinking_level or FORMATTER_THINKING_LEVEL
        offending = []
        if executor_level != "DYNAMIC":
            offending.append(f"thinking_level={executor_level}")
        if overrides.critic_enabled and critic_level != "DYNAMIC":
            offending.append(f"critic_thinking_level={critic_level}")
        if formatter_level != "DYNAMIC":
            offending.append(f"formatter_thinking_level={formatter_level}")
        if offending:
            why = (
                "(Claude thinks adaptively; Anthropic has no named levels)"
                if overrides.model.startswith("claude-")
                else "(Vertex answers 400 INVALID_ARGUMENT)"
            )
            raise ValueError(
                f"sandbox: model {overrides.model!r} does not accept a named thinking level "
                f"{why}, but {', '.join(offending)} would apply "
                "to this run. Set thinking_level, formatter_thinking_level (and "
                "critic_thinking_level, unless critic_enabled is false) to DYNAMIC for this "
                "model — or add the model to SANDBOX_THINKING_LEVEL_MODELS if it does accept "
                "named levels."
            )
    if overrides.temperature is not None and not 0.0 <= overrides.temperature <= 2.0:
        raise ValueError(
            f"sandbox: temperature {overrides.temperature} is outside 0.0..2.0. "
            "The value is NOT clamped — a clamp would silently distort an A/B."
        )
    if overrides.executor_instruction is not None:
        _check_prompt_size(overrides.executor_instruction, "executor_instruction")
    if overrides.critic_instruction is not None:
        _check_prompt_size(overrides.critic_instruction, "critic_instruction")
    if overrides.formatter_instruction is not None:
        _check_prompt_size(overrides.formatter_instruction, "formatter_instruction")
    # Variant names are resolved (and so validated) eagerly: a bad name must
    # fail the run, not the fourth model call halfway through an answer.
    for role in ("executor", "critic", "formatter"):
        name = getattr(overrides, f"{role}_variant")
        if name is not None:
            _resolve_variant(name, role)  # type: ignore[arg-type]


def read_overrides(state: Any) -> SandboxOverrides:
    """Validated overrides from session state.

    Returns an empty object when the sandbox is off or no `sandbox` key is
    present. Unknown keys warn and are ignored; an invalid value of a known key
    raises ValueError with a message that says what to do about it.
    """
    if not config.SANDBOX_ENABLED:
        # Prod engine: full ignore. No warn either — prod must not be noisy
        # about a key that was meant for somebody's sandbox run.
        return _EMPTY

    raw = None
    if state is not None:
        try:
            raw = state.get(STATE_KEY)
        except AttributeError:
            raw = None
    if raw is None:
        return _EMPTY
    if isinstance(raw, SandboxOverrides):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(
            f"sandbox: state[{STATE_KEY!r}] must be an object, got {type(raw).__name__}."
        )

    unknown = sorted(set(raw) - set(SandboxOverrides.model_fields))
    if unknown:
        logger.warning(
            "sandbox: ignoring unknown override key(s) %s — known keys: %s",
            ", ".join(unknown),
            ", ".join(sorted(SandboxOverrides.model_fields)),
        )

    try:
        overrides = SandboxOverrides.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"sandbox: invalid override — {_explain(exc)}") from exc

    _validate(overrides)
    return overrides


def _prompt_for(overrides: SandboxOverrides, role: Role) -> tuple[str | None, str]:
    """(override text, provenance source) for a role.

    Inline text wins over a variant name; when both are given we warn and keep
    both in the provenance, so a confused A/B is legible after the fact.
    """
    inline = getattr(overrides, f"{role}_instruction")
    variant = getattr(overrides, f"{role}_variant")
    if inline is not None and variant is not None:
        logger.warning(
            "sandbox: both %s_instruction and %s_variant=%r given — inline text wins",
            role,
            role,
            variant,
        )
        return inline, f"inline (ignored variant {variant})"
    if inline is not None:
        return inline, "inline"
    if variant is not None:
        return _resolve_variant(variant, role), variant
    return None, "baseline"


def sandbox_instruction(
    base_provider: Callable[[ReadonlyContext], str],
    role: Role,
) -> Callable[[ReadonlyContext], str]:
    """Wrap an InstructionProvider so a sandbox run can replace the prompt.

    With no override the base provider is called and its text returned
    unchanged. With one, the override text plus `current_date_note()` — the date
    block is not part of the experiment, and a prompt without an accurate "now"
    would fail every recency question for reasons unrelated to the variant.
    """

    def provider(ctx: ReadonlyContext) -> str:
        overrides = read_overrides(_state_of(ctx))
        text, _source = _prompt_for(overrides, role)
        if text is None:
            return base_provider(ctx)
        return f"{text}\n\n{current_date_note()}"

    return provider


def _thinking_config(level: str) -> genai_types.ThinkingConfig:
    """A FRESH ThinkingConfig for a level. Fresh matters: the planner assigns its
    own long-lived object into `llm_request.config.thinking_config`
    (`planners/built_in_planner.py:57-70`), so mutating that object in place
    would rewrite the module-level planner and leak the override into every
    later call on the engine."""
    if level == "DYNAMIC":
        # Matches build_thinking_planner(None) — unbounded, model decides.
        return genai_types.ThinkingConfig(
            thinking_budget=-1,
            include_thoughts=config.EMIT_THINKING,
        )
    return genai_types.ThinkingConfig(
        thinking_level=level,
        include_thoughts=config.EMIT_THINKING,
    )


def sandbox_before_model(callback_context: Any, llm_request: Any, *, role: Role) -> None:
    """ADK before_model_callback: apply the model / thinking / temperature
    overrides to THIS request. Returns None so the call proceeds (returning a
    response here would intercept it).

    Runs first in the callback chain by design: it only writes `model` and
    fields of `config`, while the round limiter may clear `config.tools` — no
    overlap in either direction.
    """
    overrides = read_overrides(_state_of(callback_context))
    if overrides.is_empty():
        return None

    if overrides.model is not None:
        llm_request.model = overrides.model

    if role == "executor":
        level = overrides.thinking_level
    elif role == "critic":
        level = overrides.critic_thinking_level
    else:
        level = overrides.formatter_thinking_level
    if level is not None or overrides.temperature is not None:
        cfg = getattr(llm_request, "config", None)
        if cfg is None:
            cfg = genai_types.GenerateContentConfig()
            llm_request.config = cfg
        if level is not None:
            cfg.thinking_config = _thinking_config(level)
        if overrides.temperature is not None:
            cfg.temperature = overrides.temperature
    return None


def resolved_config(overrides: SandboxOverrides) -> dict[str, Any]:
    """Flat, JSON-safe provenance for one run: what actually ran, not what was
    asked for. Flat on purpose — the batch runner (04/#33) writes one row per
    run and the UI (03/#32) renders one column per key.

    `*_prompt_sha256` is the sha256 prefix of the OVERRIDE text; it is None for
    a baseline prompt, whose identity is the deployed revision. The deliberate
    baseline arm of an A/B goes through the registry's `baseline` variant and so
    gets a hash like any other variant.
    """
    executor_text, executor_source = _prompt_for(overrides, "executor")
    critic_text, critic_source = _prompt_for(overrides, "critic")
    formatter_text, formatter_source = _prompt_for(overrides, "formatter")

    def _sha(text: str | None) -> str | None:
        if text is None:
            return None
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    return {
        "label": overrides.label,
        "model": overrides.model or config.GEMINI_MODEL,
        "temperature": overrides.temperature,
        "thinking_level": overrides.thinking_level or EXECUTOR_THINKING_LEVEL,
        "critic_thinking_level": overrides.critic_thinking_level or CRITIC_THINKING_LEVEL,
        "formatter_thinking_level": overrides.formatter_thinking_level or FORMATTER_THINKING_LEVEL,
        "critic_enabled": overrides.critic_enabled,
        "executor_prompt_source": executor_source,
        "executor_prompt_sha256": _sha(executor_text),
        "critic_prompt_source": critic_source,
        "critic_prompt_sha256": _sha(critic_text),
        "formatter_prompt_source": formatter_source,
        "formatter_prompt_sha256": _sha(formatter_text),
        "overridden_keys": overrides.overridden_keys(),
    }


class SandboxEcho(BaseAgent):
    """First sub-agent of the pipeline: records what this run resolved to.

    Emits an Event carrying a `state_delta` rather than writing through
    `callback_context.state` (see module docstring, finding 3) — a state_delta
    on an emitted event is the one write ADK commits to the session, and it also
    reaches the caller's event stream, which is what the batch runner reads.

    Emits NOTHING when the sandbox is off or the overrides are empty: an extra
    event would change the trace shape of an ordinary run, and "an empty
    `sandbox` is indistinguishable from today" is the invariant the epic rests
    on.
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        overrides = read_overrides(_state_of(ctx))
        if overrides.is_empty():
            return
        resolved = resolved_config(overrides)
        logger.info("sandbox: resolved %s", resolved)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(state_delta={RESOLVED_STATE_KEY: resolved}),
        )


sandbox_echo = SandboxEcho(name="sandbox_echo")
