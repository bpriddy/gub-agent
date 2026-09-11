"""
critic.py — quality-control specialist for the GUB pipeline.

Runs after the executor on every iteration of the LoopAgent. Reads the
conversation + executor's response and emits a structured verdict
(sufficient, reason, feedback). Verdict drives loop exit and retry:

- sufficient=true  → escalator_agent triggers actions.escalate=True →
                     LoopAgent exits the loop early
- sufficient=false → LoopAgent runs the next iteration; the executor's
                     prompt reads critic_verdict.feedback and addresses
                     the issue

The critic is deliberately narrow: it doesn't second-guess data values
it can't verify, and — since the answer contract (blend 03) — it doesn't
judge the answer's shape or grounding either: those are enforced in code
by the format gate (agents/format_gate.py). What remains is the one
judgement that needs an LLM: information sufficiency (wrong tool choice,
missing multi-entity decomposition, no tool call where data was needed).

This is the load-bearing critic-before-commit pattern from the Agentic
RAG architecture; we keep just this one specialist instead of the full
planner/rewriter/fanout fleet because at our scale it's the one piece
that genuinely improves dependability.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event, EventActions
from pydantic import BaseModel, Field

from ..config import AGENT_NAME, build_model, build_thinking_planner
from ..instruction_utils import current_date_note
from ..prompts import CRITIC_INSTRUCTION
from ..sandbox import (
    CRITIC_THINKING_LEVEL,
    read_overrides,
    sandbox_before_model,
    sandbox_instruction,
)
from .context_pruning import strip_prior_turn_tool_parts


class CriticVerdict(BaseModel):
    """Critic's decision on the executor's response.

    One real axis since blend 03 — information sufficiency; the old Axis 2
    fields remain so the wire shape (and everything reading it: the bot, the
    batch runner, the debug client) is unchanged. The critic reasons through
    its checks (was a tool called, does each question-entity map to a covering
    call) in thinking tokens — see the instruction — and emits only this
    decision, not the intermediate working.
    """

    info_sufficient: bool = Field(
        description=(
            "Axis 1: did the executor's tool calls gather enough to answer "
            "this question? False if any entity the question needs wasn't "
            "retrieved, the wrong tool/operation was used, a multi-part "
            "question wasn't fully queried, or no tool was called when data "
            "was needed."
        ),
    )
    answer_satisfies: bool = Field(
        description=(
            "Set EQUAL to info_sufficient. Answer shape and grounding — the "
            "old Axis 2 — are enforced in code by the format gate "
            "(agents/format_gate.py); the field remains for wire compatibility."
        ),
    )
    sufficient: bool = Field(
        description=("Same value as info_sufficient. This gates the loop: false triggers a retry."),
    )
    reason: str = Field(
        description="One short sentence explaining the verdict.",
    )
    feedback: str = Field(
        default="",
        description=(
            "If sufficient=false, specific actionable guidance for the "
            "executor's retry. Empty when sufficient=true."
        ),
    )


def _executor_made_tool_call(ctx: ReadonlyContext) -> bool:
    """Deterministic: did the executor emit ANY function_call this run?

    The executor's tool calls are real events in the session — whether one was
    made is a FACT, not a judgement. We compute it here and hand it to the
    critic, instead of asking the LLM to read it off the transcript.
    """
    for event in ctx.session.events:
        if event.author != AGENT_NAME:
            continue
        parts = event.content.parts if event.content and event.content.parts else []
        if any(getattr(part, "function_call", None) for part in parts):
            return True
    return False


def _critic_instruction(ctx: ReadonlyContext) -> str:
    """InstructionProvider — base critic prompt + current date + the
    deterministically-computed 'was a tool called this turn' fact."""
    if _executor_made_tool_call(ctx):
        tool_fact = (
            "TOOL CALL THIS TURN: yes — the executor made at least one tool "
            "call, so the data path was exercised. Judge sufficiency on what "
            "the results actually cover."
        )
    else:
        tool_fact = (
            "TOOL CALL THIS TURN: no — the executor made NO tool call. Any "
            "claim about a specific account, campaign, person, count, or status "
            "is therefore from memory, so info_sufficient is FALSE — unless the "
            "question genuinely needed no data (a greeting, 'what can you do?')."
        )
    return (
        f"{CRITIC_INSTRUCTION}\n\n"
        f"{current_date_note()}\n\n"
        "## Deterministic facts (computed for you — not your judgement)\n"
        f"{tool_fact}"
    )


def _critic_before_model(callback_context, llm_request):
    """Chain the critic's model-level hooks: sandbox overrides first (model,
    critic thinking level, temperature — a no-op without state["sandbox"]),
    then the prior-turn tool-payload pruning the critic has always had."""
    sandbox_before_model(callback_context, llm_request, role="critic")
    return strip_prior_turn_tool_parts(callback_context, llm_request)


critic_agent = LlmAgent(
    # Retry-with-backoff on 429/5xx, same as the executor (config.build_model).
    model=build_model(),
    name="critic",
    # InstructionProvider — injects the current date AND the deterministic
    # "was any tool called this turn" fact (computed from the event stream),
    # so the critic reasons FROM a given fact instead of re-deriving it.
    # Wrapped for the sandbox: state["sandbox"].critic_instruction (or
    # .critic_variant) replaces the critic's prompt for that run only.
    instruction=sandbox_instruction(_critic_instruction, role="critic"),
    # The critic REASONS through its remaining checks (entity↔call coverage,
    # closure, grounding, recency) in thinking tokens, then emits only the
    # two-axis decision below — a bare output_schema leaves no room to reason.
    # Thinking capped at LOW: it's a checklist judge, and unbounded thinking
    # measured 13-16s per turn (~29% of total latency) with no observed
    # quality benefit over a short deliberation. Executor keeps dynamic.
    # Thought summaries are emitted for debugging when EMIT_THINKING is set.
    # (CRITIC_THINKING_LEVEL, so the sandbox provenance can't drift from what
    # actually runs; a sandbox run overrides it per call in _critic_before_model.)
    planner=build_thinking_planner(thinking_level=CRITIC_THINKING_LEVEL),
    output_schema=CriticVerdict,
    output_key="critic_verdict",
    # Same pruning as the executor: the critic verifies THIS turn's grounding
    # against THIS turn's tool results — prior turns' payloads are noise.
    before_model_callback=_critic_before_model,
)


def _last_executor_text(ctx: InvocationContext) -> str:
    """The executor's most recent visible response text (thoughts excluded)."""
    for event in reversed(ctx.session.events):
        if event.author != AGENT_NAME:
            continue
        parts = event.content.parts if event.content and event.content.parts else []
        text = "".join(
            part.text
            for part in parts
            if getattr(part, "text", None) and not getattr(part, "thought", False)
        )
        if text.strip():
            return text
    return ""


class CriticGate(BaseAgent):
    """Deterministic pre-check in front of the critic LLM.

    The critic's instruction already auto-passes exact non-answers (the
    NO_COMPANY_RECORDS abstention marker) — but reaching that verdict cost a
    full LLM pass with thinking on every abstention turn (~observed 60-119s
    turns whose entire output is one marker word). The marker is detectable
    with a string check, so: when the executor's response IS the abstention,
    write the sufficient verdict to state directly and skip the critic LLM
    entirely. Every other response runs the critic exactly as before —
    zero change to answer quality by construction.

    The marker check mirrors the bot's own gubAbstained detection
    (trim → upper → startswith).

    The gate is also where `state["sandbox"].critic_enabled=false` takes effect
    (sandbox.py): the critic is construction-time wiring, so turning it off for
    one run means skipping it here rather than rebuilding the pipeline.
    """

    def _pass_event(self, ctx: InvocationContext, reason: str) -> Event:
        """The sufficient verdict written WITHOUT running the critic LLM.

        Shape matches CriticVerdict exactly — the escalator reads `sufficient`
        off it to exit the loop, so a missing key would silently cost a second
        executor pass.
        """
        return Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            actions=EventActions(
                state_delta={
                    "critic_verdict": {
                        "info_sufficient": True,
                        "answer_satisfies": True,
                        "sufficient": True,
                        "reason": reason,
                        "feedback": "",
                    }
                }
            ),
        )

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        # Sandbox: an experiment measuring the executor alone turns the critic
        # off. Same skip mechanism as the abstention pass — a verdict is still
        # written, so the loop still exits after one iteration instead of
        # running the executor twice for nothing.
        if not read_overrides(ctx.session.state).critic_enabled:
            yield self._pass_event(ctx, "sandbox: critic disabled")
            return

        text = _last_executor_text(ctx)
        if text.strip().upper().startswith("NO_COMPANY_RECORDS"):
            yield self._pass_event(
                ctx,
                "Deterministic pass: exact NO_COMPANY_RECORDS abstention (no critic LLM run).",
            )
            return

        # The abstention can also arrive as the answer contract's typed form:
        # the format gate (which runs before this gate) wrote an
        # AnswerPayload with kind="abstain" into state. Same deterministic
        # pass — an abstention needs no information-sufficiency judge.
        payload = ctx.session.state.get("answer_payload")
        if isinstance(payload, dict) and payload.get("kind") == "abstain":
            yield self._pass_event(
                ctx,
                "Deterministic pass: abstain AnswerPayload (no critic LLM run).",
            )
            return
        async for event in self.sub_agents[0].run_async(ctx):
            yield event


class EscalateIfSufficient(BaseAgent):
    """Loop-exit sub-agent.

    ADK's LoopAgent treats `actions.escalate=True` on any sub-agent's
    emitted event as the signal to exit the loop. An LlmAgent with
    structured output can't emit that directly, so this thin BaseAgent
    reads the critic's verdict from session state and emits the escalate
    signal when sufficient.
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        verdict = ctx.session.state.get("critic_verdict")
        sufficient = False
        if isinstance(verdict, dict):
            sufficient = bool(verdict.get("sufficient"))
        elif verdict is not None:
            sufficient = bool(getattr(verdict, "sufficient", False))

        if sufficient:
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                actions=EventActions(escalate=True),
            )


escalator_agent = EscalateIfSufficient(name="loop_escalator")

# The loop wires the GATE (not the raw critic): deterministic abstention pass,
# critic LLM for everything else.
critic_gate = CriticGate(name="critic_gate", sub_agents=[critic_agent])
