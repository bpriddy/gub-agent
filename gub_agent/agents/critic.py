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

import json
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event, EventActions
from google.genai import types as genai_types
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
from .context_pruning import (
    strip_prior_turn_tool_parts,
    strip_prior_turn_tool_text,
    trim_to_recent_turns,
)


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

    "This run" is THIS invocation, and the filter is what makes it so. Session
    events are never trimmed, and a scan of all of them answered "yes" as soon
    as ANY earlier turn had called a tool — which switched off the critic's
    "you answered from memory" guard for the rest of the session. In a thread
    that lives for 200 turns that is close to certain. Both executor passes of
    one turn share an invocation id, so a retry still counts the first pass's
    calls, as it always has.
    """
    for event in ctx.session.events:
        if event.invocation_id != ctx.invocation_id:
            continue
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
    then the conversation window, then the prior-turn pruning.

    memory-00 §3.5 left the critic unwindowed on purpose, and its reason was
    right for the counter it had in mind: ADK feeds the critic the executor's
    work as foreign-context contents, so with two prior turns its request
    already carries ~18 _has_user_text boundaries, and a window counting THOSE
    at N=5 would cut inside the current turn and remove the user's question.
    The window never counted them — `_turn_starts` excludes foreign context —
    so the cut lands at or before the current question, and every tool result
    of THIS turn survives: the evidence the critic exists to judge. Pinned by
    tests/unit/test_router_critic_context.py against ADK's own content builder.

    It has to be windowed now. A thread keeps up to 200 turns, and the critic
    re-read every one of them — including, until `strip_prior_turn_tool_text`,
    every earlier turn's full tool results, which ADK hands it as text that
    `strip_prior_turn_tool_parts` cannot see. The critic judges this turn; the
    earlier turns' payloads were only ever cost."""
    sandbox_before_model(callback_context, llm_request, role="critic")
    trim_to_recent_turns(callback_context, llm_request, role="critic")
    strip_prior_turn_tool_parts(callback_context, llm_request)
    return strip_prior_turn_tool_text(callback_context, llm_request)


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
    # Same bounds as the executor — the per-session window, then prior-turn
    # pruning: the critic verifies THIS turn's grounding against THIS turn's
    # tool results — prior turns' payloads are noise.
    before_model_callback=_critic_before_model,
)


def _last_executor_text(ctx: InvocationContext) -> str:
    """The executor's most recent visible response text (thoughts excluded),
    from THIS invocation.

    Unfiltered, a turn whose executor produced no text — the run died inside
    the engine — read the PREVIOUS turn's answer instead. Two callers act on
    that: the format gate would format and ship the earlier answer as this
    turn's (its own comment says an empty draft must emit nothing, "so the
    trace shows the failure instead of a fabricated answer"), and CriticGate
    would skip the critic whenever that earlier answer happened to be a
    NO_COMPANY_RECORDS abstention. Both got likelier with every turn a session
    lives, and a thread lives long.
    """
    for event in reversed(ctx.session.events):
        if event.invocation_id != ctx.invocation_id:
            continue
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
    (trim → upper → startswith). An abstain PAYLOAD with no tool call this
    turn is a draft from memory: the gate sends it back for one re-query
    itself — see the comment at the check.

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

    def _requery_event(self, ctx: InvocationContext) -> Event:
        """The insufficient verdict for a draft written from memory, WITHOUT
        running the critic LLM.

        Authored as the critic and carrying the verdict as text, like the
        critic LLM's own event, because both readers key on that: the
        executor's retry sees it as "[critic] said: {...}" and reads the
        feedback from there (nothing injects state into its prompt), and the
        bot counts a critic iteration and restarts the streamed pass on a
        `sufficient: false` from author "critic" — pass one's prose is the
        from-memory draft and must not stay in the bubble.
        """
        verdict = {
            "info_sufficient": False,
            "answer_satisfies": False,
            "sufficient": False,
            "reason": (
                "Deterministic: no tool call this turn and the format gate "
                "abstained, so the draft was written from memory (no critic LLM run)."
            ),
            "feedback": (
                "You made NO tool call this turn, so your draft repeated earlier "
                "answers from memory and nothing in it could be grounded. Query GUB "
                "now for THIS question - do not reuse prior turns - and answer only "
                "from what the tools return."
            ),
        }
        return Event(
            invocation_id=ctx.invocation_id,
            author=self.sub_agents[0].name,
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=json.dumps(verdict))]
            ),
            actions=EventActions(state_delta={"critic_verdict": verdict}),
        )

    def _already_sent_back(self, ctx: InvocationContext) -> bool:
        """This turn's executor already had its from-memory retry."""
        critic = self.sub_agents[0].name
        for event in ctx.session.events:
            if event.invocation_id != ctx.invocation_id or event.author != critic:
                continue
            delta = event.actions.state_delta if event.actions else None
            verdict = (delta or {}).get("critic_verdict")
            if isinstance(verdict, dict) and verdict.get("sufficient") is False:
                return True
        return False

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
        # pass — an abstention needs no information-sufficiency judge —
        # PROVIDED the executor looked this turn.
        #
        # Without a tool call, an abstain payload is not the executor's
        # abstention but the format gate refusing a draft written from memory:
        # it finds no evidence this pass and abstains. Seen live 2026-09-24
        # when a reader asked "whats new" a sixth time: the executor copied its
        # previous answer and the reader got NO_COMPANY_RECORDS. The bare
        # marker above still passes without a tool call — that one IS the
        # executor's own "nothing to look up".
        #
        # Such a draft is sent back for a re-query HERE, in code, not by the
        # critic LLM. The first fix (e48bec5) ran the critic on it and relied on
        # its "TOOL CALL THIS TURN: no" guard; live, the critic read the earlier
        # turns' identical answers in its window and passed the copy twice out
        # of two ("the executor gathered sufficient recent company data").
        # "No tool call and nothing citable" is a fact, not a judgement — the
        # same reasoning that computes the tool-call fact for the critic.
        #
        # Once per turn: if the retry ALSO answers from memory (a question
        # about the conversation itself, "what did I just ask?"), the abstain
        # passes as it always did rather than spending a verdict on a loop that
        # has no iteration left.
        payload = ctx.session.state.get("answer_payload")
        if isinstance(payload, dict) and payload.get("kind") == "abstain":
            if not _executor_made_tool_call(ctx) and not self._already_sent_back(ctx):
                yield self._requery_event(ctx)
                return
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
