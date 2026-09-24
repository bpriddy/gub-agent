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

Two wirings, chosen by CRITIC_PARALLEL (agent.py:build_deep_agent):

- serial (0):   executor → format_gate → CriticGate → escalator. The gate
                reads the formatter's payload from state, then runs the LLM.
- parallel (1): executor → ParallelAgent(format_gate, SpeculativeCritic)
                → CriticResolver → escalator. The critic LLM runs while the
                formatter does, and the resolver — named `critic_gate`, so
                its events are authored as before — makes CriticGate's
                decisions after the join, on the payload of THIS pass.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

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
from ..tenant import label_of
from .context_pruning import (
    strip_prior_turn_tool_parts,
    strip_prior_turn_tool_text,
    trim_to_recent_turns,
)

logger = logging.getLogger(__name__)


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


def _pass_reason_before_format(ctx: InvocationContext) -> str | None:
    """Why the critic LLM need not run, when the executor's pass alone says so.

    Neither check reads anything the format gate produces, so each holds
    before the gate has run exactly as it does after. That is what lets the
    parallel wiring skip the speculative critic call for these turns instead
    of spending it and dropping the verdict. The abstain PAYLOAD is the one
    check that has to wait for the gate (`CriticGate._abstain_verdict`).
    """
    # Sandbox: an experiment measuring the executor alone turns the critic
    # off. Same skip mechanism as the abstention pass — a verdict is still
    # written, so the loop still exits after one iteration instead of
    # running the executor twice for nothing.
    if not read_overrides(ctx.session.state).critic_enabled:
        return "sandbox: critic disabled"

    text = _last_executor_text(ctx)
    if text.strip().upper().startswith("NO_COMPANY_RECORDS"):
        return "Deterministic pass: exact NO_COMPANY_RECORDS abstention (no critic LLM run)."
    return None


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

    def _critic_name(self) -> str:
        """The critic LLM's name: the author of its verdicts, and of ours when
        a verdict has to read as the critic's (`_requery_event`)."""
        return self.sub_agents[0].name

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
            author=self._critic_name(),
            content=genai_types.Content(
                role="model", parts=[genai_types.Part(text=json.dumps(verdict))]
            ),
            actions=EventActions(state_delta={"critic_verdict": verdict}),
        )

    def _already_sent_back(self, ctx: InvocationContext) -> bool:
        """This turn's executor already had its from-memory retry."""
        critic = self._critic_name()
        for event in ctx.session.events:
            if event.invocation_id != ctx.invocation_id or event.author != critic:
                continue
            delta = event.actions.state_delta if event.actions else None
            verdict = (delta or {}).get("critic_verdict")
            if isinstance(verdict, dict) and verdict.get("sufficient") is False:
                return True
        return False

    def _abstain_verdict(self, ctx: InvocationContext, payload: object) -> Event | None:
        """The verdict an abstain payload settles in code, or None when the
        payload leaves it to the critic LLM."""
        # The abstention can also arrive as the answer contract's typed form:
        # the format gate (which runs before this gate — or beside the critic,
        # in the parallel wiring, with the payload read after the join) wrote
        # an AnswerPayload with kind="abstain". Same deterministic pass — an
        # abstention needs no information-sufficiency judge — PROVIDED the
        # executor looked this turn.
        #
        # Without a tool call, an abstain payload is not the executor's
        # abstention but the format gate refusing a draft written from memory:
        # it finds no evidence this pass and abstains. Seen live 2026-09-24
        # when a reader asked "whats new" a sixth time: the executor copied its
        # previous answer and the reader got NO_COMPANY_RECORDS. The bare
        # marker (`_pass_reason_before_format`) still passes without a tool
        # call — that one IS the executor's own "nothing to look up".
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
        if isinstance(payload, dict) and payload.get("kind") == "abstain":
            if not _executor_made_tool_call(ctx) and not self._already_sent_back(ctx):
                return self._requery_event(ctx)
            return self._pass_event(
                ctx,
                "Deterministic pass: abstain AnswerPayload (no critic LLM run).",
            )
        return None

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        reason = _pass_reason_before_format(ctx)
        if reason is not None:
            yield self._pass_event(ctx, reason)
            return
        verdict = self._abstain_verdict(ctx, ctx.session.state.get("answer_payload"))
        if verdict is not None:
            yield verdict
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

# The loop wires a GATE around the critic, never the raw LLM: deterministic
# passes in code, the critic LLM for everything else. Which gate — CriticGate,
# or SpeculativeCritic + CriticResolver — is decided where the loop is built
# (agent.py:build_deep_agent, by CRITIC_PARALLEL), because ADK gives an agent
# one parent and `critic_agent` can only be adopted by the one that runs it.


# ── the parallel wiring (CRITIC_PARALLEL) ────────────────────────────────────


@dataclass
class _Speculation:
    """What the critic LLM produced beside the format gate, held for the
    resolver: its events in order, or the exception that ended it."""

    events: list[Event] = field(default_factory=list)
    error: Exception | None = None


# Keyed on invocation id, like the evidence index's stores: the branch and the
# resolver are two agents of one invocation in one process, and session state
# is the very channel whose timing this hand-off exists to control.
_SPECULATIONS: OrderedDict[str, _Speculation] = OrderedDict()
_MAX_TRACKED = 256


def _hold(invocation_id: str, run: _Speculation) -> None:
    _SPECULATIONS[invocation_id] = run
    _SPECULATIONS.move_to_end(invocation_id)
    while len(_SPECULATIONS) > _MAX_TRACKED:
        _SPECULATIONS.popitem(last=False)


def _is_bookkeeping(event: Event) -> bool:
    """ADK's own agent-state events (resumable runs only) — nobody's output."""
    actions = event.actions
    return bool(actions and (actions.agent_state is not None or actions.end_of_agent))


def _this_pass_payload(ctx: InvocationContext, authors: tuple[str, ...]) -> dict | None:
    """The AnswerPayload the format gate produced in THIS pass, or None.

    Read off this pass's events, not `state["answer_payload"]`. State holds
    whatever payload was written LAST, and a gate that emits nothing — an
    executor that produced no text, which the gate answers with silence on
    purpose — leaves it holding an earlier one: the previous turn's, because
    session state outlives the turn. That payload's abstention would then
    settle this turn's verdict.

    The pass's gate events are the ones after the executor's last event: the
    resolver runs straight after the join, and the critic's events are still
    held, so the first event of this invocation by any other author is the
    edge of the pass.
    """
    for event in reversed(ctx.session.events):
        if event.invocation_id != ctx.invocation_id or _is_bookkeeping(event):
            continue
        if event.author not in authors:
            return None
        delta = event.actions.state_delta if event.actions else None
        payload = (delta or {}).get("answer_payload")
        if isinstance(payload, dict):
            return payload
    return None


class SpeculativeCritic(BaseAgent):
    """The critic LLM, run beside the format gate instead of after it.

    It is the format gate's sibling in a ParallelAgent (agent.py). ADK gives
    each branch its own conversation branch and shares session state
    (parallel_agent.py:_create_branch_ctx_for_sub_agent): this one sees every
    event that led to the fork — the executor's calls, results and text — and
    none of the formatter's. The critic judges only the executor's work
    (prompts/critic.py calls the formatter's JSON "pipeline plumbing"), so
    nothing it needs is missing.

    Its events are HELD, not yielded, for three reasons:

    - its verdict must not reach state before the resolver has decided. On an
      abstain payload CriticGate passes WITHOUT the LLM, and a speculative
      `sufficient: false` committed through output_key would be read by the
      escalator as a retry;
    - the bot restarts its streamed pass on a complete `sufficient: false`
      authored `critic`, and counts one critic iteration per such event
      (gub-gchat-bot src/agent/client.ts), so a verdict the resolver drops must
      never be streamed;
    - the stream keeps its order: payload, then verdict, then escalator.

    An exception is held as well, and re-raised by the resolver only if it
    needs the verdict. A turn the gate settles in code never ran the critic
    in the serial order, so a critic failure could not fail it; running the
    critic speculatively must not change that.

    The payload-independent passes (sandbox off, the bare NO_COMPANY_RECORDS
    marker) are known before the fork, so for them no call is made at all.
    Only an abstain payload costs a call whose verdict is then dropped.
    """

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        _SPECULATIONS.pop(ctx.invocation_id, None)  # never an earlier pass's
        if _pass_reason_before_format(ctx) is not None:
            return
        run = _Speculation()
        try:
            async for event in self.sub_agents[0].run_async(ctx):
                run.events.append(event)
        except Exception as exc:
            run.error = exc  # the resolver's to raise, if it needs the verdict
        _hold(ctx.invocation_id, run)
        return
        yield  # an async generator that holds everything it produces


class CriticResolver(CriticGate):
    """CriticGate's decisions, made after the join (the parallel wiring).

    Named `critic_gate` in the tree, so a deterministic pass is authored as it
    always was, and the verdicts it relays keep the critic's own author. The
    checks run in CriticGate's order — the payload-independent passes, then
    the abstain payload of THIS pass (`_this_pass_payload`), and only then the
    critic's verdict. A held verdict the checks overrule is dropped without
    reaching the stream or state, exactly as if the critic had never run.
    """

    #: The critic LLM's name — the author of its verdicts and of a re-query.
    critic_name: str = "critic"
    #: The authors of this pass's payload: the format gate and its formatter.
    payload_authors: tuple[str, ...] = ()

    def _critic_name(self) -> str:
        return self.critic_name

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        held = _SPECULATIONS.pop(ctx.invocation_id, None)
        reason = _pass_reason_before_format(ctx)
        verdict = (
            self._pass_event(ctx, reason)
            if reason is not None
            else self._abstain_verdict(ctx, _this_pass_payload(ctx, self.payload_authors))
        )
        if verdict is not None:
            if held is not None:
                # The measurable price of running the critic beside the gate:
                # count these against critic calls for the wasted share.
                logger.info(
                    "critic_speculation: verdict discarded (inv=%s) tenant=%s — %s",
                    ctx.invocation_id,
                    label_of(ctx),
                    verdict.actions.state_delta["critic_verdict"]["reason"],
                )
            yield verdict
            return

        if held is None:
            # The branch holds a result whenever it had a call to make, so this
            # is a pass it never ran (a resumed invocation, or an entry evicted
            # from the store). Fall back to the serial order rather than write
            # a verdict nobody reached: run the critic now, after the payload.
            logger.warning(
                "critic_speculation: nothing held (inv=%s) tenant=%s — running the critic serially",
                ctx.invocation_id,
                label_of(ctx),
            )
            critic = self.root_agent.find_agent(self.critic_name)
            if critic is None:
                raise RuntimeError(f"critic_gate: no agent named {self.critic_name!r} in the tree")
            async for event in critic.run_async(ctx):
                yield event
            return

        if held.error is not None:
            raise held.error
        for event in held.events:
            yield event
