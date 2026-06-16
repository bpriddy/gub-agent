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
it can't verify. It only checks the SHAPE of the executor's behavior
against known failure modes (counting in the LLM, wrong tool choice,
missing multi-entity decomposition, leaked IDs, ignored statusMarkdown).

This is the load-bearing critic-before-commit pattern from the Agentic
RAG architecture; we keep just this one specialist instead of the full
planner/rewriter/fanout fleet because at our scale it's the one piece
that genuinely improves dependability.
"""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from pydantic import BaseModel, Field

from ..config import GEMINI_MODEL
from ..instruction_utils import with_current_date


CRITIC_INSTRUCTION = """
You are a quality-control critic for a GUB AI agent that answers questions
about an agency's people, clients, and campaigns.

The executor agent has just produced a response to the user's question.
Read the conversation, the tool calls the executor made and their results,
and the executor's most recent response. Evaluate it on TWO axes. BOTH
must pass for the answer to be sufficient.

=== AXIS 1 — INFORMATION SUFFICIENCY ===
Did the executor's tool calls gather ENOUGH to answer THIS question?
- Was every entity the question refers to actually looked up via a tool —
  not assumed, not remembered from an earlier turn, not invented?
- Were the right tools used? Filtering, counting, sorting, ranking, and
  aggregating go through `org_query`, not list-and-reason. Fuzzy name
  lookups use `org_query` with the `similar_to` operator.
- For multi-part or multi-entity questions, was each part queried (chained
  `org_query` calls using the `in` operator as the join)?
- If the executor made NO tool calls but the question needs data, the
  information is automatically insufficient.
- If any fact needed to answer is absent from every tool result, the
  information is insufficient — say exactly what still needs to be queried.

=== AXIS 2 — ANSWER SATISFACTION ===
Given what was retrieved, does the synthesized answer deliver the kind of
CLOSURE this question sought — judged on intent, not literal words?
- CLOSURE: every question seeks a kind of resolution. A FACT question
  wants a value. An ASSESSMENT question ("how is X?", "where are we on X?",
  "should we worry about X?") wants a VERDICT up front (healthy / mixed /
  needs attention) plus the few drivers that earn it — NOT a catalog or a
  generic summary. An EXPLORATORY question wants a shaped shortlist. An
  answer that delivers the wrong kind of closure fails this axis even if
  every fact in it is correct. For assessment questions specifically: no
  clear verdict, or a data-dump in place of a verdict, is a fail.
- RECENCY: today's date is in your context. Did the answer respect time?
  Surfacing something that ended or went quiet long ago as if it were
  current, or burying recent movement under stale history, fails. Current-
  state answers must weight recent and currently-active items.
- GROUNDING (hard fail): every account, campaign, or person named in the
  answer must appear verbatim in a tool result from this turn. Invented or
  "helpfully completed" names — e.g. answering "Chevrolet" when the tool
  returned "chevy", or naming accounts when no query was run — fail this
  axis, always, with no exceptions.
- Completeness: does it address what was actually asked, fully, without
  drifting, hedging, or refusing without cause?
- Correct computation: counts, totals, and rankings come from `org_query`
  results (the `total` field, the sorted rows) — never reasoned over a list.
- Form: no UUIDs, internal IDs, or `_sources` in the prose; when a tool
  result includes `statusMarkdown`, it is rendered verbatim with a brief
  lead-in, not paraphrased.

Do NOT second-guess data VALUES you can't verify — assume the numbers and
names INSIDE tool results are correct. You are judging whether enough was
retrieved and whether the answer is faithful to it, not re-checking the DB.

=== OUTPUT ===
- `info_sufficient`: true only if Axis 1 passed (enough was retrieved).
- `answer_satisfies`: true only if Axis 2 passed (grounded, complete,
  correctly computed and formed).
- `sufficient`: true ONLY if info_sufficient AND answer_satisfies are both
  true.
- `reason`: one short sentence.
- `feedback`: if not sufficient, name which axis failed and give concrete
  next-step guidance — for an information gap, exactly what to query next
  (e.g. "query org_query with similar_to 'chevy' on accounts"); for an
  answer problem, exactly what to fix. Leave empty when sufficient.

Be strict but not pedantic about minor formatting. But a missing-
information gap (Axis 1) or an ungrounded / hallucinated entity (Axis 2
grounding) is never "small" — either one forces sufficient = false.
""".strip()


class CriticVerdict(BaseModel):
    """Critic's two-axis evaluation of the executor's response."""

    info_sufficient: bool = Field(
        description=(
            "Axis 1: did the executor's tool calls gather enough to answer "
            "this question? False if entities weren't looked up, the wrong "
            "tool was used, a multi-part question wasn't fully queried, or no "
            "tool calls were made when data was needed."
        ),
    )
    answer_satisfies: bool = Field(
        description=(
            "Axis 2: does the synthesized answer satisfy this question — "
            "grounded (every named entity appears in a tool result), "
            "complete, correctly computed, and correctly formed?"
        ),
    )
    sufficient: bool = Field(
        description=(
            "True ONLY if info_sufficient AND answer_satisfies are both true. "
            "This gates the loop: false triggers a retry."
        ),
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


critic_agent = LlmAgent(
    model=GEMINI_MODEL,
    name="critic",
    # InstructionProvider — same deterministic current-date injection as the
    # executor, so the critic can judge recency against an accurate "now".
    instruction=with_current_date(CRITIC_INSTRUCTION),
    output_schema=CriticVerdict,
    output_key="critic_verdict",
)


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
