"""
format_gate.py — deterministic gate around the formatter (blend 03).

Runs the formatter, checks its AnswerPayload IN CODE, retries with exact
feedback, and falls back to a template render — the critic's old
`answer_satisfies` axis (shape, grounding) turned from an LLM judgement into
checks:

1. `citations ⊆ evidence_index` — "unknown citation".
2. Every number in the answer (citation markers stripped) appears in some
   evidence value — "ungrounded number".
3. Every entity-name-shaped token appears in the index — "ungrounded entity";
   this is the critic's old HARD FAIL ("Chevrolet" for a table that says
   "chevy") moved into code.
4. A payload that fails pydantic validation (filler, budgets, table shape —
   `schemas/answer.py`) surfaces as a ValidationError from the formatter's
   `output_schema` and is treated the same way: feedback + retry.

On failure the gate writes `state["format_feedback"]` as a state_delta event
(trace-visible) and re-runs the formatter, at most twice; after that it emits
a template render authored by the gate itself (executor's first line as the
headline, the top evidence rows as bullets with their ids) — the turn always
ends with SOME payload for the bot.

Shaped like `CriticGate` (`agents/critic.py`), and deliberately NOT a nested
LoopAgent: ADK's LoopAgent yields sub-agent events upward and exits on
`event.actions.escalate` (`loop_agent.py:114-122`), so an inner loop's
escalate would break the OUTER pipeline loop and skip the critic. This gate
never sets `escalate`.

`kind="abstain"` and `kind="clarify"` skip checks 2-3 — an abstention grounds
nothing and a clarification question legitimately echoes the user's own
(unresolved) entity name. Check 1 still applies: a non-answer citing evidence
it does not have is a bug, not a style choice.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types as genai_types
from pydantic import ValidationError

from ..schemas import AnswerPayload, BulletBlock, Fact, TextBlock
from ..schemas.answer import HEADLINE_MAX_WORDS, TEXT_BLOCK_MAX_WORDS
from .critic import _last_executor_text
from .evidence_index import (
    answer_draft,
    evidence_index,
    set_format_feedback,
    set_formatter_brief,
)
from .formatter import ANSWER_STATE_KEY, formatter_agent

logger = logging.getLogger(__name__)

# Initial run + at most two feedback-driven retries, then the template.
MAX_FORMAT_ATTEMPTS = 3

# How many evidence rows the template render lists.
TEMPLATE_ROWS = 7

# `[evidence_id]` citation markers (markdown links `[t](u)` survive).
_CITE_MARKER_RE = re.compile(r"\[[^\]]+\](?!\()")
_NUMBER_RE = re.compile(r"\d[\d,\.]*")
# A capitalized word (Latin or Cyrillic), possibly hyphen/apostrophe-joined.
_CAP_RUN_RE = re.compile(r"[A-ZА-ЯЁ][\w'’&-]*(?:[ \t][A-ZА-ЯЁ][\w'’&-]*)*")

# Words that capitalize for reasons other than being an entity name. Stripped
# from the EDGES of a candidate run; a run reduced to nothing is not a name.
_STOPWORDS = frozenset(
    """
    the a an in on at of to for and or but with from by no not yes it its this
    that these those all any most more some last next new recent recently
    currently today now overall however note also see verdict status live
    ended awarded pitch lost active inactive prospect campaign campaigns
    account accounts staff piece pieces idea ideas budget total gub ai ok usd
    january february march april may june july august september october
    november december monday tuesday wednesday thursday friday saturday sunday
    q1 q2 q3 q4
    в на и не нет да это за по от до у с о как что кто когда сейчас недавно
    итог статус кампания кампании аккаунт бюджет январь февраль март апрель
    май июнь июль август сентябрь октябрь ноябрь декабрь
    """.split()
)


def _strip_markers(text: str) -> str:
    return _CITE_MARKER_RE.sub(" ", text)


def _prose_and_cells(payload: AnswerPayload, index: dict[str, dict]) -> tuple[str, str]:
    """(prose, table cells) to ground — citation markers stripped, cells that
    ARE evidence ids dropped (the table's source-id column cites, it doesn't
    claim; its digits would trip the number check)."""
    prose: list[str] = [payload.headline]
    cells: list[str] = []
    for block in payload.blocks:
        if block.kind == "text":
            prose.append(block.text)
        elif block.kind == "bullets":
            prose.extend(block.items)
        else:
            for row in block.rows:
                cells.extend(cell for cell in row if cell not in index)
    return _strip_markers(" \n ".join(prose)), _strip_markers(" \n ".join(cells))


def _evidence_blob(index: dict[str, dict]) -> str:
    return " \n ".join(str(entry.get("value", "")) for entry in index.values()).lower()


def _numbers_of(text: str) -> set[str]:
    return {match.group(0).rstrip(".,").replace(",", "") for match in _NUMBER_RE.finditer(text)}


def _entity_runs(text: str) -> list[str]:
    """Candidate entity names: runs of capitalized words, stopwords trimmed
    from the edges, single sentence-openers skipped."""
    runs: list[str] = []
    for match in _CAP_RUN_RE.finditer(text):
        words = match.group(0).split()
        while words and words[0].lower() in _STOPWORDS:
            words = words[1:]
        while words and words[-1].lower() in _STOPWORDS:
            words = words[:-1]
        if not words:
            continue
        if len(words) == 1:
            # A lone capitalized word that merely opens a sentence (or a
            # bullet — the newline counts as a boundary) is grammar, not
            # necessarily a name.
            before = text[: match.start()].rstrip(" \t")
            if not before or before[-1] in ".!?:;•-—\n":
                continue
        runs.append(" ".join(words))
    return runs


def gate_problems(payload: AnswerPayload, index: dict[str, dict]) -> list[str]:
    """The deterministic checks — empty list means the payload passes."""
    problems: list[str] = []

    unknown = [c for c in payload.citations if c not in index]
    if unknown:
        problems.append(
            f"unknown citation: {', '.join(sorted(unknown))} — cite only ALLOWED_EVIDENCE ids"
        )

    if payload.kind != "answer":
        return problems  # nothing to ground in an abstention / clarification

    prose, cells = _prose_and_cells(payload, index)
    blob = _evidence_blob(index)
    blob_numbers = blob.replace(",", "")

    ungrounded_numbers = sorted(
        n for n in _numbers_of(prose) | _numbers_of(cells) if n not in blob_numbers
    )
    if ungrounded_numbers:
        problems.append(
            f"ungrounded number: {', '.join(ungrounded_numbers)} — every number must "
            "appear verbatim in a tool result; copy it, never reformat or recompute"
        )

    ungrounded_entities = sorted(
        {
            run
            for run in _entity_runs(prose) + _entity_runs(cells)
            if run.lower() not in blob and not all(w.lower() in blob for w in run.split())
        }
    )
    if ungrounded_entities:
        problems.append(
            f"ungrounded entity: {', '.join(ungrounded_entities)} — every named entity must "
            "appear verbatim in a tool result from this turn; use the exact name the tool "
            "returned, never complete or translate it"
        )
    return problems


# ── the brief (the formatter's model input) ──────────────────────────────────


def compose_brief(executor_text: str, index: dict[str, dict], feedback: str) -> str:
    lines = [
        "EXECUTOR ANSWER (render this — do not add facts):",
        executor_text.strip(),
        "",
        "ALLOWED_EVIDENCE (the ONLY citable ids, with their values):",
    ]
    if index:
        for evidence_id, entry in index.items():
            lines.append(f"- {evidence_id} = {entry.get('value', '')}")
    else:
        lines.append("(none — this turn retrieved nothing citable)")
    if feedback:
        lines += ["", f"FORMAT_FEEDBACK (your previous payload was rejected): {feedback}"]
    return "\n".join(lines)


# ── template fallback ────────────────────────────────────────────────────────


def _shorten(text: str, max_chars: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


# Words a truncated headline must not end on — they promise a continuation
# that the ellipsis cannot supply.
_DANGLING = frozenset(
    "a an the and or but with of to in on for from at by as is are was were has have had"
    " its their his her our that this these those than then over under into".split()
)


def _headline_from(line: str) -> str:
    """A headline that reads as a finished thought.

    Cutting at word `HEADLINE_MAX_WORDS` mid-clause is what produced live
    answers ending "...truck portfolio, with a" (2026-09-11). So: the line's
    first SENTENCE when it fits the budget, otherwise a hard truncation that at
    least announces itself with an ellipsis.
    """
    sentence = re.match(r"\s*(.+?[.!?])(?:\s|$)", line)
    candidate = sentence.group(1).strip() if sentence else line.strip()
    words = candidate.split()
    if words and len(words) <= HEADLINE_MAX_WORDS:
        return " ".join(words)

    head = line.split()[:HEADLINE_MAX_WORDS]
    # Prefer a clause boundary inside the budget over the budget's own edge:
    # cutting at word 20 landed a live answer on "...portfolio, with a".
    for i in range(len(head) - 1, 0, -1):
        if head[i].endswith((",", ";", ":")):
            head = head[: i + 1]
            head[-1] = head[-1].rstrip(",;:")
            break
    else:
        # No clause break — at least do not end on a dangling function word.
        while len(head) > 1 and head[-1].lower() in _DANGLING:
            head.pop()
    return " ".join(head) + " …"


def template_payload(executor_text: str, index: dict[str, dict]) -> AnswerPayload:
    """The deterministic render used after the formatter failed twice.

    The body is the EXECUTOR'S PROSE, not the evidence. The previous version
    listed the top evidence rows as bullets, and an evidence `value` is the
    tool's response row, so a real answer reached users as a chopped sentence
    over a list of raw JSON (`{"id": "...", "accountId": ...` — 2026-09-11).
    The executor's text in those turns was perfectly good prose; the fallback
    threw it away. Evidence still travels, in `citations` (the bot's
    attribution chips) and `facts` (the conflict filter's input, blend 05) —
    both machine-read, neither rendered as body text.

    Bullets of evidence remain only as a last resort, for a turn that produced
    no prose at all: something cited beats nothing.

    Built with model_construct: the template quotes the executor and the tools
    verbatim, so the contract's style validators (filler, word budgets) must
    not be able to reject the fallback itself.
    """
    lines = [line.strip() for line in executor_text.splitlines() if line.strip()]
    # The executor writes Markdown; a headline is styled by the renderer, so
    # emphasis/heading markers here would render as literal asterisks.
    first_line = lines[0].strip("*#_ ").strip() if lines else ""
    headline = _headline_from(first_line) if first_line else "Company-records answer"

    # The body continues AFTER what the headline already said, so the reader is
    # not handed the same sentence twice: the later lines when there are any,
    # else whatever is left of the single line once the headline took its first
    # sentence. `kind="answer"` needs at least one block, so when nothing is
    # left the line itself is repeated — mild duplication beats both an invalid
    # payload and a wall of JSON.
    if len(lines) > 1:
        rest = " ".join(lines[1:])
    else:
        consumed = headline.rstrip(" …")
        tail = first_line[len(consumed) :] if first_line.startswith(consumed) else first_line
        rest = tail.strip(" .,;:") or first_line
    body_words = _strip_markers(rest).split()[:TEXT_BLOCK_MAX_WORDS]

    entity_rows = [(eid, e) for eid, e in index.items() if e.get("field") is None]
    if not entity_rows:
        entity_rows = list(index.items())
    entity_rows = entity_rows[:TEMPLATE_ROWS]

    citations = [eid for eid, _ in entity_rows]
    facts = [
        Fact(
            evidence_id=eid,
            entity_id=e.get("entity_id"),
            field=e.get("field"),
            value=_shorten(str(e.get("value", ""))),
        )
        for eid, e in entity_rows
    ]

    if body_words:
        blocks = [TextBlock.model_construct(kind="text", text=" ".join(body_words))]
    elif entity_rows:
        # The true last resort: the executor produced no prose at all, so cited
        # rows are the only thing left to show. This is the ONLY path that can
        # put a tool's raw value in front of a reader.
        blocks = [
            BulletBlock.model_construct(
                kind="bullets",
                items=[f"{_shorten(str(e.get('value', '')))} [{eid}]" for eid, e in entity_rows],
            )
        ]
    else:
        blocks = [TextBlock.model_construct(kind="text", text=headline)]

    return AnswerPayload.model_construct(
        kind="answer",
        headline=headline,
        blocks=blocks,
        citations=citations,
        facts=facts,
        assumptions=[],
        follow_ups=[],
        candidates=[],
    )


def _abstain_payload() -> AnswerPayload:
    return AnswerPayload(kind="abstain", headline="NO_COMPANY_RECORDS")


def _explain_validation(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(item) for item in err.get("loc", ())) or "payload"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "invalid payload — " + "; ".join(parts[:6])


class FormatGate(BaseAgent):
    """Runs the formatter, validates in code, retries with feedback, and
    guarantees the turn ends with a payload (see module docstring)."""

    def _payload_event(self, ctx: InvocationContext, payload: AnswerPayload) -> Event:
        """A gate-authored payload event: the JSON as content text (the bot's
        answer channel parses formatter/format_gate text, last one wins) AND a
        state_delta so `state["answer_payload"]` matches what was emitted —
        the critic gate reads it for abstain recognition."""
        dumped = payload.model_dump(exclude_none=True)
        return Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            content=genai_types.Content(
                role="model",
                parts=[genai_types.Part(text=json.dumps(dumped, ensure_ascii=False))],
            ),
            actions=EventActions(state_delta={ANSWER_STATE_KEY: dumped}),
        )

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        # The fast path (blend 04) has no executor prose — it leaves its
        # deterministic draft in the per-invocation store and calls this gate,
        # which is the ONLY way an AnswerPayload leaves that path. On the deep
        # path there is no draft and the executor's last text is read exactly
        # as before.
        executor_text = answer_draft(ctx.invocation_id) or _last_executor_text(ctx)
        if not executor_text.strip():
            # A run that produced no executor text (died inside the engine) —
            # nothing to format; emit nothing so the trace shows the failure
            # instead of a fabricated answer.
            return

        # Deterministic abstention, same marker check as CriticGate / the bot:
        # converting one marker word into an abstain payload needs no LLM.
        if executor_text.strip().upper().startswith("NO_COMPANY_RECORDS"):
            yield self._payload_event(ctx, _abstain_payload())
            return

        index = evidence_index(ctx.invocation_id)
        set_formatter_brief(ctx.invocation_id, compose_brief(executor_text, index, feedback=""))

        for attempt in range(MAX_FORMAT_ATTEMPTS):
            captured: dict | None = None
            feedback = ""
            try:
                async for event in self.sub_agents[0].run_async(ctx):
                    delta = (event.actions.state_delta or {}) if event.actions else {}
                    if isinstance(delta.get(ANSWER_STATE_KEY), dict):
                        captured = delta[ANSWER_STATE_KEY]
                    yield event
            except ValidationError as exc:
                if exc.title != AnswerPayload.__name__:
                    # Not the contract speaking — some OTHER pydantic model
                    # failed inside the formatter run (live case: the genai
                    # SDK's Schema type rejecting the response_schema).
                    # Retrying would loop on an infra bug and mislabel it as
                    # payload feedback; fail the turn loudly instead.
                    raise
                # output_schema validation failed INSIDE the formatter run —
                # the contract's own validators (filler, budgets, table shape)
                # speaking. Same retry path as a grounding failure.
                feedback = _explain_validation(exc)

            if not feedback:
                if captured is None:
                    feedback = "no AnswerPayload was produced — emit exactly one JSON payload"
                else:
                    problems = gate_problems(AnswerPayload.model_validate(captured), index)
                    if not problems:
                        return  # the formatter's own event already carried the payload
                    feedback = "; ".join(problems)

            if attempt + 1 >= MAX_FORMAT_ATTEMPTS:
                break

            logger.warning(
                "format_gate: attempt %d rejected (inv=%s) — %s",
                attempt + 1,
                ctx.invocation_id,
                feedback,
            )
            # Feedback travels twice on purpose: the state_delta event makes
            # the rejection visible in the trace (and to the batch runner);
            # the in-process store is what the next formatter run's brief
            # actually reads — deterministic, no commit-timing dependency.
            set_format_feedback(ctx.invocation_id, feedback)
            set_formatter_brief(ctx.invocation_id, compose_brief(executor_text, index, feedback))
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                actions=EventActions(state_delta={"format_feedback": feedback}),
            )

        # Two retries spent — deterministic template render, authored by the
        # gate. Later than every failed formatter event, so the bot's
        # last-payload-wins routing picks it up.
        logger.warning(
            "format_gate: %d attempts spent (inv=%s) — emitting the template render",
            MAX_FORMAT_ATTEMPTS,
            ctx.invocation_id,
        )
        yield self._payload_event(ctx, template_payload(executor_text, index))


format_gate = FormatGate(name="format_gate", sub_agents=[formatter_agent])
