"""
answer.py — the answer contract (blend 03, gub-agent#22).

The answer's shape used to be a request in the executor prompt policed by the
critic's `answer_satisfies` axis: probabilistic, one extra LLM call, and a
retry that doubled the turn (#6). This module makes the shape a TYPED contract:
the formatter agent runs with `output_schema=AnswerPayload`, so every violation
below is a pydantic ValidationError the format gate catches deterministically —
no judge, no vibes.

What the validators enforce (and why here, not in the gate):

- FILLER — no preamble words (EN + RU) in headline / text / bullets. A filler
  opener survives any downstream check because it is "correct", just useless;
  rejecting it at parse time makes the retry feedback exact.
- Budgets — headline ≤ 20 words, text block ≤ 60, bullet ≤ 25, ≤ 250 total.
  Nothing used to bound length at all (#5).
- Table shape — every row exactly as wide as the header. Google Chat renders
  no Markdown tables, so tables only exist as structured blocks the bot can
  turn into cards.
- `{f.evidence_id for f in facts} == set(citations)` — the facts echo the
  cited evidence rows verbatim so the bot's Workspace conflict filter (blend
  05) can compare values. A payload that cites what it does not carry would
  silently disable that filter, so it is invalid, not merely incomplete.

What the validators deliberately do NOT check: whether citations exist in the
turn's evidence index, and whether numbers/entities are grounded — those need
the index, which is per-invocation state, and live in the format gate
(`agents/format_gate.py`).

The bot's zod mirror is `gub-gchat-bot/src/chat/render-answer.ts`; the eval
reader is `gub-sandbox-ui/src/lib/batch/metrics.ts`. Field renames here are
wire changes — change all three together.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Preamble words an answer must never open with (or contain) — EN + RU. The
# formatter prompt forbids them too, but the prompt is a request and this is
# the check. Word-bounded so "ensure"/"measure" and the like stay legal.
FILLER_RE = re.compile(
    r"\b(sure|certainly|as an ai|here is|конечно|разумеется|отличный вопрос|вот что)\b",
    re.IGNORECASE,
)

# Budgets. Starting values — blend 06 ratifies them.
HEADLINE_MAX_WORDS = 20
TEXT_BLOCK_MAX_WORDS = 60
BULLET_MAX_WORDS = 25
TOTAL_MAX_WORDS = 250


def count_words(text: str) -> int:
    """Words = whitespace-separated tokens carrying at least one letter or
    digit (same definition as the eval side's countWords, so budgets here and
    metrics there agree)."""
    return sum(1 for token in text.split() if re.search(r"[^\W_]", token, re.UNICODE))


def _reject_filler(text: str, where: str) -> str:
    match = FILLER_RE.search(text)
    if match:
        raise ValueError(
            f"{where} contains filler ({match.group(0)!r}) — answer directly, no preamble"
        )
    return text


def _reject_over_budget(text: str, limit: int, where: str) -> str:
    words = count_words(text)
    if words > limit:
        raise ValueError(f"{where} is {words} words, over the {limit}-word budget")
    return text


class TableBlock(BaseModel):
    """A real table (2-6 columns, 1-10 rows) — the bot renders it as a cardsV2
    section because Chat renders no Markdown tables. The contract's convention
    puts the source evidence id in the last column."""

    kind: Literal["table"] = "table"
    columns: list[str] = Field(min_length=2, max_length=6)
    rows: list[list[str]] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def _rows_match_columns(self) -> TableBlock:
        width = len(self.columns)
        for i, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(f"table row {i} has {len(row)} cells, expected {width} (columns)")
        return self


class BulletBlock(BaseModel):
    kind: Literal["bullets"] = "bullets"
    items: list[str] = Field(min_length=1, max_length=7)

    @field_validator("items")
    @classmethod
    def _items_clean(cls, items: list[str]) -> list[str]:
        for i, item in enumerate(items):
            _reject_filler(item, f"bullet {i}")
            _reject_over_budget(item, BULLET_MAX_WORDS, f"bullet {i}")
        return items


class TextBlock(BaseModel):
    kind: Literal["text"] = "text"
    text: str

    @field_validator("text")
    @classmethod
    def _text_clean(cls, text: str) -> str:
        _reject_filler(text, "text block")
        return _reject_over_budget(text, TEXT_BLOCK_MAX_WORDS, "text block")


class Fact(BaseModel):
    """One cited evidence row, echoed verbatim from the turn's evidence index
    (`agents/evidence_index.py`) so the bot's conflict filter (blend 05) has
    the VALUES, not just the ids."""

    evidence_id: str
    entity_id: str | None = None
    field: str | None = None
    value: str | None = None


class Candidate(BaseModel):
    """kind="clarify" only — one interpretation the user is asked to pick from
    (consumed by the fast-path router, blend 04). Field names rhyme with the
    bot's disambiguation-card candidates (blend 02)."""

    name: str
    entity_id: str | None = None
    entity_type: str | None = None
    hint: str | None = None


Block = Annotated[TableBlock | BulletBlock | TextBlock, Field(discriminator="kind")]


class AnswerPayload(BaseModel):
    """THE answer, as a contract.

    `headline` IS the answer — the value for a fact question, the verdict for
    an assessment — not a title over it. `blocks` carry the drivers/detail.
    `kind="abstain"` is the typed form of the NO_COMPANY_RECORDS marker;
    `kind="clarify"` asks the user to disambiguate (blend 04) and carries
    `candidates`.
    """

    kind: Literal["answer", "abstain", "clarify"] = "answer"
    headline: str
    blocks: list[Block] = Field(default_factory=list, max_length=4)
    citations: list[str] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list, max_length=2)
    follow_ups: list[str] = Field(default_factory=list, max_length=3)
    candidates: list[Candidate] = Field(default_factory=list)

    @field_validator("headline")
    @classmethod
    def _headline_clean(cls, text: str) -> str:
        _reject_filler(text, "headline")
        return _reject_over_budget(text, HEADLINE_MAX_WORDS, "headline")

    @model_validator(mode="after")
    def _contract(self) -> AnswerPayload:
        if self.kind == "answer":
            if not self.blocks:
                raise ValueError('kind="answer" needs at least one block')
            if not self.citations:
                raise ValueError('kind="answer" needs at least one citation (an evidence id)')
        # facts ↔ citations, both directions: 05 filters Workspace prose
        # against `facts`, so a payload that cites what it does not carry
        # would silently disable that filter.
        fact_ids = {fact.evidence_id for fact in self.facts}
        cited = set(self.citations)
        if fact_ids != cited:
            missing = sorted(cited - fact_ids)
            extra = sorted(fact_ids - cited)
            detail = []
            if missing:
                detail.append(f"cited but not echoed in facts: {', '.join(missing)}")
            if extra:
                detail.append(f"in facts but not cited: {', '.join(extra)}")
            raise ValueError(
                "facts must echo exactly the cited evidence rows — " + "; ".join(detail)
            )
        if self.candidates and self.kind != "clarify":
            raise ValueError('candidates are for kind="clarify" only')

        total = count_words(self.headline)
        for block in self.blocks:
            if isinstance(block, TextBlock):
                total += count_words(block.text)
            elif isinstance(block, BulletBlock):
                total += sum(count_words(item) for item in block.items)
            else:
                total += sum(count_words(cell) for row in block.rows for cell in row)
                total += sum(count_words(col) for col in block.columns)
        total += sum(count_words(a) for a in self.assumptions)
        total += sum(count_words(f) for f in self.follow_ups)
        if total > TOTAL_MAX_WORDS:
            raise ValueError(f"answer is {total} words total, over the {TOTAL_MAX_WORDS} budget")
        return self
