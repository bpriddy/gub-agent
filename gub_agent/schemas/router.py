"""
router.py — the routing contract (blend 04, gub-agent#23).

One LLM call classifies the question BEFORE any retrieval happens, and the
dispatcher (`agents/dispatcher.py`) picks a branch from the result in plain
code. The point is latency: a FACT question with a known entity is one HTTP
call, but every question used to pay 1-3 executor rounds at MEDIUM thinking
plus the critic (20-45 s, mean 31 s — CHANGELOG 2026-08-12). Latency is model
turns, so the only way to cut it is to not take them.

Two rules this schema encodes, both load-bearing:

- `entity_surface` is the entity AS WRITTEN. The router never resolves,
  expands or corrects a name: resolution is a deterministic search with
  thresholds (`agents/fast_path.py`), and a router that "helpfully" turned
  "chevy" into "Chevrolet" would hand the fast path a name the index cannot
  ground.
- `entity_id` is only ever copied — from the `"User selected campaign <uuid>"`
  prefix the bot's disambiguation card writes (blend 02). An id the router
  invented would send a deterministic lookup at a nonexistent row.

`slots` is a TYPED object, not the free `dict[str, str]` the spec sketched.
Measured 2026-09-10 on `gemini-3.5-flash`: the free-map shape is accepted by
Vertex structured output but the model leaves it EMPTY — over six runs, in the
pipeline and in an isolated call, "сколько live кампаний" produced
`slots: {}` every time (twice even listing `entity` in `missing_slots`
instead), so every count question fell through to the deep path and the
fast path's third-largest intent was dead on arrival. With the same prompt and
the named fields below the model fills them correctly, which is the whole
difference between a routed count and a 30-second one.

`Slots.complete` is the guard the named fields need. A closed field set cannot
express every constraint a sentence can state ("с бюджетом больше миллиона"),
and a builder that silently dropped the part it could not express would answer
a DIFFERENT question than the one asked — the one failure worse than being
slow. So the router declares whether the fields cover the sentence, and the
builder refuses the fast path unless they do (`agents/fast_path.py`).

Unlike `answer.py` this contract is INTERNAL: nothing outside the engine reads
a RouterDecision, so the field names here are not a wire change.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The ten intents. Ordered as the prompt lists them: the five FACT-shaped ones
# the fast path can serve, then the ones that need the executor, then the two
# that need no data at all.
Intent = Literal[
    "campaign_status",
    "campaign_facts",
    "account_facts",
    "staff_lookup",
    "count_or_rank",
    "assessment",
    "exploratory",
    "market_enrichment",
    "workspace_personal",
    "smalltalk",
]

# Intents a single deterministic lookup can answer. Everything else — an
# assessment's verdict, an open exploration, a market question — needs the
# executor's judgement about WHICH calls to make, which is what the deep path
# is for.
FAST_INTENTS: frozenset[str] = frozenset(
    {
        "campaign_status",
        "campaign_facts",
        "account_facts",
        "staff_lookup",
        "count_or_rank",
    }
)


class Slots(BaseModel):
    """The parameters a `count_or_rank` question states, as NAMED fields — the
    catalog the builder validates against (`agents/fast_path.py`).

    `office` is here to be REFUSED: an office constraint needs an office id the
    fast path has no deterministic way to resolve, so a question carrying one
    belongs to the executor. Representing it is what makes refusing it possible
    — an unrepresentable constraint would just vanish.
    """

    entity: Literal["campaigns", "accounts", "staff", "pieces"] | None = None
    status: str | None = None
    industry: str | None = None
    # A client name as written; the builder resolves it to an id and filters on
    # the id, never with `similar_to` (which v1 requires to be a sole filter).
    account: str | None = None
    # An explicit year or quarter only ("2026", "2026 Q3"); a relative period
    # is a judgement, and judgements are the deep path's job.
    period: str | None = None
    metric: str | None = None
    limit: int | None = None
    group_by: str | None = None
    office: str | None = None
    complete: bool = False
    """True only when the fields above express EVERY constraint the sentence
    states. Default False: an unset flag must cost the deep path, never a
    confidently wrong count."""


class RouterDecision(BaseModel):
    """What the router emits (as JSON text, authored by `router`, which the
    bot's author routing ignores — gub-gchat-bot `agent/client.ts:209-217`)."""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    # As the user wrote it. NOT resolved — see the module docstring.
    entity_surface: str | None = None
    # Copied from the bot's "User selected campaign <uuid>" prefix, never
    # invented.
    entity_id: str | None = None
    slots: Slots = Field(default_factory=Slots)
    missing_slots: list[str] = Field(default_factory=list)
    language: Literal["ru", "en"] = "en"


# The decision the dispatcher uses when the router's output does not validate
# (or never arrived): the deep path, at zero confidence. A bad router must cost
# latency, never the turn — `agents/dispatcher.py`.
FALLBACK_DECISION = RouterDecision(intent="exploratory", confidence=0.0)
