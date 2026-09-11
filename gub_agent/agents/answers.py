"""
answers.py — the deterministic payloads (blend 04, gub-agent#23).

Some turns need no model at all: a personal-Workspace question GUB must bow
out of, a greeting, an intent the router could not pin down, a lookup that
came back 403 or 404. This module builds those `AnswerPayload`s in code, plus
the one event shape that carries a payload to the bot.

Three rules encoded here, each with a reason:

1. **Author.** A payload event is authored `format_gate`, never `dispatcher`.
   The bot routes event text by AUTHOR and its answer channel accepts exactly
   `formatter` and `format_gate` — `dispatcher` is on the ignore list
   (`gub-gchat-bot/src/agent/client.ts:209-217`). A dispatcher-authored
   payload would be dropped on the floor and the user would see nothing. The
   format gate already emits its deterministic abstention this way
   (`format_gate.py:_payload_event`), so this is the same author for the same
   kind of thing: a payload the engine wrote itself, with no LLM in the loop.
2. **Templates ground themselves.** These texts name no entity and carry no
   number, so they pass the gate's grounding checks against an EMPTY evidence
   index (`test_dispatcher.py` pins it). The user's own surface — the one
   string here that comes from outside — travels in `assumptions`, which the
   gate does not ground (`format_gate._prose_and_cells` reads the headline and
   the blocks only), exactly as a clarification legitimately echoes the name
   the user typed.
3. **`model_construct` for the two uncitable answers.** The contract requires
   a citation for `kind="answer"`, and a greeting or a permission denial has
   nothing to cite — the same bind the gate's template render is in
   (`format_gate.template_payload`), resolved the same way. `kind="abstain"`
   and `kind="clarify"` need no bypass and get none.

Why not `kind="abstain"` for the 403/404 pair: the bot HIDES the whole GUB
section on an abstention (`gub-gchat-bot/src/chat/cards.ts:660-675`), so "no
access" would reach the user as silence. A dedicated `kind="error"` is a wire
change across three repos (`schemas/answer.py` docstring) and belongs to blend
05/06, not here.
"""

from __future__ import annotations

import json
from typing import Literal

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

from ..schemas import AnswerPayload, BulletBlock, Candidate, TextBlock
from ..schemas.router import Intent
from .formatter import ANSWER_STATE_KEY

# See rule 1 in the module docstring — this is the bot's answer channel, not a
# cosmetic name.
ANSWER_AUTHOR = "format_gate"

Language = Literal["ru", "en"]


def payload_event(ctx: InvocationContext, payload: AnswerPayload) -> Event:
    """A payload event in the shape the bot already parses: the JSON as content
    text (its answer channel reads formatter / format_gate text, last one wins)
    plus a `state_delta` so `state["answer_payload"]` matches what was
    emitted. Mirrors `FormatGate._payload_event`."""
    dumped = payload.model_dump(exclude_none=True)
    return Event(
        invocation_id=ctx.invocation_id,
        author=ANSWER_AUTHOR,
        content=genai_types.Content(
            role="model",
            parts=[genai_types.Part(text=json.dumps(dumped, ensure_ascii=False))],
        ),
        actions=EventActions(state_delta={ANSWER_STATE_KEY: dumped}),
    )


# ── the payloads ─────────────────────────────────────────────────────────────


def abstain_payload() -> AnswerPayload:
    """`workspace_personal`: GUB gets out of the way and the Workspace spoke
    owns the question. Same marker the executor used to spend a whole turn
    emitting — here it costs zero model calls and zero tool calls."""
    return AnswerPayload(kind="abstain", headline="NO_COMPANY_RECORDS")


_SMALLTALK = {
    "ru": (
        "Привет — спросите про кампании, аккаунты, людей или бюджеты",
        "Отвечаю по данным компании: статусы, бюджеты, сроки, кто над чем работает.",
    ),
    "en": (
        "Hi — ask me about campaigns, accounts, people or budgets",
        "I answer from company records: statuses, budgets, dates, who works on what.",
    ),
}


def smalltalk_payload(language: Language = "en") -> AnswerPayload:
    """A greeting answered by a template: no retrieval to do, so no model call
    to make. `model_construct` — see rule 3 in the module docstring."""
    headline, text = _SMALLTALK.get(language, _SMALLTALK["en"])
    return AnswerPayload.model_construct(
        kind="answer",
        headline=headline,
        blocks=[TextBlock.model_construct(kind="text", text=text)],
        citations=[],
        facts=[],
        assumptions=[],
        follow_ups=[],
        candidates=[],
    )


_NO_ACCESS = {
    "ru": (
        "Нет доступа к этой записи в данных компании",
        "Доступ выдаётся по грантам GUB — попросите владельца записи открыть его.",
    ),
    "en": (
        "No access to that record in company records",
        "Access follows your GUB grants — ask the record's owner to widen them.",
    ),
}

_NOT_FOUND = {
    "ru": (
        "Такой записи в данных компании не нашлось",
        "Проверьте название или уточните, о чём речь — поищу ещё раз.",
    ),
    "en": (
        "No such record in company records",
        "Check the name or say a bit more about it and I will look again.",
    ),
}

_SEARCHED_FOR = {"ru": "Искал:", "en": "Looked for:"}


def _lookup_failure(
    table: dict[str, tuple[str, str]],
    language: Language,
    surface: str | None,
) -> AnswerPayload:
    headline, text = table.get(language, table["en"])
    assumptions = []
    if surface:
        # In `assumptions` on purpose: the gate grounds the headline and the
        # blocks, and the user's own surface is not in this turn's evidence.
        assumptions.append(f"{_SEARCHED_FOR.get(language, _SEARCHED_FOR['en'])} {surface}")
    return AnswerPayload.model_construct(
        kind="answer",
        headline=headline,
        blocks=[TextBlock.model_construct(kind="text", text=text)],
        citations=[],
        facts=[],
        assumptions=assumptions,
        follow_ups=[],
        candidates=[],
    )


def no_access_payload(language: Language = "en", surface: str | None = None) -> AnswerPayload:
    """A 403 from the deterministic lookup: answer it now. Paying 30 s of deep
    path to hear the same 403 helps nobody."""
    return _lookup_failure(_NO_ACCESS, language, surface)


def not_found_payload(language: Language = "en", surface: str | None = None) -> AnswerPayload:
    """A 404 from the deterministic lookup — the id does not exist (or is not
    visible), which the deep path would rediscover at the same cost."""
    return _lookup_failure(_NOT_FOUND, language, surface)


# ── intent clarification ─────────────────────────────────────────────────────

# Short human labels per intent, in both languages. These are what the user
# reads on a low-confidence turn, so they describe the QUESTION, not the code.
_INTENT_LABELS: dict[str, dict[str, str]] = {
    "campaign_status": {"ru": "статус кампании", "en": "campaign status"},
    "campaign_facts": {"ru": "детали кампании", "en": "campaign details"},
    "account_facts": {"ru": "детали аккаунта", "en": "account details"},
    "staff_lookup": {"ru": "профиль человека", "en": "someone's profile"},
    "count_or_rank": {"ru": "сколько или топ-N", "en": "a count or a top-N"},
    "assessment": {"ru": "оценка, как идут дела", "en": "an assessment of how it is going"},
    "exploratory": {"ru": "разобраться в целом", "en": "an open look at it"},
    "market_enrichment": {"ru": "внешняя информация о рынке", "en": "outside market information"},
    "workspace_personal": {"ru": "мои письма и файлы", "en": "my own mail and files"},
    "smalltalk": {"ru": "просто поговорить", "en": "just chatting"},
}

# The alternatives offered when confidence is below the floor: the router's own
# pick first, then the intents it is actually confusable with. Deliberately
# small — three options a person can read, not ten.
_INTENT_ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "campaign_status": ("campaign_status", "campaign_facts", "assessment"),
    "campaign_facts": ("campaign_facts", "campaign_status", "assessment"),
    "account_facts": ("account_facts", "assessment", "count_or_rank"),
    "staff_lookup": ("staff_lookup", "exploratory"),
    "count_or_rank": ("count_or_rank", "assessment"),
    "assessment": ("assessment", "campaign_status", "campaign_facts"),
    "exploratory": ("exploratory", "assessment", "count_or_rank"),
    "market_enrichment": ("market_enrichment", "exploratory"),
    "workspace_personal": ("workspace_personal", "exploratory"),
    "smalltalk": ("smalltalk", "exploratory"),
}

_CLARIFY_HEADLINE = {
    "ru": "Уточните, что именно нужно",
    "en": "Which of these did you mean?",
}


def intent_options(intent: str) -> tuple[str, ...]:
    """The intents offered on a low-confidence turn, the router's pick first."""
    return _INTENT_ALTERNATIVES.get(intent, ("exploratory", "assessment"))


def clarify_intent_payload(
    intent: Intent | str,
    language: Language = "en",
    surface: str | None = None,
) -> AnswerPayload:
    """`confidence < CONFIDENCE_FLOOR`: ask which question was meant instead of
    guessing one. The options are BOTH prose (the bullets — what the bot's
    renderer shows) and structured `candidates` (what a future card could
    offer); `entity_type="intent"` distinguishes them from blend 02's entity
    candidates.

    `kind="clarify"` skips the gate's grounding checks by design — a
    clarification echoes the user's own, unresolved words
    (`format_gate.gate_problems`)."""
    labels = [
        _INTENT_LABELS.get(option, {}).get(language)
        or _INTENT_LABELS.get(option, {}).get("en")
        or option
        for option in intent_options(intent)
    ]
    return AnswerPayload(
        kind="clarify",
        headline=_CLARIFY_HEADLINE.get(language, _CLARIFY_HEADLINE["en"]),
        blocks=[BulletBlock(kind="bullets", items=labels)],
        candidates=[Candidate(name=label, entity_type="intent", hint=surface) for label in labels],
    )
