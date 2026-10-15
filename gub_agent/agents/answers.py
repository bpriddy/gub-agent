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
import logging
import re
from datetime import date
from typing import Literal

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

from .. import config
from ..schemas import AnswerPayload, BulletBlock, Candidate, TextBlock
from ..schemas.router import Intent
from .formatter import ANSWER_STATE_KEY

logger = logging.getLogger(__name__)

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


#: Author of the fast path's progress events. Deliberately NOT one of the
#: names the bot treats as internal (`agent/client.ts:216` — critic,
#: loop_escalator, critic_gate, sandbox_echo, router, dispatcher, validator):
#: those are filtered out before tool activity is read, and the progress line
#: is exactly what these events exist to drive.
FAST_PATH_AUTHOR = "fast_path"


def tool_activity_event(ctx: InvocationContext, tool: str, *, done: bool = False) -> Event:
    """A "I am calling this tool now" event, purely so the bot's bubble can
    advance its progress line.

    The fast path calls its tool as a plain Python coroutine — no LLM, so no
    `function_call` part is ever produced, so the bot sees no activity and its
    bubble sits on one generic phrase for the whole turn (reported 2026-09-16:
    "streaming output does not work + no status updates arrive"). The deep
    path gets its progress for free precisely because its calls go through a
    model.

    So the fast path says so itself, in the shape the bot already parses
    (`agent/client.ts:extractToolActivity` reads `function_call.name` and
    `function_response.name`, and maps the name to a phrase via `TOOL_STEPS`).

    **`partial=True` is load-bearing, not a detail.** ADK appends an event to
    the session only when it is not partial (`runners.py:881`). A non-partial
    function_call here would be written into the conversation history, and the
    deep path — which runs whenever the fast path declines — would then see a
    tool call that never happened, paired with a response that carries no data.
    That is a lie told to a model that is about to answer a question. Partial
    keeps the event on the wire to the bot and out of the history.

    It also carries no text part, so it cannot reach the answer channel: the
    bot contributes only `extractText(evt)` to the reply.
    """
    part = (
        genai_types.Part(function_response=genai_types.FunctionResponse(name=tool, response={}))
        if done
        else genai_types.Part(function_call=genai_types.FunctionCall(name=tool, args={}))
    )
    return Event(
        invocation_id=ctx.invocation_id,
        author=FAST_PATH_AUTHOR,
        content=genai_types.Content(role="model", parts=[part]),
        partial=True,
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

# Starter questions for a greeting turn. The bot renders `follow_ups` as
# tappable suggestion chips (gub-gchat-bot, chat/cards.ts), so a greeting
# answers with somewhere to GO rather than a description of what could be
# asked. Each one is deliberately aggregate — no entity name, so the
# clarification layer has nothing to disambiguate and the tap cannot open
# with a card.
_SMALLTALK_FOLLOW_UPS = {
    "ru": [
        "Сколько сейчас активных кампаний?",
        "Какие у нас клиентские аккаунты?",
        "Что нового за эту неделю?",
    ],
    "en": [
        "How many live campaigns do we have?",
        "Which client accounts do we have?",
        "What's new this week?",
    ],
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
        follow_ups=_SMALLTALK_FOLLOW_UPS.get(language, _SMALLTALK_FOLLOW_UPS["en"]),
        candidates=[],
    )


_NO_ACCESS = {
    "ru": (
        "Нет доступа к этой записи в данных компании",
        "У вас пока нет доступа к этой записи — попросите её владельца открыть вам доступ.",
    ),
    "en": (
        "No access to that record in company records",
        "You don't have access to this record yet — ask its owner to share it.",
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
    """`confidence < ROUTER_CONFIDENCE_FLOOR`: ask which question was meant instead of
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


# ── reader text: no record ids, no raw ISO timestamps (latency-05 W1-5) ──────

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
UUID_RE = re.compile(_UUID, re.IGNORECASE)

# An ISO-8601 timestamp, date and time with the seconds, fraction and zone
# optional. The lookarounds keep it off the middle of a longer token. A
# date-only string ("2026-06-04") is not matched: it names no instant, and it
# is not what a record's createdAt looks like.
_ISO_TS = re.compile(
    r"(?<![\w-])(?>(\d{4})-(\d{2})-(\d{2})T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)(?!\w)"
)

# Spans the scrub never reads or rewrites, so a machine handle that merely
# contains a UUID or a timestamp survives:
#  * a marker: `[1]`, `[2, 3]`, `[1-3]` (source numbers) or `[<tool>:<id>[:field]]`
#    (an evidence id: no space inside), with its `(url)` when it is a link; and
#    a markdown link `[text](url)` as a unit. NOT any bracketed span: the format
#    gate treats every `[...]` as a citation marker and so never looks inside
#    one, which is exactly where an id or a timestamp written as bracketed
#    prose would hide from it. The spans are bounded (200 characters), so a run
#    of unmatched `[` cannot make the search quadratic;
#  * a URL;
#  * an evidence id written outside brackets (`<tool>:<uuid>[:field]`).
# Only non-capturing groups: the splitter below wraps this in ONE group.
_PROTECTED = (
    r"\[(?:\d+(?:[ \t]*[,\-\u2013][ \t]*\d+)*|[A-Za-z_]\w*:[^\]\s]{1,200})\](?:\([^)\s]*\))?"
    r"|\[[^\]\n]{0,200}\]\([^)\s]*\)"
    r"|https?://[^\s)\]>]+"
    rf"|(?<![\w-])(?!(?:id|uuid):)[A-Za-z_]\w*:{_UUID}(?::[\w.\-]+)*"
)
_PROTECTED_SPLIT = re.compile(f"({_PROTECTED})", re.IGNORECASE)

# `(id: <uuid>)` / `(ID: <uuid>)` / `(ID <uuid>)` with the space before it, then
# `id: <uuid>`, then a bare UUID. A `/` or `=` before a UUID, or a `/` after it,
# marks a path or a query value that is not wrapped in a URL: that stays.
_PAREN_ID = re.compile(
    rf"(?<![ \t])[ \t]*\([ \t]*(?:id|uuid)[ \t]*[:=]?[ \t]*{_UUID}[ \t]*\)", re.IGNORECASE
)
_LABELLED_ID = re.compile(rf"(?<![\w-])(?:id|uuid)[ \t]*[:=][ \t]*{_UUID}(?![\w/])", re.IGNORECASE)
_BARE_ID = re.compile(rf"(?<![\w/=]){_UUID}(?![\w/])", re.IGNORECASE)

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

_SPACES = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"(?<![ \t])[ \t]+([,.;:!?)])")
_EMPTY_EMPHASIS = re.compile(r"(\*\*|__|`)[ \t]*\1")
_COMMA_BEFORE_STOP = re.compile(r",[ \t]*([.!?])")
_EMPTY_PARENS = re.compile(r"\([ \t]*\)")
_DANGLING_COMMA = re.compile(r",[ \t]*\)")
_LEADING_COMMA = re.compile(r"\([ \t]*,[ \t]*")

#: A reader-visible string longer than this is not an answer's text (a text
#: block is a few hundred words, a cell a few words): it is left as written
#: rather than searched, so no input can hold the event loop.
_MAX_SCRUB_CHARS = 10_000

#: What a table cell reads when an id was ALL it said.
_EMPTY_CELL = "\u2014"

# What a sentence reads like when the id it ended on was taken out of it
# ("not found for <uuid>." -> "not found for."): a preposition, article, copula
# or separator last. The scrub keeps such a string as written. An id the reader
# asked about, or their own words echoed back, is not a leak, and a garbled
# sentence is worse than an id they can read in their own question.
_DANGLING_END = re.compile(
    r"(?:\b(?:is|are|was|were|be|for|to|of|with|by|at|on|in|and|or|as|from|the|a|an)\b"
    r"|[,:;(\u2013\u2014-])[ \t]*+[.!?]*+[ \t]*+$",
    re.IGNORECASE,
)


def is_record_id_field(key: str, value: object) -> bool:
    """A record's own identifier as a tool returns it: `id`, `accountId`,
    `created_by_id`, with a UUID for a value. A machine handle, never a fact a
    reader asked for; the fast path's draft leaves it out when READER_TEXT_SCRUB
    is on (it stays in the evidence rows, where it is cited and echoed)."""
    return (
        isinstance(value, str)
        and (key == "id" or key.endswith(("Id", "_id")))
        and UUID_RE.fullmatch(value.strip()) is not None
    )


def _human_date(match: re.Match[str]) -> str:
    """ "Jun 4, 2026" for an ISO timestamp, read as WRITTEN: no time zone is
    applied, so the day is the one the timestamp names. An impossible date
    (month 13) is left as it was."""
    year, month, day = (int(g) for g in match.groups())
    try:
        date(year, month, day)
    except ValueError:
        return match.group(0)
    return f"{_MONTHS[month - 1]} {day}, {year}"


def _tidy(text: str) -> str:
    """After an id came out: no empty or dangling brackets, no double space,
    no space before punctuation."""
    text = _EMPTY_PARENS.sub("", text)
    text = _DANGLING_COMMA.sub(")", text)
    text = _LEADING_COMMA.sub("(", text)
    text = _EMPTY_EMPHASIS.sub("", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _COMMA_BEFORE_STOP.sub(r"\1", text)
    return _SPACES.sub(" ", text)


class _Scrub:
    """One payload's scrub: the evidence ids it must not touch, and what it
    has taken out so far."""

    def __init__(self, keep: re.Pattern[str] | None) -> None:
        self.ids = 0
        self.dates = 0
        self._split = (
            _PROTECTED_SPLIT
            if keep is None
            else re.compile(f"({_PROTECTED}|{keep.pattern})", re.IGNORECASE)
        )

    @staticmethod
    def _dates(part: str) -> tuple[str, int]:
        """`part` with its ISO timestamps rewritten, and how many REALLY were:
        an impossible date is left as written and is not counted."""
        count = 0

        def rewrite(match: re.Match[str]) -> str:
            nonlocal count
            human = _human_date(match)
            if human != match.group(0):
                count += 1
            return human

        return _ISO_TS.sub(rewrite, part), count

    def text(self, text: str, *, cell: bool = False) -> str:
        """One reader-visible string, its unprotected stretches scrubbed. A
        string that would be left EMPTY (it was only an id) stays as it was, so
        no headline or block goes blank, except a table cell, which reads
        "\u2014"; so does one the removal left ending on a dangling word or
        mark (`_DANGLING_END`). Unchanged text comes back as the same string."""
        if len(text) > _MAX_SCRUB_CHARS:
            return text
        ids = dates = 0
        out: list[str] = []
        for i, part in enumerate(self._split.split(text)):
            if i % 2:  # a protected span
                out.append(part)
                continue
            removed = 0
            for pattern in (_PAREN_ID, _LABELLED_ID, _BARE_ID):
                part, n = pattern.subn("", part)
                removed += n
            part, rewritten = self._dates(part)
            out.append(_tidy(part) if removed else part)
            ids += removed
            dates += rewritten
        if not (ids or dates):
            return text
        new = "".join(out)
        if ids:
            new = new.strip(" \t")  # an id at an edge leaves its space behind
        if not new.strip():
            if not cell:
                return text
            new = _EMPTY_CELL
        elif ids and _DANGLING_END.search(new) and not _DANGLING_END.search(text):
            return text  # the id was the end of the sentence: leave the sentence whole
        self.ids += ids
        self.dates += dates
        return new


def _evidence_id_pattern(payload: AnswerPayload) -> re.Pattern[str] | None:
    """The payload's own evidence ids that carry a UUID, as one pattern: they
    are citations (`citations`, `facts`), so wherever one is written it stays."""
    ids = {c for c in payload.citations if UUID_RE.search(c)}
    ids |= {f.evidence_id for f in payload.facts if UUID_RE.search(f.evidence_id)}
    if not ids:
        return None
    return re.compile("|".join(re.escape(i) for i in sorted(ids, key=len, reverse=True)))


def scrub_reader_text(payload: AnswerPayload) -> tuple[AnswerPayload, int, int]:
    """(payload, ids removed, dates rewritten): `payload` with the record ids
    and raw ISO timestamps taken out of what a reader sees: the headline, text
    blocks, bullets and table cells. Never `facts`, `citations`, source ids,
    `[n]` / `[evidence id]` markers, table column names, assumptions,
    follow-ups or candidates: those are machine fields or the user's own
    words. The SAME object comes back when nothing changed; otherwise a copy
    built without re-validation (the word budgets were judged on what the
    formatter wrote; this only removes words, except that a date is three
    tokens where the timestamp was one)."""
    scrub = _Scrub(_evidence_id_pattern(payload))
    headline = scrub.text(payload.headline)
    blocks = []
    for block in payload.blocks:
        if block.kind == "text":
            text = scrub.text(block.text)
            if text is not block.text:
                block = block.model_copy(update={"text": text})
        elif block.kind == "bullets":
            items = [scrub.text(item) for item in block.items]
            if any(new is not old for new, old in zip(items, block.items, strict=True)):
                block = block.model_copy(update={"items": items})
        else:
            rows = [[scrub.text(cell, cell=True) for cell in row] for row in block.rows]
            if any(
                new is not old
                for new_row, row in zip(rows, block.rows, strict=True)
                for new, old in zip(new_row, row, strict=True)
            ):
                block = block.model_copy(update={"rows": rows})
        blocks.append(block)
    if not (scrub.ids or scrub.dates):
        return payload, 0, 0
    return (
        payload.model_copy(update={"headline": headline, "blocks": blocks}),
        scrub.ids,
        scrub.dates,
    )


def scrub_for_emit(
    payload: AnswerPayload, invocation_id: str, tenant: str
) -> tuple[AnswerPayload, bool]:
    """The scrub as the format gate calls it, at the point a payload leaves
    it: (payload, whether it changed). With READER_TEXT_SCRUB off the payload
    comes back untouched. Logs one line, from here, only when something was
    scrubbed (the gate's own log lines carry its file name, and the gate's
    drop and repair rates are read as `textPayload:"format_gate"`)."""
    if not config.READER_TEXT_SCRUB:
        return payload, False
    try:
        scrubbed, ids, dates = scrub_reader_text(payload)
    except Exception as exc:  # noqa: BLE001 — a scrub must never cost the answer
        logger.warning(
            "reader_text: scrub failed (%s) — payload emitted as written inv=%s tenant=%s",
            type(exc).__name__,
            invocation_id,
            tenant,
        )
        return payload, False
    if scrubbed is payload:
        return payload, False
    logger.info(
        "reader_text: scrubbed ids=%d dates=%d inv=%s tenant=%s", ids, dates, invocation_id, tenant
    )
    return scrubbed, True
