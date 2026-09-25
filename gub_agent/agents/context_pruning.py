"""context_pruning.py — keep the conversation, drop the receipts.

Every model call re-reads the whole session transcript, tool payloads
included — so a prior turn's 47-campaign list taxes every round of every
LATER turn (tokens × remaining calls). But prior turns' tool payloads are
dead weight BY DOCTRINE: the executor must re-query for entity facts
("do NOT rely on prior turns"), and the critic verifies grounding against
the CURRENT turn's results only.

This before_model_callback strips function_call / function_response parts
from all content BEFORE the current turn (= everything before the last
user message that carries real text). Prior answers' prose survives, so
follow-ups still resolve ("what about the other campaign?"); the current
turn's raw tool data is untouched, so synthesis-over-raw-data and the
critic's grounding checks are unaffected.

`trim_to_recent_turns` (memory-00 §3) is the second, coarser bound in the
same place: a sliding window of the last N real turns. The pruner above
keeps a long conversation from carrying its tool payloads; the window keeps
it from carrying every turn. Together they are what let the bot's 5-minute
idle session reset be relaxed to a day — until one of them is live, that
timer is the only thing bounding how much transcript a model round re-reads.

Since thread-topics N is PER SESSION: the bot writes `context_turn_window`
into session state (10 for the main DM stream, 200 for a thread), and
CONTEXT_TURN_WINDOW is only the default for a session that carries none.

`strip_prior_turn_tool_text` is the router's and critic's version of the
pruner. Those two agents never see a function part at all — ADK hands them
the executor's tool traffic as TEXT — so the pruner above removes nothing
from their requests, and without this they re-read every earlier turn's full
tool results.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from google.genai import types as genai_types

from ..config import CONTEXT_TURN_WINDOW
from ..tenant import _state_of, label_of

logger = logging.getLogger(__name__)


def _is_function_part(part: Any) -> bool:
    return bool(getattr(part, "function_call", None) or getattr(part, "function_response", None))


def _has_user_text(content: Any) -> bool:
    """True for a user-role content carrying real text (a typed question).

    Tool responses also arrive role="user" in the genai format, but as
    function_response parts, not text — they must not count as a turn
    boundary.
    """
    if content.role != "user":
        return False
    return any(getattr(p, "text", None) and not _is_function_part(p) for p in (content.parts or []))


# Keys that are pure tool plumbing — the executor/critic prompts already tell
# the model these are for grounding infrastructure and MUST NOT appear in prose.
# `_sources` (Drive file citations) measured at 512k chars / 128k tokens for ONE
# account overview (3,257 file refs) — re-sent every ReAct round. Dead weight.
#
# `_cited` and `_sourcesTotal` (blend 08 §5.1) are the same kind of thing: a
# fileId → name map and a count that exist so the BOT can bind links. The
# formatter gets the source ids it copies from the evidence brief's `sources:`
# lines, never from here — so to the model this is 87 opaque Drive ids per
# account overview, re-sent every round, that it is forbidden to use.
_PLUMBING_KEYS = ("_sources", "_sourcesTotal", "_cited")


def _without_plumbing(obj: Any) -> Any:
    """Return `obj` with plumbing keys dropped, COPY-ON-WRITE: a new dict/list is
    built only where a key was actually removed; unchanged subtrees (and all
    scalars) are shared. So this never mutates the input in place — the
    function_response payload is shared with ADK session history, and mutating a
    nested field there corrupts it. The FunctionResponse holding the payload is
    shared too, which is why `strip_source_metadata` never assigns to it."""
    if isinstance(obj, dict):
        new: dict = {}
        changed = False
        for k, v in obj.items():
            if k in _PLUMBING_KEYS:
                changed = True
                continue
            nv = _without_plumbing(v)
            changed = changed or nv is not v
            new[k] = nv
        return new if changed else obj
    if isinstance(obj, list):
        rebuilt = [_without_plumbing(v) for v in obj]
        return rebuilt if any(n is not o for n, o in zip(rebuilt, obj)) else obj
    return obj


def _masked(part: Any) -> Any:
    """`part` with the plumbing dropped from its tool response: a NEW part
    around a NEW function_response when there was any to drop, else `part`
    itself."""
    fr = getattr(part, "function_response", None)
    resp = getattr(fr, "response", None) if fr is not None else None
    if not isinstance(resp, dict):
        return part
    scrubbed = _without_plumbing(resp)
    if scrubbed is resp:
        return part
    try:
        return part.model_copy(
            update={"function_response": fr.model_copy(update={"response": scrubbed})}
        )
    except Exception:  # noqa: BLE001 — best-effort across ADK versions: unmasked, never corrupted
        return part


def strip_source_metadata(callback_context: Any, llm_request: Any) -> None:
    """Observation masking: drop `_sources` citation plumbing from every tool
    response in the request. The model is instructed to ignore it (see
    prompts/executor.py, prompts/critic.py), yet it dominates prompt size on
    portfolio questions — one account overview carried 128k tokens of file refs,
    re-sent each round. Stripping it is loss-free for the answer and roughly
    halves prompt tokens on the heavy questions.

    Copy-on-write all the way up: a masked part is a new Part around a new
    FunctionResponse, in a new Content, in a new `contents` list. Nothing the
    request was built from is written to — not the payload, not the
    FunctionResponse, not the Part. The FunctionResponse IS the session's:
    ADK 2.9.2 copies each Part for the request but hands its function_response
    over by reference unless it has an `adk-` id to rewrite
    (`flows/llm_flows/contents.py:_copy_content_for_request`), and
    gemini-3.5-flash issues its own call ids (1740 of 2522 tool responses in
    saved production sessions). Assigning `fr.response` there, as this did,
    rewrote the stored event: the stream and the store lost `_sources`,
    `_cited` and `_sourcesTotal` — the bot's source chips, its [n] list and the
    blend-08 names — wherever an event is serialised after the next round
    starts (a kept SPECULATIVE_DEEP run's held events; InMemorySessionService,
    which stores the object itself). The deploy's requirements.txt does not
    pin google-adk, so this writes to nothing it does not own, whatever the
    installed ADK copies."""
    contents = llm_request.contents or []
    masked: list[Any] = []
    changed = False
    for content in contents:
        parts = content.parts or []
        new_parts = [_masked(part) for part in parts]
        if any(new is not old for new, old in zip(new_parts, parts)):
            content = content.model_copy(update={"parts": new_parts})
            changed = True
        masked.append(content)
    if changed:
        llm_request.contents = masked
    return None


def strip_prior_turn_tool_parts(callback_context: Any, llm_request: Any) -> None:
    """Drop function_call/function_response parts from pre-current-turn content."""
    contents = llm_request.contents or []
    if not contents:
        return None

    # The current turn starts at the LAST user content with real text.
    boundary = None
    for i in range(len(contents) - 1, -1, -1):
        if _has_user_text(contents[i]):
            boundary = i
            break
    if boundary is None or boundary == 0:
        return None  # single-turn request (or nothing to prune) — leave as-is

    pruned: list[Any] = []
    for i, content in enumerate(contents):
        if i >= boundary:
            pruned.append(content)
            continue
        kept_parts = [p for p in (content.parts or []) if not _is_function_part(p)]
        if kept_parts:
            pruned.append(genai_types.Content(role=content.role, parts=kept_parts))
        # A content that was ONLY tool payload disappears entirely.

    llm_request.contents = pruned
    return None


# ── Conversation window (memory-00 §3) ───────────────────────────────────────


# The opening words of ADK's foreign-agent context preamble.
#
# A PREFIX, not the whole string, because ADK has already reworded it once and
# the wording is not a contract:
#
#   2.6.1  flows/llm_flows/contents.py — the first part is exactly
#          "For context:".
#   2.9.2  flows/llm_flows/_fencing.py — OTHER_AGENT_CONTEXT_PREAMBLE, which
#          opens "For context: below is a transcript of what another agent
#          did, quoted between <<<BEGIN_QUOTED_AGENT_CONTENT>>> and ..." and
#          runs on for several sentences.
#
# `google-adk` is UNPINNED (pyproject: >=1.0.0), so CI and the deploy install
# whatever is newest at build time — this repo has already been burned by that
# once, when a rebuild flipped stream_query to camelCase and the bot's readers
# went blind. An exact-equality check against 2.6.1's wording matched NOTHING
# on 2.9.2, which would have shipped a window counting 40 boundaries where
# there are 8 real turns.
#
# Prefix-matching also fails in the SAFE direction if it ever over-matches. A
# false positive (a reader genuinely opening a message with "For context:")
# drops one boundary, so `starts[-window]` moves EARLIER and the request keeps
# MORE history — and the current turn survives regardless, because it sits
# after the cut. A false negative inflates the count and cuts too late, which
# is the failure that loses the user's question.
_FOREIGN_CONTEXT_PREFIX = "For context:"


def _is_foreign_context(content: Any) -> bool:
    """True for an ADK foreign-agent context content.

    ADK rewrites another agent's events into a role="user" content whose FIRST
    part is a "For context: ..." preamble, followed by "[author] said: ..."
    parts. Those satisfy _has_user_text but they are NOT turn boundaries.

    This matters more here than in a single-agent app: the pipeline runs
    router, executor, critic, formatter and format gate, so ONE completed turn
    emits several of these. Counting them inflates the turn count 4-7x and
    silently shrinks the window to a fraction of CONTEXT_TURN_WINDOW — no
    exception, no malformed request, just a model that can no longer resolve
    "the other one".

    A SUBTRACTIVE filter layered on _has_user_text rather than a second copy of
    the "is this a real user message" rule: tool responses also arrive
    role="user" (as function_response parts), and a second home for that rule
    is the first thing to drift.
    """
    parts = content.parts or []
    if not parts:
        return False
    return (getattr(parts[0], "text", None) or "").lstrip().startswith(_FOREIGN_CONTEXT_PREFIX)


def _turn_starts(contents: list[Any]) -> list[int]:
    """Indices of REAL user turns — the only indices it is safe to cut at."""
    return [
        i
        for i, content in enumerate(contents)
        if _has_user_text(content) and not _is_foreign_context(content)
    ]


#: The session-state key the bot writes the window into (thread-topics). The
#: bot sets it when it CREATES an agent session and pushes it again on every
#: reused turn through its state-refresh appendEvent, so a session created
#: before the key existed picks it up on its next turn. Pinned on both sides of
#: the wire — gub-gchat-bot writes exactly this name.
WINDOW_STATE_KEY = "context_turn_window"


def _as_window(raw: Any) -> int | None:
    """`raw` as a usable window, or None when it is not one.

    An int >= 0. `bool` is refused although it IS an int in Python: `True`
    would silently become a one-turn window, which is the harshest setting
    there is, and no bot means that by it.

    An integral float is accepted, and it is not leniency. The value crosses a
    protobuf Struct on its way through the Vertex session store, and a Struct
    has exactly one number type — double — so the bot's `10` can legitimately
    come back as `10.0`. Refusing it would drop every thread to the default
    window with nothing but a warning to show for it. `10.5` is refused: no
    rounding rule is obviously the one the sender meant.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    if isinstance(raw, float) and raw.is_integer() and raw >= 0:
        return int(raw)
    return None


def resolve_turn_window(callback_context: Any) -> tuple[int, str]:
    """The window for THIS request, and where it came from: `"state"` or `"env"`.

    Per session, because the two conversation kinds want different memories —
    a main DM stream rolls over its last 10 exchanges, a thread keeps up to
    200 — and both reach this one engine. Read per request, so a value the bot
    pushes on a reused turn takes effect on that turn.

    Precedence, and why each rule is the one it is:

    1. CONTEXT_TURN_WINDOW <= 0 wins over everything. It is the operator's
       rollback ("the window is cutting inside a turn: set it to 0"), and a
       rollback that a per-session value could outvote would do nothing for
       exactly the traffic it was pulled for — every bot session carries one.
    2. A usable `state["context_turn_window"]`. 0 there keeps its old meaning,
       "window disabled" for that session — never "unlimited by design"; the
       bot has no reason to send it.
    3. Otherwise CONTEXT_TURN_WINDOW. ABSENT is the common case, not an edge
       one: the Chevy tenant bot runs a pre-memory-00 image against this same
       production engine and sends no window, and neither do gub-sandbox-ui
       and the /blend harness. That is why the default is a finite number and
       must stay one.

    A present-but-unusable value (a string, a bool, a negative number) is
    logged and replaced by the default rather than trusted: it means the bot
    and this engine disagree about the contract, and a guessed window in
    either direction is worse than the one that is known to be safe.

    `CONTEXT_TURN_WINDOW` is read from the module at CALL time, not bound at
    definition, so a test's monkeypatch reaches it. `callback_context=None` is
    a supported input — every unit test calls it that way — and resolves to
    the default.
    """
    default = CONTEXT_TURN_WINDOW
    if default <= 0:
        return default, "env"
    state = _state_of(callback_context)
    if state is None:
        return default, "env"
    try:
        raw = state.get(WINDOW_STATE_KEY)
    except AttributeError:
        return default, "env"
    if raw is None:
        return default, "env"
    window = _as_window(raw)
    if window is None:
        logger.warning(
            "turn_window: ignored state[%s]=%.40r (want an int >= 0) — using the env "
            "default %d (inv=%s) tenant=%s",
            WINDOW_STATE_KEY,
            raw,
            default,
            getattr(callback_context, "invocation_id", None) or "-",
            label_of(callback_context),
        )
        return default, "env"
    return window, "state"


def trim_to_recent_turns(
    callback_context: Any, llm_request: Any, *, role: str = "executor"
) -> None:
    """Sliding window: show the model the last N complete turns.

    N is per session — `resolve_turn_window` — and `role` names the agent
    whose request this is, for the logs only (executor / router / critic, the
    same vocabulary as `sandbox_before_model`).

    This is the bound that replaces the bot's 5-minute idle session reset. Once
    a session lives as long as the conversation does, every model round of every
    later turn re-reads the whole transcript; strip_prior_turn_tool_parts already
    keeps that growth to prose rather than tool payloads, so it is a slow leak
    and not a runaway — but it is unbounded, and input volume is the primary
    cost driver on this workload.

    A window, not a TTL, because time was never the thing that mattered: a
    follow-up needs the last few exchanges ("the other campaign", "and Q3?"),
    and that is equally true of a 30-second pause and an overnight one.

    Per-request only. ADK rebuilds `contents` from the session's events on every
    round, so nothing is destroyed: raising N restores the older turns on the
    very next call, and a badly chosen N is a config change rather than a lost
    conversation. That is the decisive advantage over a bot-owned truncated
    transcript.

    Orphan safety — why there is no repair pass here. A message-counted window
    can split a function_call from its function_response, which is not a smaller
    request but an INVALID one (Gemini answers 400 INVALID_ARGUMENT, which
    inside Agent Engine reaches the caller as an empty 200 stream). This design
    cannot produce that, for two independent reasons: the cut is always AT a
    turn boundary, and a call/response pair always lives between two consecutive
    boundaries; and strip_prior_turn_tool_parts runs immediately after and
    removes every pre-current-turn function part anyway. An earlier draft
    specified a _drop_orphan_responses helper — it would be dead code in all
    three chains as wired, and a later reader would trust it.

    The same holds for the router and the critic, which are windowed too since
    thread-topics: their current turn starts at the user's real question
    (foreign-context contents are never boundaries), and cut <= that index, so
    the critic still reads all of THIS turn's tool evidence.
    """
    window, source = resolve_turn_window(callback_context)
    contents = llm_request.contents or []
    starts = _turn_starts(contents)
    # `naive` is what the count would be WITHOUT _is_foreign_context, and it is
    # logged so the rollout gate diagnoses itself rather than asking a reader to
    # know what "inflated" looks like. On this five-agent pipeline a healthy
    # multi-turn request has naive several times turns; `naive == turns` on a
    # conversation that has had more than one turn means the filter matched
    # nothing — which is exactly what an ADK version bump did once already
    # (2.9.2 reworded the preamble the filter keys on).
    naive = sum(1 for c in contents if _has_user_text(c))

    kept = contents
    # window <= 0 is "explicitly disabled" — the rollback, with no redeploy.
    # len(starts) <= window is shorter than the window: a no-op, no copies made.
    if window > 0 and len(starts) > window:
        # starts[-1] IS the current turn and cut <= starts[-1] for any window
        # >= 1, so the current turn survives whole however much tool traffic it
        # has accumulated: the window bounds HISTORY, never the work in progress.
        cut = starts[-window]

        # Anything before the FIRST real boundary is preamble, not a turn. With
        # `instruction=` the system text rides in config.system_instruction and
        # this slice is empty; an agent configured with `static_instruction` has
        # ADK put instruction contents in `contents`, and those must keep
        # leading the request.
        preamble = contents[: starts[0]]
        kept = preamble + contents[cut:]

        # The memory-00 rollout line, byte for byte and EXECUTOR ONLY. Log
        # readers already filter and count on this text, and until thread-topics
        # it could only come from the executor; letting the router and critic
        # emit it too would change what an existing count means without
        # changing a character of it. The per-call line below is the one that
        # covers every agent.
        if role == "executor":
            logger.info(
                "context_window: turns=%d naive=%d window=%d kept=%d dropped=%d",
                len(starts),
                naive,
                window,
                len(kept),
                len(contents) - len(kept),
            )
        llm_request.contents = kept

    # One line per windowed model call, whether or not it cut anything. The
    # line above fires only when trimming happens, so a thread — 200 turns,
    # almost never reached — would never appear in it at all, and it carries
    # no tenant: on a shared engine a proportion built from it mixes two bots.
    #
    # A NEW message rather than new fields on the old one (the log-schema
    # rule: add, never rename). Deliberately not spelled with "context_window"
    # inside it, so an existing `textPayload:"context_window"` filter keeps
    # matching exactly the lines it matched before. Count calls with
    # `textPayload:"turn_window: agent="`; `tenant=` goes last, as on the
    # dispatcher line, and numerator and denominator need the same clause.
    logger.info(
        "turn_window: agent=%s window=%d source=%s turns=%d naive=%d kept=%d dropped=%d "
        "(inv=%s) tenant=%s",
        role,
        window,
        source,
        len(starts),
        naive,
        len(kept),
        len(contents) - len(kept),
        getattr(callback_context, "invocation_id", None) or "-",
        label_of(callback_context),
    )
    return None


# ── Prior turns' tool traffic, as the router and the critic see it ───────────


# How ADK renders ANOTHER agent's function_call / function_response parts when
# it folds that agent's events into this agent's request as foreign context
# (2.9.2: flows/llm_flows/_fencing.py, _present_other_agent_message):
#
#   [gub_agent] called tool `org_query` with parameters:\n<<<BEGIN_QUOTED…
#   [gub_agent] `org_query` tool returned result:\n<<<BEGIN_QUOTED…
#
# 2.6.1 (flows/llm_flows/contents.py) used the same two openings with the
# payload on the same line, so this matches both. Prose is "[x] said:" and
# thinking is "[x] thought:", neither of which this matches.
#
# If a future ADK rewords these, the match misses and NOTHING is stripped —
# today's behaviour, more tokens but no lost context — and the per-session
# window still bounds it. tests/unit/test_router_critic_context.py builds the
# request through ADK's own `_get_contents` and asserts both that ADK still
# renders tool traffic this way and that the stripping actually happened, so
# a rewording fails CI instead of quietly re-inflating every router call.
_FOREIGN_TOOL_TEXT = re.compile(r"\[[^\]\n]+\] (?:called tool `|`[^`\n]*` tool returned result:)")


def _is_foreign_tool_text(part: Any) -> bool:
    text = getattr(part, "text", None)
    return bool(text) and _FOREIGN_TOOL_TEXT.match(text) is not None


def strip_prior_turn_tool_text(callback_context: Any, llm_request: Any) -> None:
    """Drop earlier turns' tool calls and tool results from the foreign context
    ADK hands the router and the critic; keep their prose.

    Why these two need it. ADK shows an agent its OWN tool traffic as
    function_call / function_response parts, which is what
    strip_prior_turn_tool_parts removes for the executor. Every OTHER agent
    gets that traffic flattened into text — "[gub_agent] `get_account_overview`
    tool returned result: {…}" — so for the router and the critic the pruner
    finds no function part to remove. Both therefore re-read every earlier
    turn's full tool results on every call, `_sources` plumbing included
    (strip_source_metadata only edits dict responses, never text). Measured
    offline on ADK 2.9.2: about 41k tokens per earlier heavy turn, re-sent on
    every call — quadratic in thread length, and at the 200-turn thread window
    it reaches the model's input limit in some 8-25 heavy turns. Over that
    limit Gemini answers 400, which inside Agent Engine reaches the caller as
    an EMPTY 200 stream.

    Why it is safe. The same doctrine that already governs the executor: an
    earlier turn's entity facts are re-queried, never trusted, and the critic
    judges THIS turn's evidence. What a follow-up needs from an earlier turn —
    "and Q3?", "the other campaign" — is the question and the answer, and those
    are prose: the user's text, the executor's draft, the formatter's payload,
    the router's own decision. Text carries no call/response pairing, so
    removing it can never orphan anything.

    The current turn is never touched. The boundary is the LAST real user turn
    (`_turn_starts`, so foreign-context contents are not boundaries — unlike
    strip_prior_turn_tool_parts, whose boundary for the critic lands on the
    last foreign content, inside the current turn). Everything from the user's
    question onward, all of this turn's tool results included, passes through
    unchanged: that is the critic's evidence.

    Only foreign-context contents are edited, so a user message that happens to
    begin like a tool line is never altered. A foreign content left holding
    nothing but ADK's preamble is dropped whole, the way ADK itself drops one
    (`_present_other_agent_message` returns None when only the preamble
    remains): a bare "For context: below is a transcript…" followed by no
    transcript is 409 characters of nothing.

    Copy-on-write: the edited contents are new objects, and nothing ADK shares
    with session history is mutated in place.
    """
    contents = llm_request.contents or []
    starts = _turn_starts(contents)
    if not starts or starts[-1] == 0:
        return None  # no earlier turn in this request — nothing to strip
    boundary = starts[-1]

    pruned: list[Any] = []
    changed = False
    for i, content in enumerate(contents):
        if i >= boundary or not _is_foreign_context(content):
            pruned.append(content)
            continue
        parts = content.parts or []
        # parts[0] is the preamble (that is what _is_foreign_context matched).
        kept_parts = [parts[0]] + [p for p in parts[1:] if not _is_foreign_tool_text(p)]
        if len(kept_parts) == len(parts):
            pruned.append(content)
            continue
        changed = True
        if len(kept_parts) > 1:
            pruned.append(genai_types.Content(role=content.role, parts=kept_parts))
        # Only the preamble left: the content disappears entirely.

    if changed:
        llm_request.contents = pruned
    return None
