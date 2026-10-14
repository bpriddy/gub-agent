"""
agents.router + the new root wiring (blend 04).

Three things are worth pinning here: the routing contract's shape, how the
decision is READ back (state, then the router's own event text, then the
fallback — a bad router must cost the deep path and nothing else), and the
pipeline structure the bot and Agentspace see at the boundary.
"""

from __future__ import annotations

import json
import logging

import pytest
from google.adk.events import Event, EventActions
from google.genai import types as genai_types
from pydantic import ValidationError

from gub_agent import config
from gub_agent.agents.router import (
    FORK_DECISION_KEY,
    ROUTER_STATE_KEY,
    BotEntityPrefix,
    decision_from,
    decision_is_this_turns,
    parse_bot_prefix,
    router_agent,
    user_text,
    with_entity_type,
)
from gub_agent.config import AGENT_NAME
from gub_agent.prompts import ROUTER_INSTRUCTION
from gub_agent.schemas.router import FALLBACK_DECISION, FAST_INTENTS, RouterDecision
from tests.helpers import invocation_ctx

INV = "inv-router"


def _decision_dict(**over) -> dict:
    base = {
        "intent": "campaign_status",
        "confidence": 0.93,
        "entity_surface": "Silverado 2026 Q3",
        "slots": {"complete": True},
        "missing_slots": [],
        "language": "ru",
    }
    base.update(over)
    return base


def _router_event(text: str) -> Event:
    return Event(
        invocation_id=INV,
        author="router",
        content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]),
    )


# ── the contract ──────────────────────────────────────────────────────────────


def test_the_decision_validates_and_keeps_the_surface_verbatim():
    decision = RouterDecision.model_validate(_decision_dict())
    assert decision.entity_surface == "Silverado 2026 Q3"
    assert decision.entity_id is None
    assert decision.slots.entity is None and decision.missing_slots == []


def test_confidence_is_bounded_and_the_intent_is_closed():
    with pytest.raises(ValidationError):
        RouterDecision.model_validate(_decision_dict(confidence=1.4))
    with pytest.raises(ValidationError):
        RouterDecision.model_validate(_decision_dict(intent="vibes"))


def test_the_fast_intents_are_the_five_fact_shaped_ones():
    assert FAST_INTENTS == {
        "campaign_status",
        "campaign_facts",
        "account_facts",
        "staff_lookup",
        "count_or_rank",
    }


def test_slots_are_named_fields_the_builder_validates_later():
    """The router fills the named fields; `fast_path.org_query_args` decides
    whether they assemble into a query. A free `dict[str, str]` was the spec's
    shape and is NOT used: gemini-3.5-flash leaves such a map empty (see
    `schemas/router.py`), which killed every count question."""
    decision = RouterDecision.model_validate(
        _decision_dict(
            intent="count_or_rank",
            slots={"entity": "campaigns", "status": "live", "complete": True},
        )
    )
    assert (decision.slots.entity, decision.slots.status) == ("campaigns", "live")
    assert decision.slots.complete is True


def test_slots_complete_defaults_to_false():
    """An unset flag must cost the deep path, never a confidently wrong count."""
    bare = {"intent": "count_or_rank", "confidence": 0.9, "language": "ru"}
    assert RouterDecision.model_validate(bare).slots.complete is False


# ── reading the decision back ─────────────────────────────────────────────────


async def test_the_decision_comes_from_state():
    ctx = await invocation_ctx(state={ROUTER_STATE_KEY: _decision_dict()}, invocation_id=INV)
    assert decision_from(ctx).intent == "campaign_status"


async def test_a_json_string_in_state_is_parsed():
    """`output_key` normally lands a dict; a string is the shape an older ADK
    (and a hand-seeded session) can produce."""
    ctx = await invocation_ctx(
        state={ROUTER_STATE_KEY: json.dumps(_decision_dict())}, invocation_id=INV
    )
    assert decision_from(ctx).intent == "campaign_status"


async def test_the_routers_own_event_is_the_fallback_source():
    """No state commit — the decision is still readable off the event the
    router emitted, fences and all."""
    ctx = await invocation_ctx(
        events=[_router_event("```json\n" + json.dumps(_decision_dict()) + "\n```")],
        invocation_id=INV,
    )
    assert decision_from(ctx).entity_surface == "Silverado 2026 Q3"


async def test_the_newest_router_event_wins():
    ctx = await invocation_ctx(
        events=[
            _router_event(json.dumps(_decision_dict(intent="assessment"))),
            _router_event(json.dumps(_decision_dict(intent="campaign_facts"))),
        ],
        invocation_id=INV,
    )
    assert decision_from(ctx).intent == "campaign_facts"


@pytest.mark.parametrize(
    "raw",
    [
        {"intent": "vibes", "confidence": 0.9},  # not in the literal set
        {"confidence": 0.9},  # no intent at all
        "not json at all",
        {},
    ],
)
async def test_anything_unusable_becomes_the_deep_path_never_an_error(raw):
    ctx = await invocation_ctx(state={ROUTER_STATE_KEY: raw}, invocation_id=INV)
    decision = decision_from(ctx)
    assert (decision.intent, decision.confidence) == ("exploratory", 0.0)
    assert decision == FALLBACK_DECISION


async def test_no_decision_at_all_becomes_the_deep_path():
    ctx = await invocation_ctx(invocation_id=INV)
    assert decision_from(ctx) == FALLBACK_DECISION


async def test_an_earlier_turns_router_event_is_never_this_turns_decision():
    """Session events are never trimmed, and a thread keeps up to 200 turns.
    When this turn's router left nothing readable, the newest-first fallback
    used to walk back into an earlier turn and return ITS decision — a fast
    path against the previous question's entity. Absent means the deep path."""
    earlier = Event(
        invocation_id="inv-earlier",
        author="router",
        content=genai_types.Content(
            role="model", parts=[genai_types.Part(text=json.dumps(_decision_dict()))]
        ),
    )
    ctx = await invocation_ctx(events=[earlier], invocation_id=INV)
    assert decision_from(ctx) == FALLBACK_DECISION

    # This turn's own event still wins over an earlier one.
    now = _router_event(json.dumps(_decision_dict(intent="campaign_facts")))
    ctx_now = await invocation_ctx(events=[earlier, now], invocation_id=INV)
    assert decision_from(ctx_now).intent == "campaign_facts"


# The three lines `decision_from` writes for a turn it cannot read are ones the
# logs are counted by (the dispatcher's reader writes them). `log=False` is for
# a second reader of the same turn (the format gate's brief) and must silence
# exactly these; the default must keep writing them.
_UNREADABLE = [
    pytest.param(None, "router: no decision for inv=inv-router — deep path", id="no-decision"),
    pytest.param("this is not json {", "router: decision is not JSON — deep path", id="not-json"),
    pytest.param(
        {"intent": "vibes", "confidence": 0.9},
        "router: decision failed validation (1) — deep path",
        id="failed-validation",
    ),
]


def _router_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "gub_agent.agents.router"]


@pytest.mark.parametrize(("raw", "line"), _UNREADABLE)
async def test_an_unreadable_decision_logs_its_line_by_default(raw, line, caplog):
    state = {} if raw is None else {ROUTER_STATE_KEY: raw}
    ctx = await invocation_ctx(state=state, invocation_id=INV)

    with caplog.at_level(logging.INFO, logger="gub_agent.agents.router"):
        assert decision_from(ctx) == FALLBACK_DECISION

    assert _router_lines(caplog) == [line]


@pytest.mark.parametrize(("raw", "line"), _UNREADABLE)
async def test_a_second_reader_asks_with_log_false_and_writes_nothing(raw, line, caplog):
    state = {} if raw is None else {ROUTER_STATE_KEY: raw}
    ctx = await invocation_ctx(state=state, invocation_id=INV)

    with caplog.at_level(logging.INFO, logger="gub_agent.agents.router"):
        assert decision_from(ctx, log=False) == FALLBACK_DECISION

    assert _router_lines(caplog) == []


async def test_log_false_changes_only_the_logging_not_what_is_read():
    ctx = await invocation_ctx(state={ROUTER_STATE_KEY: _decision_dict()}, invocation_id=INV)
    assert decision_from(ctx, log=False) == decision_from(ctx)


# ── whose decision is in state ────────────────────────────────────────────────


def _committed(invocation_id: str) -> Event:
    """The router's event as an output_key write leaves it: the decision in its
    state_delta."""
    return Event(
        invocation_id=invocation_id,
        author="router",
        actions=EventActions(state_delta={ROUTER_STATE_KEY: _decision_dict()}),
    )


async def test_a_decision_this_invocation_committed_is_this_turns():
    ctx = await invocation_ctx(events=[_committed(INV)], invocation_id=INV)
    assert decision_is_this_turns(ctx)


async def test_a_decision_an_earlier_invocation_committed_is_not():
    """The state key outlives the turn: with this turn's router silent, state
    still holds the previous turn's decision."""
    ctx = await invocation_ctx(events=[_committed("inv-earlier")], invocation_id=INV)
    assert ctx.session.state[ROUTER_STATE_KEY] == _decision_dict()
    assert not decision_is_this_turns(ctx)


async def test_a_decision_seeded_in_state_alone_is_not_provably_this_turns():
    ctx = await invocation_ctx(state={ROUTER_STATE_KEY: _decision_dict()}, invocation_id=INV)
    assert not decision_is_this_turns(ctx)


async def test_a_forks_stamp_makes_its_state_decision_this_turns():
    stamped = {ROUTER_STATE_KEY: _decision_dict(), FORK_DECISION_KEY: INV}
    ctx = await invocation_ctx(state=stamped, invocation_id=INV)
    assert decision_is_this_turns(ctx)

    other = await invocation_ctx(
        state={ROUTER_STATE_KEY: _decision_dict(), FORK_DECISION_KEY: "inv-other"},
        invocation_id=INV,
    )
    assert not decision_is_this_turns(other)


# ── the bot's entity prefixes (BE-6) ──────────────────────────────────────────
#
# The wire contract with gub-gchat-bot. The engine reads the entity TYPE off the
# message in code, so the bot's literal strings are pinned here: the first four
# below are what its own `selectionPrefix` / `assumptionPrefix`
# (src/entity/resolve.ts, as of 52089be) return, copied verbatim. If the bot
# rewords a HEAD (the keyword, the type, the id, the opening quote of the name),
# this fails here, and the cheap fix is the regex — the alternative is every
# id-bearing turn silently losing the fast path.

PIECE_ID = "5e1c7d2a-9b3f-4c68-8a41-0d2f6b7e9c13"
ACCOUNT_ID = "fdb3f9ff-8094-4851-8ad4-a62f82761ad9"
CAMPAIGN_ID = "7f3a9c10-2b4e-4d56-9a78-1c3e5f7a9b0d"
STAFF_ID = "0b6d2e41-7c58-4a93-b1f6-3e8a5c9d7f20"

SELECTED_PIECE = (
    f'User selected piece {PIECE_ID} "Hero Spot 30s" (Chevy | Holiday 2026). '
    "Original question: статус Hero Spot?"
)
SELECTED_ACCOUNT = (
    f'User selected account {ACCOUNT_ID} "Chevy". Original question: How is Chevy doing?'
)
GUESS_CAMPAIGN = (
    f'Best guess: campaign {CAMPAIGN_ID} "Silverado" (Chevy) — the user did not choose this. '
    "Ignore the guess if the question is not about it. Original question: какой статус?"
)
GUESS_STAFF = (
    f'Best guess: staff {STAFF_ID} "Alex Kim" — the user did not choose this. '
    "Ignore the guess if the question is not about it. "
    "Original question: what does Alex work on?"
)


def _selected(kind: str, entity_id: str, name: str, question: str, account: str | None = None):
    """`selectionPrefix`, restated for the combinations the four literals do not
    cover. The literals above are what pins the wording; this only varies it."""
    acct = f" ({account})" if account else ""
    return f'User selected {kind} {entity_id} "{name}"{acct}. Original question: {question}'


def _guess(kind: str, entity_id: str, name: str, question: str, account: str | None = None):
    acct = f" ({account})" if account else ""
    return (
        f'Best guess: {kind} {entity_id} "{name}"{acct} — the user did not choose this. '
        f"Ignore the guess if the question is not about it. Original question: {question}"
    )


@pytest.mark.parametrize(
    ("text", "kind", "entity_id"),
    [
        pytest.param(SELECTED_PIECE, "piece", PIECE_ID, id="selected-piece-with-account"),
        pytest.param(SELECTED_ACCOUNT, "account", ACCOUNT_ID, id="selected-account-no-account"),
        pytest.param(GUESS_CAMPAIGN, "campaign", CAMPAIGN_ID, id="guess-campaign-with-account"),
        pytest.param(GUESS_STAFF, "staff", STAFF_ID, id="guess-staff-no-account"),
    ],
)
def test_the_bots_own_prefixes_parse_to_their_type_and_id(text, kind, entity_id):
    assert parse_bot_prefix(text) == BotEntityPrefix(kind, entity_id)


@pytest.mark.parametrize("kind", ["campaign", "account", "piece", "staff"])  # CLARIFIABLE_TYPES
@pytest.mark.parametrize("make", [_selected, _guess], ids=["selected", "guess"])
def test_every_type_the_bot_can_offer_parses_under_both_heads(make, kind):
    text = make(kind, CAMPAIGN_ID, "Name", "status?", account="Acct")
    assert parse_bot_prefix(text) == BotEntityPrefix(kind, CAMPAIGN_ID)


@pytest.mark.parametrize("make", [_selected, _guess], ids=["selected", "guess"])
@pytest.mark.parametrize(
    "question",
    [
        "какой сейчас статус у Silverado 2026 Q3?",
        "сколько live кампаний у этого клиента — и сколько в питче?",
        'как дела у "Silverado"? (срочно)',
        "статус:\nбюджет:\nсроки:",  # a multi-line question
        "",  # nothing after the prefix at all
    ],
)
def test_the_original_question_never_changes_what_the_prefix_says(make, question):
    """The question is the reader's own words — Russian, quoted, bracketed,
    multi-line — and none of it may reach the type."""
    text = make("piece", PIECE_ID, "Hero Spot 30s", question, account="Chevy | Holiday 2026")
    assert parse_bot_prefix(text) == BotEntityPrefix("piece", PIECE_ID)


def test_a_name_with_quotes_and_brackets_does_not_move_the_id():
    text = _selected("campaign", CAMPAIGN_ID, 'The "Big" One (2026)', "статус?", account="A (B)")
    assert parse_bot_prefix(text) == BotEntityPrefix("campaign", CAMPAIGN_ID)


def test_a_prefix_quoted_inside_the_question_is_not_the_bots():
    """Only the HEAD of the message counts. A reader who pastes (or a thread
    that quotes) a prefix later in the text must not retype the entity."""
    inner = _selected("account", ACCOUNT_ID, "Chevy", "x")
    assert parse_bot_prefix(_selected("piece", PIECE_ID, "Hero", inner)) == BotEntityPrefix(
        "piece", PIECE_ID
    )
    assert parse_bot_prefix(f"Привет! Скопировал вот это: {inner}") is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "статус Silverado 2026 Q3",  # the common turn: no prefix at all
        "Who is Alex Kim?",
        f'User selected {PIECE_ID} "Hero Spot" Original question: x',  # no type
        f"User selected piece {PIECE_ID} Hero Spot. Original question: x",  # unquoted name
        f'user selected piece {PIECE_ID} "Hero Spot". Original question: x',  # not the bot's case
        f'Best guess piece {PIECE_ID} "Hero Spot". Original question: x',  # head lacks its colon
        f'User selected Piece {PIECE_ID} "Hero Spot". Original question: x',  # types are lower-case
        f'User chose piece {PIECE_ID} "Hero Spot". Original question: x',
    ],
)
def test_text_that_is_not_the_bots_prefix_has_no_type(text):
    assert parse_bot_prefix(text) is None


def test_the_router_prompt_documents_the_same_two_heads_the_parser_reads():
    """The prompt tells the model to copy the uuid from these two heads; the
    parser reads the type from them. If one moves, the other must."""
    assert '"User selected <type> <uuid>"' in ROUTER_INSTRUCTION
    assert '"Best guess: <type> <uuid>"' in ROUTER_INSTRUCTION


@pytest.mark.parametrize(
    ("text", "entity_id", "expected"),
    [
        pytest.param(SELECTED_PIECE, PIECE_ID, "piece", id="router-copied-the-id"),
        pytest.param(GUESS_CAMPAIGN, CAMPAIGN_ID, "campaign", id="a-guess-the-router-kept"),
        # The router decided the guess is not about the question and dropped it
        # (prompts/router.py): there is no id, so there is nothing to type.
        pytest.param(GUESS_CAMPAIGN, None, None, id="router-dropped-the-guess"),
        # An id the prefix does not carry — invented, or mangled while copying —
        # is an id nobody can vouch for. Never typed by guesswork.
        pytest.param(SELECTED_PIECE, ACCOUNT_ID, None, id="another-id"),
        pytest.param(SELECTED_PIECE, PIECE_ID[:8] + "…", None, id="a-truncated-copy"),
        pytest.param(SELECTED_PIECE, PIECE_ID.upper(), None, id="a-re-cased-copy"),
        pytest.param("статус Hero Spot?", PIECE_ID, None, id="an-id-with-no-prefix-at-all"),
    ],
)
def test_the_type_is_set_only_when_the_routers_id_is_the_prefixs(text, entity_id, expected):
    decision = RouterDecision.model_validate(
        _decision_dict(intent="campaign_facts", entity_id=entity_id)
    )
    typed = with_entity_type(decision, text)
    assert typed.entity_type == expected
    assert typed.entity_id == entity_id  # the id itself is never rewritten


def test_the_type_is_derived_never_trusted():
    """A decision built (or, one day, emitted) WITH a type has it replaced by
    what the message says — including None when the message says nothing."""
    claimed = RouterDecision.model_validate(
        _decision_dict(entity_id=PIECE_ID, entity_type="campaign")
    )
    assert claimed.entity_type == "campaign"  # the field validates…
    assert with_entity_type(claimed, SELECTED_PIECE).entity_type == "piece"  # …and is overwritten
    assert with_entity_type(claimed, "статус Hero Spot?").entity_type is None
    no_id = RouterDecision.model_validate(_decision_dict(entity_type="campaign"))
    assert with_entity_type(no_id, SELECTED_PIECE).entity_type is None


def test_with_entity_type_returns_a_copy_and_leaves_the_original_alone():
    """The decision may be the object held in session state."""
    decision = RouterDecision.model_validate(_decision_dict(entity_id=PIECE_ID))
    typed = with_entity_type(decision, SELECTED_PIECE)
    assert typed is not decision and decision.entity_type is None
    assert with_entity_type(typed, SELECTED_PIECE) is typed  # nothing to change: no copy


async def test_decision_from_reads_the_type_off_this_turns_message():
    ctx = await invocation_ctx(
        state={ROUTER_STATE_KEY: _decision_dict(intent="campaign_facts", entity_id=PIECE_ID)},
        invocation_id=INV,
        user_text=SELECTED_PIECE,
    )
    decision = decision_from(ctx)
    assert (decision.intent, decision.entity_id, decision.entity_type) == (
        "campaign_facts",
        PIECE_ID,
        "piece",
    )
    # …whichever way the decision reached us: the event text is a source too.
    via_event = await invocation_ctx(
        events=[_router_event(json.dumps(_decision_dict(entity_id=CAMPAIGN_ID)))],
        invocation_id=INV,
        user_text=GUESS_CAMPAIGN,
    )
    assert decision_from(via_event).entity_type == "campaign"


async def test_the_type_survives_the_speculative_roots_this_turn_only_stand_in():
    """`speculation._decision_of_this_turn` asks `decision_from` about a bare
    stand-in with no `user_content` and an empty state when the router
    committed nothing; the prefix is then read from the session's user event,
    the way `user_text` always falls back to it."""
    from types import SimpleNamespace

    user = Event(
        invocation_id=INV,
        author="user",
        content=genai_types.Content(role="user", parts=[genai_types.Part(text=SELECTED_PIECE)]),
    )
    router = _router_event(json.dumps(_decision_dict(intent="campaign_facts", entity_id=PIECE_ID)))
    stand_in = SimpleNamespace(
        invocation_id=INV, session=SimpleNamespace(state={}, events=[user, router])
    )
    assert decision_from(stand_in).entity_type == "piece"


async def test_a_turn_with_no_bot_id_is_untouched():
    """The common turn — no prefix, no id — is exactly the decision the router
    emitted, and no type appears from anywhere."""
    ctx = await invocation_ctx(
        state={ROUTER_STATE_KEY: _decision_dict()}, invocation_id=INV, user_text="статус Silverado"
    )
    decision = decision_from(ctx)
    assert decision == RouterDecision.model_validate(_decision_dict())
    assert decision.entity_id is None and decision.entity_type is None
    # And the router's fallback never carries one.
    assert decision_from(await invocation_ctx(invocation_id=INV, user_text=GUESS_STAFF)) == (
        FALLBACK_DECISION
    )


def test_the_router_never_sees_or_stores_the_type():
    """`entity_type` is code's, not the model's: the structured-output schema
    the router is given, and the decision it stores, are what they were before
    the field existed."""
    assert set(RouterDecision.model_json_schema()["properties"]) == {
        "intent",
        "confidence",
        "entity_surface",
        "entity_id",
        "slots",
        "missing_slots",
        "language",
    }
    typed = RouterDecision.model_validate(_decision_dict(entity_id=PIECE_ID, entity_type="piece"))
    assert typed.entity_type == "piece"
    assert "entity_type" not in typed.model_dump()
    assert "entity_type" not in typed.model_dump_json()


# ── the question ──────────────────────────────────────────────────────────────


async def test_user_text_prefers_user_content_and_falls_back_to_the_events():
    ctx = await invocation_ctx(invocation_id=INV, user_text="статус Silverado 2026 Q3")
    assert user_text(ctx) == "статус Silverado 2026 Q3"

    seeded = await invocation_ctx(
        invocation_id=INV,
        events=[
            Event(
                invocation_id=INV,
                author="user",
                content=genai_types.Content(
                    role="user", parts=[genai_types.Part(text="сколько live кампаний")]
                ),
            )
        ],
    )
    assert user_text(seeded) == "сколько live кампаний"


# ── the agent + the root wiring ───────────────────────────────────────────────


def test_the_router_is_a_no_tool_typed_classifier():
    assert router_agent.name == "router"  # the author the bot ignores
    assert router_agent.tools == []
    assert router_agent.output_schema is RouterDecision
    assert router_agent.output_key == ROUTER_STATE_KEY


def test_the_root_is_echo_then_router_then_dispatcher():
    """The boundary contract: same engine, same stream_query, and sandbox_echo
    still first so the provenance event precedes any work. With
    SPECULATIVE_DEEP (the default) the router and the dispatcher sit inside
    `speculative_dispatch` (tests/unit/test_speculation.py pins both shapes)."""
    from gub_agent.agent import deep_agent, root_agent
    from gub_agent.agents.speculation import SPECULATIVE_DEEP

    names = [a.name for a in root_agent.sub_agents]
    if SPECULATIVE_DEEP:
        assert names == ["sandbox_echo", "speculative_dispatch"]
        names = [names[0], *(a.name for a in root_agent.sub_agents[1].sub_agents)]
    assert names == ["sandbox_echo", "router", "dispatcher"]
    # The deep path is the pipeline as it was, minus the echo that moved up —
    # with the format gate and the critic side by side unless CRITIC_PARALLEL=0
    # (tests/unit/test_critic_parallel.py pins both shapes).
    middle = ["format_and_critic"] if config.CRITIC_PARALLEL else ["format_gate"]
    assert [a.name for a in deep_agent.sub_agents] == [
        AGENT_NAME,
        *middle,
        "critic_gate",
        "loop_escalator",
    ]
    assert deep_agent.max_iterations == 2


def test_the_executor_keeps_its_guardrail_callbacks():
    """The per-pass budget resets and the round limiter stay attached to the
    executor — the fast path must not have moved them onto the new root."""
    from gub_agent.agent import executor_agent

    assert executor_agent.before_agent_callback is not None
    assert executor_agent.before_model_callback is not None
    assert executor_agent.before_tool_callback is not None
    assert executor_agent.after_tool_callback is not None


def test_both_destinations_are_in_the_dispatchers_subtree():
    from gub_agent.agent import dispatcher

    assert [a.name for a in dispatcher.sub_agents] == ["fast_path", "gub_pipeline"]
