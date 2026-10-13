"""
The rundown shape (BE-5) and the executor's FORMAT (FE-6b) — the formatter's
RUNDOWN rule, the budgets it runs under, and which turns get them.

THE PROBLEM. A broad question ("how is Chevy doing?") is answered by an
executor that wrote a paragraph per campaign. The formatter then had to fit it
into 25-word bullets and 250 words in all (follow-ups and assumptions counted),
and a table-first shape rule that said "two or more entities across two or more
metrics -> one table": the narrative had no home and 3 of 4 live Chevy
exploratory payloads carried no text or bullets at all.

THE FIX, pinned here:
  * the formatter prompt gets a RUNDOWN rule AHEAD of the table rule: three or
    more entities with narrative -> one bullet per entity, 40 words, up to 12 in
    at most two blocks, 450 words in all, never tables alone over 1,500 chars of
    narrative;
  * the rundown's budgets are the contract's, not only the prompt's, and they
    apply to the turns the router calls exploratory or assessment (the question's
    own words when there is no intent). Every other turn keeps today's budgets
    and today's feedback;
  * they reach ADK's own validation of the formatter's reply, and the early
    abort's scan, and nothing else — no leak into the pipeline around the gate;
  * the executor writes in the shape the synthesizer will (bold opener, a bullet
    per entity, no tables), so the voice does not flip at the end.

The gate-flow tests run the real `FormatGate`; two of them run the REAL
formatter LlmAgent through ADK with only the model stubbed, because "the
budgets reach ADK's in-run validation" cannot be shown against a stand-in.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from google.adk.agents import SequentialAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from pydantic import ValidationError

from gub_agent import config
from gub_agent.agents import format_gate as format_gate_module
from gub_agent.agents.evidence_index import (
    reset_evidence_index,
    set_answer_draft,
)
from gub_agent.agents.format_gate import (
    RUNDOWN_INTENTS,
    FormatGate,
    wants_rundown,
)
from gub_agent.agents.formatter import formatter_agent
from gub_agent.config import AGENT_NAME, GEMINI_MODEL
from gub_agent.models import VendorRouter
from gub_agent.prompts import EXECUTOR_INSTRUCTION, FORMATTER_INSTRUCTION
from gub_agent.schemas import AnswerPayload
from gub_agent.schemas.router import Intent

from .test_brief_question import EXECUTOR, PREFIXED, _ctx, _real
from .test_format_gate import (
    INV,
    LONG_BULLET,
    _bulleted,
    _Executor,
    _scripted,
    _seed_index,
    _StreamingModel,
    _valid_dict,
)


@pytest.fixture(autouse=True)
def _no_evidence_rows_event(monkeypatch):
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", False)
    yield
    reset_evidence_index(SimpleNamespace(invocation_id=INV))  # the draft store is process-wide


def _flat(text: str) -> str:
    return " ".join(text.split())


# ── the formatter's RUNDOWN rule ─────────────────────────────────────────────


def test_the_rundown_rule_comes_before_the_table_rule_in_a_first_match_wins_list():
    prompt = _flat(FORMATTER_INSTRUCTION)

    first_match = prompt.index("FIRST match wins")
    rundown = prompt.index("RUNDOWN")
    table = prompt.index("Three or more fields of one entity")

    assert first_match < rundown < table
    # …and it says so, so a model that matches both knows which wins
    assert "This rule comes BEFORE the table rule" in prompt


def test_the_rundown_rule_states_its_numbers_and_its_gate():
    prompt = _flat(FORMATTER_INSTRUCTION)
    rule = prompt[prompt.index("RUNDOWN") : prompt.index("Three or more fields of one entity")]

    for needle in (
        "ONE BULLET PER ENTITY",
        "THREE OR MORE entities",
        "40 words or fewer each",
        "At most 12 bullets, in at most 2 bullet blocks",
        "450 words in all",
        "more than 1,500 characters of narrative",
        "INTENT is exploratory or assessment",
        "with no INTENT line",
    ):
        assert needle in rule, needle
    # each bullet cites its own entity, as the list rule's bullets do
    assert "an evidence id of THAT entity" in rule


def test_the_prompt_says_what_the_question_and_intent_lines_are():
    prompt = _flat(FORMATTER_INSTRUCTION)

    assert "QUESTION: (what the reader asked) and INTENT:" in prompt
    assert "context, never evidence" in prompt
    assert "No INTENT line means the classification is unknown" in prompt


def test_a_fact_question_keeps_todays_shape_rules():
    """The lines a lookup is shaped by — the table rule, the list rule, the
    assessment rule, the single-fact rule — are the ones that were there, with
    their budgets, and the rundown rule does not claim them."""
    prompt = _flat(FORMATTER_INSTRUCTION)

    assert (
        "Three or more fields of one entity, or two or more entities across two or more "
        "metrics → one table whose last column is the source id." in prompt
    )
    assert (
        "A LIST of records — three or more entities, even with a single attribute each → "
        "ONE BULLET PER RECORD (at most 7), 25 words or fewer each" in prompt
    )
    assert "ASSESSMENT question → 2-4 bullets, 25 words or fewer each" in prompt
    assert "Otherwise — a single FACT → at most one text block (60 words or fewer)." in prompt
    assert "headline IS the answer: the value for a fact question" in prompt
    # the rundown is limited to the intents that read as one
    assert RUNDOWN_INTENTS == {"exploratory", "assessment"}


# ── the executor's FORMAT (FE-6b) ────────────────────────────────────────────


def test_the_executor_writes_in_the_shape_the_synthesizer_will():
    """The synthesizer opens with the answer in one bold sentence, then a bullet
    per named thing — the name in bold, an em dash, the point — and never a
    table or a heading (a chat client renders neither). The executor's draft is
    what the reader watches before that, so it is told the same."""
    prompt = _flat(EXECUTOR_INSTRUCTION)

    assert "one sentence on its own line, in bold" in prompt
    assert "one bullet per entity: its name in bold, an em dash, then the point" in prompt
    assert '"- **<name>** — <status>, <dates and figures>; <what is at risk or next>."' in prompt
    assert "No markdown headings and no tables" in prompt
    # …and no longer told to write a table the chat bubble shows as raw pipes
    assert "brief tables" not in prompt
    assert "Use bullet points or brief" not in prompt


def test_the_executors_example_is_a_pattern_not_a_real_record():
    """The prompt's example must not be a name the model can echo back as a
    fact (and the repo is public): placeholders, nothing from a real account."""
    prompt = _flat(EXECUTOR_INSTRUCTION)
    start = prompt.index("The pattern, with the placeholders")

    assert "<name>" in prompt[start : start + 300]
    for real in ("Silverado", "Chevy", "Utah"):
        assert real not in prompt[start : start + 300]


# ── which turns are rundowns ─────────────────────────────────────────────────


@pytest.mark.parametrize("intent", ["exploratory", "assessment"])
def test_the_routers_open_read_intents_are_rundowns(intent):
    assert wants_rundown("anything at all", intent) is True
    assert wants_rundown("when does the Q3 push end?", intent) is True


@pytest.mark.parametrize(
    "intent",
    [
        "campaign_status",
        "campaign_facts",
        "account_facts",
        "staff_lookup",
        "count_or_rank",
        "file_lookup",
        "market_enrichment",
        "workspace_personal",
        "smalltalk",
    ],
)
def test_every_other_intent_is_not_a_rundown_whatever_the_words_say(intent):
    """The router's classification wins over the question's words: a lookup
    phrased "how is…" is still a lookup."""
    assert wants_rundown("how is chevy doing?", intent) is False


def test_the_intent_labels_checked_here_are_the_routers_whole_set():
    """If the router grows an intent, this list must be looked at: a rundown is
    an opt-in for exactly two of them."""
    routers = set(Intent.__args__)
    assert RUNDOWN_INTENTS < routers
    assert len(routers) == 11


@pytest.mark.parametrize(
    "question",
    [
        "how is chevy doing?",
        "How's Chevy?",
        "How are we doing on Chevy this quarter?",
        "what's new with Chevy",
        "What is going on at Chevy?",
        "what changed on chevy this week",
        "where do we stand on Chevy",
        "should we worry about Chevy?",
        "catch me up on Chevy",
        "give me an overview of Chevy",
        "tell me about the Chevy account",
        "как дела у Chevy?",
        "что нового по Chevy?",
        "расскажи про Chevy",
        # the bot's card-pick prefix, then the reader's own words
        PREFIXED,
    ],
)
def test_no_intent_an_open_read_question_is_a_rundown(question):
    assert wants_rundown(question, None) is True


@pytest.mark.parametrize(
    "question",
    [
        "",
        "is the Q3 push live?",
        "what is the budget of Red Tag?",
        "how many campaigns are live?",
        "how much did Red Tag cost",
        "when does the Q3 push end?",
        "who leads Red Tag?",
        "which campaigns are live?",
        "what is the status of Red Tag?",
        "сколько кампаний у Chevy?",
        "когда заканчивается Red Tag?",
        # an open read that also asks for one fact keeps the standard budgets
        "how is chevy doing and how many campaigns are live?",
    ],
)
def test_no_intent_anything_but_a_clear_open_read_keeps_the_standard_budgets(question):
    assert wants_rundown(question, None) is False


def test_the_card_pick_prefix_is_not_read_as_the_question():
    """Both of the bot's prefixes end in `Original question:` — the words
    before it are an entity name and an instruction, not the reader's."""
    named_like_a_fact = (
        'User selected campaign 0f1bd315-6c13-4fea-8dd6-6a93bb0fb6da "Who Knew | When" '
        "(Chevy). Original question: how is it going?"
    )
    named_like_an_open_read = (
        'User selected campaign 0f1bd315-6c13-4fea-8dd6-6a93bb0fb6da "How Is It Going" '
        "(Chevy). Original question: what is the budget?"
    )
    guess = (
        'Best guess: campaign 0f1bd315-6c13-4fea-8dd6-6a93bb0fb6da "Chevy | Red Tag 2026" '
        "— the user did not choose this. Ignore the guess if the question is not about it. "
        "Original question: when does it end?"
    )

    assert wants_rundown(named_like_a_fact, None) is True
    assert wants_rundown(named_like_an_open_read, None) is False
    assert wants_rundown(guess, None) is False


# ── the gate: the budgets follow the question ────────────────────────────────

# 30 words: over the standard 25-word budget, inside the rundown's 40.
THIRTY = "the Q3 push is live and " + " ".join(["moving"] * 24)


def _long(first: str = THIRTY) -> dict:
    """A grounded payload whose bullets are over the standard budget and
    inside the rundown's: `_valid_dict`'s evidence, bullets of THIRTY words."""
    payload = _valid_dict()
    payload["blocks"] = [{"kind": "bullets", "items": [first, first, first]}]
    return payload


async def _run(ctx, script, draft: str | None = None):
    """Run the real gate over a scripted formatter: (events, the formatter)."""
    _seed_index()
    if draft is not None:
        set_answer_draft(INV, draft)
    formatter = _scripted(script)
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    return [e async for e in gate.run_async(ctx)], formatter


def _feedback(formatter) -> list[str]:
    return [b.split("FORMAT_FEEDBACK", 1)[1] for b in formatter.briefs if "FORMAT_FEEDBACK" in b]


async def test_a_rundown_turn_accepts_bullets_over_the_standard_budget():
    events, formatter = await _run(
        await _ctx("how is chevy doing?", _real("assessment")), [_long()]
    )

    assert len(formatter.runs) == 1  # accepted at the first attempt
    assert _feedback(formatter) == []
    assert [e.author for e in events] == ["formatter"]


async def test_a_fact_turn_rejects_the_same_payload_with_todays_feedback():
    """The identical payload on a lookup: today's budget, today's words."""
    _, formatter = await _run(
        await _ctx("is the Q3 push live?", _real("campaign_facts")), [_long()]
    )

    assert len(formatter.runs) == 3  # all three attempts, then the template
    assert "bullet 0 is 30 words, over the 25-word budget" in _feedback(formatter)[0]


async def test_no_intent_the_question_decides_which_budgets():
    open_read, lookup = "how is chevy doing?", "is the Q3 push live?"

    _, as_rundown = await _run(await _ctx(open_read, None), [_long()])
    _, as_lookup = await _run(await _ctx(lookup, None), [_long()])

    assert len(as_rundown.runs) == 1
    assert len(as_lookup.runs) == 3


async def test_the_fast_path_is_never_a_rundown():
    """Its draft is a deterministic lookup, whatever the router labelled it —
    exploratory included (a declined fast decision is formatted on the deep
    path, with its label; a fast-path draft is not)."""
    draft = (
        "DETERMINISTIC LOOKUP (fast path) — intent=exploratory, tool=get_campaign. "
        "Answer the question from these fields; nothing else was retrieved.\n"
        "QUESTION: how is chevy doing?\nRESULT:\nstatus: live"
    )
    _, formatter = await _run(
        await _ctx("how is chevy doing?", _real("exploratory")), [_long()], draft=draft
    )

    assert len(formatter.runs) == 3
    assert "25-word budget" in _feedback(formatter)[0]


async def test_a_rundown_still_fails_the_checks_that_are_not_budgets():
    """Wider words, same gate: an ungrounded name is rejected on a rundown
    exactly as anywhere (`Tesla` is in no tool result)."""
    bad = _long("the Q3 push for Tesla is live and " + " ".join(["moving"] * 22))

    _, formatter = await _run(await _ctx("how is chevy doing?", _real("exploratory")), [bad])

    assert len(formatter.runs) == 3
    assert "ungrounded entity: Tesla" in _feedback(formatter)[0]


async def test_a_rundown_over_its_own_budget_is_told_the_rundowns_numbers():
    forty_one = "the Q3 push is live and " + " ".join(["moving"] * 35)  # 41 words
    _, formatter = await _run(
        await _ctx("how is chevy doing?", _real("exploratory")), [_long(forty_one), _long()]
    )

    assert "bullet 0 is 41 words, over the 40-word budget" in _feedback(formatter)[0]
    assert len(formatter.runs) == 2  # …and the retry, inside 40, is accepted


async def test_the_gate_leaves_the_pipeline_around_it_on_the_standard_budgets():
    """The budgets are in force for the gate's own validations and the formatter's
    step — not between the events it yields, where the rest of the pipeline
    runs (and where a stale setting would loosen anything else that validates)."""
    _seed_index()
    formatter = _scripted([_long()])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    ctx = await _ctx("how is chevy doing?", _real("exploratory"))

    def verdict() -> str:
        try:
            AnswerPayload.model_validate(_long())
        except ValidationError:
            return "standard"
        return "loose"

    between = []
    async for _ in gate.run_async(ctx):
        between.append(verdict())

    assert len(formatter.runs) == 1  # the rundown path ran: its budgets were set…
    assert between == ["standard"] and verdict() == "standard"  # …and are not left set


# ── the real formatter: ADK's own validation, and the early abort's scan ────


async def _through_adk(monkeypatch, *, decision: dict | None, question: str, replies: list[str]):
    """The real formatter LlmAgent (callbacks, output_schema, ADK's flow) behind
    the real gate on an InMemoryRunner with SSE, only the model stubbed.
    Returns (model, the session's `format_feedback` events, the log lines)."""
    model = _StreamingModel(
        model="stub", replies=replies, requests=[], sent=[], finished=[], closed=[]
    )
    formatter = formatter_agent.clone(
        update={"model": VendorRouter(model=GEMINI_MODEL, gemini=model)}
    )
    root = SequentialAgent(
        name="root",
        sub_agents=[
            _Executor(name=AGENT_NAME),
            FormatGate(name="format_gate", sub_agents=[formatter]),
        ],
    )
    runner = InMemoryRunner(agent=root, app_name="gub")
    session = await runner.session_service.create_session(
        app_name="gub",
        user_id="u",
        state={"router_decision": decision} if decision is not None else {},
    )
    [
        _
        async for _ in runner.run_async(
            user_id="u",
            session_id=session.id,
            new_message=genai_types.Content(role="user", parts=[genai_types.Part(text=question)]),
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        )
    ]
    stored = (
        await runner.session_service.get_session(app_name="gub", user_id="u", session_id=session.id)
    ).events
    feedback = [
        e.actions.state_delta["format_feedback"]
        for e in stored
        if e.author == "format_gate" and e.actions and "format_feedback" in e.actions.state_delta
    ]
    payloads = [
        e.actions.state_delta["answer_payload"]
        for e in stored
        if e.actions and "answer_payload" in (e.actions.state_delta or {})
    ]
    return model, feedback, payloads


@pytest.mark.parametrize("flag", ["0", "contract"])
async def test_adk_validates_a_rundown_reply_under_the_rundowns_budgets(monkeypatch, caplog, flag):
    """The 26-word bullet the contract rejects on a lookup (`LONG_BULLET`) is
    accepted by the formatter's own `output_schema` validation on an open-read
    question — one model call, no feedback — and, with the early abort on, the
    scan does not close a valid rundown mid-stream."""
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(format_gate_module, "FORMAT_GATE_EARLY_ABORT", flag)
    reply = json.dumps(_bulleted(LONG_BULLET), indent=2)

    model, feedback, payloads = await _through_adk(
        monkeypatch,
        decision=_real("assessment"),
        question="how is chevy doing?",
        replies=[reply, json.dumps(_bulleted(), indent=2)],
    )

    assert len(model.requests) == 1 and model.finished == [True] and model.closed == [None]
    assert feedback == []
    assert payloads[-1]["blocks"][0]["items"][0] == LONG_BULLET
    assert not [r for r in caplog.records if "early abort" in r.getMessage()]
    # the brief the model read says what was asked (BE-4's lines, unchanged)
    brief = json.dumps(model.requests[0]["contents"], ensure_ascii=False)
    assert "QUESTION: how is chevy doing?" in brief and "INTENT: assessment" in brief


@pytest.mark.parametrize("flag", ["0", "contract"])
async def test_adk_still_rejects_it_on_a_lookup(monkeypatch, flag):
    monkeypatch.setattr(format_gate_module, "FORMAT_GATE_EARLY_ABORT", flag)

    model, feedback, payloads = await _through_adk(
        monkeypatch,
        decision=_real("campaign_status"),
        question="is the Q3 push live?",
        replies=[json.dumps(_bulleted(LONG_BULLET), indent=2), json.dumps(_bulleted(), indent=2)],
    )

    assert len(model.requests) == 2  # rejected, then the retry inside 25
    assert feedback and "bullet 0 is 26 words, over the 25-word budget" in feedback[0]
    assert payloads[-1]["blocks"][0]["items"][0] != LONG_BULLET


# ── the brief is the one BE-4 wrote ──────────────────────────────────────────


async def test_the_rundown_adds_nothing_to_the_brief():
    """The budget class is the gate's decision: the brief keeps its two header
    lines and gains no third (A2-3's contract: byte for byte)."""
    _, formatter = await _run(
        await _ctx("how is chevy doing?", _real("exploratory")), [_valid_dict()]
    )

    [brief] = formatter.briefs
    assert brief.startswith(
        "QUESTION: how is chevy doing?\nINTENT: exploratory\n\nEXECUTOR ANSWER (render this"
    )
    assert f"EXECUTOR ANSWER (render this — do not add facts):\n{EXECUTOR}\n" in brief
    assert "RUNDOWN" not in brief and "450" not in brief
