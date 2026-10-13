"""
The uncapped grounding store (BE-3): `agents/evidence_index.py` records it, the
gate's checks 2-3 (`agents/format_gate.py`) ground in it.

THE PROBLEM. The brief and the gate's grounding blob were both the capped index
(~400 entries, ~33-40 records). The executor writes from EVERYTHING it
retrieved, so whatever the cap did not admit — a refused row, a nested
`vehicle` dict, a bare `find` list, the tail of a status text — was readable in
the executor's answer and unwritable in the final one: the 2026-09-22 live
rejections of `MY27 Silverado HD LT` and `Cacti Green` were exactly that.

THE FIX, pinned here:
  * a per-pass store holds every scalar of every tool result (nested, bare list,
    refused row, long status text), resets with the index, and is NEVER briefed,
    emitted or citable — `citations ⊆ index` still means the capped index;
  * a number or name the capped index does not hold grounds when it is in the
    store AND in the executor's answer (whole numbers, whole words);
  * what grounded in the index before still grounds, with no new demand;
  * the true positives (`Tesla`, `Ford`, `York`, `York City`) still fail, also
    against look-alike substrings (`Affordable`, `Yorkshire`) in a big store.

The gate-flow tests run the real `FormatGate` over a scripted formatter, so each
reads the same on the old code (the gate rejects, retries, ships a template).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gub_agent import config
from gub_agent.agents import evidence_index as ei
from gub_agent.agents import format_gate as fg
from gub_agent.agents.evidence_index import (
    evidence_index,
    evidence_rows,
    record_evidence,
    reset_evidence_index,
)
from gub_agent.agents.format_gate import MAX_FORMAT_ATTEMPTS, FormatGate, compose_brief
from gub_agent.prompts.formatter import FORMATTER_INSTRUCTION
from gub_agent.schemas import AnswerPayload

from .test_format_gate import INV, _gate_ctx, _scripted

FIXTURE = Path(__file__).parents[1] / "fixtures" / "recorded_chevy_turn_2026-09-17.json"
UUID = "0f1bd315-6c13-4fea-8dd6-6a93bb0fb6da"


@pytest.fixture(autouse=True)
def _no_evidence_rows_event(monkeypatch):
    monkeypatch.setattr(config, "EVIDENCE_ROWS_EVENT", False)


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _start(invocation_id: str = INV) -> SimpleNamespace:
    ctx = SimpleNamespace(invocation_id=invocation_id)
    reset_evidence_index(ctx)
    return ctx


def _detail() -> dict:
    """A campaign detail. The index reads the row's own scalars; everything under
    `vehicle` is a nested dict it never reads — a value only the store holds."""
    return {
        "id": "c9",
        "name": "Truck push",
        "status": "live",
        "budget": 1200000,
        "vehicle": {
            "trim": "MY27 Silverado HD LT",
            "finish": "Cacti Green",
            "msrp": 48250,
            "parent": UUID,
        },
    }


def _seed(ctx: SimpleNamespace | None = None) -> SimpleNamespace:
    ctx = ctx or _start()
    record_evidence(_tool("get_campaign"), {}, ctx, copy.deepcopy(_detail()))
    return ctx


def _index_blob() -> str:
    return fg._evidence_blob(evidence_index(INV))


def _store() -> str:
    return ei.grounding_text(INV)


def _payload(text: str, headline: str = "the Truck push is live", cite: str | None = None) -> dict:
    cite = cite or "get_campaign:c9:status"
    return {
        "kind": "answer",
        "headline": headline,
        "blocks": [{"kind": "text", "text": text}],
        "citations": [cite],
        "facts": [{"evidence_id": cite, "entity_id": "c9", "field": "status", "value": "live"}],
    }


async def _gate(executor_text: str, payload: dict) -> SimpleNamespace:
    """Run the real gate over a formatter that always returns `payload`."""
    formatter = _scripted([payload])
    gate = FormatGate(name="format_gate", sub_agents=[formatter])
    events = [e async for e in gate.run_async(await _gate_ctx(executor_text))]
    gated = [e for e in events if e.author == "format_gate" and e.actions]
    deltas = [e.actions.state_delta or {} for e in gated]
    feedback = [d["format_feedback"] for d in deltas if d.get("format_feedback")]
    return SimpleNamespace(
        runs=len(formatter.runs),
        feedback=feedback,
        accepted=events[-1].author == "formatter",
        events=events,
    )


def _rejected_on_every_attempt(result: SimpleNamespace) -> None:
    assert result.runs == MAX_FORMAT_ATTEMPTS, f"{result.runs} formatter runs: {result.feedback}"
    assert not result.accepted  # the template shipped


# ── the store: what it holds ──────────────────────────────────────────────────


def test_the_store_holds_what_the_capped_index_never_reads():
    ctx = _seed()
    # a bare list (`find`'s hits): the index drops one, the store wraps it
    record_evidence(
        _tool("find"), {}, ctx, [{"type": "piece", "id": "p1", "name": "Hero Spot", "score": 0.93}]
    )

    store = _store()
    for value in ("my27 silverado hd lt", "cacti green", "48250", "hero spot", "0.93"):
        assert value in store, value
        assert value not in _index_blob(), f"{value!r} was in the index; it proves nothing"
    # the index itself is untouched by the list: still only the campaign's rows
    assert set(evidence_index(INV)) == {
        "get_campaign:c9",
        "get_campaign:c9:name",
        "get_campaign:c9:status",
        "get_campaign:c9:budget",
    }


def test_the_store_takes_what_the_cap_refused(monkeypatch):
    monkeypatch.setattr(ei, "MAX_ENTRIES", 3)
    ctx = _start()
    record_evidence(
        _tool("org_query"), {}, ctx, {"results": [{"id": "r0", "name": "Alpha"}], "total": 1}
    )
    assert len(evidence_index(INV)) == 3  # full: the entity row, its one field, `total`
    record_evidence(
        _tool("org_query"),
        {},
        ctx,
        {"results": [{"id": "z1", "name": "Cacti Green", "budget": 48250}]},
    )

    assert not any("cacti green" in str(e["value"]).lower() for e in evidence_index(INV).values())
    assert ei.dropped_by_tool(INV) == {"org_query": 3}  # refused, and counted as before
    assert "cacti green" in _store()
    assert "48250" in _store()


def test_the_store_skips_plumbing_and_noise():
    ctx = _start()
    record_evidence(
        _tool("get_campaign"),
        {},
        ctx,
        {
            "id": "c1",
            "name": "Kept",
            "_sources": ["Plumbing Source"],
            "_cited": {"1DriveFileAAA": "Hidden File Name"},
            "live": True,
            "owner": None,
            "ref": UUID,
            "notes": {"summary": "Launch holds. [src: 1DriveFileBBB]", "_x": "Skipped Too"},
        },
    )
    record_evidence(
        _tool("get_campaign"), {}, ctx, {"error": True, "status": 404, "message": "Not Found Here"}
    )

    store = _store()
    assert "kept" in store and "launch holds." in store
    for absent in (
        "plumbing source",
        "hidden file name",
        "skipped too",
        "not found here",  # an error body is not evidence
        "1drivefilebbb",  # a source marker is not prose
        UUID,
        "true",
        "none",
    ):
        assert absent not in store, absent


def test_the_store_holds_a_status_text_whole_past_the_entity_row_cap():
    """The index's whole-entity value is cut at MAX_ENTITY_VALUE_CHARS and a
    nested status text is one field row; the store holds the lot."""
    ctx = _start()
    tail = "Closing risk: the Cacti Green run slips to June."
    long_text = "Status paragraph. " * 400 + tail
    record_evidence(
        _tool("get_account_overview"),
        {},
        ctx,
        {"id": "a1", "name": "Chevy", "currentState": {"status_markdown": long_text}},
    )
    assert len(long_text) > ei.MAX_ENTITY_VALUE_CHARS
    assert tail.lower() in _store()


# ── the store: what it is not ─────────────────────────────────────────────────


def test_the_store_is_never_briefed_or_emitted():
    _seed()
    index = evidence_index(INV)
    brief = compose_brief("The Truck push is live.", index, "", ei.provenance(INV))
    rows = json.dumps(evidence_rows(index), ensure_ascii=False)
    for secret in ("MY27 Silverado HD LT", "Cacti Green", "48250"):
        assert secret.lower() in _store()
        assert secret not in brief, f"{secret!r} leaked into the brief"
        assert secret not in rows, f"{secret!r} leaked into the rows event"


async def test_a_store_row_is_never_citable():
    """Citations stay a subset of the CAPPED index: an id built from a value only
    the store holds is an unknown citation, whatever grounds the prose."""
    _seed()
    cite = "get_campaign:c9:vehicle.trim"
    assert cite not in evidence_index(INV)
    result = await _gate(
        "The Truck push is live. The hero is the MY27 Silverado HD LT.",
        _payload("The hero is the MY27 Silverado HD LT.", cite=cite),
    )
    _rejected_on_every_attempt(result)
    assert "unknown citation" in result.feedback[0]
    assert cite in result.feedback[0]


def test_the_store_resets_with_the_index_and_is_per_invocation():
    first = _seed(_start("inv-a"))
    record_evidence(_tool("find"), {}, first, [{"name": "Pass One Only"}])
    second = _start("inv-b")
    record_evidence(_tool("get_campaign"), {}, second, {"id": "c2", "name": "Second Pass"})

    assert "pass one only" in ei.grounding_text("inv-a")
    assert "second pass" not in ei.grounding_text("inv-a")
    assert "pass one only" not in ei.grounding_text("inv-b")
    assert "second pass" in ei.grounding_text("inv-b")

    # a critic retry's fresh executor pass: the reset forgets the old results
    reset_evidence_index(first)
    assert ei.grounding_text("inv-a") == ""
    assert "second pass" in ei.grounding_text("inv-b")  # another invocation is untouched
    record_evidence(_tool("get_campaign"), {}, first, {"id": "c3", "name": "Retry Pass"})
    assert set(ei.grounding_text("inv-a").split("\n")) == {"c3", "retry pass"}


# ── checks 2-3: names ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "sentence"),
    [
        ("MY27 Silverado HD LT", "The hero is the MY27 Silverado HD LT."),
        ("Cacti Green", "The finish is Cacti Green."),
    ],
)
async def test_the_2026_09_22_names_pass_when_a_detail_payload_holds_them(name, sentence):
    _seed()
    assert name.lower() not in _index_blob()
    result = await _gate(f"The Truck push is live. {sentence}", _payload(sentence))
    assert result.accepted and result.runs == 1, result.feedback


async def test_a_name_in_a_tool_result_but_not_in_the_executor_answer_is_rejected():
    """The store widens what the EXECUTOR may say, never what the formatter may
    add on its own."""
    _seed()
    result = await _gate(
        "The Truck push is live.", _payload("The finish is Cacti Green, says the record.")
    )
    _rejected_on_every_attempt(result)
    assert "ungrounded entity" in result.feedback[0] and "Cacti Green" in result.feedback[0]
    assert "executor answer" in result.feedback[0]


@pytest.mark.parametrize("name", ["Tesla", "Ford", "York", "York City"])
async def test_the_true_positives_still_fail(name):
    """Even when the executor wrote the name too, and even with look-alike
    substrings in the store (`Affordable` holds `ford`, `Yorkshire` holds `york`)."""
    ctx = _seed()
    record_evidence(
        _tool("find"), {}, ctx, [{"type": "campaign", "name": "Affordable Yorkshire Pudding Co"}]
    )
    sentence = f"Rivals include {name}."
    result = await _gate(f"The Truck push is live. {sentence}", _payload(sentence))
    _rejected_on_every_attempt(result)
    assert "ungrounded entity" in result.feedback[0] and name in result.feedback[0]
    # the premise: the store holds the SUBSTRINGS, so only a whole-word match stops it
    assert "ford" in _store() and "york" in _store()


async def test_a_name_found_by_a_bare_list_grounds():
    ctx = _seed()
    record_evidence(_tool("find"), {}, ctx, [{"type": "piece", "id": "p1", "name": "Hero Spot"}])
    sentence = "The piece to watch is Hero Spot."
    result = await _gate(f"The Truck push is live. {sentence}", _payload(sentence))
    assert result.accepted and result.runs == 1, result.feedback


async def test_possessive_plural_and_cyrillic_spellings_ground_against_the_store():
    ctx = _seed()
    record_evidence(
        _tool("find"),
        {},
        ctx,
        [{"name": "Blazer EV"}, {"name": "Серебристый Силверадо"}],
    )
    for executor, sentence in (
        ("The Truck push sells Blazer EVs.", "The Truck push sells Blazer EVs."),
        ("The Truck push has a Blazer EV's trim.", "The Truck push has a Blazer EV's trim."),
        (
            "The push is в цвете Серебристый Силверадо.",
            "The push is в цвете Серебристый Силверадо.",
        ),
    ):
        result = await _gate(executor, _payload(sentence))
        assert result.accepted and result.runs == 1, (sentence, result.feedback)


# ── checks 2-3: numbers ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("1200000", "1200000"),
        ("1200000.0", "1200000"),
        ("1200.50", "1200.5"),
        ("100", "100"),
        ("100.00", "100"),
        ("09", "9"),
        ("0.5", "0.5"),
        ("0", "0"),
        ("48250", "48250"),
    ],
)
def test_numbers_match_the_store_by_value_not_by_spelling(raw, norm):
    assert fg._norm_number(raw) == norm


async def test_a_number_only_the_store_holds_grounds():
    _seed()
    assert "48250" not in _index_blob()
    for written in ("48,250", "48250"):
        result = await _gate(
            f"The Truck push is live. The trim starts at {written}.",
            _payload(f"The trim starts at {written}."),
        )
        assert result.accepted and result.runs == 1, (written, result.feedback)


async def test_a_number_the_store_holds_but_the_executor_did_not_write_is_rejected():
    _seed()
    result = await _gate("The Truck push is live.", _payload("The trim starts at 48,250."))
    _rejected_on_every_attempt(result)
    assert "ungrounded number" in result.feedback[0] and "48250" in result.feedback[0]
    assert "executor answer and in a tool result" in result.feedback[0]


async def test_a_number_the_executor_wrote_but_no_tool_returned_is_rejected():
    _seed()
    result = await _gate(
        "The Truck push is live. The trim starts at 52,000.", _payload("The trim starts at 52,000.")
    )
    _rejected_on_every_attempt(result)
    assert "ungrounded number" in result.feedback[0] and "52000" in result.feedback[0]


async def test_a_number_grounds_as_a_whole_number_not_as_a_substring():
    """4825 is a substring of 48250 and 315 of a UUID; neither is a number the
    store holds."""
    _seed()
    for number in ("4,825", "315"):
        result = await _gate(
            f"The Truck push is live. It is {number}.", _payload(f"It is {number}.")
        )
        _rejected_on_every_attempt(result)
        assert "ungrounded number" in result.feedback[0], number
    assert "48250" in _store() and UUID not in _store()  # the premise


# ── nothing that grounded before is rejected now ──────────────────────────────


async def test_what_the_index_grounds_needs_no_executor_word():
    _seed()
    result = await _gate(
        "Everything is fine.",
        _payload("The Truck push has a 1,200,000 budget.", headline="the Truck push is live"),
    )
    assert result.accepted and result.runs == 1, result.feedback


def test_without_a_grounding_the_gate_is_the_capped_index_gate():
    _seed()
    payload = AnswerPayload.model_validate(_payload("The finish is Cacti Green at 48,250."))
    problems = fg.gate_problems(payload, evidence_index(INV))
    assert any("ungrounded number" in p for p in problems)
    assert any("ungrounded entity" in p and "Cacti Green" in p for p in problems)
    granted = fg.gate_problems(
        payload,
        evidence_index(INV),
        fg.Grounding(_store(), "The finish is Cacti Green at 48,250."),
    )
    assert granted == []


# ── the rules text ────────────────────────────────────────────────────────────


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_brief_states_the_rephrased_rules():
    brief = _flat(compose_brief("x", {"org_query:a": {"value": "v"}}, ""))
    assert "Every number you write must appear in the executor answer and in a tool result" in brief
    assert (
        "Every capitalized word or phrase you write must appear in the executor answer "
        "and in a tool result" in brief
    )
    assert "must appear in a value above" not in brief


def test_the_formatter_prompt_states_the_same_rule_and_the_bullet_rule():
    prompt = _flat(FORMATTER_INSTRUCTION)
    assert (
        "Every capitalized word or phrase you write must appear in the executor answer "
        "and in a tool result" in prompt
    )
    assert "A bullet cites a row of the entity it is about" in prompt
    assert "must appear in the tool results" not in prompt


# ── the recorded Chevy turn ───────────────────────────────────────────────────


def _replay() -> list[dict]:
    ctx = _start()
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for r in recorded:
        record_evidence(_tool(r["name"]), {}, ctx, r["response"])
    return recorded


def test_the_recorded_turn_store_holds_what_the_cap_refused():
    recorded = _replay()
    blob = _index_blob()
    store = _store()
    # `get_account_overview` stubs the cap refused or evicted: named in the store.
    for name in ("chevrolet | ev bridge 2025", "chevrolet | blazer ev ss film"):
        assert name not in blob and name in store, name
    # …and every status text, nested or not, whole — however far past the
    # 2,000-character entity row cap it runs.
    texts = [r["response"]["statusMarkdown"] for r in recorded if r["name"] == "get_campaign"] + [
        next(r for r in recorded if r["name"] == "get_account_overview")["response"][
            "currentState"
        ]["status_markdown"]
    ]
    assert max(len(t) for t in texts) > ei.MAX_ENTITY_VALUE_CHARS
    for text in texts:
        assert ei.SRC_MARKER_RE.sub(" ", text).strip().lower() in store
    # the true positives are in no result of the turn
    for word in ("tesla", "ford", "york"):
        assert word not in store


async def test_a_name_the_recorded_turn_cap_refused_grounds_in_the_final_answer():
    recorded = _replay()
    campaign = next(r["response"] for r in recorded if r["name"] == "get_campaign")
    cite = f"get_campaign:{campaign['id']}"
    assert cite in evidence_index(INV)
    assert "blazer ev ss film" not in _index_blob()
    sentence = "An older campaign was Chevrolet | Blazer EV SS Film."
    payload = {
        "kind": "answer",
        "headline": "the account has older work",
        "blocks": [{"kind": "text", "text": sentence}],
        "citations": [cite],
        "facts": [{"evidence_id": cite, "entity_id": campaign["id"], "value": campaign["name"]}],
    }
    result = await _gate(f"On the account: {sentence}", payload)
    assert result.accepted and result.runs == 1, result.feedback
