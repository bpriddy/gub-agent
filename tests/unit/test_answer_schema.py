"""
schemas.answer — the answer contract's validators (blend 03).

What used to be prompt requests policed by the critic LLM is now pydantic:
filler, budgets, table shape, and the facts↔citations equality that the bot's
Workspace conflict filter (blend 05) depends on. These tests are the contract's
pin — a payload the executor pipeline emits and the bot renders must be
constructible exactly under these rules.
"""

from __future__ import annotations

import typing

import pytest
from pydantic import ValidationError

from gub_agent.schemas import AnswerPayload
from gub_agent.schemas.answer import Block, answer_budgets, count_words


def _answer(**over) -> dict:
    base = {
        "kind": "answer",
        "headline": "7 live campaigns",
        "blocks": [{"kind": "text", "text": "chevy runs 7 live campaigns."}],
        "citations": ["org_query:c1:total"],
        "facts": [{"evidence_id": "org_query:c1:total", "value": "7"}],
    }
    base.update(over)
    return base


async def test_a_valid_answer_parses_and_dumps_wire_shape():
    payload = AnswerPayload.model_validate(_answer())
    dumped = payload.model_dump(exclude_none=True)
    assert dumped["kind"] == "answer"
    assert dumped["blocks"][0] == {"kind": "text", "text": "chevy runs 7 live campaigns."}


@pytest.mark.parametrize(
    "headline",
    [
        "Sure, chevy has 7 campaigns",
        "Here is the campaign count",
        "Конечно, у chevy 7 кампаний",
        "Отличный вопрос про chevy",
    ],
)
async def test_filler_fails_the_headline(headline):
    with pytest.raises(ValidationError, match="filler"):
        AnswerPayload.model_validate(_answer(headline=headline))


async def test_filler_is_word_bounded_not_substring():
    """'ensure'/'measure' must not trip the 'sure' pattern."""
    AnswerPayload.model_validate(_answer(headline="Measures ensure 7 live campaigns"))


async def test_filler_fails_text_blocks_and_bullets():
    with pytest.raises(ValidationError, match="filler"):
        AnswerPayload.model_validate(
            _answer(blocks=[{"kind": "text", "text": "Here is what chevy runs."}])
        )
    with pytest.raises(ValidationError, match="filler"):
        AnswerPayload.model_validate(
            _answer(blocks=[{"kind": "bullets", "items": ["вот что нашлось"]}])
        )


async def test_headline_word_budget():
    with pytest.raises(ValidationError, match="20-word"):
        AnswerPayload.model_validate(_answer(headline=" ".join(["word"] * 21)))


async def test_text_block_word_budget():
    with pytest.raises(ValidationError, match="60-word"):
        AnswerPayload.model_validate(
            _answer(blocks=[{"kind": "text", "text": " ".join(["word"] * 61)}])
        )


async def test_bullet_word_budget_and_count():
    with pytest.raises(ValidationError, match="25-word"):
        AnswerPayload.model_validate(
            _answer(blocks=[{"kind": "bullets", "items": [" ".join(["word"] * 26)]}])
        )
    with pytest.raises(ValidationError):
        AnswerPayload.model_validate(
            _answer(blocks=[{"kind": "bullets", "items": [f"item {i}" for i in range(8)]}])
        )


async def test_total_word_budget():
    long_block = {"kind": "text", "text": " ".join(["word"] * 60)}
    bullets = {"kind": "bullets", "items": [" ".join(["word"] * 24)] * 7}
    with pytest.raises(ValidationError, match="250"):
        AnswerPayload.model_validate(_answer(blocks=[long_block, bullets, bullets]))


async def test_table_rows_must_match_columns():
    with pytest.raises(ValidationError, match="cells"):
        AnswerPayload.model_validate(
            _answer(
                blocks=[
                    {
                        "kind": "table",
                        "columns": ["Campaign", "Budget", "Source"],
                        "rows": [["Q3 push", "1200000"]],
                    }
                ]
            )
        )


async def test_table_cells_do_not_spend_the_prose_budget():
    """The 250-word budget governs how much the answer SAYS. Counting table
    cells made it the real cap on table length: a twenty-row ranking blew the
    budget on data alone and the table shrank instead (live 2026-09-16)."""
    big = _answer(
        blocks=[
            {
                "kind": "table",
                "columns": ["Campaign", "Budget"],
                "rows": [[f"Chevrolet campaign number {i}", "5,130,085.67"] for i in range(60)],
            }
        ]
    )
    payload = AnswerPayload.model_validate(big)
    assert len(payload.blocks[0].rows) == 60


async def test_table_row_count_is_not_capped():
    """A "top 20" answer capped at ten rows contradicted its own headline on
    screen (live 2026-09-16). The contract no longer bounds row count at all:
    the only real limit is Chat's widget budget, and that is the renderer's to
    enforce visibly rather than the schema's to enforce silently."""
    for n in (20, 50, 200):
        payload = AnswerPayload.model_validate(
            _answer(blocks=[{"kind": "table", "columns": ["a", "b"], "rows": [["x", "y"]] * n}])
        )
        assert len(payload.blocks[0].rows) == n


async def test_table_column_and_row_bounds():
    with pytest.raises(ValidationError):
        AnswerPayload.model_validate(
            _answer(blocks=[{"kind": "table", "columns": ["one"], "rows": [["x"]]}])
        )
    # No row ceiling any more — see test_table_row_count_is_not_capped.


async def test_facts_must_echo_exactly_the_citations():
    # cited but not carried — would silently disable 05's conflict filter
    with pytest.raises(ValidationError, match="cited but not echoed"):
        AnswerPayload.model_validate(_answer(facts=[]))
    # carried but not cited
    with pytest.raises(ValidationError, match="not cited"):
        AnswerPayload.model_validate(
            _answer(
                facts=[
                    {"evidence_id": "org_query:c1:total", "value": "7"},
                    {"evidence_id": "org_query:c9:budget", "value": "5"},
                ]
            )
        )


async def test_an_answer_needs_blocks_and_citations():
    with pytest.raises(ValidationError, match="at least one block"):
        AnswerPayload.model_validate(_answer(blocks=[]))
    with pytest.raises(ValidationError, match="citation"):
        AnswerPayload.model_validate(_answer(citations=[], facts=[]))


async def test_abstain_is_valid_with_nothing_but_a_headline():
    payload = AnswerPayload.model_validate({"kind": "abstain", "headline": "NO_COMPANY_RECORDS"})
    assert payload.blocks == []
    assert payload.citations == []


async def test_candidates_only_on_clarify():
    candidate = {"name": "chevy (account)"}
    clarify = AnswerPayload.model_validate(
        {"kind": "clarify", "headline": "Which chevy?", "candidates": [candidate]}
    )
    assert clarify.candidates[0].name == "chevy (account)"
    with pytest.raises(ValidationError, match="clarify"):
        AnswerPayload.model_validate(_answer(candidates=[candidate]))


async def test_payload_converts_to_a_genai_response_schema():
    """The formatter runs with output_schema=AnswerPayload, so the genai SDK
    must be able to turn the model into a Vertex Schema. A discriminated union
    (oneOf + discriminator) fails this conversion with extra_forbidden — the
    formatter then dies CLIENT-side in ~200ms on every attempt and the gate
    only ever ships template renders (live-verified 2026-09-10). The Block
    union must stay a plain union (anyOf)."""
    from google.genai import _transformers

    schema = _transformers.t_schema(None, AnswerPayload)
    assert schema.properties["blocks"].items.any_of is not None
    assert len(schema.properties["blocks"].items.any_of) == 3


async def test_count_words_matches_the_eval_definition():
    """Tokens need a letter or digit to count — same rule as metrics.ts."""
    assert count_words("chevy has 7 live campaigns") == 5
    assert count_words("— · [] ") == 0
    assert count_words("бюджет 1,200,000 ₽") == 2


# ── follow-ups and assumptions are not the answer (BE-5) ─────────────────────


def _words(n: int) -> str:
    return " ".join(["word"] * n)


def _bullets(*sizes: int) -> dict:
    return {"kind": "bullets", "items": [_words(n) for n in sizes]}


def _prose(total: int, **over) -> dict:
    """An answer whose headline + text + bullets are exactly `total` words: a
    20-word headline, a 60-word text block, and bullets of at most 25 for the
    rest (so the standard per-bullet budget holds)."""
    rest = total - 20 - 60
    sizes = [25] * (rest // 25) + ([rest % 25] if rest % 25 else [])
    blocks = [{"kind": "text", "text": _words(60)}]
    for i in range(0, len(sizes), 7):  # a bullet block holds at most 7 items
        blocks.append(_bullets(*sizes[i : i + 7]))
    return _answer(headline=_words(20), blocks=blocks, **over)


async def test_follow_ups_and_assumptions_do_not_spend_the_total():
    """Two to three follow-ups are REQUIRED on every answer and up to two
    assumptions are copied in, and they used to be counted against the 250: a
    full-length answer lost ~25 words of body to its own buttons."""
    payload = AnswerPayload.model_validate(
        _prose(
            250,
            follow_ups=[_words(10)] * 3,  # 30 words
            assumptions=[_words(12)] * 2,  # 24 words
        )
    )
    assert payload.follow_ups and payload.assumptions


async def test_the_standard_total_is_250_of_headline_text_and_bullets():
    AnswerPayload.model_validate(_prose(250))
    with pytest.raises(ValidationError, match="251 words total, over the 250 budget"):
        AnswerPayload.model_validate(_prose(251))


# ── the rundown's budgets (BE-5) ─────────────────────────────────────────────


def _rundown_answer(bullets_per_block: list[list[int]], headline: int = 15, **over) -> dict:
    return _answer(
        headline=_words(headline),
        blocks=[_bullets(*sizes) for sizes in bullets_per_block],
        **over,
    )


async def test_a_rundown_accepts_twelve_bullets_across_two_blocks():
    """The rundown: up to 12 entity bullets across two blocks, each over the
    standard 25-word budget. 12 x 35 + a 15-word headline = 435 words."""
    data = _rundown_answer([[35] * 6, [35] * 6])
    with answer_budgets(rundown=True):
        payload = AnswerPayload.model_validate(data)
    assert sum(len(b.items) for b in payload.blocks) == 12
    # the same wire as any other bullets answer: no new kind, no new field
    assert {block["kind"] for block in payload.model_dump()["blocks"]} == {"bullets"}


async def test_a_rundown_bullet_may_be_forty_words_not_forty_one():
    with answer_budgets(rundown=True):
        AnswerPayload.model_validate(_rundown_answer([[40, 40, 40]]))
        with pytest.raises(ValidationError, match="bullet 0 is 41 words, over the 40-word budget"):
            AnswerPayload.model_validate(_rundown_answer([[41, 40, 40]]))


async def test_a_rundown_total_is_450_so_twelve_full_forty_word_bullets_do_not_fit():
    """R1 sets the total at 450 and the bullet at 40; twelve of them are 480, so
    the total binds before the count does. 11 x 40 + a 10-word headline is
    exactly 450. (The prompt tells the formatter: ~35 words each at 12.)"""
    with answer_budgets(rundown=True):
        AnswerPayload.model_validate(_rundown_answer([[40] * 6, [40] * 5], headline=10))
        with pytest.raises(ValidationError, match="over the 450 budget"):
            AnswerPayload.model_validate(_rundown_answer([[40] * 6, [40] * 6], headline=10))


async def test_a_rundown_has_at_most_twelve_bullets():
    with answer_budgets(rundown=True):
        AnswerPayload.model_validate(_rundown_answer([[10] * 6, [10] * 6]))
        with pytest.raises(ValidationError, match="13 bullets, over the 12-bullet rundown cap"):
            AnswerPayload.model_validate(_rundown_answer([[10] * 7, [10] * 6]))


async def test_a_rundown_keeps_the_other_budgets():
    """Only the bullet, the total and the count differ: a headline of 21 words
    or a text block of 61 is over budget in a rundown as anywhere."""
    with answer_budgets(rundown=True):
        with pytest.raises(ValidationError, match="20-word"):
            AnswerPayload.model_validate(_rundown_answer([[30]], headline=21))
        with pytest.raises(ValidationError, match="60-word"):
            AnswerPayload.model_validate(
                _answer(blocks=[{"kind": "text", "text": _words(61)}, _bullets(30)])
            )


async def test_every_other_answer_keeps_todays_budgets():
    """A fact, a count, a ranking — and any caller that never heard of
    rundowns — gets the standard budgets, with today's feedback."""
    twelve = _rundown_answer([[35] * 6, [35] * 6])
    with pytest.raises(ValidationError, match="bullet 0 is 35 words, over the 25-word budget"):
        AnswerPayload.model_validate(twelve)
    with answer_budgets(rundown=False):
        with pytest.raises(ValidationError, match="25-word budget"):
            AnswerPayload.model_validate(twelve)
    # the cap on the bullet count is the rundown's own: 14 short bullets are fine
    AnswerPayload.model_validate(_rundown_answer([[10] * 7, [10] * 7]))


async def test_the_budgets_are_restored_when_the_block_ends_even_on_an_error():
    twelve = _rundown_answer([[35] * 6, [35] * 6])
    with pytest.raises(RuntimeError):
        with answer_budgets(rundown=True):
            AnswerPayload.model_validate(twelve)
            raise RuntimeError("boom")
    with pytest.raises(ValidationError, match="25-word budget"):
        AnswerPayload.model_validate(twelve)


async def test_the_rundown_adds_no_block_kind_and_no_field():
    """The bot's zod mirror (`render-answer.ts`) is hash-pinned and parses
    `blocks` as a discriminated union over exactly these kinds: a new kind
    would fail that parse and drop the payload to prose. The rundown is a
    budget, not a shape — the wire is the one the bot already reads."""
    kinds = {block.model_fields["kind"].default for block in typing.get_args(Block)}
    assert kinds == {"table", "bullets", "text"}
    assert set(AnswerPayload.model_fields) == {
        "kind",
        "headline",
        "blocks",
        "citations",
        "facts",
        "assumptions",
        "follow_ups",
        "candidates",
    }
    schema = AnswerPayload.model_json_schema()
    assert schema["properties"]["blocks"]["maxItems"] == 4
    assert schema["$defs"]["BulletBlock"]["properties"]["items"]["maxItems"] == 7
