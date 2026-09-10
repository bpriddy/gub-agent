"""
schemas.answer — the answer contract's validators (blend 03).

What used to be prompt requests policed by the critic LLM is now pydantic:
filler, budgets, table shape, and the facts↔citations equality that the bot's
Workspace conflict filter (blend 05) depends on. These tests are the contract's
pin — a payload the executor pipeline emits and the bot renders must be
constructible exactly under these rules.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gub_agent.schemas import AnswerPayload
from gub_agent.schemas.answer import count_words


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


async def test_table_column_and_row_bounds():
    with pytest.raises(ValidationError):
        AnswerPayload.model_validate(
            _answer(blocks=[{"kind": "table", "columns": ["one"], "rows": [["x"]]}])
        )
    with pytest.raises(ValidationError):
        AnswerPayload.model_validate(
            _answer(
                blocks=[
                    {
                        "kind": "table",
                        "columns": ["a", "b"],
                        "rows": [["x", "y"]] * 11,
                    }
                ]
            )
        )


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


async def test_count_words_matches_the_eval_definition():
    """Tokens need a letter or digit to count — same rule as metrics.ts."""
    assert count_words("chevy has 7 live campaigns") == 5
    assert count_words("— · [] ") == 0
    assert count_words("бюджет 1,200,000 ₽") == 2
