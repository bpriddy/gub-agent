"""
agents.evidence_index — a whole-entity row's value is ONE JSON OBJECT (integration fix).

The whole-entity row `<tool>:<id>` carries the record's scalar fields as JSON,
and the formatter echoes it into `facts[].value` for each record a bullet cites.
The bot's conflict filter opens that string with a JSON parser to read the
record's fields (`unpackEntityRow`, gub-gchat-bot `src/blend/claim-filter.ts`):
a string that does not parse yields NO facts for that record, and a turn whose
cited rows all fail that way skips the filter as `no_facts`.

`_compact_row` used to cap the row by slicing the serialized text at
MAX_ENTITY_VALUE_CHARS. A record with a long `statusMarkdown` was cut in the
middle of that string: five of the six `get_campaign` responses of the recorded
Chevy turn. Those rows were refused by the index before BE-2 and are admitted
(so citable) now, which is what made the cut reachable. Pinned here:

  * every whole-entity row of the recorded turn parses to an object, within the
    cap, in the index, in the rows event and in the formatter's brief;
  * the surface reads facts off all six recorded campaign rows (five gave none);
  * a row over the cap is shortened where its length is: its longest strings are
    cut and marked, every field stays in the row's order, no number is touched,
    and the cut text stays whole in its own field row;
  * a row under the cap is byte for byte what it always was;
  * a row that cannot fit even so gives up its last fields, never its validity.
"""

from __future__ import annotations

import json
import random
import re
import string
from pathlib import Path
from types import SimpleNamespace

import pytest

from gub_agent.agents import evidence_index as ei
from gub_agent.agents.evidence_index import (
    MAX_ENTITY_VALUE_CHARS,
    evidence_index,
    evidence_rows,
    record_evidence,
    reset_evidence_index,
)
from gub_agent.agents.format_gate import compose_brief

FIXTURE = Path(__file__).parents[1] / "fixtures" / "recorded_chevy_turn_2026-09-17.json"
INV = "inv-entity-value"
MARK = "…"


def _calls() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _record(calls: list[tuple[str, dict]]) -> dict[str, dict]:
    ctx = SimpleNamespace(invocation_id=INV)
    reset_evidence_index(ctx)
    for name, response in calls:
        record_evidence(SimpleNamespace(name=name), {}, ctx, response)
    return evidence_index(INV)


def _replayed() -> tuple[dict[str, dict], list[dict]]:
    calls = _calls()
    return _record([(c["name"], c["response"]) for c in calls]), calls


def _scalars(row: dict) -> dict:
    return {
        k: v
        for k, v in row.items()
        if isinstance(v, (str, int, float, bool)) and not k.startswith("_")
    }


def _campaigns(calls: list[dict]) -> list[dict]:
    return [c["response"] for c in calls if c["name"] == "get_campaign"]


# ── the recorded turn ────────────────────────────────────────────────────────


def test_every_whole_entity_row_of_the_recorded_turn_is_a_json_object_within_the_cap():
    index, calls = _replayed()
    campaigns = _campaigns(calls)
    # The premise, carried by the fixture: five of the six detail responses have a
    # status long enough that their scalars do not fit the cap whole.
    assert len(campaigns) == 6
    over = [c for c in campaigns if len(json.dumps(_scalars(c), ensure_ascii=False)) > 2_000]
    assert len(over) == 5

    rows = {key: entry for key, entry in index.items() if entry["field"] is None}
    assert len(rows) > 50
    for key, entry in rows.items():
        parsed = json.loads(entry["value"])  # a cut string raises here
        assert isinstance(parsed, dict), key
        assert len(entry["value"]) <= MAX_ENTITY_VALUE_CHARS, key


def test_the_rows_event_and_the_formatters_brief_carry_the_same_parseable_rows():
    index, _ = _replayed()
    rows = [r for r in evidence_rows(index) if r["field"] is None]
    assert len(rows) > 50
    for row in rows:
        assert isinstance(json.loads(row["value"]), dict), row["evidence_id"]

    brief_lines = compose_brief("Executor answer.", index, "", ei.provenance(INV)).splitlines()
    for row in rows:
        [line] = [ln for ln in brief_lines if ln.startswith(f"- {row['evidence_id']} = ")]
        assert isinstance(json.loads(line.split(" = ", 1)[1]), dict), row["evidence_id"]


def test_a_cut_campaign_keeps_every_field_in_order_and_shortens_only_its_status_text():
    index, calls = _replayed()
    cut = 0
    for response in _campaigns(calls):
        scalars = _scalars(response)
        entry = index[f"get_campaign:{response['id']}"]
        value = json.loads(entry["value"])

        assert list(value) == list(scalars), response["name"]  # every field, same order
        for key, original in scalars.items():
            if key != "statusMarkdown":
                assert value[key] == original, (response["name"], key)  # numbers and names whole

        status, full = value["statusMarkdown"], scalars["statusMarkdown"]
        if status == full:
            continue
        cut += 1
        assert status.endswith(MARK) and len(status) < len(full)
        assert full.startswith(status[: -len(MARK)])
        # The cut is in the entity row only: the status stays whole as its own
        # field row, and the row's sources are the whole response's, not the cut
        # text's (the backend's resolved `_cited` map, here).
        assert index[f"get_campaign:{response['id']}:statusMarkdown"]["value"] == full
        assert entry["source_file_ids"] == list(response["_cited"])
    assert cut == 5


# What the bot's conflict filter reads out of a cited whole-entity row. A mirror of
# `unpackEntityRow` in gub-gchat-bot `src/blend/claim-filter.ts` (the consumer this
# value is shaped for): update it with that function, not around it.
_UNCOMPARABLE = re.compile(r"^(id|.*Id|createdAt|updatedAt|createdByName)$")
_MAX_FIELDS_PER_ROW = 12


def _surface_facts(value: str) -> list[tuple[str, str]]:
    try:
        row = json.loads(value)
    except ValueError:
        return []
    if not isinstance(row, dict):
        return []
    out: list[tuple[str, str]] = []
    for field, item in row.items():
        if len(out) >= _MAX_FIELDS_PER_ROW:
            break
        if item is None or isinstance(item, (dict, list)) or _UNCOMPARABLE.match(field):
            continue
        text = str(item).strip()
        if text:
            out.append((field, text))
    return out


def test_the_surface_reads_facts_off_every_recorded_campaign_row():
    """At the epic head this was one row of six (the BHAC status fits the cap): the
    other five were cut mid-string and read as no facts at all."""
    index, calls = _replayed()
    for campaign in _campaigns(calls):
        facts = _surface_facts(index[f"get_campaign:{campaign['id']}"]["value"])
        assert {"name", "status", "statusMarkdown"} <= {field for field, _ in facts}, campaign[
            "name"
        ]


# ── the compaction itself ────────────────────────────────────────────────────


def _campaign(**extra) -> dict:
    return {
        "id": "c1",
        "accountId": "a1",
        "name": "Red Tag 2026",
        "status": "live",
        "budget": 1234567.89,
        "createdAt": "2026-01-02",
        **extra,
    }


def _entity_value(row: dict) -> str:
    return _record([("get_campaign", row)])[f"get_campaign:{row['id']}"]["value"]


def test_a_row_under_the_cap_is_serialized_exactly_as_it_always_was():
    row = _campaign(note="Résumé — “quoted”", live=True, count=0)

    assert _entity_value(row) == json.dumps(_scalars(row), ensure_ascii=False)
    assert _entity_value(row).startswith('{"id": "c1", "accountId": "a1", "name": "Red Tag 2026"')


def test_a_row_exactly_at_the_cap_is_not_touched():
    pad = MAX_ENTITY_VALUE_CHARS - len(json.dumps({**_scalars(_campaign()), "pad": ""}))
    row = _campaign(pad="p" * pad)
    value = _entity_value(row)

    assert len(value) == MAX_ENTITY_VALUE_CHARS
    assert json.loads(value)["pad"] == "p" * pad


def test_a_long_status_is_cut_to_the_longest_length_at_which_the_row_fits():
    status = "s" * 9_000  # no whitespace: the cut lands exactly on the cap
    value = _entity_value(_campaign(statusMarkdown=status))

    parsed = json.loads(value)
    assert len(value) == MAX_ENTITY_VALUE_CHARS
    assert parsed["statusMarkdown"].endswith(MARK)
    assert status.startswith(parsed["statusMarkdown"][: -len(MARK)])
    assert {k: v for k, v in parsed.items() if k != "statusMarkdown"} == _scalars(_campaign())


def test_the_longest_strings_give_up_the_most_and_a_short_one_is_left_alone():
    row = _campaign(
        note="n" * 300,  # fits whole at the level the two long ones settle at
        description="d" * 4_000,
        statusMarkdown="s" * 9_000,
    )

    parsed = json.loads(_entity_value(row))

    assert parsed["note"] == "n" * 300
    assert len(parsed["description"]) == len(parsed["statusMarkdown"]) > 300
    assert parsed["description"].endswith(MARK) and parsed["statusMarkdown"].endswith(MARK)
    # the level is the largest that fits: one more character each would not
    assert MAX_ENTITY_VALUE_CHARS - 1 <= len(_entity_value(row)) <= MAX_ENTITY_VALUE_CHARS


def test_numbers_and_booleans_are_never_shortened():
    row = _campaign(budget=987654321.5, active=True, count=0, statusMarkdown="s " * 5_000)

    parsed = json.loads(_entity_value(row))

    assert (parsed["budget"], parsed["active"], parsed["count"]) == (987654321.5, True, 0)
    assert parsed["statusMarkdown"].endswith(MARK)
    assert not parsed["statusMarkdown"][: -len(MARK)].endswith(" ")  # no dangling space


def test_a_status_full_of_characters_json_escapes_still_parses_and_fits():
    """The cut is chosen on the SERIALIZED length: quotes, newlines, backslashes
    and non-ASCII text change it, and the object must fit and parse anyway."""
    status = ('He said "go"\n- next\\line é—😀 [src: 1AbC_dEf] ') * 300
    value = _entity_value(_campaign(statusMarkdown=status))

    parsed = json.loads(value)
    assert len(value) <= MAX_ENTITY_VALUE_CHARS
    assert parsed["statusMarkdown"].endswith(MARK)
    assert status.startswith(parsed["statusMarkdown"][: -len(MARK)])
    assert "é—😀" in value  # raw characters, as before: no \u escapes


def test_a_row_of_many_short_fields_gives_up_its_last_fields_not_its_validity():
    fields = {f"field{i:03d}": f"value {i}" for i in range(400)}
    value = _entity_value({"id": "c1", **fields})

    parsed = json.loads(value)
    names = [k for k in parsed if k != "id"]
    assert len(value) <= MAX_ENTITY_VALUE_CHARS
    assert 50 < len(names) < 400
    assert names == [f"field{i:03d}" for i in range(len(names))]  # the leading ones, in order
    assert all(parsed[k] == fields[k] for k in names)  # …whole


def test_a_row_of_many_long_texts_cuts_each_to_the_floor_then_drops_its_last_fields():
    fields = {f"text{i:02d}": "x" * 3_000 for i in range(40)}
    value = _entity_value({"id": "c1", **fields})

    parsed = json.loads(value)
    names = [k for k in parsed if k != "id"]
    assert len(value) <= MAX_ENTITY_VALUE_CHARS
    assert 0 < len(names) < 40
    assert names == [f"text{i:02d}" for i in range(len(names))]
    assert {len(parsed[k]) for k in names} == {ei._MIN_CUT_CHARS + len(MARK)}


def test_a_row_of_nothing_but_numbers_over_the_cap_is_still_an_object():
    value = _entity_value({"id": "c1", **{f"n{i:04d}": 1_000_000 + i for i in range(500)}})

    parsed = json.loads(value)
    assert len(value) <= MAX_ENTITY_VALUE_CHARS
    assert 0 < len(parsed) < 500
    assert all(parsed[k] == 1_000_000 + int(k[1:]) for k in parsed if k != "id")


@pytest.mark.parametrize("cap", [300, 700, 1_500])
def test_whatever_the_cap_a_cut_row_is_an_object_within_it(monkeypatch, cap):
    monkeypatch.setattr(ei, "MAX_ENTITY_VALUE_CHARS", cap)
    index, _ = _replayed()

    for key, entry in index.items():
        if entry["field"] is None:
            assert isinstance(json.loads(entry["value"]), dict), key
            assert len(entry["value"]) <= cap, key


def _random_row(rng: random.Random) -> dict:
    alphabet = string.ascii_letters + string.digits + ' \n\t"\\{}[],:é—😀…'
    row: dict = {}
    for i in range(rng.randint(1, 40)):
        kind = rng.random()
        key = f"k{i}"
        if kind < 0.45:
            size = rng.choice([0, 1, 5, 30, 64, 65, 200, 900, 3_000, 9_000])
            row[key] = "".join(rng.choices(alphabet, k=size))
        elif kind < 0.7:
            row[key] = rng.randint(-(10**9), 10**9)
        elif kind < 0.85:
            row[key] = rng.random() * 10 ** rng.randint(0, 12)
        else:
            row[key] = rng.random() < 0.5
    return {"id": "r1", **row}


def test_a_row_of_any_shape_compacts_to_an_object_within_the_cap_whose_fields_are_the_rows(
    monkeypatch,
):
    """Seeded, so the same rows every run: strings of every length and escape,
    numbers, booleans, 1-40 fields, caps from 2 to 5,000 characters."""
    rng = random.Random(20261014)
    cut = dropped = 0
    for _ in range(250):
        cap = rng.choice([2, 10, 80, 300, 700, 2_000, 2_000, 5_000])
        monkeypatch.setattr(ei, "MAX_ENTITY_VALUE_CHARS", cap)
        row = _random_row(rng)

        value = ei._compact_row(row)

        parsed = json.loads(value)
        assert isinstance(parsed, dict) and len(value) <= cap, (cap, value[:60])
        assert list(parsed) == list(row)[: len(parsed)]  # the row's leading fields, in order
        if len(json.dumps(row, ensure_ascii=False)) <= cap:
            assert value == json.dumps(row, ensure_ascii=False)  # under the cap: as it was
            continue
        cut += len(parsed) == len(row)
        dropped += len(parsed) < len(row)
        for key, shown in parsed.items():
            original = row[key]
            if isinstance(original, str):
                assert shown == original or (
                    shown.endswith(MARK) and original.startswith(shown[: -len(MARK)])
                ), key
            else:
                assert shown == original and type(shown) is type(original), key
    assert cut > 20 and dropped > 20  # both ways of fitting a row were exercised
