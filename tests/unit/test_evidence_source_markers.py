"""
agents.evidence_index — do the `[src: <driveFileId>]` markers reach the agent?

blend 08 §3 check 2. The per-bullet provenance the whole feature rests on is
written into `status_markdown` by the status-synthesis prompt and travels to
the engine as an ordinary field of `get_campaign`'s result. Nothing parses it
today, so the only question is whether it SURVIVES indexing — and the answer
has a sharp edge:

  * the FIELD row `<tool>:<id>:statusMarkdown` stores `str(value)` whole;
  * the WHOLE-ENTITY row `<tool>:<id>` is `_compact_row`, capped at
    MAX_ENTITY_VALUE_CHARS (2 000).

Measured on `gub_prod_copy` 2026-09-17: 30 of the 47 campaigns carrying
markers have a `status_markdown` longer than that cap, and only 364 of the 689
markers (52.8 %) fall inside the first 2 000 characters. So a reader of the
entity row sees about half the provenance and the half it sees is the head of
the document — parsing must read the FIELD row. That is what these pin.

The fixture is a recorded `get_campaign` body (real marker syntax, real file
ids, both marker flavours — `[src: …]` and the `[expires: …]` that shares its
bracket grammar); no live turn is involved.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

from gub_agent.agents.evidence_index import (
    MAX_ENTITY_VALUE_CHARS,
    evidence_index,
    record_evidence,
    reset_evidence_index,
)

SRC_MARKER_RE = re.compile(r"\[src:\s*([A-Za-z0-9_-]+)\]")

FIXTURE = Path(__file__).parents[1] / "fixtures" / "recorded_get_campaign.json"


def _recorded() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _fresh(invocation_id: str = "inv-src") -> SimpleNamespace:
    ctx = SimpleNamespace(invocation_id=invocation_id)
    reset_evidence_index(ctx)
    return ctx


async def test_src_markers_survive_whole_on_the_status_markdown_field_row():
    ctx = _fresh()
    recorded = _recorded()
    record_evidence(SimpleNamespace(name="get_campaign"), {}, ctx, recorded)

    entry = evidence_index("inv-src")["get_campaign:cmp-ev-social:statusMarkdown"]
    assert entry["field"] == "statusMarkdown"
    assert entry["entity_id"] == "cmp-ev-social"
    # Byte-for-byte: the field row is not compacted, capped or re-encoded.
    assert entry["value"] == recorded["statusMarkdown"]
    assert SRC_MARKER_RE.findall(entry["value"]) == [
        "1rRAWMbDltWlshELSJb3QELxNiSBylximIAzpNrd1L1w",
        "1RsPYpRiFaV6T1ClhnREl9DC2V4X2qssz_r3FtvbfJQY",
    ]


async def test_the_whole_entity_row_truncates_and_loses_the_later_markers():
    """Why §5.2 must parse the FIELD row. A long status is cut mid-document,
    so the entity row's marker set is the head of the file only — on live data
    that is 52.8 % of them."""
    ctx = _fresh()
    recorded = _recorded()
    head = recorded["statusMarkdown"]
    # A status long enough to be cut, with a marker on the far side of the cap.
    padding = "\n- filler bullet that carries no marker at all." * 120
    recorded["statusMarkdown"] = (
        head + padding + "\n- Late bullet. [src: 1LATEmarkerPastTheEntityCap_00000000000]"
    )
    record_evidence(SimpleNamespace(name="get_campaign"), {}, ctx, recorded)
    index = evidence_index("inv-src")

    entity_value = index["get_campaign:cmp-ev-social"]["value"]
    assert len(entity_value) == MAX_ENTITY_VALUE_CHARS
    assert "1LATEmarkerPastTheEntityCap_00000000000" not in entity_value

    field_value = index["get_campaign:cmp-ev-social:statusMarkdown"]["value"]
    assert field_value == recorded["statusMarkdown"]
    assert "1LATEmarkerPastTheEntityCap_00000000000" in field_value
    assert len(SRC_MARKER_RE.findall(field_value)) == 3


async def test_sources_plumbing_stays_out_while_the_markers_stay_in():
    """The two must not be confused: `_sources` is masked as attribution
    plumbing (context_pruning), the markers ride inside an ordinary field."""
    ctx = _fresh()
    record_evidence(SimpleNamespace(name="get_campaign"), {}, ctx, _recorded())
    index = evidence_index("inv-src")
    assert "get_campaign:cmp-ev-social:_sources" not in index
    assert "0AMkVeL1ln2XhUk9PVA" not in index["get_campaign:cmp-ev-social"]["value"]
    assert "[src:" in index["get_campaign:cmp-ev-social:statusMarkdown"]["value"]
