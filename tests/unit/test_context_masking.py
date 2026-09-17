"""
context_pruning.strip_source_metadata — `_sources` masking, copy-on-write.

The masking must NOT mutate the function_response payload in place: ADK shares
that nested field with session history (and trace consumers read `_sources` from
it), so an in-place pop can corrupt it. These tests pin that the request loses
`_sources` while the original response object is left untouched.
"""

from __future__ import annotations

from types import SimpleNamespace

from gub_agent.agents.context_pruning import strip_source_metadata


def _request_with(resp: dict) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Build a minimal llm_request wrapping one function_response = resp.
    Returns (request, function_response) so tests can inspect the payload."""
    fr = SimpleNamespace(response=resp)
    part = SimpleNamespace(function_response=fr)
    content = SimpleNamespace(role="user", parts=[part])
    return SimpleNamespace(contents=[content]), fr


async def test_strips_sources_without_mutating_the_original():
    resp = {
        "id": "a1",
        "campaigns": [{"id": "c1", "name": "Launch"}],
        "_sources": [{"fileId": "f1"}, {"fileId": "f2"}],
    }
    req, fr = _request_with(resp)

    strip_source_metadata(None, req)

    # The request's payload no longer carries the plumbing...
    assert "_sources" not in fr.response
    assert fr.response["campaigns"] == [{"id": "c1", "name": "Launch"}]  # content intact
    # ...but the ORIGINAL response object was not touched (copy-on-write).
    assert "_sources" in resp
    assert fr.response is not resp


async def test_strips_nested_sources():
    resp = {"account": {"name": "Chevy", "_sources": [{"fileId": "x"}]}, "_sources": []}
    req, fr = _request_with(resp)

    strip_source_metadata(None, req)

    assert "_sources" not in fr.response
    assert "_sources" not in fr.response["account"]
    assert fr.response["account"]["name"] == "Chevy"
    assert "_sources" in resp["account"]  # original untouched


async def test_noop_when_no_sources_shares_the_object():
    resp = {"id": "a1", "campaigns": []}
    req, fr = _request_with(resp)

    strip_source_metadata(None, req)

    # Nothing to strip → no needless copy; the same object is kept.
    assert fr.response is resp


async def test_strips_the_blend_08_plumbing_beside_sources():
    """`_cited` / `_sourcesTotal` (blend 08 §5.1) exist for the BOT to bind
    links; the formatter copies source ids off the evidence brief, never off
    these. To the model they are ~90 opaque Drive ids per account overview,
    re-sent every ReAct round — the same dead weight `_sources` was masked for.
    Same copy-on-write contract: the original response is left whole."""
    resp = {
        "id": "a1",
        "name": "Chevy",
        "_sources": [{"fileId": "f1"}],
        "_sourcesTotal": 3308,
        "_cited": {"1rRAWMbDl": {"name": "Brief v2", "mimeType": "application/pdf"}},
        "campaigns": [{"id": "c1", "_cited": {"x": {"name": "n"}}}],
    }
    req, fr = _request_with(resp)
    strip_source_metadata(None, req)
    for key in ("_sources", "_sourcesTotal", "_cited"):
        assert key not in fr.response
    assert "_cited" not in fr.response["campaigns"][0]
    assert fr.response["name"] == "Chevy"  # real fields untouched
    for key in ("_sources", "_sourcesTotal", "_cited"):
        assert key in resp  # original untouched
