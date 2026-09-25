"""
context_pruning.strip_source_metadata — `_sources` masking, copy-on-write.

The masking must NOT write to anything the request was built from: ADK shares
the function_response payload AND the FunctionResponse object itself with
session history whenever the call id is the model's own (not `adk-`), and the
bot reads `_sources` / `_cited` / `_sourcesTotal` from the stream and the
stored session. These tests pin that the request loses the plumbing while the
original part, its FunctionResponse and its payload are left untouched —
including through ADK's own request copy (`_copy_content_for_request`).
"""

from __future__ import annotations

import pytest
from google.adk.flows.llm_flows.contents import _copy_content_for_request
from google.adk.models.llm_request import LlmRequest
from google.genai import types as genai_types

from gub_agent.agents.context_pruning import strip_source_metadata


def _request_with(resp: dict):
    """A request with one tool-response content carrying resp. Returns
    (request, part, function_response, payload) — payload is the dict the
    FunctionResponse holds (pydantic copies it on construction) — so tests
    can check both sides."""
    fr = genai_types.FunctionResponse(name="org_query", response=resp)
    part = genai_types.Part(function_response=fr)
    content = genai_types.Content(role="user", parts=[part])
    return LlmRequest(contents=[content]), part, fr, fr.response


def _sent(request: LlmRequest) -> dict:
    """The payload the request now carries."""
    return request.contents[0].parts[0].function_response.response


async def test_strips_sources_without_mutating_the_original():
    resp = {
        "id": "a1",
        "campaigns": [{"id": "c1", "name": "Launch"}],
        "_sources": [{"fileId": "f1"}, {"fileId": "f2"}],
    }
    req, part, fr, payload = _request_with(resp)

    strip_source_metadata(None, req)

    # The request's payload no longer carries the plumbing...
    assert "_sources" not in _sent(req)
    assert _sent(req)["campaigns"] == [{"id": "c1", "name": "Launch"}]  # content intact
    # ...but NOTHING it was built from was touched: payload, FunctionResponse, Part.
    assert "_sources" in payload
    assert fr.response is payload
    assert part.function_response is fr
    assert req.contents[0].parts[0] is not part


async def test_strips_nested_sources():
    resp = {"account": {"name": "Chevy", "_sources": [{"fileId": "x"}]}, "_sources": []}
    req, _, fr, payload = _request_with(resp)

    strip_source_metadata(None, req)

    assert "_sources" not in _sent(req)
    assert "_sources" not in _sent(req)["account"]
    assert _sent(req)["account"]["name"] == "Chevy"
    assert "_sources" in payload["account"]  # original untouched
    assert fr.response is payload


async def test_noop_when_no_sources_shares_the_object():
    resp = {"id": "a1", "campaigns": []}
    req, part, _, payload = _request_with(resp)
    contents = req.contents

    strip_source_metadata(None, req)

    # Nothing to strip → no needless copy; the same objects are kept.
    assert req.contents is contents
    assert req.contents[0].parts[0] is part
    assert _sent(req) is payload


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
    req, _, _, payload = _request_with(resp)
    strip_source_metadata(None, req)
    for key in ("_sources", "_sourcesTotal", "_cited"):
        assert key not in _sent(req)
    assert "_cited" not in _sent(req)["campaigns"][0]
    assert _sent(req)["name"] == "Chevy"  # real fields untouched
    for key in ("_sources", "_sourcesTotal", "_cited"):
        assert key in payload  # original untouched


async def test_other_parts_and_contents_pass_through_as_they_were():
    question = genai_types.Content(role="user", parts=[genai_types.Part(text="how is chevy?")])
    req, _, _, _ = _request_with({"id": "a1", "_sources": [{"fileId": "f1"}]})
    text = genai_types.Part(text="note")
    req.contents[0].parts.append(text)
    req.contents.insert(0, question)

    strip_source_metadata(None, req)

    assert req.contents[0] is question
    assert req.contents[1].parts[1] is text
    assert "_sources" not in req.contents[1].parts[0].function_response.response


@pytest.mark.parametrize(
    "call_id",
    [
        pytest.param("rp8i0001", id="model-issued-id"),  # as gemini-3.5-flash sends them
        pytest.param("adk-7f3e", id="adk-id"),
    ],
)
async def test_the_session_event_keeps_its_plumbing_through_adks_request_copy(call_id):
    """The request as ADK really builds it. `_copy_content_for_request` copies
    each Part but hands a function_response over BY REFERENCE unless its id is
    an `adk-` one it rewrites — so with the model's own id, writing to the
    request's FunctionResponse is writing to the stored event."""
    fr = genai_types.FunctionResponse(
        name="org_query",
        id=call_id,
        response={"results": [{"id": "a1"}], "_sources": [{"fileId": "f1"}], "_sourcesTotal": 1},
    )
    stored = genai_types.Content(role="user", parts=[genai_types.Part(function_response=fr)])
    payload = fr.response
    request = LlmRequest(
        contents=[_copy_content_for_request(stored, strip_client_function_call_ids=True)]
    )
    shared = request.contents[0].parts[0].function_response is stored.parts[0].function_response
    assert shared is (not call_id.startswith("adk-"))  # ADK's copy rule, as described

    strip_source_metadata(None, request)

    assert "_sources" not in _sent(request)
    assert stored.parts[0].function_response is fr
    assert fr.response is payload
    assert sorted(payload) == ["_sources", "_sourcesTotal", "results"]
