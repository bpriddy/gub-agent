"""
Client lifecycle and failure-path tests — no backend involved.

Covers the shared-client cache semantics (re-create after aclose,
re-bind on event-loop change) and the two no-backend failure paths
(unreachable host, no JWT source configured).
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from gub_agent.tools import _client
from gub_agent.tools._client import (
    _friendly_error,
    _get_client,
    aclose_client,
    gub_get,
    gub_post,
)


async def test_unreachable_backend_returns_status_zero(monkeypatch):
    monkeypatch.setattr(_client, "GUB_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(_client, "GUB_SERVICE_JWT", "test-jwt")

    result = await gub_get("/org/accounts")

    assert result["error"] is True
    assert result["status"] == 0
    assert result["message"] == "Could not reach company records right now."
    # The model may paraphrase this into the reader's answer: no system name.
    assert "gub" not in result["message"].lower()


async def test_unreachable_backend_on_post_returns_status_zero(monkeypatch):
    monkeypatch.setattr(_client, "GUB_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(_client, "GUB_SERVICE_JWT", "test-jwt")

    result = await gub_post("/org/query", {"entity": "campaigns"})

    assert result["error"] is True
    assert result["status"] == 0
    assert result["message"] == "Could not reach company records right now."
    assert "gub" not in result["message"].lower()


class _RaisingClient:
    """Stands in for the shared httpx client: every request fails."""

    def __init__(self, exc: Exception):
        self._exc = exc

    async def get(self, *args, **kwargs):
        raise self._exc

    async def post(self, *args, **kwargs):
        raise self._exc


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("hostname gcp-universal-backend-dev-abc.run.app does not match"),
        httpx.ReadTimeout(""),
    ],
)
@pytest.mark.parametrize("verb", ["get", "post"])
async def test_unreachable_message_never_carries_the_exception_text(monkeypatch, exc, verb):
    # The detail stays in the server-side log; the model gets one fixed sentence
    # (no hostname, no dangling ": " when the exception text is empty).
    monkeypatch.setattr(_client, "GUB_SERVICE_JWT", "test-jwt")
    monkeypatch.setattr(_client, "_get_client", lambda: _RaisingClient(exc))

    if verb == "get":
        result = await gub_get("/org/accounts")
    else:
        result = await gub_post("/org/query", {"entity": "campaigns"})

    assert result == {
        "error": True,
        "status": 0,
        "message": "Could not reach company records right now.",
    }


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 502, 503])
def test_friendly_error_never_names_the_backend(status):
    # These strings go back to the model as the tool result; it can quote them.
    message = _friendly_error(status, "/org/campaigns/abc")

    assert "gub" not in message.lower()
    assert "universal backend" not in message.lower()


def test_friendly_error_keeps_the_status_wording_the_model_keys_on():
    assert _friendly_error(401, "/org/staff").startswith("Authentication failed")
    assert "may have expired" in _friendly_error(401, "/org/staff")
    assert _friendly_error(403, "/org/staff").startswith("Access denied to /org/staff")
    assert _friendly_error(404, "/org/staff") == "Resource not found at /org/staff."
    # The unexpected-error fallback still carries the status and the path.
    fallback = _friendly_error(502, "/org/staff")
    assert fallback == "Company records returned an unexpected error (502) for /org/staff."


async def test_missing_jwt_raises(monkeypatch):
    monkeypatch.setattr(_client, "GUB_SERVICE_JWT", "")

    with pytest.raises(RuntimeError, match="No GUB JWT available"):
        await gub_get("/org/accounts")


async def test_shared_client_recreated_after_aclose():
    first = _get_client()
    assert _get_client() is first

    await aclose_client()
    assert first.is_closed

    second = _get_client()
    assert second is not first
    assert not second.is_closed


async def test_client_rebinds_when_loop_changes():
    async def grab():
        return _get_client()

    # A client cached by a different (now-finished) loop, never aclosed —
    # is_closed is still False, so only the loop check can catch it.
    first = await asyncio.to_thread(asyncio.run, grab())
    assert not first.is_closed

    second = _get_client()
    assert second is not first
