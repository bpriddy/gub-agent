"""
Client lifecycle and failure-path tests — no backend involved.

Covers the shared-client cache semantics (re-create after aclose,
re-bind on event-loop change) and the two no-backend failure paths
(unreachable host, no JWT source configured).
"""

from __future__ import annotations

import asyncio

import pytest

from gub_agent.tools import _client
from gub_agent.tools._client import _get_client, aclose_client, gub_get


async def test_unreachable_backend_returns_status_zero(monkeypatch):
    monkeypatch.setattr(_client, "GUB_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(_client, "GUB_SERVICE_JWT", "test-jwt")

    result = await gub_get("/org/accounts")

    assert result["error"] is True
    assert result["status"] == 0
    assert result["message"].startswith("Could not reach GUB backend")


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
