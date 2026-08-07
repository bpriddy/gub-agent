"""
Integration fixtures: a real localhost MockGub server behind the client.
"""

from __future__ import annotations

import pytest

from gub_agent.tools import _client
from gub_agent.tools._client import aclose_client
from tests.helpers import MockGub


@pytest.fixture
async def gub(monkeypatch):
    server = MockGub()
    await server.start()
    monkeypatch.setattr(_client, "GUB_BASE_URL", f"http://127.0.0.1:{server.port}")
    monkeypatch.setattr(_client, "GUB_SERVICE_JWT", "test-jwt")
    yield server
    # Close the client first: 3.12's Server.wait_closed() waits for the
    # httpx keep-alive connection, which only drops when the client closes.
    await aclose_client()
    await server.stop()
