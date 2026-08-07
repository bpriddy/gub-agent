"""
Suite-wide fixtures: every test runs with a fresh shared client.

The MockGub server fixture lives in tests/integration/conftest.py — unit
tests never open a socket to a real endpoint.
"""

from __future__ import annotations

import pytest

from gub_agent.tools._client import aclose_client


@pytest.fixture(autouse=True)
async def _close_shared_client():
    yield
    await aclose_client()
