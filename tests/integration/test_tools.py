"""
Tool-layer behavior tests: get_account_overview's single-fetch semantics
(campaigns now ride embedded in the account detail), get_staff_profile's
double-fetch merge/error semantics, and the cold-session auth path (exactly
one token exchange per session, no matter how the fetches fan out).
"""

from __future__ import annotations

from gub_agent.tools import _client
from gub_agent.tools.accounts import get_account_overview
from gub_agent.tools.staff import get_staff_profile
from tests.helpers import FakeToolContext


async def test_get_staff_profile_merges_profile_and_metadata(gub):
    gub.routes[("GET", "/org/staff/s1")] = (200, {"id": "s1", "name": "Alex"})
    gub.routes[("GET", "/org/staff/s1/metadata")] = (200, {"metadata": [{"type": "skill"}]})

    result = await get_staff_profile("s1")

    assert result == {"id": "s1", "name": "Alex", "metadata": [{"type": "skill"}]}


async def test_get_staff_profile_returns_profile_error(gub):
    gub.routes[("GET", "/org/staff/s1")] = (403, {})
    gub.routes[("GET", "/org/staff/s1/metadata")] = (200, {"metadata": []})

    result = await get_staff_profile("s1")

    assert result["error"] is True
    assert result["status"] == 403


async def test_get_account_overview_single_fetch_returns_embedded_campaigns(gub):
    # The backend now embeds campaigns[] (with statusSummary) in the account
    # detail, so the overview is ONE fetch — no separate /campaigns call.
    account = {
        "id": "a1",
        "name": "Chevy",
        "campaigns": [
            {"id": "c1", "name": "Launch", "status": "live", "statusSummary": "On track."}
        ],
    }
    gub.routes[("GET", "/org/accounts/a1")] = (200, account)

    result = await get_account_overview("a1")

    assert result == account
    # The redundant campaigns-by-account call is gone.
    assert ("GET", "/org/accounts/a1/campaigns") not in gub.requests


async def test_get_account_overview_defaults_campaigns_when_absent(gub):
    # Defensive: if the stub-embedding backend isn't deployed yet, the tool
    # still returns a stable shape (empty campaigns) rather than a missing key.
    gub.routes[("GET", "/org/accounts/a1")] = (200, {"id": "a1", "name": "Chevy"})

    result = await get_account_overview("a1")

    assert result == {"id": "a1", "name": "Chevy", "campaigns": []}


async def test_get_account_overview_returns_error(gub):
    gub.routes[("GET", "/org/accounts/a1")] = (403, {})

    result = await get_account_overview("a1")

    assert result["error"] is True
    assert result["status"] == 403


async def test_cold_session_double_fetch_exchanges_token_once(gub, monkeypatch):
    monkeypatch.setattr(_client, "GUB_SERVICE_JWT", "")
    gub.routes[("POST", "/auth/google/access-token-exchange")] = (200, {"accessToken": "gub-jwt-1"})
    gub.routes[("GET", "/org/staff/s1")] = (200, {"id": "s1", "name": "Alex"})
    gub.routes[("GET", "/org/staff/s1/metadata")] = (200, {"metadata": []})
    ctx = FakeToolContext(**{_client.GUB_AUTHORIZATION_ID: "google-access-token"})

    result = await get_staff_profile("s1", ctx)

    assert result == {"id": "s1", "name": "Alex", "metadata": []}
    exchanges = gub.requests.count(("POST", "/auth/google/access-token-exchange"))
    assert exchanges == 1
    assert ctx.state["gub_jwt"] == "gub-jwt-1"
