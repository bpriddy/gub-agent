"""
Tool-layer behavior tests: the double-fetch tools' merge and error
semantics, the file search's response normalisation, and the cold-session
auth path (exactly one token exchange per session, no matter how the
fetches fan out).
"""

from __future__ import annotations

from gub_agent.tools import _client
from gub_agent.tools._client import ACCOUNT_SCOPE_NOTICE, SCOPE_HEADER
from gub_agent.tools.accounts import get_account_overview
from gub_agent.tools.files import find_files
from gub_agent.tools.staff import get_staff_profile
from tests.helpers import FakeToolContext

FILE_HIT = {
    "fileId": "f1",
    "name": "BHAC Hits The Road - 30s Teaser.mov",
    "mimeType": "video/quicktime",
    "accountId": "a1",
    "campaignId": None,
    "modifiedTime": "2026-08-02T10:00:00Z",
    "similarity": 0.378,
    "coverage": 0.81,
}


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


async def test_get_account_overview_degrades_on_campaigns_error(gub):
    gub.routes[("GET", "/org/accounts/a1")] = (200, {"id": "a1", "name": "Chevy"})
    gub.routes[("GET", "/org/accounts/a1/campaigns")] = (500, {})

    result = await get_account_overview("a1")

    assert result == {"id": "a1", "name": "Chevy", "campaigns": []}


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


# ── find_files: one response shape, whatever GUB answered with ───────────────


async def test_find_files_normalises_the_bare_array(gub):
    """GUB answers with a BARE ARRAY. gub-gchat-bot reads `files` off this
    response, so the array is given that key here rather than at the reader."""
    gub.routes[("GET", "/org/files/search")] = (200, [FILE_HIT])

    result = await find_files("BHAC 30 second teaser")

    assert result == {"files": [FILE_HIT]}


async def test_find_files_reports_no_match_as_an_empty_list(gub):
    """Empty is a CORRECT answer here (search-01 §6): the reported failure was
    eight unrelated files, not silence. It must arrive as the same shape, so
    the model reads "not found by name" and not a broken response."""
    gub.routes[("GET", "/org/files/search")] = (200, [])

    assert await find_files("final OnStar pitch pre-read doc") == {"files": []}


async def test_find_files_keeps_the_key_when_the_scope_notice_wraps_the_array(gub):
    """A tenant-10 scoped read wraps the bare array so the notice has somewhere
    to live — under `_ARRAY_KEYS["/org/files/search"]`, which is this same
    key. The scoped turn and the unscoped one hand the model one shape."""
    gub.routes[("GET", "/org/files/search")] = (200, [FILE_HIT], {SCOPE_HEADER: "1"})

    result = await find_files("BHAC 30 second teaser")

    assert result["files"] == [FILE_HIT]
    assert result["account_scope_notice"] == ACCOUNT_SCOPE_NOTICE


async def test_find_files_passes_an_error_through_untouched(gub):
    """Not dressed up as an empty file list: the model reads an empty list as
    "we do not have it", which is a claim about the world made from an HTTP
    failure."""
    gub.routes[("GET", "/org/files/search")] = (403, {})

    result = await find_files("BHAC 30 second teaser")

    assert result["error"] is True
    assert result["status"] == 403
    assert "files" not in result


async def test_find_files_omits_an_unset_account_scope(gub):
    """`gub_get` drops None params, so an unscoped call must not send an empty
    `accountId` — the backend would read that as a scope matching nothing."""
    gub.routes[("GET", "/org/files/search")] = (200, [])

    await find_files("BHAC teaser")
    await find_files("BHAC teaser", account_id="a1")

    targets = [t for m, t in gub.targets if m == "GET"]
    assert "accountId" not in targets[0]
    assert "accountId=a1" in targets[1]
