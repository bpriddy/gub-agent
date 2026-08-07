"""
HTTP behavior of the shared client against a live localhost server:
connection pooling across request types and friendly error mapping.
"""

from __future__ import annotations

from gub_agent.tools._client import gub_get, gub_post


async def test_get_post_and_404_share_one_connection(gub):
    gub.routes[("GET", "/org/accounts")] = (200, {"accounts": [{"id": "a1"}]})
    gub.routes[("POST", "/org/query")] = (200, {"results": [], "total": 0, "truncated": False})

    accounts = await gub_get("/org/accounts", q=None, limit=20)
    query = await gub_post("/org/query", {"entity": "campaigns"})
    missing = await gub_get("/org/nope")

    assert accounts == {"accounts": [{"id": "a1"}]}
    assert query["total"] == 0
    assert missing == {"error": True, "status": 404, "message": "Resource not found at /org/nope."}
    assert gub.connections == 1


async def test_friendly_error_mapping(gub):
    gub.routes[("GET", "/org/staff")] = (401, {})
    gub.routes[("GET", "/org/resourcing")] = (403, {})

    unauthorized = await gub_get("/org/staff")
    forbidden = await gub_get("/org/resourcing")

    assert unauthorized["status"] == 401
    assert "Authentication failed" in unauthorized["message"]
    assert forbidden["status"] == 403
    assert "Access denied to /org/resourcing" in forbidden["message"]
