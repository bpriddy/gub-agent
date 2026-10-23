"""
A tenant turn without an app scope fails hard (UBScope, gub #119, step B2b).

A tenant bot never runs unscoped. With `state["tenant"]` set, `app_scope` must
be a string; anything else raises ValueError — the same way a malformed tenant
label does — and writes one `tenant_scope: missing … tenant=<label>` line.

There are two doors and both are pinned: the tenant instruction provider (the
deep path) and `tools/_client.py` before any request. The fast path never runs
an instruction provider, so a check in the provider alone would let a fast-path
tenant turn call the backend unscoped. With no tenant, nothing changes.

Labels are synthetic: this repo is public.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest

from gub_agent import agent as _agent
from gub_agent.agents import critic as _critic
from gub_agent.agents import fast_path as fp
from gub_agent.agents import formatter as _formatter
from gub_agent.agents import router as _router
from gub_agent.tenant import (
    require_app_scope,
    tenant_instruction,
    tenant_note_formatter,
)
from gub_agent.tools import _client
from gub_agent.tools._client import _resolve_gub_jwt, gub_get, gub_post
from gub_agent.tools.accounts import get_account_overview, list_accounts

SCOPE = '{"account":[{"id":"11111111-1111-1111-1111-111111111111","name":"Acme Roadworks"}]}'

MISSING_STATES = [
    pytest.param({"tenant": "somebot"}, id="absent"),
    pytest.param({"tenant": "somebot", "app_scope": None}, id="none"),
    pytest.param({"tenant": "somebot", "app_scope": ""}, id="empty-string"),
    pytest.param({"tenant": "somebot", "app_scope": {"account": []}}, id="object-not-string"),
    pytest.param({"tenant": "somebot", "app_scope": 7}, id="number"),
]


def _ctx(state: dict) -> SimpleNamespace:
    return SimpleNamespace(state=state)


class _Recorder:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json={"ok": True})

        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture
def recorder(monkeypatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(_client, "GUB_SERVICE_JWT", "test-jwt")
    monkeypatch.setattr(_client, "_get_client", lambda: rec.client)
    return rec


# ── door 1: the instruction provider (deep path) ────────────────────────────


@pytest.mark.parametrize("state", MISSING_STATES)
def test_the_tenant_provider_raises_without_a_scope(state) -> None:
    provider = tenant_instruction(lambda _c: "BASE")

    with pytest.raises(ValueError, match="app_scope"):
        provider(_ctx(state))


@pytest.mark.parametrize("state", MISSING_STATES)
def test_the_formatter_provider_raises_without_a_scope(state) -> None:
    provider = tenant_instruction(lambda _c: "BASE", note=tenant_note_formatter)

    with pytest.raises(ValueError, match="app_scope"):
        provider(_ctx(state))


@pytest.mark.parametrize(
    "agent",
    [_agent.executor_agent, _router.router_agent, _critic.critic_agent, _formatter.formatter_agent],
    ids=["executor", "router", "critic", "formatter"],
)
def test_every_real_agent_raises_on_a_scopeless_tenant_turn(agent) -> None:
    state = {"gub_jwt": "jwt", "tenant": "somebot"}
    ctx = SimpleNamespace(state=state, session=SimpleNamespace(state=state, events=[]))

    with pytest.raises(ValueError, match="app_scope"):
        agent.instruction(ctx)


def test_a_scoped_tenant_turn_renders() -> None:
    provider = tenant_instruction(lambda _c: "BASE")

    assert "Acme Roadworks" in provider(_ctx({"tenant": "somebot", "app_scope": SCOPE}))


def test_no_tenant_needs_no_scope_and_changes_nothing() -> None:
    provider = tenant_instruction(lambda _c: "BASE")

    assert provider(_ctx({"gub_jwt": "jwt"})) == "BASE"
    assert provider(_ctx({})) == "BASE"
    assert provider(_ctx(None)) == "BASE"
    require_app_scope({"gub_jwt": "jwt"})
    require_app_scope(None)


def test_a_malformed_tenant_still_raises_its_own_error_first() -> None:
    with pytest.raises(ValueError, match="lowercase slug"):
        require_app_scope({"tenant": "Not A Slug", "app_scope": SCOPE})


# ── door 2: the client, before any request (fast path included) ─────────────


@pytest.mark.parametrize("state", MISSING_STATES)
async def test_gub_get_raises_before_any_request(recorder, state) -> None:
    with pytest.raises(ValueError, match="app_scope"):
        await gub_get("/org/accounts", _ctx({"gub_jwt": "cached", **state}))

    assert recorder.requests == []


@pytest.mark.parametrize("state", MISSING_STATES)
async def test_gub_post_raises_before_any_request(recorder, state) -> None:
    with pytest.raises(ValueError, match="app_scope"):
        await gub_post("/org/query", {"entity": "campaigns"}, _ctx({"gub_jwt": "cached", **state}))

    assert recorder.requests == []


async def test_a_cold_tenant_session_does_not_even_exchange_a_token(recorder, monkeypatch) -> None:
    """The exchange is a GUB call too: it must not happen for a scopeless tenant turn."""

    class _State(dict):
        def to_dict(self) -> dict:
            return dict(self)

    monkeypatch.setattr(_client, "GUB_AUTHORIZATION_ID", "auth-id")
    state = _State({"auth-id": "google-token", "tenant": "somebot"})

    with pytest.raises(ValueError, match="app_scope"):
        await _resolve_gub_jwt(SimpleNamespace(state=state))
    with pytest.raises(ValueError, match="app_scope"):
        await gub_get("/org/accounts", SimpleNamespace(state=state))

    assert recorder.requests == []
    assert "gub_jwt" not in state


async def test_the_fast_path_shim_raises_before_any_request(recorder) -> None:
    """The fast path runs no instruction provider: the client is its only gate."""
    ctx = SimpleNamespace(
        session=SimpleNamespace(state={"gub_jwt": "cached", "tenant": "somebot"}),
        invocation_id="inv-1",
    )

    with pytest.raises(ValueError, match="app_scope"):
        await list_accounts(query="x", tool_context=fp._ToolShim(ctx))

    assert recorder.requests == []


async def test_the_warm_up_tools_raise_before_their_warm_up(recorder) -> None:
    """`get_account_overview` warms the token cache directly, before its fan-out."""
    tool_context = _ctx({"gub_jwt": "cached", "tenant": "somebot"})

    with pytest.raises(ValueError, match="app_scope"):
        await get_account_overview("acct-1", tool_context)

    assert recorder.requests == []


async def test_a_scoped_tenant_turn_calls_the_backend_with_the_header(recorder) -> None:
    state = {"gub_jwt": "cached", "tenant": "somebot", "app_scope": SCOPE}

    await gub_get("/org/accounts", _ctx(state))
    await gub_post("/org/query", {"entity": "campaigns"}, _ctx(state))

    assert [r.headers["x-app-scope"] for r in recorder.requests] == [SCOPE, SCOPE]


async def test_no_tenant_no_scope_is_the_request_it_always_was(recorder) -> None:
    await gub_get("/org/accounts", _ctx({"gub_jwt": "cached"}), limit=3)
    await gub_get("/org/accounts")

    for req in recorder.requests:
        assert "x-app-scope" not in req.headers


# ── the log line ────────────────────────────────────────────────────────────


def test_the_missing_scope_line_ends_with_the_tenant_label(caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="gub_agent.tenant"):
        with pytest.raises(ValueError):
            require_app_scope({"tenant": "somebot"})

    (line,) = [r.getMessage() for r in caplog.records if "tenant_scope:" in r.getMessage()]
    assert line.startswith("tenant_scope: missing")
    assert line.endswith("tenant=somebot")


def test_the_log_line_never_carries_the_value(caplog) -> None:
    """A non-string scope may hold labels (or anything): only its TYPE is logged."""
    state = {"tenant": "somebot", "app_scope": {"account": [{"id": "i", "name": "Acme Roadworks"}]}}

    with caplog.at_level(logging.ERROR, logger="gub_agent.tenant"):
        with pytest.raises(ValueError) as excinfo:
            require_app_scope(state)

    text = " ".join(r.getMessage() for r in caplog.records) + str(excinfo.value)
    assert "Acme Roadworks" not in text
    assert "dict" in text


def test_a_scoped_or_tenantless_turn_logs_nothing(caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="gub_agent.tenant"):
        require_app_scope({"tenant": "somebot", "app_scope": SCOPE})
        require_app_scope({})

    assert not [r for r in caplog.records if "tenant_scope" in r.getMessage()]
