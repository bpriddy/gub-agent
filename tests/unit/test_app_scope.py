"""
app_scope — the agent half of UBScope (gub #119), read-only by structure.

The bot declares which records a tenant surface covers and puts that in session
state as `app_scope`, a STRING (its serialized JSON). The agent's whole job is
to forward that exact string as `X-App-Scope` on every records call, and to
read the labels out of a COPY for the persona. Pinned here:

  * the header equals the state value byte for byte — escapes, key order and
    unicode escapes survive, because nothing parses and re-serialises it;
  * absent means no header: the request is the one the agent has always sent;
  * the fast path (a `_ToolShim` over plain-dict session state) forwards it too;
  * the token exchange never carries it;
  * no tool can be handed a scope: no parameter of any registered tool is
    scope-like;
  * the persona names the scope's labels, and falls back to today's slug text.

Labels are synthetic: this repo is public and the engine is multi-company.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import httpx
import pytest

from gub_agent.agents import fast_path as fp
from gub_agent.tenant import (
    read_scope_labels,
    tenant_instruction,
    tenant_note,
    tenant_note_formatter,
)
from gub_agent.tools import ALL_TOOLS, _client
from gub_agent.tools._client import gub_get, gub_post
from gub_agent.tools.accounts import list_accounts

# Deliberately awkward: non-sorted keys, escaped quote, escaped non-ASCII,
# an escaped slash and extra spaces. Any parse/dump round trip changes it.
RAW_SCOPE = (
    '{"office":[{"name":"Annex \\"B\\"","id":"22222222-2222-2222-2222-222222222222"}],'
    '"account":[{"name":"Caf\\u00e9 Co","id":"11111111-1111-1111-1111-111111111111"},'
    '{"id":"33333333-3333-3333-3333-333333333333","name":"A\\/B"}]  }'
)


class _Recorder:
    """A shared-client stand-in that records the requests it is handed."""

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


def _tool_context(state: dict) -> SimpleNamespace:
    # `gub_jwt` cached: no token exchange, so each call is exactly one request.
    return SimpleNamespace(state={"gub_jwt": "cached-jwt", **state})


# ── the header ──────────────────────────────────────────────────────────────


async def test_get_sends_the_state_string_byte_for_byte(recorder) -> None:
    await gub_get("/org/accounts", _tool_context({"app_scope": RAW_SCOPE}))

    (req,) = recorder.requests
    assert req.headers.get_list("x-app-scope") == [RAW_SCOPE]
    # The raw wire bytes, not just httpx's decoded view.
    assert (b"x-app-scope", RAW_SCOPE.encode("ascii")) in [
        (k.lower(), v) for k, v in req.headers.raw
    ]
    assert req.headers["authorization"] == "Bearer cached-jwt"


async def test_post_sends_the_state_string_byte_for_byte(recorder) -> None:
    await gub_post("/org/query", {"entity": "campaigns"}, _tool_context({"app_scope": RAW_SCOPE}))

    (req,) = recorder.requests
    assert req.headers.get_list("x-app-scope") == [RAW_SCOPE]
    assert json.loads(req.content) == {"entity": "campaigns"}


async def test_the_scope_is_never_reserialised(recorder) -> None:
    """The value a round trip would produce differs from the stored one — so a
    header equal to the stored one proves nothing parsed it."""
    assert json.dumps(json.loads(RAW_SCOPE), separators=(",", ":")) != RAW_SCOPE

    await gub_get("/org/search", _tool_context({"app_scope": RAW_SCOPE}), q="x")

    assert recorder.requests[0].headers["x-app-scope"] == RAW_SCOPE


@pytest.mark.parametrize("state", [{}, {"app_scope": None}, {"tenant": "somebot"}])
async def test_no_scope_means_no_header(recorder, state) -> None:
    await gub_get("/org/accounts", _tool_context(state), limit=5)
    await gub_post("/org/query", {"entity": "campaigns"}, _tool_context(state))

    for req in recorder.requests:
        assert "x-app-scope" not in req.headers
        assert set(req.headers) >= {"authorization"}


async def test_without_a_tool_context_the_request_is_unchanged(recorder) -> None:
    await gub_get("/org/accounts")

    assert "x-app-scope" not in recorder.requests[0].headers


async def test_a_non_string_scope_is_not_forwarded(recorder) -> None:
    """An object cannot be sent byte for byte, and guessing a serialisation is
    the one thing the client must not do."""
    await gub_get("/org/accounts", _tool_context({"app_scope": {"account": []}}))

    assert "x-app-scope" not in recorder.requests[0].headers


async def test_the_fast_path_shim_forwards_it_too(recorder) -> None:
    """The fast path calls the same tool functions with a `_ToolShim` over the
    session's plain-dict state — no instruction provider, no ToolContext."""
    ctx = SimpleNamespace(
        session=SimpleNamespace(state={"gub_jwt": "cached-jwt", "app_scope": RAW_SCOPE}),
        invocation_id="inv-1",
    )

    await list_accounts(query="x", tool_context=fp._ToolShim(ctx))

    (req,) = recorder.requests
    assert req.headers["x-app-scope"] == RAW_SCOPE


class _State(dict):
    """ADK's State, as far as the token lookup is concerned."""

    def to_dict(self) -> dict:
        return dict(self)


async def test_the_token_exchange_never_carries_the_scope(recorder, monkeypatch) -> None:
    monkeypatch.setattr(_client, "GUB_AUTHORIZATION_ID", "auth-id")
    monkeypatch.setattr(_client, "GUB_BASE_URL", "http://backend.test")

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.requests.append(request)
        if request.url.path.endswith("/access-token-exchange"):
            return httpx.Response(200, json={"accessToken": "fresh-jwt"})
        return httpx.Response(200, json={})

    recorder.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    state = _State({"auth-id": "google-token", "app_scope": RAW_SCOPE})

    await gub_get("/org/accounts", SimpleNamespace(state=state))

    exchange, records = recorder.requests
    assert exchange.url.path.endswith("/auth/google/access-token-exchange")
    assert "x-app-scope" not in exchange.headers
    assert records.headers["x-app-scope"] == RAW_SCOPE
    assert records.headers["authorization"] == "Bearer fresh-jwt"


# ── no tool takes a scope ───────────────────────────────────────────────────


def test_there_are_tools_to_check() -> None:
    assert len(ALL_TOOLS) >= 10


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.__name__)
def test_no_tool_takes_a_scope_parameter(tool) -> None:
    """The scope is set by the bot and never by the model: a tool that could be
    handed one could be talked into widening or dropping it."""
    for name in inspect.signature(tool).parameters:
        assert "scope" not in name.lower(), f"{tool.__name__}({name})"
        assert "tenant" not in name.lower(), f"{tool.__name__}({name})"


def test_nothing_in_the_tools_package_writes_the_scope() -> None:
    import pathlib

    import gub_agent.tools as tools_pkg

    for path in pathlib.Path(tools_pkg.__file__).parent.glob("*.py"):
        text = path.read_text()
        assert 'state["app_scope"] =' not in text, path.name
        assert "state['app_scope'] =" not in text, path.name


# ── labels for the persona ──────────────────────────────────────────────────


def test_labels_are_read_in_declaration_order() -> None:
    assert read_scope_labels({"app_scope": RAW_SCOPE}) == ['Annex "B"', "Café Co", "A/B"]


def test_labels_never_include_ids() -> None:
    joined = " ".join(read_scope_labels({"app_scope": RAW_SCOPE}))
    assert "11111111" not in joined


def test_reading_labels_does_not_touch_the_stored_string() -> None:
    state = {"app_scope": RAW_SCOPE}
    read_scope_labels(state)
    assert state == {"app_scope": RAW_SCOPE}


@pytest.mark.parametrize(
    "state",
    [
        None,
        {},
        {"app_scope": None},
        {"app_scope": {"account": [{"id": "1", "name": "Obj"}]}},  # not a string
        {"app_scope": "not json"},
        {"app_scope": "[1, 2]"},
        {"app_scope": '{"account": "x"}'},
        {"app_scope": '{"account": [{"id": "1"}, 7, {"name": 3}, {"name": "  "}]}'},
    ],
)
def test_unreadable_or_absent_scope_yields_no_labels(state) -> None:
    assert read_scope_labels(state) == []


def test_labels_are_one_line_and_bounded() -> None:
    raw = json.dumps({"account": [{"id": "1", "name": "Line\nbreak  " + "x" * 500}]})
    (label,) = read_scope_labels({"app_scope": raw})
    assert "\n" not in label
    assert len(label) <= 128


def _ctx(state: dict) -> SimpleNamespace:
    return SimpleNamespace(state=state)


def _scope(*names: str) -> str:
    return json.dumps({"account": [{"id": f"id-{i}", "name": n} for i, n in enumerate(names)]})


def test_the_note_names_the_scope_labels_not_the_slug() -> None:
    provider = tenant_instruction(lambda _c: "BASE")
    out = provider(_ctx({"tenant": "somebot", "app_scope": _scope("Acme Roadworks")}))

    assert "branded for the Acme Roadworks account" in out
    assert "Treat Acme Roadworks as the default subject" in out
    assert "it is about Acme Roadworks." in out
    # The slug still names the surface, but is no longer passed off as the account.
    assert "the somebot account" not in out


def test_several_labels_read_as_a_list() -> None:
    provider = tenant_instruction(lambda _c: "BASE")
    out = provider(_ctx({"tenant": "somebot", "app_scope": _scope("Acme", "Bolt", "Crest")}))

    assert "branded for the Acme, Bolt and Crest accounts" in out


def test_the_formatter_note_names_the_labels_too() -> None:
    provider = tenant_instruction(lambda _c: "BASE", note=tenant_note_formatter)
    out = provider(_ctx({"tenant": "somebot", "app_scope": _scope("Acme Roadworks")}))

    assert "scoped to the Acme Roadworks account" in out


def test_without_a_scope_the_note_is_todays_slug_text() -> None:
    provider = tenant_instruction(lambda _c: "BASE")

    assert provider(_ctx({"tenant": "somebot"})) == f"BASE\n\n{tenant_note('somebot')}"
    assert "branded for the somebot account" in provider(_ctx({"tenant": "somebot"}))
    fmt = tenant_instruction(lambda _c: "BASE", note=tenant_note_formatter)
    assert fmt(_ctx({"tenant": "somebot"})) == f"BASE\n\n{tenant_note_formatter('somebot')}"


def test_an_unreadable_scope_falls_back_to_the_slug_text() -> None:
    provider = tenant_instruction(lambda _c: "BASE")

    out = provider(_ctx({"tenant": "somebot", "app_scope": "not json"}))

    assert out == f"BASE\n\n{tenant_note('somebot')}"


def test_no_tenant_ignores_a_scope_entirely() -> None:
    """Without a tenant every instruction is the base text, byte for byte —
    a stray scope in state changes nothing."""
    provider = tenant_instruction(lambda _c: "BASE")

    assert provider(_ctx({"app_scope": _scope("Acme")})) == "BASE"


def test_braces_in_a_label_reach_the_prompt_verbatim() -> None:
    """Provider-returned text bypasses ADK's `{state}` injection; the bot also
    rejects braces, but the agent must not crash or rewrite one that arrives."""
    provider = tenant_instruction(lambda _c: "BASE")
    out = provider(_ctx({"tenant": "somebot", "app_scope": _scope("Odd {Name}")}))

    assert "Odd {Name}" in out
