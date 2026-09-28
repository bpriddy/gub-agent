"""
SESSION_CLIENT_REUSE — one Vertex client per event loop for the agentengine
session service (services.py at the repo root).

ADK 2.9.2's VertexAiSessionService builds a new `vertexai.Client(...).aio` for
every session call and closes it when the call's `async with` exits: ~100 ms of
CPU and three SSL contexts per build, then ADC, a token fetch and a new TLS
connection before the one request. A deep turn makes ~12 appends and a
get_session, each awaited by the Runner before the agent continues.

These tests run against a fake `vertexai` module (CI does not install
google-cloud-aiplatform), so they count exactly what the base class does with
the client it is handed: how many are built, which requests go out with which
arguments, and when each client is closed.
"""

from __future__ import annotations

import asyncio
import datetime
import inspect
import os
import re
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest
from google.adk.agents import BaseAgent
from google.adk.cli import service_registry
from google.adk.cli.utils.service_factory import create_session_service_from_options
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService
from google.genai import types as genai_types

import services

REPO = Path(__file__).resolve().parents[2]
ENGINE = "9136379226620952576"
URI = f"agentengine://projects/proj-x/locations/us-central1/reasoningEngines/{ENGINE}"
USER = "user-1"
SESSION = "4273866877588996096"
STAMP = datetime.datetime(2026, 9, 25, 13, 13, 10, tzinfo=datetime.UTC)


# ── A fake `vertexai` that records what the service does with its clients ────


class _Recorder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.clients: list[_FakeAsyncClient] = []  # every client built, in order
        self.client_kwargs: list[dict] = []  # vertexai.Client(**kwargs) of each
        self.calls: list[tuple] = []  # (api method, kwargs, client index)

    def record(self, client: _FakeAsyncClient, method: str, kwargs: dict) -> None:
        with self.lock:
            self.calls.append((method, kwargs, self.clients.index(client)))

    def api_calls(self) -> list[tuple[str, dict]]:
        """The requests as the server would see them, without the client index."""
        return [(method, kwargs) for method, kwargs, _ in self.calls]

    @property
    def closes(self) -> int:
        return sum(c.closed for c in self.clients)


class _Events:
    def __init__(self, client: _FakeAsyncClient) -> None:
        self._client = client

    async def append(self, **kwargs):
        self._client.recorder.record(self._client, "events.append", kwargs)

    async def list(self, **kwargs):
        self._client.recorder.record(self._client, "events.list", kwargs)

        async def none():
            return
            yield

        return none()


class _Sessions:
    def __init__(self, client: _FakeAsyncClient) -> None:
        self._client = client
        self.events = _Events(client)

    def _session(self, name: str = f"reasoningEngines/{ENGINE}/sessions/{SESSION}"):
        return types.SimpleNamespace(
            name=name, user_id=USER, update_time=STAMP, session_state={"tenant": "anomaly"}
        )

    async def create(self, **kwargs):
        self._client.recorder.record(self._client, "sessions.create", kwargs)
        return types.SimpleNamespace(response=self._session())

    async def get(self, **kwargs):
        self._client.recorder.record(self._client, "sessions.get", kwargs)
        return self._session()

    async def list(self, **kwargs):
        self._client.recorder.record(self._client, "sessions.list", kwargs)
        sessions = [self._session()]

        async def each():
            for s in sessions:
                yield s

        return each()

    async def delete(self, **kwargs):
        self._client.recorder.record(self._client, "sessions.delete", kwargs)


class _FakeAsyncClient:
    """`vertexai.Client(...).aio`: `async with` closes it, like the real one."""

    def __init__(self, recorder: _Recorder) -> None:
        self.recorder = recorder
        self.closed = 0
        self.agent_engines = types.SimpleNamespace(sessions=_Sessions(self))

    async def aclose(self) -> None:
        self.closed += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        await self.aclose()


@pytest.fixture
def fake_vertexai(monkeypatch):
    recorder = _Recorder()

    class Client:
        def __init__(self, **kwargs):
            self.aio = _FakeAsyncClient(recorder)
            with recorder.lock:
                recorder.clients.append(self.aio)
                recorder.client_kwargs.append(kwargs)

    module = types.ModuleType("vertexai")
    module.Client = Client
    monkeypatch.setitem(sys.modules, "vertexai", module)
    return recorder


@pytest.fixture
def registry():
    """The process-wide ADK registry, restored after the test registers into it."""
    reg = service_registry.get_service_registry()
    saved = dict(reg._session_factories)
    yield reg
    reg._session_factories.clear()
    reg._session_factories.update(saved)


def _stock() -> VertexAiSessionService:
    return VertexAiSessionService(project="proj-x", location="us-central1", agent_engine_id=ENGINE)


def _kept() -> services.PerLoopClientSessionService:
    return services.PerLoopClientSessionService(
        project="proj-x", location="us-central1", agent_engine_id=ENGINE
    )


def _event(i: int, author: str = "gub_agent") -> Event:
    return Event(
        id=f"evt-{i}",
        invocation_id="e-57f8453d",
        author=author,
        timestamp=1790342000.0 + i,
        content=genai_types.Content(role="model", parts=[genai_types.Part(text=f"round {i}")]),
    )


async def _deep_turn_calls(service: VertexAiSessionService, appends: int) -> None:
    """A deep turn's session traffic: one get_session, then the appends."""
    session = await service.get_session(app_name="gub_agent", user_id=USER, session_id=SESSION)
    for i in range(appends):
        await service.append_event(session, _event(i))


# ── Reuse on one loop ─────────────────────────────────────────────────────────


def test_one_loop_reuses_one_client_for_every_call(fake_vertexai):
    service = _kept()

    async def turn():
        async with service._get_api_client() as first:
            pass
        async with service._get_api_client() as second:
            pass
        assert second is first
        await _deep_turn_calls(service, appends=12)
        assert len(fake_vertexai.clients) == 1, "one client for the whole loop"
        assert fake_vertexai.closes == 0, "nothing closes it while the loop runs"

    asyncio.run(turn())

    assert {index for *_, index in fake_vertexai.calls} == {0}  # 13 calls, one client
    assert len(fake_vertexai.calls) == 1 + 1 + 12  # sessions.get + events.list + appends
    assert fake_vertexai.closes == 1, "closed once, when asyncio.run shut the loop down"
    assert service._clients == {}, "the finished loop is not kept"


def test_leaving_the_async_with_does_not_close_the_client(fake_vertexai):
    service = _kept()

    async def turn():
        async with service._get_api_client() as client:
            pass
        assert client.closed == 0
        session = await service.get_session(app_name="gub_agent", user_id=USER, session_id=SESSION)
        await service.append_event(session, _event(0))
        assert client.closed == 0
        return client

    client = asyncio.run(turn())
    assert client.closed == 1


def test_the_stock_service_builds_and_closes_a_client_per_call(fake_vertexai):
    """What the flag turns off — ADK's own behaviour, pinned so the contrast
    above is measured against the real base class, not an assumption."""
    service = _stock()

    asyncio.run(_deep_turn_calls(service, appends=12))

    assert len(fake_vertexai.clients) == 13
    assert [c.closed for c in fake_vertexai.clients] == [1] * 13


# ── Never shared across loops ─────────────────────────────────────────────────


def test_each_loop_gets_its_own_client_and_closes_it(fake_vertexai):
    service = _kept()

    async def grab():
        async with service._get_api_client() as client:
            return client

    first = asyncio.run(grab())
    second = asyncio.run(grab())

    assert first is not second
    assert (first.closed, second.closed) == (1, 1)
    assert service._clients == {}


def test_concurrent_turns_on_two_threads_never_share_a_client(fake_vertexai):
    """The engine runs each stream_query turn in asyncio.run on its own thread
    (ADK Runner.run); two users' turns overlap."""
    service = _kept()
    both_inside = threading.Barrier(2, timeout=5)
    seen: dict[str, object] = {}

    async def turn(name: str):
        async with service._get_api_client() as client:
            seen[name] = client
        await asyncio.to_thread(both_inside.wait)  # both loops alive at once
        await _deep_turn_calls(service, appends=3)
        async with service._get_api_client() as again:
            assert again is seen[name]

    threads = [threading.Thread(target=asyncio.run, args=(turn(n),)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert seen["a"] is not seen["b"]
    assert len(fake_vertexai.clients) == 2
    for index in (0, 1):  # each client carried only its own loop's calls
        assert sum(1 for *_, i in fake_vertexai.calls if i == index) == 2 + 3
    assert fake_vertexai.closes == 2
    assert service._clients == {}


def test_fifty_turns_leave_nothing_behind(fake_vertexai):
    service = _kept()
    for _ in range(50):
        asyncio.run(_deep_turn_calls(service, appends=2))

    assert len(fake_vertexai.clients) == 50
    assert fake_vertexai.closes == 50
    assert service._clients == {}


def test_a_call_after_the_loop_client_was_closed_gets_a_client_of_its_own(fake_vertexai):
    """A task still cleaning up while asyncio.run shuts its loop down must not
    be handed the client that was just closed."""
    service = _kept()

    async def turn():
        async with service._get_api_client() as kept:
            pass
        _, closer = service._clients[asyncio.get_running_loop()]
        await asyncio.sleep(0)  # the closer has started, as it has by any real request
        closer.cancel()  # what asyncio.run does at shutdown
        with pytest.raises(asyncio.CancelledError):
            await closer
        assert kept.closed == 1
        async with service._get_api_client() as late:
            assert late is not kept
        assert late.closed == 1, "closed by its own async with, the base behaviour"

    asyncio.run(turn())
    assert len(fake_vertexai.clients) == 2


def test_a_closer_cancelled_before_it_started_still_forgets_the_loop(fake_vertexai):
    """asyncio.run cancels every pending task at shutdown; one that never took
    a step never runs its body. The loop must still be dropped and retired."""
    service = _kept()

    async def turn():
        async with service._get_api_client() as kept:
            pass
        loop = asyncio.get_running_loop()
        _, closer = service._clients[loop]
        closer.cancel()  # before its first step
        with pytest.raises(asyncio.CancelledError):
            await closer
        assert loop not in service._clients
        async with service._get_api_client() as late:
            assert late is not kept
        assert late.closed == 1

    asyncio.run(turn())
    assert service._clients == {}


def test_a_first_call_from_a_task_cancelled_at_shutdown_is_closed_like_the_base(fake_vertexai):
    """A task whose CancelledError handler makes the loop's FIRST session call
    while asyncio.run is cancelling the pending tasks: a closer created then is
    never cancelled (asyncio.run already took its list), so the call gets the
    base class's client, closed by its own async with, and nothing is kept."""
    service = _kept()

    async def turn():
        async def cleanup_on_cancel():
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                async with service._get_api_client() as late:
                    pass
                assert late.closed == 1, "closed by its own async with"
                raise

        asyncio.get_running_loop().create_task(cleanup_on_cancel())
        await asyncio.sleep(0)

    asyncio.run(turn())

    assert len(fake_vertexai.clients) == 1
    assert fake_vertexai.closes == 1
    assert service._clients == {}, "the finished loop is not kept"


def test_a_first_call_from_an_async_generator_finalizer_is_not_kept(fake_vertexai):
    """The same, from an async generator that asyncio.run finalizes after it
    cancelled the tasks (shutdown_asyncgens): the closed loop is not kept."""
    service = _kept()
    gens = []

    async def turn():
        async def gen():
            try:
                yield 1
            finally:
                async with service._get_api_client():
                    pass

        g = gen()
        gens.append(g)  # kept alive, so only shutdown_asyncgens finalizes it
        await g.__anext__()

    asyncio.run(turn())
    # Any later call (another turn) drops a loop that closed with its entry.
    asyncio.run(_deep_turn_calls(service, appends=1))

    assert service._clients == {}, "no closed loop is kept"
    assert fake_vertexai.closes == len(fake_vertexai.clients)


# ── The same requests as the base class ───────────────────────────────────────


async def _scripted(service: VertexAiSessionService) -> None:
    created = await service.create_session(
        app_name="gub_agent", user_id=USER, state={"tenant": "anomaly"}, session_id=SESSION
    )
    session = await service.get_session(app_name="gub_agent", user_id=USER, session_id=created.id)
    for i in range(3):
        await service.append_event(session, _event(i, author=("router", "gub_agent")[i % 2]))
    await service.list_sessions(app_name="gub_agent", user_id=USER)
    await service.delete_session(app_name="gub_agent", user_id=USER, session_id=SESSION)


@pytest.mark.parametrize(
    "ctor, env",
    [
        (dict(project="proj-x", location="us-central1", agent_engine_id=ENGINE), {}),
        (
            dict(agent_engine_id=ENGINE, express_mode_api_key="express-key"),
            {"GOOGLE_GENAI_USE_ENTERPRISE": "1"},
        ),
    ],
    ids=["project-location", "express-mode"],
)
def test_the_same_api_calls_and_client_arguments_as_the_base_class(
    monkeypatch, fake_vertexai, ctor, env
):
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    asyncio.run(_scripted(VertexAiSessionService(**ctor)))
    stock_calls = fake_vertexai.api_calls()
    stock_kwargs = fake_vertexai.client_kwargs[:]
    fake_vertexai.calls.clear()
    fake_vertexai.client_kwargs.clear()

    asyncio.run(_scripted(services.PerLoopClientSessionService(**ctor)))

    assert fake_vertexai.api_calls() == stock_calls
    assert [c[0] for c in stock_calls] == [
        "sessions.create",
        "sessions.get",
        "events.list",
        "events.append",
        "events.append",
        "events.append",
        "sessions.list",
        "sessions.get",
        "sessions.delete",
    ]
    # create, get (sessions.get + events.list), 3 appends, list, delete
    # (sessions.get + sessions.delete): 7 clients before, 1 now.
    assert len(stock_kwargs) == 7 and len(fake_vertexai.client_kwargs) == 1
    assert fake_vertexai.client_kwargs[0] == stock_kwargs[0]
    assert all(kwargs == stock_kwargs[0] for kwargs in stock_kwargs)


def test_a_runner_turn_builds_one_client_instead_of_one_per_event(fake_vertexai):
    """The engine's own path: AdkApp.stream_query -> ADK's sync Runner.run,
    which runs the turn in asyncio.run on a thread of its own. A deep turn
    persists ~12 events (p50 of 77 production deep turns, 2026-09-17..25)."""

    class Rounds(BaseAgent):
        async def _run_async_impl(self, ctx):
            for i in range(11):
                yield Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    content=genai_types.Content(
                        role="model", parts=[genai_types.Part(text=f"round {i}")]
                    ),
                )

    def one_turn(service):
        runner = Runner(
            app_name="gub_agent", agent=Rounds(name="gub_agent"), session_service=service
        )
        message = genai_types.Content(role="user", parts=[genai_types.Part(text="Q3 status?")])
        return list(runner.run(user_id=USER, session_id=SESSION, new_message=message))

    streamed = one_turn(_stock())
    stock_built, stock_calls = len(fake_vertexai.clients), fake_vertexai.calls[:]
    fake_vertexai.calls.clear()
    fake_vertexai.clients.clear()

    assert len(one_turn(_kept())) == len(streamed) == 11

    # get_session + the user message + 11 events = 13 clients before, 1 now.
    assert stock_built == 13
    assert len(fake_vertexai.clients) == 1
    assert fake_vertexai.clients[0].closed == 1
    assert [c[0] for c in fake_vertexai.calls] == [c[0] for c in stock_calls]
    assert [c[1]["author"] for c in fake_vertexai.calls if c[0] == "events.append"] == ["user"] + [
        "gub_agent"
    ] * 11


# ── Registration: the flag, the factory, the URI ──────────────────────────────


@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE"])
def test_the_flag_on_registers_the_subclass(monkeypatch, fake_vertexai, registry, tmp_path, value):
    monkeypatch.setenv(services.FLAG, value)
    assert services.register() is True

    service = create_session_service_from_options(base_dir=tmp_path, session_service_uri=URI)

    assert type(service) is services.PerLoopClientSessionService
    assert (service._project, service._location, service._agent_engine_id) == (
        "proj-x",
        "us-central1",
        ENGINE,
    )


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off"])
def test_the_flag_off_leaves_adks_own_factory_and_service(
    monkeypatch, fake_vertexai, registry, tmp_path, value
):
    if value is None:
        monkeypatch.delenv(services.FLAG, raising=False)
    else:
        monkeypatch.setenv(services.FLAG, value)
    builtin = registry._session_factories[services.SCHEME]

    assert services.register() is False

    assert registry._session_factories[services.SCHEME] is builtin
    service = create_session_service_from_options(base_dir=tmp_path, session_service_uri=URI)
    assert type(service) is VertexAiSessionService
    asyncio.run(_deep_turn_calls(service, appends=2))
    assert len(fake_vertexai.clients) == 3 and fake_vertexai.closes == 3


def test_the_factory_parses_the_uri_like_the_builtin_one(monkeypatch, fake_vertexai, registry):
    """Both URI forms: the full resource name the deploy passes, and a bare id
    resolved from GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION."""
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-env")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
    builtin = registry._session_factories[services.SCHEME]
    for uri in (URI, f"agentengine://{ENGINE}"):
        stock = builtin(uri, agents_dir="/nonexistent")
        ours = services._session_factory(uri, agents_dir="/nonexistent")
        assert type(stock) is VertexAiSessionService
        assert (ours._project, ours._location, ours._agent_engine_id) == (
            stock._project,
            stock._location,
            stock._agent_engine_id,
        )


def test_every_use_of_the_client_in_adk_is_an_async_with():
    """The no-op __aexit__ is the whole mechanism: it holds only while every
    call site in the installed VertexAiSessionService takes the client through
    `async with`. An ADK upgrade that uses it any other way must fail here."""
    source = inspect.getsource(VertexAiSessionService)
    uses = source.count("self._get_api_client()")
    assert uses == 5  # create, get, list, delete, append
    assert source.count("async with self._get_api_client() as api_client:") == uses


def test_services_py_never_imports_gub_agent():
    """ADK imports services.py at server start, before the first request loads
    the agent; gub_agent.config rewrites GOOGLE_CLOUD_LOCATION at import."""
    for flag in ("0", "1"):
        env = {**os.environ, services.FLAG: flag, "GOOGLE_CLOUD_LOCATION": "us-central1"}
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, sys, services; "
                "print(any(m == 'gub_agent' or m.startswith('gub_agent.') for m in sys.modules),"
                " os.environ['GOOGLE_CLOUD_LOCATION'])",
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert out.stdout.split() == ["False", "us-central1"], out.stderr


# ── The deploys stage services.py and state the flag ──────────────────────────


def _env_value(path: Path, key: str) -> str | None:
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith(f"{key}="):
            return line.partition("=")[2].strip()
    return None


@pytest.mark.parametrize(
    "script, anchor",
    [
        (REPO / ".github" / "workflows" / "deploy.yml", "$GITHUB_WORKSPACE/"),
        (REPO / "deployment" / "deploy-sandbox.sh", "$REPO_ROOT/"),
    ],
    ids=["prod-workflow", "sandbox-script"],
)
def test_both_deploys_stage_services_py_by_absolute_path(script, anchor):
    """Without --extra_packages the engine never sees this file and silently
    runs the stock service. Absolute, because adk deploy chdir()s before it
    resolves paths — the lesson of the relative --env_file."""
    found = re.findall(r"--extra_packages=\"?([^\s\"]+)\"?", script.read_text())
    assert found == [f"{anchor}services.py"], found
    assert (REPO / "services.py").is_file()


@pytest.mark.parametrize("env_file", ["deploy-prod.env", "deploy-sandbox.env"])
def test_both_deploy_env_files_state_the_flag(env_file):
    assert _env_value(REPO / env_file, services.FLAG) in ("0", "1")
