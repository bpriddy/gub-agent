"""
services.py — ADK service registration for the deployed engine (SESSION_CLIENT_REUSE).

Why this file exists: ADK 2.9.2's `VertexAiSessionService._get_api_client`
builds a brand-new `vertexai.Client(...).aio` for EVERY session call and the
`async with` around each call closes it again. Construction alone is ~100 ms of
CPU (three SSL contexts), and every fresh client then loads ADC, fetches an
access token and opens a new TLS connection before its one request. The Runner
awaits each non-partial `append_event` before the agent continues, and a deep
turn makes ~12 of them plus one `get_session` — all on the turn's critical path.

With SESSION_CLIENT_REUSE on, the `agentengine://` session service keeps ONE
client per running event loop instead: every call on that loop reuses it (its
connection pool and its token included), and it is closed when the loop shuts
down. The requests are the same, argument for argument; only the client object
that sends them is kept. Off (the default) registers nothing, so ADK's own
factory builds the stock `VertexAiSessionService` exactly as before.

How it is loaded — the only supported hook on this path. The engine runs
`adk api_server ... "/app/agents"` (the CMD `adk deploy agent_engine` writes
into its Dockerfile). `get_fast_api_app` calls
`service_registry.load_services_module("/app/agents")`, which does
`importlib.import_module("services")` BEFORE it builds the session service from
`--session_service_uri=agentengine://…`. The deploy only copies the agent folder
into /app/agents/gub_agent/, so this file is staged with
`adk deploy agent_engine --extra_packages=<this file>`: ADK copies it to
/app/services.py and puts /app on PYTHONPATH (deploy.yml, deploy-sandbox.sh).
Locally, `adk web` / `adk api_server` run from the repo root load it the same
way (the repo root is their agents dir).

Why it lives outside gub_agent/ and never imports it: this module is imported
at server start, before the first request loads the agent. `gub_agent.config`
rewrites GOOGLE_CLOUD_LOCATION at import and `gub_agent/__init__` builds the
whole agent tree, so importing the package here would change what the server
does at start (the Gemini Enterprise adapter reads GOOGLE_CLOUD_LOCATION right
after the services are built). It depends on ADK and vertexai only.

Why per event loop: the engine answers `stream_query` through ADK's sync
`Runner.run`, which runs every turn in `asyncio.run` on a thread of its own —
a fresh loop per turn — and the client's HTTP sessions belong to the loop that
opened them. One client per loop is therefore one client per turn, never shared
between turns running at the same time.

Rollback: SESSION_CLIENT_REUSE=0 and redeploy. A failure to import this file is
logged by ADK as a warning and leaves the built-in service in place.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import weakref
from typing import Any
from urllib.parse import urlparse

from google.adk.cli import service_registry
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService

logger = logging.getLogger(__name__)

FLAG = "SESSION_CLIENT_REUSE"
SCHEME = "agentengine"


def reuse_enabled() -> bool:
    """SESSION_CLIENT_REUSE, parsed the way gub_agent/config.py parses its flags."""
    return os.environ.get(FLAG, "0").strip().lower() in ("1", "true", "yes")


class _KeptClient:
    """What `_get_api_client()` hands the base class while reuse is on.

    Every call site in ADK 2.9.2's VertexAiSessionService is
    `async with self._get_api_client() as api_client:` (pinned by
    tests/unit/test_session_client_reuse.py). Entering yields the loop's client;
    leaving does nothing, so the client outlives the call. The loop's shutdown
    closes it (`PerLoopClientSessionService._close_with_loop`).
    """

    __slots__ = ("_client",)

    def __init__(self, client: Any) -> None:
        self._client = client

    async def __aenter__(self) -> Any:
        return self._client

    async def __aexit__(self, *exc_info: object) -> None:
        return None


class PerLoopClientSessionService(VertexAiSessionService):
    """VertexAiSessionService with one API client per running event loop.

    Everything but `_get_api_client` is the base class's: the same requests
    with the same arguments, the same retries, the same in-memory session
    updates. The client itself is built by the base class's own
    `_get_api_client` (project, location, express-mode key, http options).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Turns run on different threads at the same time, each on its own loop.
        self._clients_lock = threading.Lock()
        # loop -> (client, the task that closes it at the loop's shutdown). The
        # task is held here because the loop keeps only a weak reference to it.
        self._clients: dict[asyncio.AbstractEventLoop, tuple[Any, asyncio.Task[None]]] = {}
        # Loops whose client has already been closed at shutdown. A session call
        # made after that point (a task cleaning up while its loop shuts down)
        # gets a client of its own that its `async with` closes, the base
        # class's behaviour, instead of the closed one. Weak, so a finished loop
        # is not kept alive by this set.
        self._retired: weakref.WeakSet[asyncio.AbstractEventLoop] = weakref.WeakSet()

    def _get_api_client(self) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # not called from a coroutine: nothing to key on
            return super()._get_api_client()
        with self._clients_lock:
            self._drop_closed_loops()
            if _contains(self._retired, loop):
                return super()._get_api_client()
            entry = self._clients.get(loop)
            if entry is None:
                if _shutting_down(loop):
                    # A closer created now would never be cancelled (asyncio.run
                    # has already collected the tasks it cancels), so the client
                    # would never be closed: the base class's own client instead.
                    return super()._get_api_client()
                client = super()._get_api_client()
                # Closed on its own loop when the loop shuts down: asyncio.run()
                # (each turn: Runner.run; each sync session method: AdkApp)
                # cancels the tasks still pending once its main coroutine
                # returns and waits for them before it closes the loop. The
                # uvicorn loop's client lives as long as the server.
                closer = loop.create_task(
                    self._close_with_loop(loop, client),
                    name="session_client_reuse: close at loop shutdown",
                )
                # Forgotten again in a done-callback: a task cancelled before
                # its first step never runs its body, but its callbacks always
                # run. (Such a client has sent no request —
                # sending one yields to the loop, which starts the task — so it
                # holds no connection to close.)
                closer.add_done_callback(
                    lambda _task, loop=loop, client=client: self._forget(loop, client)
                )
                entry = self._clients[loop] = (client, closer)
        return _KeptClient(entry[0])

    def _drop_closed_loops(self) -> None:
        """Backstop, under the lock: a loop that closed with its entry still here
        (its closer never ran) is dropped, so a finished loop is never kept."""
        for closed in [lp for lp in self._clients if lp.is_closed()]:
            del self._clients[closed]

    def _forget(self, loop: asyncio.AbstractEventLoop, client: Any) -> None:
        with self._clients_lock:
            entry = self._clients.get(loop)
            if entry is not None and entry[0] is client:
                del self._clients[loop]
            try:
                self._retired.add(loop)
            except TypeError:  # a loop type without weakref support
                pass

    async def _close_with_loop(self, loop: asyncio.AbstractEventLoop, client: Any) -> None:
        try:
            await loop.create_future()  # never resolved: only cancellation ends it
        finally:
            # Retired BEFORE the close is awaited, so a call made meanwhile
            # gets a client of its own, never the one being closed. The
            # done-callback repeats this (idempotent) for a task that never ran.
            self._forget(loop, client)
            try:
                await client.aclose()
            except Exception:  # closing must never fail the shutdown
                logger.debug("session_client_reuse: closing a loop's client failed", exc_info=True)


def _shutting_down(loop: asyncio.AbstractEventLoop) -> bool:
    """Whether asyncio.run is already past the point where it cancels tasks.

    True from a task whose cancellation was requested (a CancelledError handler
    at shutdown, or a turn being cancelled) and from an async generator being
    finalized by `loop.shutdown_asyncgens()`. Either way the call just gets the
    base class's client, closed by its own `async with`.
    """
    task = asyncio.current_task(loop)
    if task is not None and task.cancelling():
        return True
    return bool(getattr(loop, "_asyncgens_shutdown_called", False))


def _contains(loops: weakref.WeakSet[asyncio.AbstractEventLoop], loop: object) -> bool:
    try:
        return loop in loops
    except TypeError:  # a loop type without weakref support is never retired
        return False


def _session_factory(uri: str, **kwargs: Any) -> VertexAiSessionService:
    """The built-in `agentengine` factory, with the per-loop subclass.

    The URI is parsed by the same ADK helper the built-in factory uses, so the
    project, location and engine id are the ones the stock service would get.
    """
    parsed = urlparse(uri)
    params = service_registry._parse_agent_engine_kwargs(
        parsed.netloc + parsed.path, kwargs.get("agents_dir")
    )
    service = PerLoopClientSessionService(**params)
    logger.info(
        "session_client_reuse: session service=%s engine=%s",
        type(service).__name__,
        params["agent_engine_id"],
    )
    return service


def register() -> bool:
    """Register the per-loop factory for `agentengine://` when the flag is on.

    Returns whether it registered. Off registers nothing: ADK's built-in
    factory stays in place and builds the stock service.
    """
    if not reuse_enabled():
        logger.info("session_client_reuse: off (%s=0), built-in %s service", FLAG, SCHEME)
        return False
    service_registry.get_service_registry().register_session_service(SCHEME, _session_factory)
    logger.info("session_client_reuse: on (%s=1), %s:// → one client per loop", FLAG, SCHEME)
    return True


register()
