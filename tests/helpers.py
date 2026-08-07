"""
Test doubles shared across the suite.

MockGub is a real localhost HTTP/1.1 keep-alive server (not a transport mock),
so tests exercise the actual network path: connection pooling, status codes,
and request bodies. FakeToolContext stands in for the ADK ToolContext that
Gemini Enterprise injects at runtime.
"""

from __future__ import annotations

import asyncio
import json


class MockGub:
    """Keep-alive HTTP server; (method, path) -> (status, json payload).

    `connections` counts TCP connections (for pooling assertions);
    `requests` records every (method, path) hit (for call-count assertions).
    """

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], tuple[int, dict]] = {}
        self.connections = 0
        self.requests: list[tuple[str, str]] = []
        self._server: asyncio.Server | None = None
        self.port: int | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        try:
            while True:
                head = (await reader.readuntil(b"\r\n\r\n")).decode()
                request_line, *header_lines = head.split("\r\n")
                method, target, _ = request_line.split(" ")
                path = target.split("?")[0]
                content_length = 0
                for line in header_lines:
                    if line.lower().startswith("content-length:"):
                        content_length = int(line.split(":", 1)[1])
                if content_length:
                    await reader.readexactly(content_length)

                self.requests.append((method, path))
                status, payload = self.routes.get((method, path), (404, {"detail": "not found"}))
                body = json.dumps(payload).encode()
                writer.write(
                    (
                        f"HTTP/1.1 {status} X\r\n"
                        "Content-Type: application/json\r\n"
                        f"Content-Length: {len(body)}\r\n"
                        "Connection: keep-alive\r\n\r\n"
                    ).encode()
                    + body
                )
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()


class FakeState(dict):
    """ADK State stand-in: dict plus the to_dict() the auth code calls."""

    def to_dict(self) -> dict:
        return dict(self)


class FakeToolContext:
    """ToolContext stand-in carrying session state (e.g. an injected OAuth token)."""

    def __init__(self, **state) -> None:
        self.state = FakeState(state)
