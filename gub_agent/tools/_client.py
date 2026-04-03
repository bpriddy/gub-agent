"""
_client.py — Authenticated HTTP client for the GUB backend.

Handles two auth modes:
  1. Gemini Enterprise runtime: Google OAuth access token injected into
     ToolContext by the platform → exchanged for a GUB JWT on first call,
     then cached for the session.
  2. Local dev / CI: GUB_SERVICE_JWT env var used directly (no exchange needed).

All GUB API calls go through gub_get() / gub_post() which pick the right mode
automatically. Tool functions never need to think about auth.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..config import GUB_AUTHORIZATION_ID, GUB_BASE_URL, GUB_SERVICE_JWT

logger = logging.getLogger(__name__)

# ── Token resolution ──────────────────────────────────────────────────────────

def _resolve_gub_jwt(tool_context: Any | None) -> str:
    """
    Resolve a valid GUB JWT for this request.

    Priority:
      1. Cached GUB JWT in ToolContext session state (avoids re-exchange per call)
      2. Google OAuth access token injected by Gemini Enterprise → exchange for GUB JWT
      3. GUB_SERVICE_JWT env var (local dev / CI fallback)
    """
    if tool_context is not None:
        # 1. Cached from a previous tool call in this session
        cached = tool_context.state.get("gub_jwt")
        if cached:
            return cached

        # 2. Google OAuth token injected by Gemini Enterprise
        auth_tokens = tool_context.state.get("auth_tokens") or {}
        gub_auth = auth_tokens.get(GUB_AUTHORIZATION_ID) or {}
        google_access_token = (gub_auth.get("token") or {}).get("access_token")

        if google_access_token:
            logger.debug("Exchanging Google access token for GUB JWT")
            gub_jwt = _exchange_google_token(google_access_token)
            # Cache for the duration of this session
            tool_context.state["gub_jwt"] = gub_jwt
            return gub_jwt

    # 3. Local dev / service account fallback
    if GUB_SERVICE_JWT:
        return GUB_SERVICE_JWT

    raise RuntimeError(
        "No GUB JWT available. Set GUB_SERVICE_JWT for local dev, or ensure "
        f"Gemini Enterprise is injecting an OAuth token under authorization ID '{GUB_AUTHORIZATION_ID}'."
    )


def _exchange_google_token(google_access_token: str) -> str:
    """
    Exchange a Google OAuth access token for a GUB JWT via the broker endpoint.
    GUB verifies the token with Google's userinfo endpoint, then issues its own JWT.
    """
    with httpx.Client(timeout=10) as client:
        resp = client.post(
            f"{GUB_BASE_URL}/auth/google/access-token-exchange",
            json={"accessToken": google_access_token},
        )
        if not resp.is_success:
            logger.error("GUB token exchange failed: %s %s", resp.status_code, resp.text)
            raise RuntimeError(f"GUB token exchange failed ({resp.status_code}): {resp.text}")

        data = resp.json()
        access_token = data.get("accessToken")
        if not access_token:
            raise RuntimeError("GUB token exchange response missing 'accessToken'")
        return access_token


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def gub_get(
    path: str,
    tool_context: Any | None = None,
    **params: Any,
) -> dict:
    """
    Make an authenticated GET request to the GUB backend.

    Non-None query params are forwarded as URL query parameters.
    Returns the parsed JSON response, or an error dict on HTTP failure.
    """
    jwt = _resolve_gub_jwt(tool_context)
    clean_params = {k: v for k, v in params.items() if v is not None}

    with httpx.Client(timeout=15) as client:
        try:
            resp = client.get(
                f"{GUB_BASE_URL}{path}",
                headers={"Authorization": f"Bearer {jwt}"},
                params=clean_params,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("GUB GET %s returned %s", path, exc.response.status_code)
            return {
                "error": True,
                "status": exc.response.status_code,
                "message": _friendly_error(exc.response.status_code, path),
            }
        except httpx.RequestError as exc:
            logger.error("GUB GET %s request error: %s", path, exc)
            return {"error": True, "status": 0, "message": f"Could not reach GUB backend: {exc}"}


def gub_post(
    path: str,
    body: dict,
    tool_context: Any | None = None,
) -> dict:
    """
    Make an authenticated POST request to the GUB backend.
    Returns the parsed JSON response, or an error dict on HTTP failure.
    """
    jwt = _resolve_gub_jwt(tool_context)

    with httpx.Client(timeout=15) as client:
        try:
            resp = client.post(
                f"{GUB_BASE_URL}{path}",
                headers={"Authorization": f"Bearer {jwt}"},
                json=body,
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("GUB POST %s returned %s", path, exc.response.status_code)
            return {
                "error": True,
                "status": exc.response.status_code,
                "message": _friendly_error(exc.response.status_code, path),
            }
        except httpx.RequestError as exc:
            logger.error("GUB POST %s request error: %s", path, exc)
            return {"error": True, "status": 0, "message": f"Could not reach GUB backend: {exc}"}


def _friendly_error(status: int, path: str) -> str:
    if status == 401:
        return "Authentication failed — the GUB session may have expired."
    if status == 403:
        return f"Access denied to {path}. The user may not have permission to view this data."
    if status == 404:
        return f"Resource not found at {path}."
    return f"GUB returned an unexpected error ({status}) for {path}."
