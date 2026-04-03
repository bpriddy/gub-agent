"""
config.py — Central configuration for the GUB ADK agent.

All values are read from environment variables (via .env for local dev).
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ── GUB backend ───────────────────────────────────────────────────────────────
# URL of the GUB backend this agent calls for data.
GUB_BASE_URL: str = os.environ.get("GUB_BASE_URL", "http://localhost:3000")

# Service JWT for local dev and CI — bypasses the Gemini Enterprise OAuth flow.
# In production (Gemini Enterprise), the JWT is obtained dynamically via
# token exchange and injected per-session.
GUB_SERVICE_JWT: str = os.environ.get("GUB_SERVICE_JWT", "")

# ── Gemini Enterprise OAuth ───────────────────────────────────────────────────
# The Authorization ID registered in Gemini Enterprise under
# Agents → Authorization. Gemini Enterprise stores the user's OAuth token in
# ToolContext at: state["auth_tokens"][GUB_AUTHORIZATION_ID]["token"]["access_token"]
# This value must match EXACTLY — any mismatch causes silent token injection failure.
GUB_AUTHORIZATION_ID: str = os.environ.get("GUB_AUTHORIZATION_ID", "gub-oauth")

# ── Agent ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
AGENT_NAME: str = os.environ.get("AGENT_NAME", "gub-agent")
