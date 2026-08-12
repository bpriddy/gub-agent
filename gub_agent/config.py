"""
config.py — Central configuration for the GUB ADK agent.

All values are read from environment variables (via .env for local dev).
"""

import os

from dotenv import load_dotenv
from google.adk.models import Gemini
from google.adk.planners import BuiltInPlanner
from google.genai import types as genai_types

load_dotenv()

# gemini-3.5-flash is served ONLY from the Vertex *global* endpoint, and the
# model location is a property of the model — not of the environment. Pin it in
# code so it survives Agent Engine's runtime default (which forces the engine's
# own region, us-central1, and 404s the model — a `.env` value does not stick).
# The engine RESOURCE stays regional (set by `adk deploy --region`); this only
# steers the genai client used for model inference.
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"

# ── GUB backend ───────────────────────────────────────────────────────────────
# URL of the GUB backend this agent calls for data.
GUB_BASE_URL: str = os.environ.get("GUB_BASE_URL", "https://gcp-universal-backend-dev-843516467880.us-central1.run.app")

# Service JWT for local dev and CI — bypasses the Gemini Enterprise OAuth flow.
# In production (Gemini Enterprise), the JWT is obtained dynamically via
# token exchange and injected per-session.
GUB_SERVICE_JWT: str = os.environ.get("GUB_SERVICE_JWT", "")

# ── Gemini Enterprise OAuth ───────────────────────────────────────────────────
# The Authorization ID registered in Gemini Enterprise under
# Agents → Authorization. Gemini Enterprise stores the user's OAuth token in
# ToolContext at: state["auth_tokens"][GUB_AUTHORIZATION_ID]["token"]["access_token"]
# This value must match EXACTLY — any mismatch causes silent token injection failure.
GUB_AUTHORIZATION_ID: str = os.environ.get("GUB_AUTHORIZATION_ID", "gub-oauth-3")

# ── Agent ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
AGENT_NAME: str = os.environ.get("AGENT_NAME", "gub_agent")

# ── Debugging ─────────────────────────────────────────────────────────────────
# When true, the executor + critic RETURN their thinking-token summaries in the
# event stream (consumed by the debug client). Thinking happens either way — this
# flag only controls whether the summaries are emitted. Keep OFF in production so
# internal reasoning never reaches end users.
EMIT_THINKING: bool = os.environ.get("EMIT_THINKING", "false").lower() in ("1", "true", "yes")


def build_thinking_planner(thinking_level: str | None = None) -> BuiltInPlanner:
    """Native thinking planner shared by the executor and critic.

    Default (thinking_level=None): dynamic budget — the model thinks as much
    as it wants. Pass a level ('MINIMAL'/'LOW'/'MEDIUM'/'HIGH', the 3-series
    knob) to cap it — the critic runs at LOW because it's a checklist judge
    whose unbounded thinking measured 13-16s/turn (~29% of a whole turn).
    Thought summaries are emitted only when EMIT_THINKING is set.
    """
    if thinking_level is not None:
        return BuiltInPlanner(
            thinking_config=genai_types.ThinkingConfig(
                thinking_level=thinking_level,
                include_thoughts=EMIT_THINKING,
            ),
        )
    return BuiltInPlanner(
        thinking_config=genai_types.ThinkingConfig(
            thinking_budget=-1,
            include_thoughts=EMIT_THINKING,
        ),
    )


def build_model() -> Gemini:
    """Gemini model wired with truncated exponential backoff + jitter on 429/5xx.

    gemini-3.5-flash on Vertex is served via Dynamic Shared Quota: a transient
    429 RESOURCE_EXHAUSTED reflects shared-pool congestion, NOT a project quota
    that can be raised. Google's required mitigation is client-side retry with
    backoff. ADK/genai default to NO retries (one attempt), so a single
    transient 429 kills the whole turn — the executor makes several model calls
    per question, so the odds of one hitting congestion are real. HttpRetryOptions
    adds tenacity-backed exponential backoff + jitter, retrying 429/408/5xx.

    Bounds are chat-latency-aware. Retry is PER CALL and a turn makes several
    model calls with NO shared retry budget, so under sustained congestion the
    per-call waits compound. attempts=3 (up to ~5s of backoff per call worst
    case — ~2s after the first failure, ~3s after the second) keeps that
    compounding well under the bot's 120s stream ceiling: a transient
    blip still clears (most do on the first retry), but a genuinely overloaded
    turn fails fast instead of dragging toward the timeout. Retriable codes are
    the genai defaults (408/429/5xx) — genuine client errors (400/403/404) are
    NOT retried. Retries are logged by genai at INFO (before_sleep); a dedicated
    retry counter is a worthwhile follow-up for prod visibility.
    """
    return Gemini(
        model=GEMINI_MODEL,
        retry_options=genai_types.HttpRetryOptions(
            attempts=3,
            initial_delay=1.0,
            max_delay=8.0,
            exp_base=2.0,
            jitter=1.0,
        ),
    )
