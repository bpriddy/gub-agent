"""
config.py — Central configuration for the GUB ADK agent.

All values are read from environment variables (via .env for local dev).
"""

import os

from dotenv import load_dotenv
from google.adk.models import Gemini
from google.adk.models.base_llm import BaseLlm
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
GUB_BASE_URL: str = os.environ.get(
    "GUB_BASE_URL", "https://gcp-universal-backend-dev-843516467880.us-central1.run.app"
)

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

# ── Sandbox (per-call overrides) ───────────────────────────────────────────────
# Master switch for the experiment sandbox (sandbox.py): when false, a `sandbox`
# key in session state is ignored completely — no warn, no error, nothing logged.
# Set it ONLY on the dev/sandbox engine; the prod engine must stay inert so an
# override aimed at a sandbox run can never reach the engine serving the Chat bot.
SANDBOX_ENABLED: bool = os.environ.get("SANDBOX_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)

# Models a sandbox run may select. An allowlist rather than a free-text field:
# `llm_request.model` is swapped inside the SAME genai client (Vertex, global
# endpoint — pinned above), so only models that client can serve are valid; a
# cross-provider model needs a different BaseLlm and is out of scope.
# The default holds only ids verified to answer from this project's global
# endpoint (2026-09-07): `gemini-3.5-pro` 404s there AND in us-central1, and a
# 404 inside the engine surfaces to the caller as an EMPTY 200 stream — so an
# unservable id in the allowlist is a silent failure, not a loud one.
SANDBOX_MODEL_ALLOWLIST: tuple[str, ...] = tuple(
    name.strip()
    for name in os.environ.get("SANDBOX_MODEL_ALLOWLIST", "gemini-3.5-flash,gemini-2.5-pro").split(
        ","
    )
    if name.strip()
)

# Models that accept a NAMED thinking level (`thinking_level=LOW|MEDIUM|HIGH`, the
# 3-series knob). Any other allowlisted model rejects it with 400 INVALID_ARGUMENT
# — verified live 2026-09-07 with gemini-2.5-pro — and inside the engine that 400
# reaches the caller as an empty 200 stream. The baseline planners pin MEDIUM
# (executor) and LOW (critic), so a sandbox run that swaps to a model outside this
# set must ALSO set both roles to DYNAMIC (`thinking_budget=-1`); sandbox.py
# refuses the run up front otherwise.
SANDBOX_THINKING_LEVEL_MODELS: tuple[str, ...] = tuple(
    name.strip()
    for name in os.environ.get("SANDBOX_THINKING_LEVEL_MODELS", "gemini-3.5-flash").split(",")
    if name.strip()
)


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


# Where Anthropic models are served for a sandbox run that selects a `claude-*`
# id (gub_agent/models.py). Claude on Vertex serves from the global endpoint;
# the genai client's own GOOGLE_CLOUD_LOCATION pin above is a separate thing.
CLAUDE_VERTEX_LOCATION: str = os.environ.get("CLAUDE_VERTEX_LOCATION", "global")


def build_model() -> BaseLlm:
    """The agents' model object: a VendorRouter (gub_agent/models.py) over a
    Gemini client, so a sandbox run that writes a `claude-*` id into
    `llm_request.model` reaches Anthropic on Vertex while every other request —
    and every prod request, where the sandbox never writes the field — goes to
    the same Gemini client as before.

    Gemini model wired with truncated exponential backoff + jitter on 429/5xx.

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
    from .models import VendorRouter  # noqa: PLC0415 — models.py imports config

    gemini = Gemini(
        model=GEMINI_MODEL,
        retry_options=genai_types.HttpRetryOptions(
            attempts=3,
            initial_delay=1.0,
            max_delay=8.0,
            exp_base=2.0,
            jitter=1.0,
        ),
    )
    return VendorRouter(model=GEMINI_MODEL, gemini=gemini, claude_location=CLAUDE_VERTEX_LOCATION)
