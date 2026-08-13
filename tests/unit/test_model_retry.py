"""
Model retry configuration — 429/Dynamic-Shared-Quota resilience, no backend.

gemini-3.5-flash on Vertex is served via Dynamic Shared Quota: a 429
RESOURCE_EXHAUSTED is transient shared-pool congestion, not a project quota that
can be raised. ADK/genai default to NO retries, so a single transient 429 kills
the turn. `config.build_model` wires HttpRetryOptions (exponential backoff +
jitter over 429/408/5xx). These tests pin that behaviour:

  * the config is present with chat-latency-aware bounds (fast, stable);
  * fed through genai's own retry builder, our options recover a transient 429,
    leave a non-retriable 4xx alone, and respect the attempt cap.

The behavioural tests override only the sleep so they stay instant — the real
delay values are asserted separately in the config test. Tests are async to
match the suite convention (asyncio_mode=auto + async autouse fixtures); the
retry logic itself is synchronous.
"""

from __future__ import annotations

import tenacity
from google.genai import _api_client as genai_api_client
from google.genai import errors as genai_errors

from gub_agent.config import build_model

# Retriable status codes genai applies when HttpRetryOptions.http_status_codes
# is left None (our case) — 429 must be among them for DSQ resilience.
_DEFAULT_RETRIABLE = {408, 429, 500, 502, 503, 504}


def _api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(
        code, {"error": {"code": code, "message": "Resource exhausted. Please try again later."}}
    )


def _fast_retry_args() -> dict:
    """genai's own retry config for our options, with the real backoff swapped
    for a near-zero sleep so the test is instant. Keeps the real stop (attempt
    cap) and retry predicate (which codes retry)."""
    args = dict(genai_api_client.retry_args(build_model().retry_options))
    args["wait"] = tenacity.wait_fixed(0)
    return args


async def test_build_model_enables_retry_with_429():
    opts = build_model().retry_options
    assert opts is not None, "retries disabled — a single transient 429 would kill the turn"
    # None http_status_codes → genai's default set, which includes 429.
    assert opts.http_status_codes is None or 429 in opts.http_status_codes
    assert 429 in _DEFAULT_RETRIABLE
    # More than one attempt, and bounds kept chat-latency-aware (< SDK's 60s max).
    assert opts.attempts is not None and opts.attempts >= 2
    assert opts.initial_delay and opts.initial_delay > 0
    assert opts.max_delay and opts.max_delay <= 16.0
    assert opts.exp_base and opts.exp_base > 1


async def test_transient_429_is_retried_and_recovers():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _api_error(429)
        return "SUCCESS"

    result = tenacity.Retrying(**_fast_retry_args())(flaky)
    assert result == "SUCCESS"
    assert calls["n"] == 3  # two 429s absorbed, third call succeeds


async def test_non_retriable_client_error_is_not_retried():
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise _api_error(400)

    try:
        tenacity.Retrying(**_fast_retry_args())(bad_request)
    except genai_errors.APIError:
        pass
    assert calls["n"] == 1  # a 400 is a caller bug — retrying only wastes quota


async def test_persistent_429_stops_at_attempt_cap():
    calls = {"n": 0}

    def always_429():
        calls["n"] += 1
        raise _api_error(429)

    reraised = False
    try:
        tenacity.Retrying(**_fast_retry_args())(always_429)
    except genai_errors.APIError:
        reraised = True

    assert reraised, "cap reached should reraise, not swallow"
    assert calls["n"] == build_model().retry_options.attempts
