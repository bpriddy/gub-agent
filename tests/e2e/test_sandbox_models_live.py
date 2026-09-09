"""
Every model the deployed sandbox engine allows, exercised end to end: a real
session with the shared sandbox subject's GUB JWT and `state.sandbox`, a real
question, and the two things a QA run must be able to trust —

  1. provenance: `sandbox_resolved.model` is the model that was ASKED for
     (invariant 4 of the epic: a run without provenance does not exist);
  2. an answer: the executor produced visible text (the engine turns an
     exception inside the run into HTTP 200 + an empty stream, so "no text"
     is exactly how a broken model id looks from outside).

Plus the two guard rails: a baseline run has NO provenance event, and a named
thinking level on a DYNAMIC-only model is refused before the first model call.

The model list comes from the engine's own env, so the parametrisation follows
the deploy — widen the allowlist, redeploy, rerun. Claude cases skip while
Anthropic is not enabled for the project in Model Garden.

    SANDBOX_E2E=1 .venv/bin/pytest tests/e2e -q -rs
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

from .conftest import (
    PROJECT,
    Engine,
    GubSession,
    _adc_token,
    claude_reachable,
    executor_text,
    sandbox_resolved,
)

QUESTION = "how many client accounts do we have, and which three have the most campaigns?"
USER_ID = "e2e:sandbox-models"

# Probed once per session; Claude arms are skipped, not failed, until enabled.
_CLAUDE_OK: dict[str, bool] = {}

# Models that are servable but known not to finish the reference question:
# gemini-2.5-flash-lite (2026-09-09) makes the tool calls, then the critic
# loop ends with "executor did not follow the instruction" and no final text.
# That is a sandbox FINDING, not an infrastructure fault, so the case is a
# non-strict xfail — it turns green the day the model (or the prompt) copes.
# Override per run: SANDBOX_E2E_XFAIL_MODELS=a,b
KNOWN_WEAK = {
    m
    for m in os.environ.get("SANDBOX_E2E_XFAIL_MODELS", "gemini-2.5-flash-lite").split(",")
    if m.strip()
}


def _claude_ok() -> bool:
    if "v" not in _CLAUDE_OK:
        _CLAUDE_OK["v"] = claude_reachable(PROJECT)
    return _CLAUDE_OK["v"]


def _config_for(engine: Engine, model: str) -> dict:
    """The smallest valid sandbox config that selects `model`: a named level on
    the 3-series ids, DYNAMIC on everything else (2.5 and Claude)."""
    level = "LOW" if model in engine.thinking_level_models else "DYNAMIC"
    return {
        "model": model,
        "thinking_level": level,
        "critic_thinking_level": level,
        "label": "e2e",
    }


def _run(engine: Engine, gub: GubSession, config: dict | None) -> tuple[list[dict], float]:
    state: dict = {"gub_jwt": gub.jwt}
    if config:
        state["sandbox"] = config
    session_id = engine.create_session(USER_ID, state)
    t0 = time.monotonic()
    events = engine.stream_query(USER_ID, session_id, QUESTION)
    return events, time.monotonic() - t0


def _allowlist() -> list[str]:
    """Parametrise from the live engine env at collection time; when the
    directory is skipped (no SANDBOX_E2E) this must not call anything."""
    if os.environ.get("SANDBOX_E2E") != "1":
        return ["(skipped)"]
    eng = Engine(
        project=PROJECT,
        region=os.environ.get("GCP_REGION", "us-central1"),
        engine_id=os.environ.get("SANDBOX_AGENT_ENGINE_ID", "9148206673200939008"),
        token=_adc_token(),
    )
    r = httpx.get(eng.base, headers=eng.headers, timeout=30)
    r.raise_for_status()
    spec = r.json().get("spec", {}).get("deploymentSpec", {})
    env = {e["name"]: e.get("value", "") for e in spec.get("env", [])}
    models = [m for m in env.get("SANDBOX_MODEL_ALLOWLIST", "").split(",") if m.strip()]
    return models or ["(empty allowlist)"]


@pytest.mark.parametrize("model", _allowlist())
def test_every_allowed_model_answers_with_its_own_provenance(
    request: pytest.FixtureRequest, engine: Engine, gub: GubSession, model: str
):
    if model.startswith("claude-") and not _claude_ok():
        pytest.skip("Anthropic models are not enabled for this project in Model Garden")
    if model in KNOWN_WEAK:
        # Still runs and reports; a pass is recorded as XPASS, a fail as XFAIL.
        request.applymarker(
            pytest.mark.xfail(
                reason=f"{model}: known not to finish the reference question "
                "(critic loop exhausted)",
                strict=False,
            )
        )
    assert model in engine.allowlist, f"{model} left the engine's allowlist — rerun collection"

    events, seconds = _run(engine, gub, _config_for(engine, model))
    resolved = sandbox_resolved(events)
    text = executor_text(events)

    assert events, (
        f"{model}: empty stream — the run died inside the engine "
        "(validator refusal or a model the project cannot serve); engine logs have the reason"
    )
    assert resolved is not None, f"{model}: no sandbox_resolved although a config was sent"
    assert resolved.get("model") == model, f"{model}: provenance says {resolved.get('model')!r}"
    assert resolved.get("label") == "e2e"
    assert text.strip(), (
        f"{model}: provenance echoed but the executor produced no text "
        f"({len(events)} events, {seconds:.0f}s)"
    )


def test_baseline_run_has_no_provenance_and_answers(engine: Engine, gub: GubSession):
    events, _ = _run(engine, gub, None)
    assert events, "baseline: empty stream"
    assert sandbox_resolved(events) is None, (
        "a run with no overrides must not emit sandbox_resolved"
    )
    assert executor_text(events).strip()


def test_named_thinking_level_on_a_dynamic_only_model_is_refused(engine: Engine, gub: GubSession):
    dynamic_only = [
        m
        for m in engine.allowlist
        if m not in engine.thinking_level_models and not m.startswith("claude-")
    ]
    if not dynamic_only:
        pytest.skip("no DYNAMIC-only Gemini model on the allowlist")
    model = dynamic_only[0]
    config = {
        "model": model,
        "thinking_level": "LOW",
        "critic_thinking_level": "LOW",
        "label": "e2e-neg",
    }
    events, seconds = _run(engine, gub, config)
    # read_overrides raises before the first model call → nothing reaches the stream.
    assert sandbox_resolved(events) is None
    assert not executor_text(events).strip(), f"{model} answered despite an invalid thinking level"
    assert seconds < 30, f"refusal took {seconds:.0f}s — that was a model call, not a validator"
