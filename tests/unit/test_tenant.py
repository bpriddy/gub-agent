"""
tenant — which branded surface a turn arrived through (gub-gchat-bot#83/#90).

One engine, more than one Chat app. The load-bearing case is the NEGATIVE one:
a turn with no `tenant` in state must get the base instruction back byte for
byte, because that is every production turn the original bot has ever sent and
will keep sending. Case 1 pins that against the real ADK instruction path, not
a fake.

The positive cases pin the two things the tenancy ruling actually bought: the
scope block reaching the prompt with the label interpolated (never hardcoded —
this engine is multi-company), and every per-turn log line carrying a `tenant=`
field so log-derived proportions stop mixing two bots' traffic.

The loud-failure case matters more than it looks: a bot that believes it is
scoped and is not is exactly the state nobody notices.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from gub_agent.tenant import (
    DEFAULT_LABEL,
    label_of,
    read_tenant,
    tenant_instruction,
    tenant_note,
)


def _ctx(state: dict | None) -> SimpleNamespace:
    """An instruction-provider-shaped context."""
    return SimpleNamespace(state=state)


def _inv_ctx(state: dict | None) -> SimpleNamespace:
    """An InvocationContext-shaped context (state hangs off `.session`)."""
    return SimpleNamespace(session=SimpleNamespace(state=state))


# ── 1. the negative case: no tenant key, nothing happens ────────────────────


def test_no_tenant_returns_the_base_text_unchanged() -> None:
    base = "BASE INSTRUCTION\n\n## Current date\nToday is 2026-09-17 (UTC)."
    provider = tenant_instruction(lambda _ctx: base)

    assert provider(_ctx({"gub_jwt": "jwt"})) == base
    assert provider(_ctx({})) == base
    assert provider(_ctx(None)) == base


def test_no_tenant_logs_the_default_label() -> None:
    assert label_of(_ctx({"gub_jwt": "jwt"})) == DEFAULT_LABEL
    assert label_of(_inv_ctx({})) == DEFAULT_LABEL
    assert label_of(None) == DEFAULT_LABEL


# ── 2. the positive case: the scope block, with the label interpolated ──────


def test_tenant_appends_the_scope_block() -> None:
    provider = tenant_instruction(lambda _ctx: "BASE")
    out = provider(_ctx({"gub_jwt": "jwt", "tenant": "chevy"}))

    assert out.startswith("BASE\n\n")
    assert "## Tenant surface" in out
    assert "chevy" in out


def test_the_block_names_no_client_and_claims_no_boundary() -> None:
    """The engine is multi-company: the label is interpolated, never baked in.
    And the block must not promise a restriction that does not exist — the
    ruling is 'branded for, not restricted to', recorded as an accepted risk."""
    note = tenant_note("someclient")

    assert note.count("someclient") >= 3
    assert "chevy" not in note.lower()
    assert "anomaly" not in note.lower()
    # It tells the model it is NOT restricted, rather than staying silent.
    assert "not restricted" in note.lower()


def test_state_read_works_through_both_context_shapes() -> None:
    assert label_of(_ctx({"tenant": "chevy"})) == "chevy"
    assert label_of(_inv_ctx({"tenant": "chevy"})) == "chevy"


# ── 3. a malformed label is loud, never a silent fall back to unscoped ──────


@pytest.mark.parametrize("bad", ["Chevy", "chevy bot", "", "-chevy", "x" * 33, 7, ["chevy"]])
def test_malformed_label_raises(bad: object) -> None:
    with pytest.raises(ValueError):
        read_tenant({"tenant": bad})


def test_log_labelling_never_raises_on_a_bad_label() -> None:
    """`label_of` is called from log statements. A bad label must not turn a
    log line into a crashed turn — the instruction path already raised."""
    assert label_of(_ctx({"tenant": "NOT A SLUG"})) == DEFAULT_LABEL


# ── 4. the log lines the CLAUDE.md queries actually match on ────────────────


def test_dispatcher_line_keeps_its_existing_fields_and_gains_tenant(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The `dispatcher: intent` line is the denominator of every per-turn
    proportion measured from the logs. `tenant=` is appended so the existing
    substring matches keep working."""
    from gub_agent.agents import dispatcher as dispatcher_module

    logger = logging.getLogger(dispatcher_module.__name__)
    with caplog.at_level(logging.INFO, logger=dispatcher_module.__name__):
        logger.info(
            "dispatcher: intent=%s confidence=%.2f branch=%s (inv=%s) tenant=%s",
            "count_or_rank",
            0.91,
            "fast",
            "inv-1",
            label_of(_inv_ctx({"tenant": "chevy"})),
        )

    line = caplog.messages[-1]
    assert line.startswith("dispatcher: intent=count_or_rank confidence=0.91 branch=fast")
    assert "(inv=inv-1)" in line
    assert line.endswith("tenant=chevy")
