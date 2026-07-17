"""
instruction_utils.py — deterministic instruction augmentation.

ADK's `instruction` accepts a callable (InstructionProvider) invoked fresh on
every request. We use that to inject the current date into the agent's
context deterministically — computed server-side, per call, with no
LLM-mediated tool. The model always has an accurate "now" for recency
reasoning, and it can never go stale (unlike a value baked into a session at
creation time).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from google.adk.agents.readonly_context import ReadonlyContext


def current_date_note() -> str:
    """The '## Current date' block, computed fresh (UTC) on each call. Shared by
    any InstructionProvider that needs an accurate, never-stale 'now'."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        "## Current date\n"
        f'Today is {today} (UTC). Use this as "now" for every recency, '
        '"recent", "this week/month", and time-window judgement. It is '
        "injected fresh on each request — always trust it over any date "
        "you might infer from the data or your own training."
    )


def with_current_date(base: str) -> Callable[[ReadonlyContext], str]:
    """Wrap a static instruction string in an InstructionProvider that appends
    today's UTC date on every invocation."""

    def provider(_ctx: ReadonlyContext) -> str:
        return f"{base}\n\n{current_date_note()}"

    return provider
