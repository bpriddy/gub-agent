"""
formatter.py — the rendering specialist (blend 03).

An LlmAgent with `output_schema=AnswerPayload` and NO tools: it turns the
executor's prose plus this turn's evidence index into the typed answer the bot
renders. It never retrieves and never adds facts.

Two wiring choices worth their comments:

- `include_contents="none"` + a before_model_callback that REPLACES the
  request contents with one user message (the "brief": executor answer,
  ALLOWED_EVIDENCE ids and values, any retry feedback — composed by the format
  gate, carried in `evidence_index.py`'s per-invocation store). The formatter
  therefore sees exactly its inputs, never the whole transcript — cheaper, and
  deterministic about what grounding means. Putting the data in the
  INSTRUCTION instead would break the sandbox: a `formatter_variant` override
  replaces the whole instruction (`sandbox_instruction`), and the data must
  survive that.
- `FORMATTER_THINKING_LEVEL` is imported from sandbox.py and used at this
  construction site, so the provenance `resolved_config()` reports can never
  drift from what actually runs (the `EXECUTOR_THINKING_LEVEL` pattern).

The retry loop lives OUTSIDE this agent, in `format_gate.py` — deliberately
not a nested LoopAgent: ADK's LoopAgent exits on `event.actions.escalate`
(`loop_agent.py:114-122`), so an inner loop's escalate would break the OUTER
pipeline loop and skip the critic.
"""

from __future__ import annotations

from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.genai import types as genai_types

from ..config import build_model, build_thinking_planner
from ..instruction_utils import current_date_note
from ..prompts import FORMATTER_INSTRUCTION
from ..sandbox import FORMATTER_THINKING_LEVEL, sandbox_before_model, sandbox_instruction
from ..schemas import AnswerPayload
from .evidence_index import formatter_brief

FORMATTER_NAME = "formatter"
# The bot writes the payload it finds under this state key… nowhere — but the
# CriticGate reads it (abstain recognition) and tests assert on it.
ANSWER_STATE_KEY = "answer_payload"


def _formatter_base_instruction(_ctx: ReadonlyContext) -> str:
    """InstructionProvider — static prompt + fresh date. The turn's data (the
    brief) travels as request content, not here; see the module docstring."""
    return f"{FORMATTER_INSTRUCTION}\n\n{current_date_note()}"


def _formatter_before_model(callback_context: Any, llm_request: Any) -> None:
    """Sandbox overrides first (model / formatter_thinking_level / temperature —
    a no-op without state["sandbox"]), then replace the request contents with
    the gate-composed brief: the formatter's input is exactly the executor's
    answer + ALLOWED_EVIDENCE, nothing else."""
    sandbox_before_model(callback_context, llm_request, role="formatter")
    invocation_id = getattr(callback_context, "invocation_id", "") or "?"
    brief = formatter_brief(invocation_id)
    if brief:
        llm_request.contents = [
            genai_types.Content(role="user", parts=[genai_types.Part(text=brief)])
        ]
    return None


formatter_agent = LlmAgent(
    # Retry-with-backoff on 429/5xx, same as the executor (config.build_model).
    model=build_model(),
    name=FORMATTER_NAME,
    # Wrapped for the sandbox: state["sandbox"].formatter_instruction (or
    # .formatter_variant) replaces the prompt for that run only.
    instruction=sandbox_instruction(_formatter_base_instruction, role="formatter"),
    # LOW: it renders given text into a given schema — no retrieval, no
    # analysis worth a deliberation budget. (FORMATTER_THINKING_LEVEL, so the
    # sandbox provenance can't drift from what actually runs.)
    planner=build_thinking_planner(thinking_level=FORMATTER_THINKING_LEVEL),
    # THE contract: every AnswerPayload validator (filler, budgets, table
    # shape, facts↔citations) runs on parse — a violation is a pydantic error
    # the format gate turns into retry feedback, not a judge's opinion.
    output_schema=AnswerPayload,
    output_key=ANSWER_STATE_KEY,
    # The brief IS the input; the transcript is noise (and tokens).
    include_contents="none",
    before_model_callback=_formatter_before_model,
)
