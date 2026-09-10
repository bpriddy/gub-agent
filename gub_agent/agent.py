"""
agent.py — GUB AI Agent pipeline.

The agent is a small multi-agent pipeline (Phase C):

  LoopAgent(max_iterations=2)
    ├─ sandbox_echo — records the resolved sandbox config on experiment runs
    │                  (silent on ordinary ones); see sandbox.py
    ├─ executor_agent — runs the existing tool-using LLM
    ├─ format_gate — renders the executor's answer into the typed
    │                  AnswerPayload (agents/formatter.py) and enforces the
    │                  contract in code (agents/format_gate.py, blend 03)
    ├─ critic_gate — evaluates information sufficiency; emits structured
    │                  verdict {sufficient, reason, feedback} into state
    └─ loop_escalator — exits the loop early when critic verdict is sufficient

This is the load-bearing critic-before-commit pattern from the Agentic RAG
architecture. On a clean answer the loop exits after one iteration; on a
flagged failure the executor runs again, sees the critic's feedback in
state, and addresses it.

root_agent is what ADK looks for at deploy time and what callers
(gub-gchat-bot, Agentspace) invoke via stream_query. Same engine ID, same
external interface — multi-agent pipeline is invisible at the boundary.

The executor's instruction text lives in `prompts/executor.py` (edit it
there); the critic's lives in `prompts/critic.py`.

Prompt, model, thinking level and temperature are overridable per call from
`state["sandbox"]` (`sandbox.py`) so experiments need no redeploy. With no
such key every override point is a no-op.
"""

from google.adk.agents import Agent, LoopAgent

from .agents.circuit_breaker import circuit_breaker, reset_tool_budget
from .agents.context_pruning import strip_prior_turn_tool_parts, strip_source_metadata
from .agents.critic import critic_gate, escalator_agent
from .agents.evidence_index import record_evidence, reset_evidence_index
from .agents.format_gate import format_gate
from .agents.round_limiter import reset_rounds, round_limit
from .config import AGENT_NAME, build_model, build_thinking_planner
from .instruction_utils import with_current_date
from .prompts import EXECUTOR_INSTRUCTION
from .sandbox import (
    EXECUTOR_THINKING_LEVEL,
    sandbox_before_model,
    sandbox_echo,
    sandbox_instruction,
)
from .tools import ALL_TOOLS


def _before_agent(callback_context):
    """before_agent_callback — fires once per executor pass (LoopAgent runs the
    executor's run_async each iteration). Reset the per-pass round + tool budgets
    so a critic-requested retry starts fresh instead of inheriting the previous
    pass's counts (which would open the retry already at the cap, tools stripped,
    unable to make the call the critic asked for). The evidence index resets on
    the same cadence: the formatter grounds against what THIS pass retrieved
    (the executor's own doctrine is to re-query, never answer from prior data)."""
    reset_rounds(callback_context)
    reset_tool_budget(callback_context)
    reset_evidence_index(callback_context)
    return None


def _before_model(callback_context, llm_request):
    """Chain the model-level guards: apply any sandbox overrides (model,
    thinking, temperature), prune prior-turn tool payloads, mask `_sources`
    citation plumbing from current-turn results, then cap ReAct rounds (strip
    tools past the budget).

    Sandbox first, deliberately: it only writes `llm_request.model` and fields
    of `config`, while the round limiter may clear `config.tools` — neither can
    undo the other. With no `sandbox` key in state it is a no-op.
    """
    sandbox_before_model(callback_context, llm_request, role="executor")
    strip_prior_turn_tool_parts(callback_context, llm_request)
    strip_source_metadata(callback_context, llm_request)
    return round_limit(callback_context, llm_request)


executor_agent = Agent(
    # Model carries retry-with-backoff on 429/5xx (Dynamic Shared Quota); see
    # config.build_model. A string here would disable retries (ADK default).
    model=build_model(),
    name=AGENT_NAME,
    # InstructionProvider — appends today's date deterministically per request.
    # Wrapped for the sandbox: a run carrying state["sandbox"].executor_instruction
    # (or .executor_variant) swaps the prompt for that call only, with no redeploy;
    # without one, the base provider's text is returned unchanged.
    instruction=sandbox_instruction(with_current_date(EXECUTOR_INSTRUCTION), role="executor"),
    # Native thinking capped at MEDIUM. Unbounded (dynamic) thinking was the top
    # latency driver (thinking_tokens ↔ elapsed r=0.86): hard questions ran away
    # to 9-13k thought tokens / one 40s pause per step (pitch 70s @ 2 calls,
    # chevy 77s). MEDIUM keeps room to reason while cutting the runaway tail.
    # (EXECUTOR_THINKING_LEVEL, so the sandbox provenance can't drift from what
    # actually runs; a sandbox run overrides it per call in _before_model.)
    planner=build_thinking_planner(EXECUTOR_THINKING_LEVEL),
    tools=ALL_TOOLS,
    # Reset the per-pass round + tool budgets at the start of each executor pass
    # (so critic-requested retries aren't born over budget). See _before_agent.
    before_agent_callback=_before_agent,
    # Prune prior-turn tool payloads + cap ReAct rounds (context_pruning.py,
    # round_limiter.py) — the round cap forces synthesis instead of endless fan-out.
    before_model_callback=_before_model,
    # Dedupe repeated calls + per-turn tool budget (circuit_breaker.py) —
    # reliability guard against true loops / runaway fan-out.
    before_tool_callback=circuit_breaker,
    # Build the turn's citable evidence index from every tool result
    # (evidence_index.py) — what the formatter may cite and the format gate
    # grounds numbers/entities against.
    after_tool_callback=record_evidence,
)

# Wrap [executor → format-gate → critic-gate → escalator] in a LoopAgent.
# The format gate (blend 03) turns the executor's prose into the typed
# AnswerPayload and enforces the contract in code — validation retries happen
# INSIDE the gate, never through this loop. The critic gate then judges
# information sufficiency only (its old shape axis moved into the format
# gate); it skips its LLM deterministically for abstentions — the bare
# NO_COMPANY_RECORDS marker or an abstain payload. On clean answers the
# critic emits sufficient=true, escalator triggers loop exit after one
# iteration. On flagged failures, the executor runs again seeing the critic's
# feedback in session state. Capped at 2 iterations.
#
# sandbox_echo leads the pipeline — the provenance event must precede any
# work: on an experiment run it records what the run resolved to (model,
# thinking, temperature, prompt hashes) as a state_delta the caller's event
# stream carries. On an ordinary run it emits nothing at all, so the trace
# shape is today's plus only the formatter stage.
root_agent = LoopAgent(
    name="gub_pipeline",
    sub_agents=[sandbox_echo, executor_agent, format_gate, critic_gate, escalator_agent],
    max_iterations=2,
)
