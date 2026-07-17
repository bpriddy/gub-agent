"""
agent.py — GUB AI Agent pipeline.

The agent is a small multi-agent pipeline (Phase C):

  LoopAgent(max_iterations=2)
    ├─ executor_agent — runs the existing tool-using LLM
    ├─ critic_agent — evaluates the executor's response; emits structured
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
"""

from google.adk.agents import Agent, LoopAgent

from .agents.critic import critic_agent, escalator_agent
from .config import AGENT_NAME, GEMINI_MODEL, build_thinking_planner
from .instruction_utils import with_current_date
from .prompts import EXECUTOR_INSTRUCTION
from .tools import ALL_TOOLS

executor_agent = Agent(
    model=GEMINI_MODEL,
    name=AGENT_NAME,
    # InstructionProvider — appends today's date deterministically per request.
    instruction=with_current_date(EXECUTOR_INSTRUCTION),
    # Native dynamic thinking; emits thought summaries when EMIT_THINKING is set.
    planner=build_thinking_planner(),
    tools=ALL_TOOLS,
)

# Wrap [executor → critic → escalator] in a LoopAgent. On clean answers
# the critic emits sufficient=true, escalator triggers loop exit after
# one iteration. On flagged failures, the executor runs again seeing the
# critic's feedback in session state. Capped at 2 iterations.
root_agent = LoopAgent(
    name="gub_pipeline",
    sub_agents=[executor_agent, critic_agent, escalator_agent],
    max_iterations=2,
)
