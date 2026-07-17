"""
prompts/ — the agent's instruction text, extracted from code for hand-editing.

Each prompt lives in its own module as a plain triple-quoted string so it can
be edited as prose without touching the wiring in `agent.py` / `agents/`. These
are `.py` (not `.md`) on purpose: `adk deploy agent_engine` packages Python
modules reliably, whereas loose data files can be dropped from the bundle.

Two things to keep in mind when editing any prompt here:

1. NO literal curly braces. ADK runs a state-substitution pass
   (`re.sub('{+[^{}]*}+', ...)`) over every instruction string. Any `{...}`
   that isn't a real session-state key crashes the agent at runtime with a
   KeyError — visible only in the Reasoning Engine logs, symptom is an empty
   reply. If you need a literal brace, reword to avoid it.
2. Don't hardcode a date. The current date is injected per-request by
   `with_current_date()` (see `instruction_utils.py`) and appended to whatever
   string you write here.
"""

from .critic import CRITIC_INSTRUCTION
from .executor import EXECUTOR_INSTRUCTION

__all__ = ["CRITIC_INSTRUCTION", "EXECUTOR_INSTRUCTION"]
