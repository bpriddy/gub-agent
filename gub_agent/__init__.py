"""
gub_agent — GUB AI Agent for Vertex AI Agent Engine.

ADK looks for `root_agent` in the module specified at deploy time.
Expose it here so both `adk deploy agent_engine --agent_module=gub_agent`
and `from gub_agent import root_agent` work.
"""

from .agent import root_agent

__all__ = ["root_agent"]
