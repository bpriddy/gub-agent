"""
variants/ — alternative prompt texts, addressable by name for an A/B run.

A sandbox run (`sandbox.py`) picks a prompt either by passing the full text
inline in session state or by naming a variant registered here:

    create_session(state={"sandbox": {"executor_variant": "v2_concise"}})

Variants are Python modules for the same reason the baseline prompts are (see
`prompts/__init__.py:5-7`): `adk deploy agent_engine` packages Python modules
reliably, while loose data files can drop out of the bundle. A variant that
exists in git but not in the deployed image would be the worst possible failure
mode here — the run would look fine and compare the baseline against itself.

Three properties this registry keeps:

1. **`baseline` is the live object, not a copy.** `VARIANTS["executor/baseline"]`
   *is* `EXECUTOR_INSTRUCTION`, so the baseline arm of an A/B cannot drift from
   what the prod engine actually runs. Copying the text here would let the two
   diverge silently at the next prompt edit.
2. **No silent fallback.** An unknown name raises and lists what is available.
   An A/B that quietly ran the baseline in both arms is worse than one that
   failed to start — it reports a tie and gets believed.
3. **Names are role-scoped.** `VARIANTS` is flat (`dict[str, str]`, keyed
   "<role>/<name>") but resolution takes the role, so both roles can have a
   `baseline` and an executor name can never be served to the critic.

Editing rules: the brace and date rules from `prompts/__init__.py:9-23` apply
here too. The date rule is unconditional — `sandbox_instruction()` appends
`current_date_note()` to an override exactly as `with_current_date()` does for
the baseline, so a hardcoded date would be a contradiction, not just staleness.
The brace rule is currently slack for the same reason it is slack for the
baseline (both instructions are callables, which bypass ADK's state
substitution), but the test suite pins it anyway: variant text reaches an
`instruction=` through more paths than the baseline does, and the failure is a
KeyError visible only in the Reasoning Engine logs.

A variant that wins does NOT get promoted automatically. Moving it into
`executor.py` is a hand-made pull request, deliberately.
"""

from __future__ import annotations

from typing import Literal

from ..critic import CRITIC_INSTRUCTION
from ..executor import EXECUTOR_INSTRUCTION
from ..formatter import FORMATTER_INSTRUCTION
from ..router import ROUTER_INSTRUCTION
from .critic_v2_calibrated import CRITIC_V2_CALIBRATED
from .critic_v3_coverage import CRITIC_V3_COVERAGE
from .executor_v2_concise import EXECUTOR_V2_CONCISE
from .executor_v3_grounding import EXECUTOR_V3_GROUNDING
from .executor_v4_brief import EXECUTOR_V4_BRIEF
from .executor_v5_query_discipline import EXECUTOR_V5_QUERY_DISCIPLINE

# Mirrors `sandbox.Role`. Declared locally rather than imported so that
# `prompts/` stays a leaf package with no dependency on the agent wiring.
Role = Literal["executor", "critic", "formatter", "router"]

ROLES: tuple[str, ...] = ("executor", "critic", "formatter", "router")

# The name every role reserves for "what production runs right now".
BASELINE = "baseline"

# Keyed "<role>/<name>". Values for `baseline` are the imported objects
# themselves — see property 1 in the module docstring.
VARIANTS: dict[str, str] = {
    "executor/baseline": EXECUTOR_INSTRUCTION,
    "executor/v2_concise": EXECUTOR_V2_CONCISE,
    "executor/v3_grounding": EXECUTOR_V3_GROUNDING,
    "executor/v4_brief": EXECUTOR_V4_BRIEF,
    "executor/v5_query_discipline": EXECUTOR_V5_QUERY_DISCIPLINE,
    "critic/baseline": CRITIC_INSTRUCTION,
    "critic/v2_calibrated": CRITIC_V2_CALIBRATED,
    "critic/v3_coverage": CRITIC_V3_COVERAGE,
    "formatter/baseline": FORMATTER_INSTRUCTION,
    "router/baseline": ROUTER_INSTRUCTION,
}


def variant_names(role: str) -> list[str]:
    """Registered variant names for one role, sorted."""
    prefix = f"{role}/"
    return sorted(key.removeprefix(prefix) for key in VARIANTS if key.startswith(prefix))


def resolve_variant(name: str, role: Role) -> str:
    """Prompt text for `name` in `role`.

    Raises ValueError — never returns the baseline as a fallback — when the
    role or the name is unknown, and names what is available so the caller can
    fix the run rather than guess.
    """
    if role not in ROLES:
        raise ValueError(
            f"unknown prompt-variant role {role!r}: variants exist for {', '.join(ROLES)}."
        )
    text = VARIANTS.get(f"{role}/{name}")
    if text is None:
        raise ValueError(
            f"unknown {role} prompt variant {name!r}. Available {role} variants: "
            f"{', '.join(variant_names(role))}. Register it as a module in "
            f"gub_agent/prompts/variants/ or pass the text inline as "
            f"{role}_instruction — there is deliberately no fallback to the "
            "baseline prompt, because an A/B that silently compares the "
            "baseline against itself reports a tie."
        )
    return text


__all__ = [
    "BASELINE",
    "CRITIC_V2_CALIBRATED",
    "CRITIC_V3_COVERAGE",
    "EXECUTOR_V2_CONCISE",
    "EXECUTOR_V3_GROUNDING",
    "EXECUTOR_V4_BRIEF",
    "EXECUTOR_V5_QUERY_DISCIPLINE",
    "ROLES",
    "VARIANTS",
    "Role",
    "resolve_variant",
    "variant_names",
]
