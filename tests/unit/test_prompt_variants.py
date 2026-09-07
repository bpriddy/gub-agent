"""
prompts.variants — the A/B prompt registry: identity, no fallback, brace safety.

Three things are load-bearing. (1) `baseline` must BE the live prompt object,
not a copy of its text: a copy would let the baseline arm of an A/B drift from
what the prod engine runs, and the drift would be invisible — both arms would
still produce answers. (2) An unknown name must raise, because a fallback to
the baseline turns a mis-typed A/B into a reported tie. (3) No variant may
carry a literal `{...}`: ADK's state-substitution pass raises KeyError on a
brace that is not a real state key, and it surfaces only as an empty reply in
the Reasoning Engine logs. Today both instructions are callables and so bypass
that pass — this test guards the variants written after someone unwraps one.

The registry is plain data plus a lookup; tests are async only to match the
suite's `asyncio_mode = "auto"`.
"""

from __future__ import annotations

import re

import pytest

from gub_agent.prompts import CRITIC_INSTRUCTION, EXECUTOR_INSTRUCTION
from gub_agent.prompts.variants import (
    BASELINE,
    ROLES,
    VARIANTS,
    resolve_variant,
    variant_names,
)

# What ADK's `re.sub('{+[^{}]*}+', ...)` would try to resolve as a state key:
# a brace followed by anything that isn't another brace (`{{` is the escape).
BRACE_HAZARD = re.compile(r"\{(?:[^{]|$)")


async def test_executor_baseline_is_the_live_prompt_object():
    """Identity, not equality — a copy could drift from production silently."""
    assert resolve_variant(BASELINE, "executor") is EXECUTOR_INSTRUCTION


async def test_critic_baseline_is_the_live_prompt_object():
    assert resolve_variant(BASELINE, "critic") is CRITIC_INSTRUCTION


async def test_every_role_has_a_baseline():
    """Every role needs a deliberate baseline arm addressable by name."""
    for role in ROLES:
        assert BASELINE in variant_names(role)


async def test_unknown_name_raises_and_lists_the_available_names():
    with pytest.raises(ValueError) as exc:
        resolve_variant("v9_does_not_exist", "executor")
    message = str(exc.value)
    assert "v9_does_not_exist" in message
    for name in variant_names("executor"):
        assert name in message


async def test_unknown_role_raises():
    with pytest.raises(ValueError, match="role"):
        resolve_variant(BASELINE, "executer")  # type: ignore[arg-type]


async def test_a_variant_is_not_reachable_from_the_other_role():
    """Role scoping is real: an executor name must not resolve for the critic."""
    with pytest.raises(ValueError, match="variant"):
        resolve_variant("v2_concise", "critic")


async def test_the_starter_variants_are_registered():
    """The two variants the sandbox epic ships with, reachable by name."""
    assert "v2_concise" in variant_names("executor")
    assert "v3_grounding" in variant_names("executor")


async def test_every_variant_is_non_empty_and_substantial():
    for key, text in VARIANTS.items():
        assert isinstance(text, str), key
        assert text.strip(), key
        # A truncated or placeholder prompt is worse than none: it would still
        # answer, just badly, and the A/B would read as a legitimate loss.
        assert len(text) > 500, key


async def test_no_variant_carries_a_brace_hazard():
    for key, text in VARIANTS.items():
        found = BRACE_HAZARD.search(text)
        assert found is None, (
            f"{key} contains a literal brace at offset {found.start()} "
            f"({text[found.start() : found.start() + 40]!r}) — ADK state "
            "substitution raises KeyError on it; see prompts/__init__.py"
        )


async def test_variant_keys_are_role_scoped():
    """VARIANTS is flat; the '<role>/<name>' key shape is what makes the role
    argument meaningful, so a key that skips it would be silently unreachable."""
    for key in VARIANTS:
        role, _, name = key.partition("/")
        assert role in ROLES, key
        assert name, key
        assert resolve_variant(name, role) is VARIANTS[key]  # type: ignore[arg-type]
