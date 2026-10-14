"""
google-adk pin consistency — requirements.txt must carry pyproject's exact pin.

Two files pin ADK for two different consumers:

  * pyproject.toml is what CI and the deploy workflow install, so it also fixes
    the version of the `adk deploy agent_engine` CLI.
  * gub_agent/requirements.txt is what the deployed engine's image installs.
    The CLI appends `google-adk[a2a]==<CLI version>` to it while staging the
    build, so a different (or ranged) pin there makes the build's requirements
    unsatisfiable, or lets the engine run an ADK nobody tested.

This agent reads ADK's internal wire and prompt shapes, so the version the
engine runs has to be the version the suite ran against. Bumping ADK is one
deliberate change to BOTH files; this test fails when only one moves.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"
REQUIREMENTS = REPO / "gub_agent" / "requirements.txt"

# `google-adk==2.9.2`, `google-adk[a2a] >= 1.0`, `google-adk==2.9.2 ; marker`.
# Group 1 = operator, group 2 = version. The name must end at the extras, a
# space or an operator, so `google-adk-extras==1` is not mistaken for ADK.
_ADK = re.compile(
    r"^google-adk(?:\[[^\]]*\])?\s*(===|==|~=|!=|>=|<=|>|<)\s*([^\s;,]+)",
    re.IGNORECASE,
)


def _adk_specs(lines: list[str]) -> list[tuple[str, str]]:
    """(operator, version) of every google-adk requirement among `lines`."""
    found: list[tuple[str, str]] = []
    for raw in lines:
        line = raw.split(" #", 1)[0].strip()  # drop a trailing comment
        if not line or line.startswith("#"):
            continue
        match = _ADK.match(line)
        if match:
            found.append((match.group(1), match.group(2)))
    return found


def _requirements_specs() -> list[tuple[str, str]]:
    return _adk_specs(REQUIREMENTS.read_text(encoding="utf-8").splitlines())


def _pyproject_specs() -> list[tuple[str, str]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return _adk_specs(data["project"]["dependencies"])


# ── the parser itself: a pin that parses to nothing would pass vacuously ─────


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["google-adk==2.9.2"], [("==", "2.9.2")]),
        (["google-adk[a2a]==2.9.2"], [("==", "2.9.2")]),
        (["google-adk >= 1.0.0"], [(">=", "1.0.0")]),
        (["google-adk==2.9.2  # tested"], [("==", "2.9.2")]),
        (["google-adk==2.9.2 ; python_version >= '3.11'"], [("==", "2.9.2")]),
        (["# google-adk==9.9.9", "google-adk==2.9.2"], [("==", "2.9.2")]),
        (["google-adk-extras==1.0"], []),
        (["google-adk"], []),  # unpinned: no operator, nothing to compare
        (["httpx[http2]>=0.27.0"], []),
    ],
)
def test_adk_spec_parser(lines: list[str], expected: list[tuple[str, str]]) -> None:
    assert _adk_specs(lines) == expected


# ── the files ────────────────────────────────────────────────────────────────


def test_pyproject_pins_adk_exactly() -> None:
    specs = _pyproject_specs()
    assert len(specs) == 1, f"expected one google-adk dependency, got {specs}"
    assert specs[0][0] == "==", f"pyproject must pin google-adk exactly, got {specs[0]}"


def test_requirements_pins_adk_exactly() -> None:
    specs = _requirements_specs()
    assert len(specs) == 1, f"expected one google-adk line in requirements.txt, got {specs}"
    assert specs[0][0] == "==", (
        f"requirements.txt must pin google-adk exactly, got {specs[0]}: the deploy CLI "
        "appends google-adk[a2a]==<its version> and a range here lets the engine drift"
    )


def test_requirements_adk_pin_equals_pyproject_pin() -> None:
    assert _requirements_specs() == _pyproject_specs()
