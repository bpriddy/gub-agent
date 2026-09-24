"""
agents.tool_gate — `find_files` is offered on one intent, behind one flag.

The gate is the whole of search-01's blast radius inside this engine: the tool
exists in `ALL_TOOLS` unconditionally, so what keeps it off the other ten kinds
of turn — and off EVERY turn until the feature is switched on — is this
callback and nothing else. Four properties are pinned here.

1. The sweep runs over `get_args(Intent)`, not a hand-written list — a new
   intent added to the schema is automatically covered, and has to be
   classified as gated or not rather than quietly inheriting one.
2. The FLAG IS THE OUTER GATE, and it ships off. With `FILE_SEARCH_ENABLED`
   off the tool is hidden on every intent, `file_lookup` included — that is
   what lets this engine deploy before the GUB that serves the search, whose
   own flag defaults to off and whose disabled endpoint answers `200 []`, a
   reply the tool would read as "no such file".
3. The DEFAULT IS DENY. A missing, unparseable or schema-invalid router
   decision — and a context the reader cannot read at all — hides the tool, so
   a broken router costs today's behaviour instead of a new one.
4. Both carriers are cleared. `config.tools` is what the model is shown and
   `tools_dict` is what ADK resolves a returned function_call against; leaving
   either one behind still surfaces the tool, one way or the other.

Driven with the `SimpleNamespace` request shape the round-limiter tests use,
plus one pass over REAL `google.genai` objects — the request internals this
touches have changed shape under this repo before, and a fake cannot notice
that. The gate is synchronous, so the tests are.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import get_args

import pytest
from google.genai import types as genai_types

from gub_agent import config
from gub_agent.agents.tool_gate import GATED_INTENT, GATED_TOOL, tool_gate
from gub_agent.schemas.router import Intent

INV = "inv-gate"


def _ctx(state: dict | None = None) -> SimpleNamespace:
    """A CallbackContext stand-in: `decision_from` reads `.session.state`,
    falls back to `.session.events`, and logs `.invocation_id`."""
    return SimpleNamespace(
        invocation_id=INV,
        session=SimpleNamespace(state=state if state is not None else {}, events=[]),
    )


def _decided(intent: str, confidence: float = 0.9) -> SimpleNamespace:
    return _ctx({"router_decision": {"intent": intent, "confidence": confidence}})


def _req() -> SimpleNamespace:
    """A fresh llm_request stand-in carrying both declarations."""
    return SimpleNamespace(
        config=SimpleNamespace(
            tools=[
                SimpleNamespace(
                    function_declarations=[
                        SimpleNamespace(name="find"),
                        SimpleNamespace(name=GATED_TOOL),
                    ]
                )
            ]
        ),
        tools_dict={"find": 1, GATED_TOOL: 2},
    )


def _declared(req) -> list[str]:
    return [d.name for tool in (req.config.tools or []) for d in (tool.function_declarations or [])]


@pytest.fixture(autouse=True)
def file_search_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """The feature flag, forced ON for every test in this file.

    Autouse and unconditional on purpose: the flag ships OFF, so without it
    every assertion below would hold for the wrong reason — the tool hidden by
    the feature switch rather than by the intent gate the test is about. The
    one test that wants it off turns it off itself.
    """
    monkeypatch.setattr(config, "FILE_SEARCH_ENABLED", True)


# ── the flag, over every intent in the schema ────────────────────────────────


@pytest.mark.parametrize("intent", get_args(Intent))
def test_the_flag_off_hides_the_tool_on_every_intent(intent: str, monkeypatch):
    """Off is today's behaviour EXACTLY: never declared, therefore never
    called, therefore a GUB with file search still disabled can never answer
    this engine `200 []` and have it read as a genuine miss. `file_lookup` is
    in the sweep because it is the whole point — the intent that earns the
    tool must not earn it while the feature is off."""
    monkeypatch.setattr(config, "FILE_SEARCH_ENABLED", False)
    req = _req()

    tool_gate(_decided(intent), req)

    assert GATED_TOOL not in _declared(req), intent
    assert GATED_TOOL not in req.tools_dict, intent
    # The flag withholds ONE tool; it does not disarm the executor.
    assert "find" in _declared(req)
    assert "find" in req.tools_dict


def test_the_flag_off_needs_no_router_decision_at_all(monkeypatch):
    """A context `decision_from` cannot read at all is still denied with the
    flag off: the flag is checked FIRST, so the feature can never be switched
    back on by whatever the router did or failed to do."""
    monkeypatch.setattr(config, "FILE_SEARCH_ENABLED", False)
    req = _req()

    tool_gate(SimpleNamespace(invocation_id=INV), req)

    assert GATED_TOOL not in _declared(req)
    assert GATED_TOOL not in req.tools_dict


# ── the gate, over every intent in the schema ────────────────────────────────


@pytest.mark.parametrize("intent", get_args(Intent))
def test_the_tool_is_offered_on_file_lookup_and_on_nothing_else(intent: str):
    req = _req()

    tool_gate(_decided(intent), req)

    offered = intent == GATED_INTENT
    assert (GATED_TOOL in _declared(req)) is offered, intent
    assert (GATED_TOOL in req.tools_dict) is offered, intent
    # Every other tool is none of this gate's business, on either branch.
    assert "find" in _declared(req)
    assert "find" in req.tools_dict


@pytest.mark.parametrize(
    ("label", "state"),
    [
        ("no decision at all", {}),
        ("schema-invalid intent", {"router_decision": {"intent": "vibes", "confidence": 0.99}}),
        ("not JSON", {"router_decision": "file_lookup"}),
        ("not a decision", {"router_decision": 42}),
    ],
)
def test_an_unreadable_decision_hides_the_tool(label: str, state: dict):
    """Default deny: `decision_from` answers `exploratory` at confidence 0 for
    all of these, which is not the gated intent — so today's behaviour (no
    file search) is what a broken router costs."""
    req = _req()

    tool_gate(_ctx(state), req)

    assert GATED_TOOL not in _declared(req), label
    assert GATED_TOOL not in req.tools_dict, label


def test_a_context_the_reader_cannot_read_hides_the_tool():
    """No `.session` at all — `decision_from` raises rather than falling back.
    The gate swallows it and denies; it must never fail the turn."""
    req = _req()

    tool_gate(SimpleNamespace(invocation_id=INV), req)

    assert GATED_TOOL not in _declared(req)
    assert GATED_TOOL not in req.tools_dict


# ── the mutation, against the real request objects ───────────────────────────


def test_it_strips_the_declaration_from_a_real_genai_tool():
    """A fake cannot catch a pydantic model refusing the assignment, and ADK
    hands us `types.Tool` objects, not namespaces."""
    req = SimpleNamespace(
        config=genai_types.GenerateContentConfig(
            tools=[
                genai_types.Tool(
                    function_declarations=[
                        genai_types.FunctionDeclaration(name="find"),
                        genai_types.FunctionDeclaration(name=GATED_TOOL),
                    ]
                )
            ]
        ),
        tools_dict={"find": 1, GATED_TOOL: 2},
    )

    tool_gate(_decided("exploratory"), req)

    assert _declared(req) == ["find"]
    assert req.tools_dict == {"find": 1}


def test_a_tool_left_with_no_declarations_is_dropped_entirely():
    """An empty `function_declarations` is not a valid Tool for the API, and
    every tool this agent declares is a plain function — so a Tool emptied by
    the gate held nothing but ours."""
    req = SimpleNamespace(
        config=SimpleNamespace(
            tools=[SimpleNamespace(function_declarations=[SimpleNamespace(name=GATED_TOOL)])]
        ),
        tools_dict={GATED_TOOL: 1},
    )

    tool_gate(_decided("smalltalk"), req)

    assert req.config.tools == []


def test_a_request_with_no_tools_is_left_alone():
    """The round limiter clears `config.tools` past the budget. The gate runs
    BEFORE it (see `agent.py:_before_model`), but nothing about this callback
    may depend on that — an already-empty request is a no-op, not a crash."""
    req = SimpleNamespace(config=SimpleNamespace(tools=[]), tools_dict={})

    tool_gate(_decided("exploratory"), req)

    assert req.config.tools == []
    assert req.tools_dict == {}


def test_a_request_with_no_config_is_survivable():
    """ADK's request shape has moved under this repo before; a gate that
    raised here would fail the turn over one withheld tool."""
    req = SimpleNamespace(config=None, tools_dict={GATED_TOOL: 1})

    tool_gate(_decided("exploratory"), req)

    assert req.tools_dict == {}
