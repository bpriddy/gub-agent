"""
tool_gate.py — before_model_callback that offers `find_files` on file turns only,
and only while file search is switched on.

Per-intent tool gating did not exist in this repo before search-01: every
executor round was offered the whole of `ALL_TOOLS`, and the only runtime
mutation of that set was the round limiter's all-or-nothing strip. This adds
the narrow version — ONE tool, withheld unless the router classified the turn
as `file_lookup`.

Why gate it at all, rather than trust the model to pick:

* Drive file-name search answers a question the other eleven tools cannot, and
  it answers NOTHING ELSE — it matches names, never content. Offered on every
  turn it is a standing invitation to answer "what does the brief say" with a
  filename that looks about right, which is exactly the confident-wrong-file
  failure search-01 exists to remove.
* The declaration is not free. Fixed preambles — instructions plus every tool
  declaration — re-transmit on EVERY ReAct round, up to four per turn, and
  input volume is the primary cost driver here.

The default is DENY. A missing, unparseable or schema-invalid router decision
hides the tool, so a broken router costs today's behaviour rather than a new
one — the same principle as `decision_from`'s fallback, which costs the deep
path and never the turn.

`config.FILE_SEARCH_ENABLED` is the OUTER condition, checked before the intent
and defaulting to off. It is what makes the three services of search-01
deployable in any order: GUB's own `FILE_SEARCH` flag defaults to off too, and
in that state `/org/files/search` answers `200 []` — a shape this side cannot
tell apart from a genuine miss, so an engine that offered the tool ahead of its
backend would report files GUB DOES hold as "not found by name". With the flag
off the tool is never declared, so it is never called, so that `200 []` can
never be read at all.

The executor prompt documents `find_files` UNCONDITIONALLY, which is safe but
worth knowing why. A model that calls a tool it was not declared does not fail
the turn: ADK 2.9.2 defers the lookup error past the before-tool callbacks and
answers the call with a "tool not found" function_response
(`flows/llm_flows/_tool_caller.py`), so the cost is one wasted round. The
prompt tells the model to work with the tools it can see precisely so that
round is not spent.

Defensive to the same degree as `round_limiter.py`, and for the same reason:
ADK reads and rewrites `llm_request` internals that have changed shape under
this repo before (`stream_query` flipped to camelCase on a rebuild; 2.9.2 moved
and reworded the foreign-context preamble). A gate that raised would fail the
turn; a gate that quietly does nothing offers one extra tool, which is where
this feature started.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import config
from .router import decision_from

logger = logging.getLogger(__name__)

#: The tool this gate withholds…
GATED_TOOL = "find_files"

#: …and the one intent that earns it (`schemas/router.py`).
GATED_INTENT = "file_lookup"


def _strip_declaration(cfg: Any) -> int:
    """Drop `GATED_TOOL` from `config.tools`, the genai `FunctionDeclaration`
    side of the request. Returns how many declarations went.

    A `types.Tool` left with NO declarations is dropped rather than kept empty:
    an empty `function_declarations` is not a valid Tool for the API, and every
    tool this agent declares is a plain Python function, so a Tool emptied here
    held nothing but ours. Entries that carry no function declarations at all
    (a built-in, were one ever added) pass through untouched.
    """
    tools = getattr(cfg, "tools", None)
    if not tools:
        return 0

    removed = 0
    kept = []
    for tool in tools:
        declarations = getattr(tool, "function_declarations", None)
        if not declarations:
            kept.append(tool)
            continue
        surviving = [d for d in declarations if getattr(d, "name", None) != GATED_TOOL]
        if len(surviving) == len(declarations):
            kept.append(tool)
            continue
        removed += len(declarations) - len(surviving)
        if not surviving:
            continue
        tool.function_declarations = surviving
        kept.append(tool)

    if removed:
        cfg.tools = kept
    return removed


def _hide(llm_request: Any) -> None:
    """Remove the tool from BOTH places a request carries it.

    Both, because either one alone can still surface it: `config.tools` is what
    the model is shown, `tools_dict` is what ADK resolves a returned
    function_call against. Leaving the dict entry behind would keep an
    undeclared tool executable; leaving the declaration behind would keep
    offering it.
    """
    cfg = getattr(llm_request, "config", None)
    if cfg is not None:
        try:
            _strip_declaration(cfg)
        except Exception:  # noqa: BLE001 — best-effort across ADK versions
            logger.warning("tool_gate: could not strip the %s declaration", GATED_TOOL)
    try:
        tools_dict = getattr(llm_request, "tools_dict", None)
        if tools_dict:
            tools_dict.pop(GATED_TOOL, None)
    except Exception:  # noqa: BLE001
        logger.warning("tool_gate: could not remove %s from tools_dict", GATED_TOOL)


def tool_gate(callback_context: Any, llm_request: Any) -> None:
    """ADK before_model_callback: offer `find_files` only on a `file_lookup`
    turn, and only while `FILE_SEARCH_ENABLED` is on.

    The flag is read off the config MODULE rather than imported by name, the
    way `sandbox.py` reads its own switch: a from-import would freeze the value
    at import time, and this one has to be flippable per test.

    The intent comes from the SAME reader the dispatcher branches on
    (`agents/router.py:decision_from` — state first, the router's own event
    text as the fallback), so the gate and the branch can never disagree about
    what this turn was classified as. It is duck-typed over `.session` and
    `.invocation_id`, both of which a CallbackContext exposes.
    """
    if not config.FILE_SEARCH_ENABLED:
        # Feature off: hide it on every intent, and do not even read the
        # router decision — there is no branch here it could change.
        _hide(llm_request)
        return None

    try:
        intent = decision_from(callback_context).intent
    except Exception:  # noqa: BLE001 — an unreadable decision is a DENY, not a crash
        logger.warning("tool_gate: no readable router decision — hiding %s", GATED_TOOL)
        intent = None

    if intent == GATED_INTENT:
        return None
    _hide(llm_request)
    return None
