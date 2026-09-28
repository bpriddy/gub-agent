"""
eager_imports.py — import at package load what ADK would import on the first
model request (EAGER_ADK_IMPORTS).

A fresh engine process pays a one-off cost on its FIRST turn, on top of the
package import: in production the first request of each engine process waited
6.2-16.4 s between arriving and its first model send (13 processes,
2026-09-17..25), against p50 0.38 s on later turns. Deploys hide it behind the
deploy workflow's warm-up; a process the platform restarts between deploys
hands it to the next real user.

What that cost is, measured in a fresh local process (ADK 2.9.2, a stub model
that never touches the network, a smalltalk turn with the speculative deep run
beside it): importing `gub_agent.agent` loads 732 modules in 1.8 s, and the
first turn then imports 2,115 MORE, of which the second turn imports none.
They are imports, not work:

- ADK's content builder asks, on every request, which installed model types
  pair tool calls with results by id (flows/llm_flows/contents.py
  `_id_pairing_model_types`, memoized). Our models are VendorRouter, not
  Gemini, so the question is asked, and answering it imports
  `google.adk.models.anthropic_llm` — the `anthropic` SDK, 1,376 modules (it is
  installed for the sandbox's Claude arms).
- The genai client ADK builds on the first call: `google.genai`'s generated
  async stack (311 modules), `google.auth.aio`, and the HTTP/2 transport.
- ADK's per-request processors and flows that import on first use: auth
  (authlib, joserfc, cryptography), compaction, the workflow runtime, tool
  declarations, `requests`/`urllib3` for the credential transport.

So this module imports that list at package load. The first request of any
kind loads the package — on Agent Engine the "No .env file found for
gub_agent" line, which is the package loading, is logged during the first
`:query` of every process, before its first `:streamQuery` — so the cost moves
off the first user turn onto whichever request comes first after a restart,
which the warm-up ping (gcp-universal-backend terraform/engine_warmup.tf) makes
sure is not a user's.

Nothing here runs: no client is constructed, no credential is read, nothing
touches the network — only `import`. A module that is missing or renamed in
another ADK/genai version is skipped (the lazy import then happens where it
always did), and the line below says which.

    eager_imports: imported=<n> skipped=<n> new_modules=<n> ms=<n> skipped_names=<a,b|->

Measure the list again after an ADK or genai upgrade:
`tests/unit/test_eager_imports.py` fails when a stub first turn imports more
than a handful of new modules with the flag on.
"""

from __future__ import annotations

import importlib
import logging
import sys
import time

logger = logging.getLogger(__name__)

#: The roots of what a fresh process imports on its first turn (measured with
#: a sys.modules diff over a stub first turn, ADK 2.9.2); importing each pulls
#: in its own dependencies. Order does not matter.
MODULES: tuple[str, ...] = (
    # contents._id_pairing_model_types() -> the anthropic SDK
    "google.adk.models.anthropic_llm",
    # the genai client ADK builds on the first call
    "google.adk.models.interactions_utils",
    "google.adk.models._prompt_cache",
    "google.genai._gaos",
    "google.genai.interactions",
    "google.auth.aio.credentials",
    "google.auth.transport.requests",
    "httpcore",
    "h2.connection",
    "anyio._backends._asyncio",
    # request processors and flows imported on first use
    "google.adk.auth.auth_preprocessor",
    "google.adk.auth.auth_handler",
    "google.adk.auth.exchanger.oauth2_credential_exchanger",
    "google.adk.auth.oauth2_credential_util",
    "google.adk.flows.llm_flows.compaction",
    "google.adk.apps.compaction",
    "google.adk.apps.llm_event_summarizer",
    "google.adk.telemetry.node_tracing",
    "google.adk.tools.google_search_tool",
    "google.adk.tools.vertex_ai_search_tool",
    "google.adk.tools.load_artifacts_tool",
    "google.adk.utils._mtls_utils",
    "google.adk.workflow._workflow",
    "google.adk.workflow._dynamic_node_scheduler",
    "google.adk.workflow._schedule_dynamic_node",
    "google.adk.workflow.utils._replay_manager",
    "google.adk.workflow.utils._workflow_hitl_utils",
    "google.adk.labs",
    "pydantic._internal._serializers",
    "importlib.resources.readers",
    "encodings.idna",
)


def import_now(modules: tuple[str, ...] = MODULES) -> dict:
    """Import `modules`; skip (and report) any that fail. Returns the counts
    it logs, for tests."""
    started = time.perf_counter()
    before = len(sys.modules)
    skipped: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 — an optional vendor or a renamed module: stay lazy
            skipped.append(name)
    result = {
        "imported": len(modules) - len(skipped),
        "skipped": skipped,
        "new_modules": len(sys.modules) - before,
        "ms": int((time.perf_counter() - started) * 1000),
    }
    logger.info(
        "eager_imports: imported=%d skipped=%d new_modules=%d ms=%d skipped_names=%s",
        result["imported"],
        len(skipped),
        result["new_modules"],
        result["ms"],
        ",".join(skipped) or "-",
    )
    return result
