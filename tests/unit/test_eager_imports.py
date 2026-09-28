"""
EAGER_ADK_IMPORTS — gub_agent/eager_imports.py.

A fresh process imports 2,115 modules on its first turn (measured with a stub
model, ADK 2.9.2) and none on its second; the flag imports them when the
package loads instead. Pinned here:

- flag on: after `import gub_agent.agent`, a stub first turn imports (almost)
  nothing new — the list still covers what ADK imports lazily;
- flag off: today's package, byte for byte in what it imports — the eager
  module is never loaded and ADK's lazily imported vendors stay unloaded;
- every listed module imports in this environment (a rename in an ADK or
  genai upgrade shows up here, not as a silently lazy import in production);
- a module that fails to import is skipped and reported, never raised;
- the flag is stated in both deploy env files, and the deploy warm-up is the
  ping that writes nothing and bills no model.

The first-turn checks run in a fresh interpreter each: sys.modules of this
test process has long since been filled by the other tests.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gub_agent import eager_imports

REPO = Path(__file__).resolve().parents[2]

# One stub first turn in a fresh interpreter: the real root agent, genai's
# async calls answered in-process (so ADK builds its requests and its genai
# client for real, and nothing reaches the network), a smalltalk decision so
# the speculative deep run beside the router builds the executor's request too.
_FIRST_TURN = r"""
import asyncio, json, os, sys
sys.path.insert(0, os.getcwd())
before = set(sys.modules)
from gub_agent import agent as agent_module
after_import = set(sys.modules)
from google.adk.runners import InMemoryRunner
from google.genai import models as genai_models, types

DECISION = json.dumps({"intent": "smalltalk", "confidence": 0.99, "slots": {}, "language": "en"})

def reply(config):
    si = str(getattr(config, "system_instruction", "") or "").lower()
    text = DECISION if ("route" in si or "intent" in si) else "Nothing to add."
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=[types.Part(text=text)]), finish_reason="STOP")])

async def generate(self, *, model, contents, config=None, **kw):
    await asyncio.sleep(0.05)
    return reply(config)

async def stream(self, *, model, contents, config=None, **kw):
    await asyncio.sleep(0.05)
    async def gen():
        yield reply(config)
    return gen()

genai_models.AsyncModels.generate_content = generate
genai_models.AsyncModels.generate_content_stream = stream

async def main():
    runner = InMemoryRunner(agent=agent_module.root_agent, app_name="gub_agent")
    s = await runner.session_service.create_session(app_name="gub_agent", user_id="u")
    m0 = set(sys.modules)
    msg = types.Content(role="user", parts=[types.Part(text="hi there")])
    async for _ in runner.run_async(user_id="u", session_id=s.id, new_message=msg):
        pass
    new = sorted(set(sys.modules) - m0)
    print(json.dumps({"at_import": sorted(after_import - before), "first_turn_new": new}))

asyncio.run(main())
"""


def _fresh(flag: str) -> dict:
    env = dict(os.environ)
    env.update(
        EAGER_ADK_IMPORTS=flag,
        GOOGLE_CLOUD_PROJECT=env.get("GOOGLE_CLOUD_PROJECT", "test-project"),
        GOOGLE_GENAI_USE_VERTEXAI="1",
        SANDBOX_ENABLED="0",
        FILE_SEARCH_ENABLED="0",
    )
    done = subprocess.run(
        [sys.executable, "-c", _FIRST_TURN],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def lazy() -> dict:
    return _fresh("0")


@pytest.fixture(scope="module")
def eager() -> dict:
    return _fresh("1")


def test_flag_off_first_turn_pays_the_lazy_imports(lazy):
    """The cost this exists for: today a first turn imports thousands of
    modules, the anthropic SDK behind ADK's content builder among them."""
    assert len(lazy["first_turn_new"]) > 1000
    assert "google.adk.models.anthropic_llm" in lazy["first_turn_new"]


def test_flag_off_loads_exactly_todays_package(lazy):
    assert "gub_agent.eager_imports" not in lazy["at_import"]
    assert "google.adk.models.anthropic_llm" not in lazy["at_import"]
    assert "anthropic" not in lazy["at_import"]


def test_flag_on_the_first_turn_imports_almost_nothing(eager, lazy):
    assert "gub_agent.eager_imports" in eager["at_import"]
    assert "google.adk.models.anthropic_llm" in eager["at_import"]
    # What the lazy first turn imported is, with the flag, already loaded.
    assert len(eager["first_turn_new"]) <= 10, eager["first_turn_new"]
    assert set(lazy["first_turn_new"]) - set(eager["at_import"]) <= set(eager["first_turn_new"])


def test_every_listed_module_imports_here():
    """A rename in an ADK or genai upgrade fails here instead of quietly
    making production lazy again."""
    result = eager_imports.import_now()
    assert result["skipped"] == [], result["skipped"]
    assert result["imported"] == len(eager_imports.MODULES)


def test_a_module_that_fails_is_skipped_and_reported(caplog):
    caplog.set_level(logging.INFO, logger="gub_agent.eager_imports")

    result = eager_imports.import_now(("json", "gub_agent_no_such_module_xyz"))

    assert result["imported"] == 1
    assert result["skipped"] == ["gub_agent_no_such_module_xyz"]
    line = next(
        r.getMessage() for r in caplog.records if r.getMessage().startswith("eager_imports:")
    )
    assert line.startswith("eager_imports: imported=1 skipped=1 new_modules=")
    assert line.endswith(" skipped_names=gub_agent_no_such_module_xyz")


def test_nothing_skipped_reads_as_a_dash(caplog):
    caplog.set_level(logging.INFO, logger="gub_agent.eager_imports")

    eager_imports.import_now(("json",))

    line = next(
        r.getMessage() for r in caplog.records if r.getMessage().startswith("eager_imports:")
    )
    assert line.endswith(" skipped_names=-")


def _env_lines(name: str) -> dict:
    out = {}
    for raw in (REPO / name).read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            out[key] = value
    return out


def test_both_deploy_envs_state_the_flag_on():
    for name in ("deploy-prod.env", "deploy-sandbox.env"):
        assert _env_lines(name).get("EAGER_ADK_IMPORTS") == "1", name


def test_the_deploy_warm_up_is_a_query_that_writes_nothing():
    """The deploy's warm-up used to be a greeting turn (a billed router call,
    and since stage 1 a cancelled speculative executor round). It is now one
    `:query` list_sessions call: it loads the package, which is the cost, and
    reaches no model and stores no event."""
    workflow = (REPO / ".github/workflows/deploy.yml").read_text()
    step = workflow[workflow.index("- name: Warm the fresh revision") :]
    step = step[: step.index("\n      - name:") if "\n      - name:" in step else len(step)]
    assert '"class_method":"list_sessions"' in step
    assert '"user_id":"engine-warmup"' in step
    assert ":streamQuery" not in step
    assert "stream_query" not in step
    assert "create_session" not in step
