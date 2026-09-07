"""
deploy env files — the sandbox flag is a property of the deploy, not of callers.

There is one production engine; the Chat bot and the Gemini Enterprise
registration point at it. Its isolation from experiments rests on a single
fact: the env file the production deploy bakes in has SANDBOX_ENABLED off, and
only the sandbox deploy's env file has it on. Both files are plain text that
anyone can edit, and production's used to be called `deploy-dev.env` — a name
that invited exactly the mistake this pins (adding a flag "to the dev file").
So the check lives in CI: read the workflow, find the file it deploys, and
refuse a build that would arm the sandbox in production, name the file "dev"
again, or pass a relative --env_file that would be silently skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEPLOY_WORKFLOW = REPO / ".github" / "workflows" / "deploy.yml"
SANDBOX_SCRIPT = REPO / "deployment" / "deploy-sandbox.sh"
PROD_ENGINE_ID = "9136379226620952576"

TRUTHY = ("1", "true", "yes")  # mirrors config.py's parsing


def _env(path: Path) -> dict[str, str]:
    """Minimal dotenv parse — the files are deliberately KEY=VALUE only."""
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _flag_on(values: dict[str, str], key: str) -> bool:
    return values.get(key, "false").lower() in TRUTHY


def _env_file_named_in(text: str) -> Path:
    found = re.findall(r"--env_file=(\S+)", text)
    assert len(found) == 1, f"expected exactly one --env_file in {text[:60]!r}..., got {found}"
    named = found[0].strip("'\"")
    # The workflow anchors the path on $GITHUB_WORKSPACE (see the absolute-path
    # test below); resolve it against the repo the same way the runner would.
    return REPO / named.removeprefix("$GITHUB_WORKSPACE/")


def _prod_env_file() -> Path:
    return _env_file_named_in(DEPLOY_WORKFLOW.read_text())


def _sandbox_env_file() -> Path:
    text = SANDBOX_SCRIPT.read_text()
    match = re.search(r'^ENV_FILE="([^"]+)"', text, re.MULTILINE)
    assert match, 'deploy-sandbox.sh must declare ENV_FILE="..."'
    # The script anchors the path on $REPO_ROOT (adk deploy chdir()s before it
    # reads --env_file, so a relative path is silently skipped). Strip it here.
    return REPO / match.group(1).removeprefix("$REPO_ROOT/")


async def test_sandbox_script_passes_an_absolute_env_file_path():
    """A relative --env_file is looked up inside adk's temp folder and silently
    dropped — the engine comes up with no env and is not a sandbox at all."""
    text = SANDBOX_SCRIPT.read_text()
    match = re.search(r'^ENV_FILE="([^"]+)"', text, re.MULTILINE)
    assert match and match.group(1).startswith(("$REPO_ROOT/", "/")), match


async def test_production_deploy_env_has_the_sandbox_explicitly_off():
    """Explicit, not defaulted: a missing key would also be off today, but the
    line is the record of intent the next editor reads."""
    path = _prod_env_file()
    values = _env(path)
    assert "SANDBOX_ENABLED" in values, f"{path.name}: SANDBOX_ENABLED must be set explicitly"
    assert not _flag_on(values, "SANDBOX_ENABLED"), (
        f"{path.name} is baked into the PRODUCTION engine by deploy.yml and "
        "would arm the sandbox for live users"
    )


async def test_sandbox_deploy_env_has_the_sandbox_on():
    path = _sandbox_env_file()
    assert _flag_on(_env(path), "SANDBOX_ENABLED"), (
        f"{path.name}: a sandbox engine without SANDBOX_ENABLED ignores every override"
    )


async def test_production_and_sandbox_deploys_use_different_env_files():
    """The two deploys must not share a file, or one edit flips both engines."""
    assert _prod_env_file().resolve() != _sandbox_env_file().resolve()


async def test_sandbox_env_never_names_the_production_engine():
    values = _env(_sandbox_env_file())
    assert PROD_ENGINE_ID not in values.get("SANDBOX_AGENT_ENGINE_ID", "")


async def test_sandbox_script_does_not_register_with_gemini_enterprise():
    """The sandbox engine must not appear in the Gemini Enterprise agent list —
    registration is what routes real users to an engine."""
    text = SANDBOX_SCRIPT.read_text()
    invoking = [
        line
        for line in text.splitlines()
        if "register_agent" in line and not line.lstrip().startswith("#")
    ]
    assert invoking == [], f"deploy-sandbox.sh invokes register_agent: {invoking}"


async def test_sandbox_script_refuses_the_production_engine_id():
    """The literal guard must be in the script, not only in the reviewer's head."""
    assert PROD_ENGINE_ID in SANDBOX_SCRIPT.read_text()


async def test_production_workflow_passes_an_absolute_env_file_path():
    """The bug this pins is silent, not loud: `adk deploy` chdir()s into its
    staging folder before reading --env_file, so a relative path resolves in
    the wrong directory and is skipped with no error. The engine then runs on
    config.py defaults and looks like a successful deploy — which is what
    happened to production for a month. Absolute, or the build fails."""
    text = DEPLOY_WORKFLOW.read_text()
    found = re.findall(r"--env_file=(\S+)", text)
    assert len(found) == 1, found
    named = found[0].strip("'\"")
    assert named.startswith(("$GITHUB_WORKSPACE/", "/")), (
        f"--env_file={named} is relative and would be silently dropped"
    )


async def test_production_env_file_is_not_named_dev():
    """The name is load-bearing documentation: for a month this file was called
    deploy-dev.env while being the file the PRODUCTION deploy bakes in."""
    name = _prod_env_file().name
    assert "dev" not in name.lower(), (
        f"{name} is deployed to production ({PROD_ENGINE_ID}) — a 'dev' name "
        "invites edits meant for a test engine"
    )


async def test_production_deploy_declares_emit_thinking_explicitly():
    """EMIT_THINKING sat at 1 in this file for its whole life and never reached
    the engine, so production has always run with thought summaries off. The
    key stays explicit so that whoever changes it is choosing, not discovering."""
    values = _env(_prod_env_file())
    assert "EMIT_THINKING" in values, (
        f"{_prod_env_file().name}: keep EMIT_THINKING explicit — the fix to the "
        "env_file path means its value now actually reaches production"
    )


async def test_production_deploy_verifies_the_env_landed():
    """A deploy that cannot fail on a skipped env file will hide the next one."""
    text = DEPLOY_WORKFLOW.read_text()
    assert "verify-engine-env.sh" in text, (
        "deploy.yml must read the env back off the engine after deploying"
    )
    assert (REPO / "deployment" / "verify-engine-env.sh").exists()
