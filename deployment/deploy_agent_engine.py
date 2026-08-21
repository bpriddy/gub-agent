#!/usr/bin/env python3
"""
deploy_agent_engine.py — update the Vertex AI Agent Engine in place via the
Python SDK (`vertexai.agent_engines.update`) instead of `adk deploy
agent_engine`.

Why the CLI had to go (gub-gchat-bot#3, token streaming): the deployed
engine's AdkApp template predates per-call `run_config`, so callers cannot
request `RunConfig(streaming_mode=SSE)` and every answer arrives as one blob
after 40-140s. The CURRENT google-cloud-aiplatform AdkApp accepts a
`run_config` dict on `stream_query`/`async_stream_query` and forwards it to
`Runner.run(run_config=RunConfig.model_validate(...))` — so redeploying with
the current template is the entire agent-side fix. The `adk deploy
agent_engine` CLI (google-adk >= 2.x) would instead rebuild the engine as an
api_server container with unpinned deps; this script keeps the classic
managed-template runtime and pins the template version it was tested with.

Streaming stays OPT-IN PER CALL. This deploy does NOT change the default
cadence: a `stream_query` without `run_config` still runs StreamingMode.NONE
and returns the same single executor blob, so existing callers (debug client,
Agentspace/Gemini Enterprise) are untouched. Only a caller that passes
  input.run_config = {"streaming_mode": "sse", "response_modalities": ["TEXT"]}
gets partial=True token deltas.

Post-deploy gate (run before trusting streaming, see the issue-3 task spec):
  GTOKEN=$(gcloud auth print-access-token) node scratchpad/cadence-test.mjs
  - run WITHOUT run_config  → must STAY one blob (no caller regression)
  - run WITH run_config sse → must show multiple partial=true events

Env (repo vars in CI, .env locally):
  GCP_PROJECT_ID        e.g. os-test-491819
  GCP_REGION            engine RESOURCE region, e.g. us-central1 (never global)
  AGENT_ENGINE_ID       numeric id of the engine to update (refuses to create)
  AGENT_STAGING_BUCKET  gs:// bucket for the SDK's pickle+deps upload (NEW —
                        the SDK deploy path requires one; the CLI did not)

Usage:
  python deployment/deploy_agent_engine.py --description "what changed"
  python deployment/deploy_agent_engine.py --dry-run   # build + validate only
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import dotenv_values, load_dotenv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# `python deployment/deploy_agent_engine.py` puts deployment/ (not the repo
# root) on sys.path — make gub_agent importable without an editable install.
sys.path.insert(0, REPO_ROOT)
load_dotenv(dotenv_path=os.path.join(REPO_ROOT, ".env"))

AGENT_PACKAGE = "gub_agent"
DISPLAY_NAME = "gub-agent"
# Deploy-time env baked into the engine (EMIT_THINKING=1 dev convention) —
# same file the old `adk deploy --env_file` step shipped.
ENV_FILE = os.path.join(REPO_ROOT, "deploy-dev.env")


def _require_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        print(f"ERROR: {name} is not set — refusing to deploy.", file=sys.stderr)
        sys.exit(1)
    return value


def build_requirements() -> list[str]:
    """Engine-runtime requirements, pinned to the versions installed HERE.

    The SDK deploy cloudpickles the AdkApp instance locally and unpickles it
    inside the engine container — a google-adk / aiplatform version skew
    between the two sides is the classic way that breaks. Pinning to the
    deploy environment's versions makes both sides identical by construction.
    """
    import google.adk
    from google.cloud import aiplatform

    requirements = [f"google-cloud-aiplatform[agent_engines]=={aiplatform.__version__}"]
    req_txt = os.path.join(REPO_ROOT, AGENT_PACKAGE, "requirements.txt")
    with open(req_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("google-adk"):
                line = f"google-adk=={google.adk.__version__}"
            requirements.append(line)
    return requirements


def build_app():
    """Wrap root_agent in the CURRENT AdkApp template (per-call run_config)."""
    from vertexai.preview.reasoning_engines import AdkApp

    from gub_agent.agent import root_agent

    return AdkApp(agent=root_agent)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update the GUB Vertex AI Agent Engine in place")
    parser.add_argument(
        "--description",
        default="",
        help="Stored as the engine revision description (default: generic SHA line)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Import the agent, build the app and requirements, print the plan, deploy nothing",
    )
    args = parser.parse_args()

    project = _require_env("GCP_PROJECT_ID")
    region = _require_env("GCP_REGION")
    engine_id = _require_env("AGENT_ENGINE_ID")  # refuse to create a new engine
    description = args.description or f"SDK deploy {os.environ.get('GITHUB_SHA', 'local')[:7]}"
    resource_name = f"projects/{project}/locations/{region}/reasoningEngines/{engine_id}"

    requirements = build_requirements()
    env_vars = {k: v for k, v in dotenv_values(ENV_FILE).items() if v is not None}
    app = build_app()

    print(f"Engine:       {resource_name}")
    print(f"Description:  {description}")
    print(f"Requirements: {requirements}")
    print(f"Baked env:    {env_vars}")
    print(f"Agent:        {type(app).__name__}(agent={app._tmpl_attrs['agent'].name})")

    if args.dry_run:
        print("Dry run — not deploying.")
        return

    staging_bucket = _require_env("AGENT_STAGING_BUCKET")
    if not staging_bucket.startswith("gs://"):
        staging_bucket = f"gs://{staging_bucket}"

    import vertexai
    from vertexai import agent_engines

    vertexai.init(project=project, location=region, staging_bucket=staging_bucket)
    updated = agent_engines.update(
        resource_name=resource_name,
        agent_engine=app,
        requirements=requirements,
        display_name=DISPLAY_NAME,
        description=description,
        env_vars=env_vars,
        extra_packages=[os.path.join(REPO_ROOT, AGENT_PACKAGE)],
    )
    print(f"Updated: {updated.resource_name}")


if __name__ == "__main__":
    main()
