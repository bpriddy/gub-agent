#!/usr/bin/env python3
"""
register_agent.py — Register (or update) the GUB ADK agent with Gemini Enterprise.

Wraps the Discovery Engine API calls described in:
https://medium.com/@sravan.gottipaty/plugging-your-custom-adk-agent-into-gemini-enterprise-part-3-3-6e8557efbbc4

Prerequisites:
  1. `adk deploy agent_engine` has been run and AGENT_ENGINE_ID is set in .env
  2. The running identity has roles/discoveryengine.admin on the project
  3. The Gemini Enterprise service account has been granted aiplatform.user:
       gcloud projects add-iam-policy-binding $PROJECT_ID \\
         --member="serviceAccount:service-$PROJECT_NUMBER@gcp-sa-discoveryengine.iam.gserviceaccount.com" \\
         --role="roles/aiplatform.user"

Usage:
  python deployment/register_agent.py             # register new agent
  python deployment/register_agent.py --update    # update existing registration
  python deployment/register_agent.py --list      # list registered agents
  python deployment/register_agent.py --delete AGENT_ID
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ID     = os.environ["GCP_PROJECT_ID"]
PROJECT_NUMBER = os.environ.get("GCP_PROJECT_NUMBER", PROJECT_ID)  # numeric, e.g. 843516467880
REGION         = os.environ.get("GCP_REGION", "global")
APP_ID         = os.environ["GEMINI_APP_ID"]
ENGINE_ID      = os.environ.get("AGENT_ENGINE_ID", "")
AGENT_NAME     = os.environ.get("AGENT_NAME", "gub-agent")

TOOL_DESCRIPTION = (
    "Use this agent to answer questions about the agency's operations. "
    "Call this agent when the user asks about: staff, team members, resourcing, "
    "who is available, capacity, skills, accounts, clients, campaigns, "
    "account overviews, budgets, or any agency organisational data."
)

# Discovery Engine base path
DE_BASE = (
    f"https://discoveryengine.googleapis.com/v1alpha"
    f"/projects/{PROJECT_ID}/locations/{REGION}"
    f"/collections/default_collection/engines/{APP_ID}"
    f"/assistants/default_assistant/agents"
)

# Full Vertex AI Agent Engine resource path — MUST use project NUMBER not ID
REASONING_ENGINE = (
    f"projects/{PROJECT_NUMBER}/locations/us-central1/reasoningEngines/{ENGINE_ID}"
    if ENGINE_ID else ""
)


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_access_token() -> str:
    """Get a GCP access token via the active gcloud identity."""
    result = subprocess.run(
        ["gcloud", "auth", "print-access-token"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def headers() -> dict:
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Content-Type": "application/json",
    }


# ── Operations ────────────────────────────────────────────────────────────────

def list_agents() -> None:
    resp = requests.get(DE_BASE, headers=headers(), timeout=30)
    resp.raise_for_status()
    agents = resp.json().get("agents", [])
    if not agents:
        print("No agents registered.")
        return
    for a in agents:
        print(f"  {a.get('name')}  ({a.get('displayName')})")


def register_agent() -> str:
    if not REASONING_ENGINE:
        print("ERROR: AGENT_ENGINE_ID is not set in .env")
        print("Run `adk deploy agent_engine` first and paste the engine ID into .env")
        sys.exit(1)

    payload = {
        "displayName": AGENT_NAME,
        "description": (
            "GUB AI — agency operations assistant. "
            "Provides resourcing, staff profiles, account overviews, and campaign history."
        ),
        "adkAgentDefinition": {
            "provisionedReasoningEngine": {
                "reasoningEngine": REASONING_ENGINE,
            },
            "toolSettings": {
                "toolDescription": TOOL_DESCRIPTION,
            },
        },
    }

    resp = requests.post(DE_BASE, headers=headers(), json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    agent_id = data.get("name", "").split("/")[-1]
    print(f"Registered agent: {data.get('name')}")
    print(f"Agent ID: {agent_id}")
    print("\nAdd to .env:")
    print(f"  REGISTERED_AGENT_ID={agent_id}")
    return agent_id


def update_agent(agent_id: str) -> None:
    if not REASONING_ENGINE:
        print("ERROR: AGENT_ENGINE_ID is not set in .env")
        sys.exit(1)

    url = f"{DE_BASE}/{agent_id}"
    payload = {
        "displayName": AGENT_NAME,
        "description": (
            "GUB AI — agency operations assistant. "
            "Provides resourcing, staff profiles, account overviews, and campaign history."
        ),
        "adkAgentDefinition": {
            "provisionedReasoningEngine": {
                "reasoningEngine": REASONING_ENGINE,
            },
            "toolSettings": {
                "toolDescription": TOOL_DESCRIPTION,
            },
        },
    }

    resp = requests.patch(url, headers=headers(), json=payload, timeout=30)
    resp.raise_for_status()
    print(f"Updated: {resp.json().get('name')}")


def delete_agent(agent_id: str) -> None:
    url = f"{DE_BASE}/{agent_id}"
    resp = requests.delete(url, headers=headers(), timeout=30)
    resp.raise_for_status()
    print(f"Deleted agent: {agent_id}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Register the GUB agent with Gemini Enterprise")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--list",   action="store_true", help="List registered agents")
    group.add_argument("--update", metavar="AGENT_ID",  help="Update an existing agent registration")
    group.add_argument("--delete", metavar="AGENT_ID",  help="Delete an agent registration")
    args = parser.parse_args()

    if args.list:
        list_agents()
    elif args.update:
        update_agent(args.update)
    elif args.delete:
        delete_agent(args.delete)
    else:
        register_agent()


if __name__ == "__main__":
    main()
