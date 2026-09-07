#!/usr/bin/env bash
# Verify that the env we baked into an Agent Engine actually reached it.
#
# The failure this exists for is SILENCE: `adk deploy` chdir()s into its
# staging folder before it reads --env_file (ADK 2.6.1 cli_deploy.py:1000 vs
# :1094), so a relative path is looked up in the wrong directory and skipped
# with no error. The engine comes up healthy, running on config.py defaults —
# indistinguishable from an intentional deploy. That is how the production
# engine ran for a month with an empty deploymentSpec.
#
# Usage: verify-engine-env.sh <env-file> [engine-id] [project] [region]
#   Falls back to GCP_PROJECT_ID / GCP_REGION / AGENT_ENGINE_ID from the env.
# Exits non-zero if any key in <env-file> is missing from the deployed engine.
set -euo pipefail

ENV_FILE="${1:?usage: verify-engine-env.sh <env-file> [engine-id] [project] [region]}"
ENGINE_ID="${2:-${AGENT_ENGINE_ID:?engine id required}}"
PROJECT="${3:-${GCP_PROJECT_ID:?project required}}"
REGION="${4:-${GCP_REGION:-us-central1}}"

[[ -f "$ENV_FILE" ]] || { echo "error: env file not found: $ENV_FILE" >&2; exit 1; }

RESOURCE="projects/$PROJECT/locations/$REGION/reasoningEngines/$ENGINE_ID"
TOKEN="$(gcloud auth application-default print-access-token 2>/dev/null || gcloud auth print-access-token)"

ENGINE_ENV="$(curl -sfS -H "Authorization: Bearer $TOKEN" \
    "https://$REGION-aiplatform.googleapis.com/v1/$RESOURCE" \
  | python3 "$(dirname "${BASH_SOURCE[0]}")/engine_env.py")"
echo "engine env: $ENGINE_ENV"

missing=()
while IFS= read -r key; do
  [[ -n "$key" ]] || continue
  [[ "$ENGINE_ENV" == *"$key="* ]] || missing+=("$key")
done < <(grep -vE '^[[:space:]]*#' "$ENV_FILE" | grep '=' | cut -d= -f1 | tr -d '[:space:]')

if ((${#missing[@]})); then
  echo "::error::these keys are in $(basename "$ENV_FILE") but never reached the engine: ${missing[*]}" >&2
  echo "       The env file was skipped — check that --env_file is an ABSOLUTE path." >&2
  exit 1
fi
echo "all keys from $(basename "$ENV_FILE") landed on the engine"
