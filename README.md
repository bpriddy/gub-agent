# gub-agent

Python ADK agent that exposes GUB (gcp-universal-backend) data to Gemini
Enterprise / Agentspace. Users talk to Gemini Enterprise; Gemini invokes this
agent's tools; the agent calls GUB's HTTP API on the user's behalf.

Sibling repos: [gcp-universal-backend](https://github.com/bpriddy/gcp-universal-backend),
[gub-admin](https://github.com/bpriddy/gub-admin),
[gub-review](https://github.com/bpriddy/gub-review).

## Stack

- Python 3.11+ with [Google ADK](https://github.com/google/adk-python)
- Deployed to Vertex AI Agent Engine
- Registered with a Gemini Enterprise app for end-user chat

## Local setup

```bash
git clone git@github.com:bpriddy/gub-agent.git
cd gub-agent

# Python env
python -m venv .venv && source .venv/bin/activate
pip install -r gub_agent/requirements.txt

# Local config
cp .env.example .env
# then edit .env and fill in GUB_BASE_URL, GUB_SERVICE_JWT, etc.

# Secret-scan pre-commit hook (required — refuses commit on detected
# API keys, tokens, JSON keys, etc.)
brew install gitleaks        # or see https://github.com/gitleaks/gitleaks#installation
git config core.hooksPath .githooks
```

## Run

```bash
# Start the agent locally against a running GUB backend (localhost:3000)
python -m gub_agent
```

## Deploy

See `deployment/` — `register_agent.py` ships the agent to Vertex AI Agent
Engine and registers it with the configured Gemini Enterprise app.

## Security

Credentials live in `.env` locally and in GCP Secret Manager in deployed
environments. Never commit `.env`, service account JSON, or any file under
`secrets/` / `keys/` — the gitleaks pre-commit hook above is the second line
of defense after `.gitignore`.
