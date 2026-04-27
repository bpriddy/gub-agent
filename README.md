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

## Secrets & rotation

Documents secrets/credentials this service uses and how to rotate them.
For company-wide incident response (escalation, post-mortem, comms), see
IT's canonical incident-response doc. This section covers system-specific
actions only.

This service is unusual in two ways:

- It runs on **Vertex AI Agent Engine**, not Cloud Run. Deploys go through
  `adk deploy agent_engine` (see `deployment/register_agent.py`), and
  Cloud Run's `--set-secrets` plumbing does not apply — Agent Engine
  injects environment via its own mechanism.
- It depends on a **separate OAuth relay** (`deployment/oauth-relay/`,
  deployed as a Cloud Function) that bridges Gemini Enterprise's OAuth
  flow with GUB's identity API. The relay has its own configuration
  surface, also documented below.

### Inventory

| Credential | Where it lives | Issued by | Used for |
|---|---|---|---|
| `GUB_SERVICE_JWT` | Agent Engine env (set during `adk deploy`); locally `.env` | Self-issued by GUB (re-use a frontend-issued access token, or generate via `scripts/test-broker-flow.mjs`) | **Local dev / CI fallback only.** Authenticated GUB API calls when the OAuth flow isn't active (e.g., direct Python REPL testing). Production traffic uses Gemini Enterprise's OAuth flow with per-user tokens — this JWT is not on the runtime hot path. |
| `GUB_AUTHORIZATION_ID` | Agent Engine env; locally `.env` | Gemini Enterprise registration | OAuth flow identity matching. **Configuration, not a secret** — it's just a string that must match exactly between the Gemini Enterprise console registration and the agent's env. |
| `GEMINI_APP_ID` | Agent Engine env; locally `.env` | Gemini Enterprise app registration | Targets the correct Gemini Enterprise app during `register_agent.py`. **Configuration, not a secret.** |
| `AGENT_ENGINE_ID` | `.env` after deploy | Output of `adk deploy agent_engine` | Pins the running agent revision so `register_agent.py` can register it. **Configuration**, populated post-deploy. |
| OAuth relay deployment config | Cloud Function env (in `deployment/oauth-relay/`) | GCP Cloud Function deploy | Redirect URL whitelist + GUB endpoint. |
| GCP service account key | None — uses Application Default Credentials (`gcloud auth application-default login` locally; the Agent Engine runtime SA in deploys) | GCP | All GCP API calls |

### Rotation procedures

#### `GUB_SERVICE_JWT`

This is a JWT issued by GUB. Rotation = re-issue a new one. The old JWT
remains technically valid until its `exp` (default 15 min for access
tokens), but is hardly used in production paths so the urgency is low.

**Preconditions.** None.

**Steps.**
1. Issue a fresh JWT. Two options:
   - Sign in via the gub-admin frontend with a service account user, then
     copy the access token from the browser's network tab.
   - Run `node scripts/test-broker-flow.mjs` from the gcp-universal-backend
     repo (the broker test flow returns a fresh JWT).
2. Update Agent Engine's env. Because Agent Engine doesn't expose a
   per-revision env override flag like Cloud Run, the cleanest path is
   to redeploy:
   ```bash
   adk deploy agent_engine    # picks up new GUB_SERVICE_JWT from .env
   ```
   For an in-place update without redeploy, use the Vertex AI console
   under Agent Engine → your agent → Edit env vars.
3. If JWT is stored locally: update `.env`. Don't commit.

**Verification.** Send a test query through Gemini Enterprise that
exercises a GUB-backed tool. Confirm the response is non-error and
includes data the user is authorized to see.

**Cleanup.** GUB-issued JWTs expire on their own — no explicit revocation
step today. **Note: this is a gap.** Until refresh-token revocation lands,
a leaked service JWT remains valid until `exp`. Mitigation: keep TTL
short (default 15 min for access tokens) and rotate immediately on any
suspected exposure.

#### OAuth relay redirect URI whitelist

The OAuth relay (`deployment/oauth-relay/main.py`, Cloud Function)
maintains a whitelist of redirect URIs to defend against open-redirect
attacks during the OAuth flow.

**Preconditions.** None — config-only change.

**Steps.**
1. Edit `deployment/oauth-relay/main.py` (or wherever the whitelist
   lives in the deployed function — verify against the actual deployed
   source if unsure).
2. Redeploy the Cloud Function:
   ```bash
   gcloud functions deploy oauth-relay --source=deployment/oauth-relay --runtime=python311 ...
   ```
   (verify the actual deploy command matches your existing infra).

**Verification.** Initiate a Gemini Enterprise → agent flow that should
succeed (whitelisted redirect) and one that should fail (an arbitrary
redirect URI). The latter must be rejected.

**Cleanup.** Old function revisions are kept by Cloud Functions for
rollback. After a few days of stable behavior, prune old revisions if
quota is a concern.

#### GCP service account / Application Default Credentials

This agent uses ADC, not a static SA key file. Rotation = the standard
GCP SA-key rotation if you've configured one for ADC; locally, run
`gcloud auth application-default login` again to refresh the user
credential. There's no per-service-key rotation procedure here because
the service has none.

### What this service does NOT have

- No JWT signing keys (it consumes JWTs verified by GUB; doesn't sign).
- No DB credentials (it talks to GUB over HTTP; never touches the DB).
- No third-party API keys beyond ADC for GCP-native APIs.

### Cut a user off

Access to this agent is controlled at the **Gemini Enterprise app**
level, not by this code. To revoke a specific user's ability to invoke
the agent:

1. Remove them from the Gemini Enterprise app's user/group binding (in
   the GCP console under Agentspace → your app → Access).
2. If the user also has access to gub-admin, follow the [Cut a user
   off](https://github.com/bpriddy/gcp-universal-backend#cut-a-user-off-revoke-admin-access)
   procedure in gcp-universal-backend.
3. Their existing GUB-issued JWTs remain valid until `exp` (see above
   gap note). For immediate cutoff, escalate to the IT process.
