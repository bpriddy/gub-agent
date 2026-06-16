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

## Architecture

`root_agent` is a small multi-agent pipeline, not a single LLM — a deliberate,
minimal borrowing from the Agentic-RAG "critic-before-commit" pattern:

```
LoopAgent(max_iterations=2)
  ├─ executor   — the tool-using LLM (gub_agent/agent.py)
  ├─ critic     — evaluates the executor's answer (gub_agent/agents/critic.py)
  └─ escalator  — exits the loop early when the critic is satisfied
```

On a clean answer the loop exits after one pass; on a flagged answer the
executor runs again, sees the critic's feedback in session state, and fixes
it. We keep just this one specialist (critic), not a planner/rewriter/fanout
fleet — at our scale the critic is the single piece that materially improves
dependability.

### Tools

- `org_query` — structured query primitive (filter / sort / group / aggregate
  / `similar_to` trigram) against GUB's `POST /org/query`. THE preferred tool
  for any filter/count/sort/aggregate question. Lightweight: returns only
  catalog fields, never large text blobs. (`gub_agent/tools/org_query.py`)
- `find_staff_for_resourcing`, `search_staff`, `get_staff_profile` — staff.
- `list_accounts`, `get_account_overview`, `get_campaign` — detail tools.
  These return the rich per-entity fields (e.g. `statusMarkdown`); `org_query`
  is for *finding* entities, the detail tools for *reading* one.

### Synthesis principles (how the agent decides what to say)

These are general — phrased to scale across question types, never tuned to a
specific phrasing. See `gub_agent/agent.py` (`SYSTEM_INSTRUCTION`) and
`gub_agent/agents/critic.py`.

- **Author to the question's intended CLOSURE, not its literal words.** A
  question seeking a FACT wants the value ("47"); an ASSESSMENT ("how is X?",
  "should we worry about X?") wants a VERDICT up front plus the few drivers
  that earn it — never a catalog; an EXPLORATORY question wants a shaped
  shortlist. "Technically answered" is a near-zero bar ("Active." technically
  answers "how is the chevy account?"); the real bar is serving the asker's
  intent. Length follows closure, not data volume.
- **Verdicts must be earned** — grounded in retrieved signals, like any claim.
- **Ground every entity.** Never name an account/campaign/person that didn't
  appear in a tool result this turn — no inventing, no "helpful completion".
- **Time is part of relevance.** Weight recent and currently-active items;
  don't surface something long-finished as if it were current.

### The critic — two-axis evaluation

`CriticVerdict` reports two independent judgments, gating `sufficient`:

1. `info_sufficient` — did the tool calls gather enough to answer this
   question? (right tools used, every entity queried, multi-part questions
   decomposed, not answered with zero tool calls).
2. `answer_satisfies` — does the answer deliver the right KIND of closure,
   grounded, recency-respected? (a catalog answer to "how is X" fails here).

`sufficient = info_sufficient AND answer_satisfies`. False triggers a retry.

### Deterministic current-date injection

Gemini doesn't reliably know "now", and a `get_current_date` tool would be
LLM-mediated (the model has to choose to call it). Instead, both the executor
and critic use an ADK **`InstructionProvider`** (a *callable* instruction,
`gub_agent/instruction_utils.py:with_current_date`) that appends
"Today is YYYY-MM-DD (UTC)" on **every** request — computed server-side,
deterministic, never stale, no tool, no caller coupling. This is what makes
"recent" / "this week" reliable.

> **ADK brace trap:** ADK runs `{var}` session-state injection over instruction
> strings (including an InstructionProvider's *output*). Any literal `{anything}`
> that isn't a real state key crashes the run with `KeyError` before the LLM is
> called — visible only in the Reasoning Engine logs. Grep prompts for `{`
> before every `adk deploy`. (f-string interpolations like `f"{base}"` are safe
> because they don't emit literal braces.)

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

The agent deploys to **Vertex AI Agent Engine via `adk deploy agent_engine`**
— NOT Cloud Build. It will never appear in `gcloud builds`. Update in place
(same engine id, so callers — the Chat bot, Agentspace — see no change):

```bash
adk deploy agent_engine \
  --project=os-test-491819 \
  --region=us-central1 \
  --agent_engine_id=9136379226620952576 \
  --display_name=gub-agent \
  --description="<what changed>" \
  gub_agent
```

The positional `gub_agent` (the package exporting `root_agent`) is required.
Agent names must be valid Python identifiers — no dashes (e.g. `gub_pipeline`,
not `gub-pipeline`), or ADK rejects the deploy. A transient `code 13
INTERNAL` from Agent Engine usually succeeds on a retry.

Then `register_agent.py` (see `deployment/`) registers the deployed engine
with the Gemini Enterprise app — a separate, occasional step.

### Debug client

`debug_client/` is a local-only Next.js tool for inspecting the agent's
decomposition trace (per-iteration tool calls, the two-axis critic verdict,
sources). Sign in with Google, ask a question, watch how the agent turns it
into queries. See `debug_client/README.md`. It calls Vertex AI server-side via
ADC; it never deploys.

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
