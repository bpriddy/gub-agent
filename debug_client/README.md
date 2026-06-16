# gub-agent debug_client

Local-only debug client for `gub-agent`. Sign in with Google, ask the agent
a business question, and inspect the **full decomposition trace** —
per-iteration tool calls, critic verdicts, and sources. Use it to iterate on
how the agent turns a question like *"how is the chevy account?"* into a set
of `org_query` calls.

This is a **development tool**. It never deploys. It holds no secrets: the
Vertex AI credential comes from your machine's ADC, and the GUB credential
comes from a browser Google Sign-In.

## How it's wired

```
browser (localhost:3002)
  ├─ Google Sign-In → /gub/auth/google/exchange → GUB JWT (held in memory)
  └─ POST /api/agent  (Authorization: Bearer <GUB JWT>)
        │  Next API route (server, Node runtime):
        ├─ ADC → Vertex AI bearer token        ← `gcloud auth application-default login`
        ├─ create_session(state.gub_jwt = <GUB JWT>)
        ├─ stream_query → collect events
        ├─ buildTrace(events)
        └─ return structured trace
              ↓
          gub-agent (Vertex AI) → GUB /org/query (as the signed-in user)
```

`/gub/*` is a Next rewrite to `GUB_BACKEND_URL` so the auth calls are
same-origin (no CORS). The agent's tools call GUB as **you** because your
GUB JWT is seeded into the session state.

## One-time setup

1. **ADC** (for the Vertex AI call):
   ```bash
   gcloud auth application-default login
   ```
   The account you log in as needs `roles/aiplatform.user` on the project.

2. **Add the origin to the Google OAuth client.** In the Cloud Console,
   edit the OAuth client in `NEXT_PUBLIC_GOOGLE_CLIENT_ID` and add
   `http://localhost:3002` to **Authorized JavaScript origins**.

3. **Add the origin to GUB's trusted_apps.** Pair `http://localhost:3002`
   with that client_id (via the gub-admin Trusted Apps UI, or the local
   admin running against your dev DB).

4. **Env:**
   ```bash
   cp .env.local.example .env.local
   # GUB_BACKEND_URL → your local or dev GUB
   # GCP_PROJECT_ID / GCP_REGION / AGENT_ENGINE_ID → the deployed engine
   # NEXT_PUBLIC_GOOGLE_CLIENT_ID → same client used by the GUB frontend
   ```

5. **Install + run:**
   ```bash
   npm install
   npm run dev
   # → http://localhost:3002
   ```

## Using it

- Sign in.
- Type a question, ⌘↵ (or click Run).
- Read the trace:
  - **Iteration cards** — the tool calls the executor made (expand to see
    args + response), the executor's prose, and the **critic verdict**
    (green = sufficient, red = insufficient + the feedback it gave for the
    retry).
  - **Sources** — Drive files any tool attributed.
  - **include raw events** — the unparsed Vertex AI stream, for deep digs.
- **new session** starts a fresh agent conversation; otherwise follow-up
  questions reuse the same session (the agent keeps context).

## What to look for when iterating

- Wrong tool pick (e.g. listing + counting instead of `org_query` with a
  count aggregate) → fix in the executor prompt or the `org_query` docstring.
- A multi-entity question answered in one shot instead of chained queries →
  executor prompt.
- The critic passing a bad answer, or nitpicking a good one → critic prompt.
- A question you can't express at all → missing operator in GUB's catalog.
