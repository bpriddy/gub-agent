# gub-agent debug_client

Local-only debug client for `gub-agent`. Sign in with Google, ask the agent
a business question, and inspect the **full decomposition trace** —
per-iteration tool calls, critic verdicts, and sources. Use it to iterate on
how the agent turns a question like *"how is the chevy account?"* into a set
of `org_query` calls — and, against the **sandbox engine**, to change the
model, thinking level, temperature or prompt per run and A/B two configs on
one question without a redeploy (see [Sandbox mode](#sandbox-mode)).

This is a **development tool**. It never deploys. It holds no secrets: the
Vertex AI credential comes from your machine's ADC, and the GUB credential
comes from a browser Google Sign-In.

## How it's wired

```
browser (localhost:3002)
  ├─ Google Sign-In → /gub/auth/google/exchange → GUB JWT (held in memory)
  ├─ GET  /api/config → engine id/target/isProd, model allowlist, variant names, defaults
  └─ POST /api/agent  (Authorization: Bearer <GUB JWT>)  body { message, sessionId?, config? }
        │  Next API route (server, Node runtime):
        ├─ ADC → Vertex AI bearer token        ← `gcloud auth application-default login`
        ├─ validate config (mirror of gub_agent/sandbox.py) → 400 with the engine's wording
        ├─ create_session(state = { gub_jwt, sandbox? })   ← config rides beside the JWT
        ├─ stream_query → collect events
        ├─ buildTrace(events)                 ← + sandbox_resolved, word/tool-call counts
        └─ return structured trace
              ↓
          gub-agent (Vertex AI, sandbox or prod engine) → GUB /org/query (as the signed-in user)
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
   # GCP_PROJECT_ID / GCP_REGION → the project the engines live in
   # AGENT_ENGINE_ID → the PRODUCTION engine (ignores sandbox configs)
   # SANDBOX_AGENT_ENGINE_ID → the sandbox engine; the default target when set
   # AGENT_ENGINE_TARGET → optional "sandbox" | "prod" to force the target
   # NEXT_PUBLIC_GOOGLE_CLIENT_ID → same client used by the GUB frontend
   ```
   The example file documents the optional list fallbacks
   (`SANDBOX_MODEL_ALLOWLIST`, `SANDBOX_THINKING_LEVEL_MODELS`, `GEMINI_MODEL`).

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

## Sandbox mode

The agent's per-call overrides (`state["sandbox"]`, `gub_agent/sandbox.py`,
gub-agent#30/#31) are honoured only by an engine deployed with
`SANDBOX_ENABLED=1` — the **sandbox engine** in `deploy-sandbox.env`. The
production engine ignores the key outright, with no warning, so the first
thing to check is the **engine badge** in the header.

### The engine badge

`GET /api/config` reports which engine this process addresses and reads the
engine resource itself (its baked-in env) when ADC allows
`aiplatform.reasoningEngines.get`:

- **green** `sandbox · gub-agent-sandbox · …939008 · SANDBOX_ENABLED=1` — experiments
  take effect here.
- **red, thick border, warning line** — `isProd`: the target is `prod`, or the id
  is the known production engine, or the engine's env says `SANDBOX_ENABLED=0`.
  The config is still sent (it is inert there) and every A/B would compare the
  baseline against itself. Fix `.env.local` before running.
- `SANDBOX_ENABLED unread` — the engine resource could not be read; prod
  detection then relies on the id alone.

Which engine: `AGENT_ENGINE_TARGET` if set, else the sandbox whenever
`SANDBOX_AGENT_ENGINE_ID` is set, else `AGENT_ENGINE_ID`. Defaulting to the
sandbox is deliberate — a silent prod default makes every experiment look like
a no-op.

### What can be turned

The collapsible **Sandbox config** panel above the input, options from
`/api/config` (nothing is hardcoded in the UI):

| knob | sent as | notes |
|---|---|---|
| model | `model` | `SANDBOX_MODEL_ALLOWLIST` of the engine. A model outside `SANDBOX_THINKING_LEVEL_MODELS` (e.g. `gemini-2.5-pro`) needs **DYNAMIC** on both roles — the panel says so and offers a one-click fix; the run is refused otherwise, same message as the engine's |
| thinking / critic thinking | `thinking_level`, `critic_thinking_level` | `MINIMAL LOW MEDIUM HIGH DYNAMIC`; defaults MEDIUM / LOW |
| temperature | `temperature` | 0.0–2.0, refused (not clamped) outside |
| executor / critic prompt | `*_variant` or `*_instruction` | a registry name (`baseline`, `v2_concise`, `v3_grounding` / `baseline`) or **inline text** (≤ 64 KB; `{braces}` are fine — the instruction is a callable; the date block is appended by the agent) |
| critic enabled | `critic_enabled` | measure the executor alone |
| label | `label` | free-form, provenance only |

"deployed default" / "deployed prompt" sends **no override** for that knob. The
config is persisted in `localStorage`.

Validation runs twice with the same code (`src/lib/sandbox.ts`): in the
browser as you type (the Run button is disabled with the message shown) and in
`POST /api/agent` (a 400). Left to the engine, an invalid override is an HTTP
200 with an empty stream — the route turns that shape into
`AGENT_EMPTY_STREAM` with the `gcloud logging read` line that shows the reason.

### Baseline, provenance and how a run is labelled

A run with **no** overrides is exactly a pre-sandbox run: the request omits the
`sandbox` key and `sandbox_echo` emits nothing. Such a run has no
`sandbox_resolved` and the UI labels it **baseline · deployed defaults**
(model / thinking / critic from `/api/config` `defaults`) — absence means
"baseline", not "broken".

A run with overrides carries `sandbox_resolved` (what actually ran: model,
thinking, temperature, prompt source + `sha256[:12]`, label, `overridden_keys`).
The chips under a trace and the A/B headers read **that**, never the form. The
registry's `baseline` variant is byte-identical to the deployed prompt but goes
through the sandbox path and gets a hash — pick it for the deliberate baseline
arm of an A/B so both arms are labelled alike.

Because the overrides are seeded into session state at creation, a changed
config cannot apply to an open session: in single-run mode the client opens a
new session for that run and says so; a request with both `sessionId` and a
non-empty `config` is refused (`CONFIG_NEEDS_NEW_SESSION`).

### A/B

Tick **A/B** in the header (off by default; single runs are unchanged):

- two config panels (**copy A → B**, then change one thing), one question,
  one **Run A/B**;
- each arm gets its **own fresh session** — reusing one would leak context from
  one run into the other and the comparison would be worthless — and both
  requests fire in parallel so GUB data cannot shift between them;
- the **diff line** names how B differs from A, computed from the two
  `sandbox_resolved` payloads (a side with none is the synthesised baseline);
  identical configs are flagged in amber, because a tie between identical arms
  measures only model variance;
- each column header: label / overridden keys, resolved chips, duration, answer
  word count, tool-call count, iteration count, critic verdict, session id,
  start time; the trace below is the same `AgentTrace` as single-run mode;
- a failed side shows its error; the other renders normally;
- follow-ups are single-run only — the aside says so instead of silently
  reusing a session.

### What duplicates the agent

There is no Python process to ask, so `src/lib/sandbox.ts` mirrors
`gub_agent/sandbox.py` and `gub_agent/prompts/variants/__init__.py`: the
thinking levels, the baseline levels, the 64 KB cap, the validation rules and
messages, the `config.py` defaults and — mirror-only, no env exists for them —
the **variant names per role**. The model allowlist, the named-thinking-level
models and the default model are taken from the engine's baked-in env when it
can be read, then `.env.local`, then the mirror. When a variant is added to the
registry, add it to `VARIANT_NAMES` here too; that is the seam that drifts
first.

## What to look for when iterating

- Wrong tool pick (e.g. listing + counting instead of `org_query` with a
  count aggregate) → fix in the executor prompt or the `org_query` docstring.
- A multi-entity question answered in one shot instead of chained queries →
  executor prompt.
- The critic passing a bad answer, or nitpicking a good one → critic prompt.
- A question you can't express at all → missing operator in GUB's catalog.
