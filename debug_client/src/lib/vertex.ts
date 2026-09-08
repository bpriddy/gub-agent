/**
 * vertex.ts — server-side Vertex AI Agent Engine client.
 *
 * Runs in the Next API routes (Node runtime). Auth via ADC
 * (`gcloud auth application-default login` locally). Requires the ADC
 * identity to have aiplatform.user on the project.
 *
 * Two engines exist since the sandbox epic (gub-agent#29): the production one
 * (`AGENT_ENGINE_ID`, `SANDBOX_ENABLED=0`, ignores `state.sandbox` outright)
 * and the sandbox one (`SANDBOX_AGENT_ENGINE_ID`). `activeEngine()` picks
 * which one every request addresses; the default is the sandbox whenever it
 * is configured, because an experiment silently sent to prod looks exactly
 * like a no-op.
 */
import { GoogleAuth } from 'google-auth-library';
import type { SandboxOverrides } from './sandbox';

const SCOPES = ['https://www.googleapis.com/auth/cloud-platform'];
const auth = new GoogleAuth({ scopes: SCOPES });

export type EngineTarget = 'sandbox' | 'prod';

export interface ActiveEngine {
  target: EngineTarget;
  id: string;
}

/**
 * The production engine id, mirrored from `deployment/deploy-sandbox.sh`
 * (PROD_ENGINE_ID). A last line of defence for the badge: if this id ends up
 * in SANDBOX_AGENT_ENGINE_ID by mistake, the UI still turns red.
 */
export const KNOWN_PROD_ENGINE_ID = '9136379226620952576';

/**
 * Which engine this process addresses.
 *
 *   AGENT_ENGINE_TARGET=sandbox|prod   explicit choice (the named id must be set)
 *   unset                              sandbox if SANDBOX_AGENT_ENGINE_ID is set, else prod
 */
export function activeEngine(): ActiveEngine {
  const prodId = process.env.AGENT_ENGINE_ID?.trim() || undefined;
  const sandboxId = process.env.SANDBOX_AGENT_ENGINE_ID?.trim() || undefined;
  const explicit = process.env.AGENT_ENGINE_TARGET?.trim().toLowerCase();

  if (explicit && explicit !== 'sandbox' && explicit !== 'prod') {
    throw new Error(`AGENT_ENGINE_TARGET must be "sandbox" or "prod", got "${explicit}"`);
  }
  const target: EngineTarget = explicit ? (explicit as EngineTarget) : sandboxId ? 'sandbox' : 'prod';
  const id = target === 'sandbox' ? sandboxId : prodId;
  if (!id) {
    throw new Error(
      target === 'sandbox'
        ? 'AGENT_ENGINE_TARGET=sandbox but SANDBOX_AGENT_ENGINE_ID is not set in .env.local'
        : 'Set AGENT_ENGINE_ID (or SANDBOX_AGENT_ENGINE_ID) in .env.local',
    );
  }
  return { target, id };
}

function projectRegion(): { project: string; region: string } {
  const project = process.env.GCP_PROJECT_ID;
  const region = process.env.GCP_REGION;
  if (!project || !region) {
    throw new Error('Set GCP_PROJECT_ID and GCP_REGION in .env.local');
  }
  return { project, region };
}

function engineBase(engine: ActiveEngine): string {
  const { project, region } = projectRegion();
  return (
    `https://${region}-aiplatform.googleapis.com/v1` +
    `/projects/${project}/locations/${region}/reasoningEngines/${engine.id}`
  );
}

async function bearer(): Promise<Record<string, string>> {
  const client = await auth.getClient();
  const tok = await client.getAccessToken();
  const token = typeof tok === 'string' ? tok : tok?.token;
  if (!token) throw new Error('Could not mint a Google access token via ADC');
  return { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
}

/**
 * Create a session. `sandbox` rides in the same state object as the JWT —
 * that is the whole transport for per-call overrides (`state["sandbox"]`,
 * gub_agent/sandbox.py). Omitted when null so a baseline request is
 * byte-identical to one made before the sandbox existed.
 */
export async function createSession(args: {
  userId: string;
  gubJwt: string;
  sandbox?: SandboxOverrides | null;
  engine?: ActiveEngine;
}): Promise<string> {
  const engine = args.engine ?? activeEngine();
  const res = await fetch(`${engineBase(engine)}:query`, {
    method: 'POST',
    headers: await bearer(),
    body: JSON.stringify({
      class_method: 'create_session',
      input: {
        user_id: args.userId,
        state: { gub_jwt: args.gubJwt, ...(args.sandbox ? { sandbox: args.sandbox } : {}) },
      },
    }),
  });
  if (!res.ok) {
    throw new Error(`create_session failed: ${res.status} ${(await res.text()).slice(0, 400)}`);
  }
  const data = (await res.json()) as { output?: { id?: string }; id?: string };
  const sessionId = data.output?.id ?? data.id;
  if (!sessionId) throw new Error(`create_session missing id: ${JSON.stringify(data).slice(0, 200)}`);
  return sessionId;
}

/** Stream the agent's response and collect all events into an array. */
export async function streamQueryCollect(args: {
  userId: string;
  sessionId: string;
  message: string;
  engine?: ActiveEngine;
}): Promise<unknown[]> {
  const engine = args.engine ?? activeEngine();
  const res = await fetch(`${engineBase(engine)}:streamQuery`, {
    method: 'POST',
    headers: await bearer(),
    body: JSON.stringify({
      class_method: 'stream_query',
      input: { user_id: args.userId, session_id: args.sessionId, message: args.message },
    }),
  });
  if (!res.ok || !res.body) {
    throw new Error(`stream_query failed: ${res.status} ${(await res.text()).slice(0, 400)}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const events: unknown[] = [];

  for (;;) {
    const { value, done } = await reader.read();
    if (value) buffer += decoder.decode(value, { stream: !done });
    if (done) break;
    let nl: number;
    while ((nl = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line) continue;
      try {
        events.push(JSON.parse(line));
      } catch {
        buffer = `${line}\n${buffer}`;
        break;
      }
    }
  }
  const tail = buffer.trim();
  if (tail) {
    try {
      const parsed = JSON.parse(tail) as unknown;
      if (Array.isArray(parsed)) events.push(...parsed);
      else events.push(parsed);
    } catch {
      /* drop */
    }
  }
  return events;
}

export interface EngineDescription {
  displayName: string | null;
  /** Deploy-time env baked into the engine (`spec.deploymentSpec.env`). */
  env: Record<string, string>;
}

/**
 * GET the engine resource: its display name and the env it was deployed with.
 * The two engines run the same code, so the baked-in env (`SANDBOX_ENABLED`,
 * `SANDBOX_MODEL_ALLOWLIST`, …) is the only ground truth about which one this
 * is — the README's "check the ID, not the behaviour" recipe, automated.
 * Best-effort: returns null when the resource cannot be read.
 */
export async function describeEngine(engine: ActiveEngine): Promise<EngineDescription | null> {
  try {
    const res = await fetch(engineBase(engine), {
      headers: await bearer(),
      signal: AbortSignal.timeout(8000),
    });
    if (!res.ok) return null;
    const data = (await res.json()) as {
      displayName?: string;
      spec?: { deploymentSpec?: { env?: Array<{ name?: string; value?: string }> } };
    };
    const env: Record<string, string> = {};
    for (const item of data.spec?.deploymentSpec?.env ?? []) {
      if (item.name && typeof item.value === 'string') env[item.name] = item.value;
    }
    return { displayName: data.displayName ?? null, env };
  } catch {
    return null;
  }
}
