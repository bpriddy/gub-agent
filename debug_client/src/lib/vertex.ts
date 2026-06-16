/**
 * vertex.ts — server-side Vertex AI Agent Engine client.
 *
 * Runs in the Next /api/agent route (Node runtime). Auth via ADC
 * (`gcloud auth application-default login` locally). Requires the ADC
 * identity to have aiplatform.user on the project.
 */
import { GoogleAuth } from 'google-auth-library';

const SCOPES = ['https://www.googleapis.com/auth/cloud-platform'];
const auth = new GoogleAuth({ scopes: SCOPES });

function engineBase(): string {
  const project = process.env.GCP_PROJECT_ID;
  const region = process.env.GCP_REGION;
  const engineId = process.env.AGENT_ENGINE_ID;
  if (!project || !region || !engineId) {
    throw new Error('Set GCP_PROJECT_ID, GCP_REGION, and AGENT_ENGINE_ID in .env.local');
  }
  return (
    `https://${region}-aiplatform.googleapis.com/v1` +
    `/projects/${project}/locations/${region}/reasoningEngines/${engineId}`
  );
}

async function bearer(): Promise<Record<string, string>> {
  const client = await auth.getClient();
  const tok = await client.getAccessToken();
  const token = typeof tok === 'string' ? tok : tok?.token;
  if (!token) throw new Error('Could not mint a Google access token via ADC');
  return { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
}

export async function createSession(args: { userId: string; gubJwt: string }): Promise<string> {
  const res = await fetch(`${engineBase()}:query`, {
    method: 'POST',
    headers: await bearer(),
    body: JSON.stringify({
      class_method: 'create_session',
      input: { user_id: args.userId, state: { gub_jwt: args.gubJwt } },
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
}): Promise<unknown[]> {
  const res = await fetch(`${engineBase()}:streamQuery`, {
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
