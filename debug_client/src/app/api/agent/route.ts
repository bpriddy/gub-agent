/**
 * POST /api/agent — server-side proxy to Vertex AI Agent Engine.
 *
 * Holds NO secrets. Uses ADC for the Vertex AI bearer (server's own
 * `gcloud auth application-default login`), and reads the caller's GUB JWT
 * from the Authorization header to seed `state.gub_jwt` — so the agent's
 * tools run as the signed-in user. Returns a structured trace.
 *
 * Node runtime required (google-auth-library is not Edge-compatible).
 */
import { NextResponse } from 'next/server';
import { createSession, streamQueryCollect } from '@/lib/vertex';
import { buildTrace } from '@/lib/trace';

export const runtime = 'nodejs';

interface Body {
  message?: string;
  sessionId?: string;
  includeRaw?: boolean;
}

/** Decode a JWT payload without verifying — only to label the Vertex
 *  session with a stable user id. The JWT itself is the auth, seeded into
 *  state and verified downstream by GUB. */
function jwtSub(jwt: string): string {
  try {
    const payload = jwt.split('.')[1];
    if (!payload) return 'debug-user';
    const json = JSON.parse(Buffer.from(payload, 'base64url').toString('utf-8')) as { sub?: string; email?: string };
    return json.sub ?? json.email ?? 'debug-user';
  } catch {
    return 'debug-user';
  }
}

export async function POST(req: Request): Promise<Response> {
  const auth = req.headers.get('authorization') ?? '';
  const gubJwt = auth.replace(/^Bearer\s+/i, '').trim();
  if (!gubJwt) {
    return NextResponse.json({ code: 'NO_TOKEN', message: 'Missing Authorization bearer token.' }, { status: 401 });
  }

  let body: Body;
  try {
    body = (await req.json()) as Body;
  } catch {
    return NextResponse.json({ code: 'BAD_JSON', message: 'Invalid JSON body.' }, { status: 400 });
  }
  const message = (body.message ?? '').trim();
  if (!message) {
    return NextResponse.json({ code: 'NO_MESSAGE', message: 'message is required.' }, { status: 400 });
  }

  const userId = jwtSub(gubJwt);
  const started = Date.now();

  try {
    const sessionId = body.sessionId ?? (await createSession({ userId, gubJwt }));
    const events = await streamQueryCollect({ userId, sessionId, message });
    const trace = buildTrace(events);

    return NextResponse.json({
      text: trace.text,
      sessionId,
      durationMs: Date.now() - started,
      iterations: trace.iterations,
      sources: trace.sources,
      ...(body.includeRaw ? { rawEvents: events } : {}),
    });
  } catch (err) {
    return NextResponse.json(
      { code: 'AGENT_ERROR', message: (err as Error).message },
      { status: 502 },
    );
  }
}
