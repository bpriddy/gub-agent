/**
 * POST /api/agent — server-side proxy to Vertex AI Agent Engine.
 *
 * Holds NO secrets. Uses ADC for the Vertex AI bearer (server's own
 * `gcloud auth application-default login`), and reads the caller's GUB JWT
 * from the Authorization header to seed `state.gub_jwt` — so the agent's
 * tools run as the signed-in user. Returns a structured trace.
 *
 * `config` (a `SandboxOverrides`, gub_agent/sandbox.py) is validated here
 * against the active engine's lists and seeded as `state.sandbox` when a NEW
 * session is created — overrides live in session state, so they cannot be
 * changed on an existing session (the request is refused rather than
 * silently ignored). An invalid value is a 400 with the engine's own wording;
 * left to the engine it would be an HTTP 200 with an empty stream.
 *
 * Node runtime required (google-auth-library is not Edge-compatible).
 */
import { NextResponse } from 'next/server';
import { activeEngine, createSession, streamQueryCollect } from '@/lib/vertex';
import { buildTrace } from '@/lib/trace';
import { validateOverrides } from '@/lib/sandbox';
import { sandboxConfig, sandboxLists } from '@/lib/sandboxServer';

export const runtime = 'nodejs';

interface Body {
  message?: string;
  sessionId?: string;
  includeRaw?: boolean;
  /** SandboxOverrides — applied only when no sessionId is given. */
  config?: unknown;
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

  let engine;
  try {
    engine = activeEngine();
  } catch (err) {
    return NextResponse.json({ code: 'ENGINE_CONFIG', message: (err as Error).message }, { status: 500 });
  }

  const check = validateOverrides(body.config, await sandboxLists());
  if (!check.ok) {
    return NextResponse.json({ code: 'BAD_CONFIG', message: check.message }, { status: 400 });
  }
  const sandbox = check.value;
  if (sandbox && body.sessionId) {
    return NextResponse.json(
      {
        code: 'CONFIG_NEEDS_NEW_SESSION',
        message: 'A sandbox config is seeded into session state at creation and cannot be applied to an existing session. Omit sessionId to start a new one.',
      },
      { status: 400 },
    );
  }

  const userId = jwtSub(gubJwt);
  const started = Date.now();

  try {
    const sessionId = body.sessionId ?? (await createSession({ userId, gubJwt, sandbox, engine }));
    const events = await streamQueryCollect({ userId, sessionId, message, engine });
    const trace = buildTrace(events);
    const { engine: engineInfo } = await sandboxConfig();

    if (trace.iterations.length === 0) {
      // Agent Engine turns an exception inside the run into HTTP 200 + an
      // empty stream (gub-agent README, "How a failed sandbox run looks").
      // The echo may have fired first; surface it so the caller sees the run
      // was accepted and then died.
      return NextResponse.json(
        {
          code: 'AGENT_EMPTY_STREAM',
          message:
            `The engine returned ${events.length} event(s) and no executor output. ` +
            'This is how a run that failed inside the engine looks (invalid override, unknown variant, a model the project cannot serve). ' +
            `Read the reason from the engine logs: gcloud logging read 'resource.type="aiplatform.googleapis.com/ReasoningEngine" AND resource.labels.reasoning_engine_id="${engine.id}" AND severity>=ERROR' --limit=5 --freshness=30m`,
          sessionId,
          resolved: trace.resolved,
          engine: engineInfo,
          ...(body.includeRaw ? { rawEvents: events } : {}),
        },
        { status: 502 },
      );
    }

    return NextResponse.json({
      text: trace.text,
      sessionId,
      durationMs: Date.now() - started,
      startedAt: new Date(started).toISOString(),
      iterations: trace.iterations,
      sources: trace.sources,
      resolved: trace.resolved,
      answerWordCount: trace.answerWordCount,
      toolCallCount: trace.toolCallCount,
      sandbox,
      engine: engineInfo,
      ...(body.includeRaw ? { rawEvents: events } : {}),
    });
  } catch (err) {
    return NextResponse.json(
      { code: 'AGENT_ERROR', message: (err as Error).message },
      { status: 502 },
    );
  }
}
