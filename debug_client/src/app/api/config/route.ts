/**
 * GET /api/config — what the active engine accepts, so the UI hardcodes
 * nothing: model allowlist, which models take a named thinking level, the
 * variant names per role, the deployed defaults, and the engine itself
 * (id, target, `isProd`).
 *
 * Lists come from the engine's baked-in env when it can be read, else from
 * .env.local, else from the mirrored config.py defaults — see
 * `lib/sandboxServer.ts` for the precedence and `lib/sandbox.ts` for the
 * duplication seam.
 */
import { NextResponse } from 'next/server';
import { sandboxConfig } from '@/lib/sandboxServer';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(): Promise<Response> {
  try {
    return NextResponse.json(await sandboxConfig());
  } catch (err) {
    return NextResponse.json({ code: 'CONFIG_ERROR', message: (err as Error).message }, { status: 500 });
  }
}
