/**
 * GET /api/batch/runs/[id] — a live job (meta, progress, records so far, live
 * summary, log) or a finished run read back from disk. When the run is
 * `run_as: me`, an Authorization header on the poll hands the job the browser's
 * freshest GUB token — that is how a long run outlives the 15-minute JWT.
 */
import { NextResponse } from 'next/server';
import { runView, updateMeToken } from '@/lib/batch/runner';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(req: Request, { params }: { params: { id: string } }): Promise<Response> {
  try {
    const jwt = (req.headers.get('authorization') ?? '').replace(/^Bearer\s+/i, '').trim();
    if (jwt) updateMeToken(params.id, jwt);
    const view = runView(params.id);
    if (!view) return NextResponse.json({ code: 'NOT_FOUND', message: `no run ${params.id}` }, { status: 404 });
    return NextResponse.json(view);
  } catch (err) {
    return NextResponse.json({ code: 'RUN_ERROR', message: (err as Error).message }, { status: 500 });
  }
}
