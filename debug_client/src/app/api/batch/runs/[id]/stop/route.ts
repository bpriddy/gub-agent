/** POST /api/batch/runs/[id]/stop — stop scheduling new cells; in-flight ones finish. */
import { NextResponse } from 'next/server';
import { requestStop } from '@/lib/batch/runner';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(_req: Request, { params }: { params: { id: string } }): Promise<Response> {
  const ok = requestStop(params.id);
  return NextResponse.json({ stopped: ok }, { status: ok ? 200 : 409 });
}
