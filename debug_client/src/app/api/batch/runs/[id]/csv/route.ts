/** GET /api/batch/runs/[id]/csv — summary.csv (recomputed live for a running job). */
import { NextResponse } from 'next/server';
import { runView } from '@/lib/batch/runner';
import { summaryCsv } from '@/lib/batch/metrics';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(_req: Request, { params }: { params: { id: string } }): Promise<Response> {
  const view = runView(params.id);
  if (!view) return NextResponse.json({ code: 'NOT_FOUND', message: `no run ${params.id}` }, { status: 404 });
  const csv = summaryCsv([...view.summary.rows, ...view.summary.byKind], { run_id: view.meta.run_id, questions_sha256: view.meta.questions_sha256, questions_file: view.meta.questions_file, engine_id: view.meta.engine_id });
  return new Response(csv, { headers: { 'Content-Type': 'text/csv; charset=utf-8', 'Content-Disposition': `attachment; filename="summary-${params.id}.csv"` } });
}
