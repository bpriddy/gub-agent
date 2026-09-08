/**
 * GET  /api/batch/runs — every run under scratchpad/runs (UI and CLI alike).
 * POST /api/batch/runs — start a run. Body: { questionIds?, arms?, parallel?,
 *   timeout?, cooldown?, runAs }. With runAs = 'me' the Authorization header
 *   carries the signed-in user's GUB token; with 'subject' the server mints
 *   the dedicated sandbox subject's token itself. Answers 202 + the run meta;
 *   progress is read from GET /api/batch/runs/[id].
 */
import { NextResponse } from 'next/server';
import { parseArms, parseQuestions, questionsPath, readArmsText, readQuestionsText, listRuns } from '@/lib/batch/store';
import { liveRunIds, startRun } from '@/lib/batch/runner';
import type { RunAs } from '@/lib/batch/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(): Promise<Response> {
  try {
    return NextResponse.json({ runs: listRuns(liveRunIds()) });
  } catch (err) {
    return NextResponse.json({ code: 'STORE_ERROR', message: (err as Error).message }, { status: 500 });
  }
}

interface StartBody { questionIds?: string[]; arms?: string[]; parallel?: number; timeout?: number; cooldown?: number; runAs?: RunAs }

export async function POST(req: Request): Promise<Response> {
  let body: StartBody;
  try { body = (await req.json()) as StartBody; } catch { return NextResponse.json({ code: 'BAD_JSON', message: 'Invalid JSON body.' }, { status: 400 }); }
  const runAs: RunAs = body.runAs === 'subject' ? 'subject' : 'me';
  const meJwt = (req.headers.get('authorization') ?? '').replace(/^Bearer\s+/i, '').trim() || null;
  const parallel = clampInt(body.parallel, 1, 6, 2);
  const timeout = clampInt(body.timeout, 30, 600, 180);
  const cooldown = clampInt(body.cooldown, 0, 900, 60);

  const questionsText = readQuestionsText();
  const parsedQ = parseQuestions(questionsText);
  if (parsedQ.problems.some((p) => p.severity === 'error')) return NextResponse.json({ code: 'INVALID_QUESTIONS', message: 'Fix the question set first.', problems: parsedQ.problems }, { status: 400 });
  const wanted = body.questionIds ? new Set(body.questionIds) : null;
  const questions = wanted ? parsedQ.questions.filter((q) => wanted.has(q.id)) : parsedQ.questions;

  const parsedA = parseArms(readArmsText());
  if (parsedA.problems.length) return NextResponse.json({ code: 'INVALID_ARMS', message: 'Fix the arms first.', problems: parsedA.problems }, { status: 400 });
  const wantedArms = body.arms ? new Set(body.arms) : null;
  const arms = parsedA.arms.filter((a) => (wantedArms ? wantedArms.has(a.name) : a.enabled !== false));

  try {
    const meta = await startRun({ questions, questionsText, questionsFile: questionsPath(), arms, parallel, timeoutS: timeout, cooldownS: cooldown, runAs, meJwt });
    return NextResponse.json(meta, { status: 202 });
  } catch (err) {
    return NextResponse.json({ code: 'START_FAILED', message: (err as Error).message }, { status: 409 });
  }
}

function clampInt(v: unknown, min: number, max: number, dflt: number): number {
  const n = typeof v === 'number' ? v : typeof v === 'string' ? Number(v) : NaN;
  if (!Number.isFinite(n)) return dflt;
  return Math.min(max, Math.max(min, Math.round(n)));
}
