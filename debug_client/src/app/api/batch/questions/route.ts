/**
 * GET/PUT /api/batch/questions — the question set (`scratchpad/questions.jsonl`).
 * PUT takes `{ text }` (raw JSONL) or `{ questions }` and refuses invalid input
 * with per-line problems instead of writing a file the runner would reject.
 */
import { NextResponse } from 'next/server';
import { parseQuestions, questionsPath, readQuestionsText, serializeQuestions, sha256, writeQuestionsText } from '@/lib/batch/store';
import type { Question } from '@/lib/batch/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function view(text: string) {
  const { questions, problems } = parseQuestions(text);
  return { text, questions, problems, sha256: sha256(text), path: questionsPath() };
}

export async function GET(): Promise<Response> {
  try {
    return NextResponse.json(view(readQuestionsText()));
  } catch (err) {
    return NextResponse.json({ code: 'STORE_ERROR', message: (err as Error).message }, { status: 500 });
  }
}

export async function PUT(req: Request): Promise<Response> {
  let body: { text?: string; questions?: Question[] };
  try { body = (await req.json()) as typeof body; } catch { return NextResponse.json({ code: 'BAD_JSON', message: 'Invalid JSON body.' }, { status: 400 }); }
  const text = typeof body.text === 'string' ? body.text : Array.isArray(body.questions) ? serializeQuestions(body.questions) : null;
  if (text === null) return NextResponse.json({ code: 'BAD_BODY', message: 'Send { text } (JSONL) or { questions }.' }, { status: 400 });
  const parsed = parseQuestions(text);
  if (parsed.problems.some((p) => p.severity === 'error')) {
    return NextResponse.json({ code: 'INVALID_QUESTIONS', message: 'The question set has errors; nothing was written.', problems: parsed.problems }, { status: 400 });
  }
  try {
    writeQuestionsText(text);
    return NextResponse.json(view(readQuestionsText()));
  } catch (err) {
    return NextResponse.json({ code: 'STORE_ERROR', message: (err as Error).message }, { status: 500 });
  }
}
