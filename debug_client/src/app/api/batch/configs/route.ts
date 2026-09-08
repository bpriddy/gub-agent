/**
 * GET/PUT /api/batch/configs — the arms (`scratchpad/configs.json`). PUT
 * validates every arm's overrides against the active engine's lists with the
 * same validator the single-run route uses, so a bad model/thinking pair is a
 * 400 here rather than an arm of empty streams later.
 */
import { NextResponse } from 'next/server';
import { configsPath, parseArms, readArmsText, writeArms } from '@/lib/batch/store';
import { sandboxLists } from '@/lib/sandboxServer';
import { validateOverrides } from '@/lib/sandbox';
import type { Arm } from '@/lib/batch/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(): Promise<Response> {
  try {
    const text = readArmsText();
    const { arms, problems } = parseArms(text);
    return NextResponse.json({ arms, problems, path: configsPath() });
  } catch (err) {
    return NextResponse.json({ code: 'STORE_ERROR', message: (err as Error).message }, { status: 500 });
  }
}

export async function PUT(req: Request): Promise<Response> {
  let body: { arms?: unknown };
  try { body = (await req.json()) as typeof body; } catch { return NextResponse.json({ code: 'BAD_JSON', message: 'Invalid JSON body.' }, { status: 400 }); }
  const { arms, problems } = parseArms(JSON.stringify(body.arms ?? null));
  if (problems.length) return NextResponse.json({ code: 'INVALID_ARMS', message: problems.map((p) => p.message).join('; '), problems }, { status: 400 });
  const lists = await sandboxLists();
  const normalised: Arm[] = [];
  for (const arm of arms) {
    const check = validateOverrides(arm.overrides, lists);
    if (!check.ok) return NextResponse.json({ code: 'BAD_CONFIG', message: `arm "${arm.name}": ${check.message}`, problems: [{ line: 0, id: arm.name, message: check.message, severity: 'error' }] }, { status: 400 });
    normalised.push({ name: arm.name, overrides: check.value ?? {}, ...(arm.enabled === false ? { enabled: false } : {}) });
  }
  try {
    writeArms(normalised);
    return NextResponse.json({ arms: normalised, problems: [], path: configsPath() });
  } catch (err) {
    return NextResponse.json({ code: 'STORE_ERROR', message: (err as Error).message }, { status: 500 });
  }
}
