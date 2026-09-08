/** GET /api/batch/subject — is a dedicated sandbox subject configured, and can this server mint for it? */
import { NextResponse } from 'next/server';
import { probeSubject } from '@/lib/batch/auth';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(): Promise<Response> {
  return NextResponse.json(await probeSubject());
}
