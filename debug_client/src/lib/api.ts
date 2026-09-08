/**
 * api.ts — browser-side types + calls for the two API routes.
 *
 * `AgentRunResponse` extends the `AgentResponse` that `AgentTrace.tsx`
 * renders (kept there untouched) with what the sandbox console needs on top:
 * provenance, the two summary counters, the engine, the start time.
 */
'use client';

import type { AgentResponse } from '@/components/AgentTrace';
import type { SandboxConfigResponse } from './sandboxServer';
import type { SandboxOverrides, SandboxResolved } from './sandbox';
import { getValidAccessToken } from './auth/tokenStore';

export type { SandboxConfigResponse };

export interface AgentRunResponse extends AgentResponse {
  startedAt: string;
  resolved: SandboxResolved | null;
  answerWordCount: number;
  toolCallCount: number;
  /** The overrides the server actually seeded (null = baseline request). */
  sandbox: SandboxOverrides | null;
  engine: SandboxConfigResponse['engine'];
}

export class ApiError extends Error {
  code: string;
  status: number;
  /** Present on AGENT_EMPTY_STREAM: the echo fired before the run died. */
  resolved: SandboxResolved | null;
  constructor(code: string, message: string, status: number, resolved: SandboxResolved | null = null) {
    super(message);
    this.code = code;
    this.status = status;
    this.resolved = resolved;
  }
}

export async function fetchConfig(): Promise<SandboxConfigResponse> {
  const res = await fetch('/api/config', { cache: 'no-store' });
  const data = (await res.json()) as SandboxConfigResponse & { code?: string; message?: string };
  if (!res.ok) throw new ApiError(data.code ?? 'CONFIG_ERROR', data.message ?? res.statusText, res.status);
  return data;
}

export async function runAgent(args: {
  message: string;
  sessionId?: string | null;
  includeRaw: boolean;
  config: SandboxOverrides | null;
}): Promise<AgentRunResponse> {
  const accessToken = await getValidAccessToken();
  const res = await fetch('/api/agent', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${accessToken}` },
    body: JSON.stringify({
      message: args.message,
      ...(args.sessionId ? { sessionId: args.sessionId } : { config: args.config ?? undefined }),
      includeRaw: args.includeRaw,
    }),
  });
  const data = (await res.json()) as AgentRunResponse & { code?: string; message?: string };
  if (!res.ok) {
    throw new ApiError(data.code ?? 'Error', data.message ?? res.statusText, res.status, data.resolved ?? null);
  }
  return data;
}
