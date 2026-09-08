/**
 * batch/engine.ts — Agent Engine transport for one cell: create_session with
 * `state.gub_jwt` (+ `state.sandbox`), then stream_query collected as events.
 *
 * Differs from `vertex.ts`'s helpers in two ways the batch needs: typed errors
 * carrying the HTTP status (auth vs transport vs other), and an AbortSignal
 * raced against every chunk so a stalled stream cannot outlive the cell budget.
 */
import { engineBaseUrl, type ActiveEngine } from '../vertex';

export type TransportKind = 'auth' | 'transport' | 'timeout' | 'error';

export class TransportError extends Error {
  kind: TransportKind;
  status: number | undefined;
  partialEvents: unknown[] | undefined;
  constructor(message: string, kind: TransportKind, status?: number, partialEvents?: unknown[]) {
    super(message);
    this.kind = kind;
    this.status = status;
    this.partialEvents = partialEvents;
  }
}

function classifyHttp(status: number, where: string, body: string): TransportError {
  if (status === 401 || status === 403) return new TransportError(`${where}: HTTP ${status} (Vertex rejected the ADC bearer) ${body}`, 'auth', status);
  if (status === 429 || status >= 500) return new TransportError(`${where}: HTTP ${status} ${body}`, 'transport', status);
  return new TransportError(`${where}: HTTP ${status} ${body}`, 'error', status);
}

const kindOf = (e: unknown): TransportKind => {
  const name = (e as { name?: string })?.name;
  return name === 'TimeoutError' || name === 'AbortError' ? 'timeout' : 'transport';
};

export async function createSession(args: {
  engine: ActiveEngine; headers: Record<string, string>; userId: string; state: Record<string, unknown>; signal: AbortSignal;
}): Promise<string> {
  let res: Response;
  try {
    res = await fetch(`${engineBaseUrl(args.engine)}:query`, {
      method: 'POST', headers: args.headers, signal: args.signal,
      body: JSON.stringify({ class_method: 'create_session', input: { user_id: args.userId, state: args.state } }),
    });
  } catch (e) {
    throw new TransportError(`create_session: ${(e as Error).name}: ${(e as Error).message}`, kindOf(e));
  }
  if (!res.ok) throw classifyHttp(res.status, 'create_session', (await res.text().catch(() => '')).slice(0, 300));
  const data = (await res.json()) as { output?: { id?: string }; id?: string };
  const id = data.output?.id ?? data.id;
  if (!id) throw new TransportError(`create_session: no session id in ${JSON.stringify(data).slice(0, 200)}`, 'error');
  return id;
}

/** NDJSON (HTTP 200, application/json), occasionally `data:`-prefixed — both handled. */
export async function streamQuery(args: {
  engine: ActiveEngine; headers: Record<string, string>; userId: string; sessionId: string; message: string; signal: AbortSignal;
}): Promise<{ events: unknown[]; firstEventMs: number | null }> {
  let res: Response;
  try {
    res = await fetch(`${engineBaseUrl(args.engine)}:streamQuery`, {
      method: 'POST', headers: args.headers, signal: args.signal,
      body: JSON.stringify({ class_method: 'stream_query', input: { user_id: args.userId, session_id: args.sessionId, message: args.message } }),
    });
  } catch (e) {
    throw new TransportError(`stream_query: ${(e as Error).name}: ${(e as Error).message}`, kindOf(e));
  }
  if (!res.ok || !res.body) throw classifyHttp(res.status, 'stream_query', (await res.text().catch(() => '')).slice(0, 300));

  const events: unknown[] = [];
  let firstEventMs: number | null = null;
  const t0 = Date.now();
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  const aborted = new Promise<never>((_, reject) => {
    const fail = () => reject(args.signal.reason ?? new DOMException('aborted', 'AbortError'));
    if (args.signal.aborted) fail(); else args.signal.addEventListener('abort', fail, { once: true });
  });
  aborted.catch(() => { /* observed through Promise.race */ });

  const handle = (raw: string): boolean => {
    let s = raw.trim();
    if (!s) return true;
    if (s.startsWith('data:')) s = s.slice(5).trim();
    if (!s || s === '[DONE]') return true;
    try {
      const parsed: unknown = JSON.parse(s);
      if (Array.isArray(parsed)) events.push(...parsed); else events.push(parsed);
      if (firstEventMs === null) firstEventMs = Date.now() - t0;
      return true;
    } catch {
      return false; // partial line — wait for the next chunk
    }
  };
  try {
    for (;;) {
      const { value, done } = await Promise.race([reader.read(), aborted]);
      if (value) buf += dec.decode(value, { stream: !done });
      let nl: number;
      while ((nl = buf.indexOf('\n')) >= 0) {
        if (!handle(buf.slice(0, nl))) break;
        buf = buf.slice(nl + 1);
      }
      if (done) break;
    }
  } catch (e) {
    reader.cancel().catch(() => {});
    throw new TransportError(`stream read: ${(e as Error).name}: ${(e as Error).message}`, kindOf(e), undefined, events);
  }
  if (buf.trim()) handle(buf);
  return { events, firstEventMs };
}
