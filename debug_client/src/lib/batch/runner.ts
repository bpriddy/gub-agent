/**
 * batch/runner.ts — the matrix executor behind POST /api/batch/runs.
 *
 * One JOB per run, kept in a module registry on `globalThis` (survives Next's
 * dev-time module reloads) while the browser polls. Semantics match the
 * headless twin (`scratchpad/batch-run.mjs`): question-major order, a fresh
 * session per cell, N workers, per-cell timeout, one retry on transport
 * errors, a shared cooldown + one retry when GUB rate-limits a cell, stop
 * after 3 consecutive auth failures, records appended to raw.jsonl as they
 * land, summary.csv written at the end.
 */
import { activeEngine, describeEngine, vertexHeaders, type ActiveEngine } from '../vertex';
import { sandboxConfig } from '../sandboxServer';
import { createSession, streamQuery, TransportError } from './engine';
import { analyze, analyzeResolved, judgeExpect, regressions, summarizeRun, summaryCsv } from './metrics';
import { JwtExpiredError, MeTokenSource, SubjectTokenSource, type TokenSource } from './auth';
import { appendRecord, ensureDirs, newRunId, readRecords, readRunMeta, writeRunMeta, writeSummaryCsv } from './store';
import type { Arm, CellRecord, Question, RunAs, RunMeta, RunProgress, RunSummary, RunView } from './types';

const MAX_CONSECUTIVE_AUTH_FAILURES = 3;
const USER_ID = 'batch-runner';

interface Job {
  meta: RunMeta;
  questions: Question[];
  arms: Arm[];
  records: CellRecord[];
  progress: RunProgress;
  log: string[];
  stopRequested: boolean;
  stopReason: string | null;
  tokenSource: TokenSource;
  cooldownUntil: number;
  done: Promise<void>;
}

type Registry = Map<string, Job>;
const g = globalThis as unknown as { __gubBatchJobs?: Registry };
const registry: Registry = (g.__gubBatchJobs ??= new Map());

export const liveRunIds = (): Set<string> => new Set([...registry.values()].filter((j) => j.meta.status === 'running').map((j) => j.meta.run_id));
export const getJob = (id: string): Job | undefined => registry.get(id);

export interface StartOptions {
  questions: Question[];
  questionsText: string;
  questionsFile: string;
  arms: Arm[];
  parallel: number;
  timeoutS: number;
  cooldownS: number;
  runAs: RunAs;
  /** The browser's GUB token when runAs = 'me'. */
  meJwt?: string | null;
}

export async function startRun(opts: StartOptions): Promise<RunMeta> {
  if (!opts.questions.length) throw new Error('no questions selected');
  if (!opts.arms.length) throw new Error('no arms selected');
  if (liveRunIds().size > 0) throw new Error('a batch run is already in progress — one at a time: the GUB rate bucket is shared');

  const cfg = await sandboxConfig();
  const engine: ActiveEngine = activeEngine();
  if (cfg.engine.isProd) throw new Error(`refusing to run a batch against the production engine (${engine.id}) — epic invariant 3`);
  if (cfg.engine.sandboxEnabled === false) throw new Error(`engine ${engine.id} is deployed with SANDBOX_ENABLED=0 — every arm would silently measure the baseline`);

  const tokenSource: TokenSource = opts.runAs === 'subject'
    ? SubjectTokenSource.fromEnv()
    : new MeTokenSource((() => { if (!opts.meJwt) throw new Error('run_as=me needs the signed-in user\'s token in the Authorization header'); return opts.meJwt; })());
  // Fail early on a subject that cannot mint, before a run directory exists.
  await tokenSource.token();

  ensureDirs();
  const headers = await vertexHeaders();
  const described = await describeEngine(engine);
  const { sha256 } = await import('./store');
  const runId = newRunId();
  const meta: RunMeta = {
    run_id: runId, source: 'ui', started_at: new Date().toISOString(), status: 'running', stop_reason: null,
    engine_id: engine.id, engine: described ? { displayName: described.displayName, env: described.env } : null,
    questions_file: opts.questionsFile, questions_sha256: sha256(opts.questionsText), n_questions: opts.questions.length,
    question_ids: opts.questions.map((q) => q.id), configs: opts.arms, parallel: opts.parallel, timeout_s: opts.timeoutS, cooldown_s: opts.cooldownS,
    run_as: opts.runAs, gub_subject: tokenSource.subject(), node: process.version,
  };
  writeRunMeta(meta);

  const cells: Array<{ q: Question; arm: Arm }> = [];
  for (const q of opts.questions) for (const arm of opts.arms) cells.push({ q, arm });

  const job: Job = {
    meta, questions: opts.questions, arms: opts.arms, records: [], log: [], stopRequested: false, stopReason: null, tokenSource, cooldownUntil: 0,
    progress: { total: cells.length, completed: 0, running: [], cooldowns: 0, cooldown_until: null, rate_limited: 0, consecutive_auth: 0, jwt_seconds_left: tokenSource.secondsLeft() },
    done: Promise.resolve(),
  };
  registry.set(runId, job);
  job.done = execute(job, cells, engine, headers).catch((e) => {
    job.log.push(`run crashed: ${(e as Error).stack ?? (e as Error).message}`);
    job.meta.status = 'failed';
    job.meta.stop_reason = (e as Error).message;
  }).finally(async () => {
    job.meta.finished_at = new Date().toISOString();
    if (job.meta.status === 'running') job.meta.status = job.stopRequested || job.stopReason ? 'stopped' : 'done';
    if (job.stopReason) job.meta.stop_reason = job.stopReason;
    finalizeSummary(job);
    writeRunMeta(job.meta);
    await job.tokenSource.close();
    if (job.tokenSource instanceof SubjectTokenSource) job.log.push(...job.tokenSource.log.splice(0));
  });
  return meta;
}

export function requestStop(id: string): boolean {
  const job = registry.get(id);
  if (!job || job.meta.status !== 'running') return false;
  job.stopRequested = true;
  job.stopReason = 'stopped from the UI — cells already in flight finished, the rest were not started';
  job.log.push(job.stopReason);
  return true;
}

/** The browser pushes its freshest token on every poll (run_as = me). */
export function updateMeToken(id: string, jwt: string): void {
  const job = registry.get(id);
  if (job && job.tokenSource instanceof MeTokenSource) job.tokenSource.update(jwt);
}

function finalizeSummary(job: Job): void {
  const names = job.arms.map((a) => a.name);
  const { rows, byKind } = summarizeRun(job.records, names);
  writeSummaryCsv(job.meta.run_id, summaryCsv([...rows, ...byKind], { run_id: job.meta.run_id, questions_sha256: job.meta.questions_sha256, questions_file: job.meta.questions_file, engine_id: job.meta.engine_id }));
}

export function buildSummary(records: CellRecord[], arms: Arm[] | string[]): RunSummary {
  const names = arms.map((a) => (typeof a === 'string' ? a : a.name));
  const { rows, byKind } = summarizeRun(records, names);
  const baseline = names.includes('baseline') ? 'baseline' : names[0] ?? 'baseline';
  return { rows, byKind, regressions: regressions(records, names, baseline) };
}

/** Live job or a finished run read back from disk. */
export function runView(id: string): RunView | null {
  const job = registry.get(id);
  if (job) {
    if (job.tokenSource instanceof SubjectTokenSource && job.tokenSource.log.length) job.log.push(...job.tokenSource.log.splice(0));
    job.progress.jwt_seconds_left = job.tokenSource.secondsLeft();
    job.progress.cooldown_until = job.cooldownUntil > Date.now() ? job.cooldownUntil : null;
    return { meta: job.meta, progress: job.meta.status === 'running' ? job.progress : null, records: job.records, summary: buildSummary(job.records, job.arms), log: job.log };
  }
  const meta = readRunMeta(id);
  if (!meta) return null;
  const records = readRecords(id);
  return { meta, progress: null, records, summary: buildSummary(records, meta.configs.length ? meta.configs : [...new Set(records.map((r) => r.config))]), log: [] };
}

// ── Execution ─────────────────────────────────────────────────────────────────

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function startCooldown(job: Job, why: string): void {
  const until = Date.now() + job.meta.cooldown_s * 1000;
  if (until > job.cooldownUntil) {
    job.cooldownUntil = until;
    job.progress.cooldowns++;
    job.log.push(`cooldown: ${why} — pausing all workers ${job.meta.cooldown_s}s so the GUB rate bucket can refill`);
  }
}
async function waitCooldown(job: Job): Promise<void> {
  while (job.cooldownUntil > Date.now() && !job.stopRequested) await sleep(Math.min(1000, job.cooldownUntil - Date.now()));
}

async function execute(job: Job, cells: Array<{ q: Question; arm: Arm }>, engine: ActiveEngine, headers: Record<string, string>): Promise<void> {
  let next = 0;
  let consecutiveAuth = 0;
  const worker = async () => {
    while (!job.stopRequested && !job.stopReason && next < cells.length) {
      const cell = cells[next++]!;
      const key = `${cell.q.id} × ${cell.arm.name}`;
      job.progress.running.push(key);
      let record: CellRecord;
      try {
        record = await runCell(job, cell.q, cell.arm, engine, headers, 1);
      } finally {
        job.progress.running = job.progress.running.filter((k) => k !== key);
      }
      job.records.push(record);
      appendRecord(job.meta.run_id, record);
      job.progress.completed++;
      if (record.status === 'rate_limited') job.progress.rate_limited++;
      if (record.status === 'auth') {
        consecutiveAuth++;
        job.progress.consecutive_auth = consecutiveAuth;
        if (record.auth_scope === 'gub_jwt' && job.tokenSource.canRenew() && consecutiveAuth < MAX_CONSECUTIVE_AUTH_FAILURES) {
          try { await job.tokenSource.renew(); } catch (e) { job.log.push(`token renewal failed — ${(e as Error).message}`); }
        }
        if (consecutiveAuth >= MAX_CONSECUTIVE_AUTH_FAILURES) {
          job.stopReason = record.auth_scope === 'gtoken'
            ? `${MAX_CONSECUTIVE_AUTH_FAILURES} consecutive Vertex auth failures — the server's ADC bearer was rejected; run gcloud auth application-default login and start again`
            : job.tokenSource.kind === 'me'
              ? `${MAX_CONSECUTIVE_AUTH_FAILURES} consecutive GUB auth failures — the signed-in user's token expired and no fresh one arrived (keep the page open, or run as the sandbox subject)`
              : `${MAX_CONSECUTIVE_AUTH_FAILURES} consecutive GUB auth failures for the sandbox subject — check its GUB user (disabled? no grants?) and the exchange`;
          job.log.push(`STOPPING: ${job.stopReason}`);
        }
      } else if (record.status === 'ok') {
        consecutiveAuth = 0;
        job.progress.consecutive_auth = 0;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(job.meta.parallel, cells.length) }, worker));
}

async function runCell(job: Job, q: Question, arm: Arm, engine: ActiveEngine, headers: Record<string, string>, attempt: number): Promise<CellRecord> {
  await waitCooldown(job);
  const started = Date.now();
  const base: Omit<CellRecord, 'status' | 'latency_ms'> = {
    run_id: job.meta.run_id, question_id: q.id, kind: q.kind, config: arm.name, attempt, started_at: new Date(started).toISOString(), engine_id: engine.id,
  };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new DOMException(`cell exceeded ${job.meta.timeout_s}s`, 'TimeoutError')), job.meta.timeout_s * 1000);

  let jwt: string;
  try {
    jwt = await job.tokenSource.token();
  } catch (e) {
    clearTimeout(timer);
    return { ...base, status: 'auth', auth_scope: 'gub_jwt', error: (e as Error).message, latency_ms: Date.now() - started };
  }
  const state: Record<string, unknown> = { gub_jwt: jwt };
  const overrides = arm.overrides as Record<string, unknown>;
  if (Object.keys(overrides).length > 0) {
    // Every experiment arm carries a label (invariant 10); the baseline arm stays
    // EMPTY so it remains an ordinary run with no sandbox_resolved.
    state.sandbox = { ...overrides, label: overrides.label ?? `batch:${job.meta.run_id}:${arm.name}` };
  }

  try {
    const sessionId = await createSession({ engine, headers, userId: USER_ID, state, signal: controller.signal });
    const { events, firstEventMs } = await streamQuery({ engine, headers, userId: USER_ID, sessionId, message: q.q, signal: controller.signal });
    const m = analyze(events);
    const resolved = analyzeResolved(events);
    const latency = Date.now() - started;
    const judged = judgeExpect(q, m);
    let status: CellRecord['status'] = 'ok';
    if (m.auth_error) status = 'auth';
    else if (m.rate_limited) {
      if (attempt === 1) {
        startCooldown(job, `${q.id} × ${arm.name}: ${m.tool_errors_429} of ${m.tool_calls} tool call(s) got HTTP 429 from GUB`);
        clearTimeout(timer);
        await waitCooldown(job);
        if (job.stopRequested) return { ...base, ...m, ...judged, status: 'rate_limited', latency_ms: latency, sandbox_resolved: resolved, error: 'rate-limited, and the run was stopped before the retry' };
        return runCell(job, q, arm, engine, headers, 2);
      }
      status = 'rate_limited';
    } else if (!m.answer) status = 'empty';
    const record: CellRecord = {
      ...base, status, session_id: sessionId, latency_ms: latency, first_event_ms: firstEventMs, ...m, ...judged,
      sandbox_requested: (state.sandbox as CellRecord['sandbox_requested']) ?? null, sandbox_resolved: resolved, expect: q.expect ?? null,
    };
    if (status === 'auth') { record.auth_scope = 'gub_jwt'; record.error = 'a tool result carried HTTP 401 — GUB rejected the JWT'; }
    if (status === 'rate_limited') record.error = `${m.tool_errors_429} of ${m.tool_calls} tool call(s) got HTTP 429 from GUB even after a ${job.meta.cooldown_s}s cooldown — the per-subject rate bucket is exhausted`;
    if (status === 'empty') record.error = `${events.length} event(s) and no executor text — how a run that failed inside the engine looks (invalid override, unknown variant, unservable model); the reason is only in the engine logs`;
    return record;
  } catch (e) {
    const latency = Date.now() - started;
    if (e instanceof TransportError) {
      if (e.kind === 'timeout') return { ...base, status: 'timeout', latency_ms: latency, error: e.message, events: e.partialEvents?.length ?? 0 };
      if (e.kind === 'auth') return { ...base, status: 'auth', auth_scope: 'gtoken', latency_ms: latency, error: e.message };
      if (e.kind === 'transport' && attempt === 1 && !job.stopRequested) {
        job.log.push(`retry: ${q.id} × ${arm.name} — ${e.message.slice(0, 140)}`);
        clearTimeout(timer);
        await sleep(e.status === 429 ? 5000 : 2000);
        return runCell(job, q, arm, engine, headers, 2);
      }
      return { ...base, status: e.kind === 'transport' ? 'transport' : 'error', latency_ms: latency, error: e.message };
    }
    if (e instanceof JwtExpiredError) return { ...base, status: 'auth', auth_scope: 'gub_jwt', latency_ms: latency, error: e.message };
    return { ...base, status: 'error', latency_ms: latency, error: `${(e as Error).name}: ${(e as Error).message}` };
  } finally {
    clearTimeout(timer);
  }
}
