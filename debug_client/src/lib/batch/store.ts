/**
 * batch/store.ts — server-only file storage for question sets, arms and runs.
 *
 * Everything lives under BATCH_DIR, which defaults to `gub-agent/scratchpad/`
 * — the SAME files the headless runner uses (`questions.jsonl`, `configs.json`,
 * `runs/<id>/`). Epic invariant 7: question sets, run outputs and local scripts
 * stay in scratchpad, uncommitted; `runs/` ignores itself (answers carry client
 * data) and no token is ever written here.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { KINDS, type Arm, type CellRecord, type Question, type RunListItem, type RunMeta } from './types';
import { vacuousExpectItems } from './metrics';

export function batchDir(): string {
  const configured = process.env.BATCH_DIR?.trim();
  return path.resolve(configured || path.join(process.cwd(), '..', 'scratchpad'));
}
export const questionsPath = () => path.join(batchDir(), 'questions.jsonl');
export const configsPath = () => path.join(batchDir(), 'configs.json');
export const runsDir = () => path.join(batchDir(), 'runs');
export const runDir = (runId: string) => {
  if (!/^[A-Za-z0-9_-]{4,64}$/.test(runId)) throw new Error(`invalid run id ${JSON.stringify(runId)}`);
  return path.join(runsDir(), runId);
};

export function ensureDirs(): void {
  fs.mkdirSync(runsDir(), { recursive: true });
  const ignore = path.join(runsDir(), '.gitignore');
  if (!fs.existsSync(ignore)) fs.writeFileSync(ignore, '# batch-run output — answers can contain client data\n*\n');
}

export const sha256 = (text: string) => crypto.createHash('sha256').update(text).digest('hex');

// ── Questions ─────────────────────────────────────────────────────────────────

export interface Problem { line: number; id?: string; message: string; severity: 'error' | 'warn' }

export function parseQuestions(text: string): { questions: Question[]; problems: Problem[] } {
  const questions: Question[] = [];
  const problems: Problem[] = [];
  const ids = new Set<string>();
  text.split('\n').forEach((raw, i) => {
    const line = i + 1;
    if (!raw.trim()) return;
    let q: unknown;
    try { q = JSON.parse(raw); } catch (e) { problems.push({ line, message: `invalid JSON: ${(e as Error).message}`, severity: 'error' }); return; }
    if (!q || typeof q !== 'object') { problems.push({ line, message: 'not an object', severity: 'error' }); return; }
    const o = q as Record<string, unknown>;
    const id = typeof o.id === 'string' ? o.id : '';
    if (!id || typeof o.q !== 'string' || !o.q.trim()) { problems.push({ line, id, message: 'needs string "id" and non-empty "q"', severity: 'error' }); return; }
    if (!(KINDS as readonly string[]).includes(String(o.kind))) { problems.push({ line, id, message: `kind must be one of ${KINDS.join(', ')}`, severity: 'error' }); return; }
    if (ids.has(id)) { problems.push({ line, id, message: `duplicate id "${id}"`, severity: 'error' }); return; }
    ids.add(id);
    const question: Question = { id, q: o.q, kind: o.kind as Question['kind'] };
    if (typeof o.note === 'string') question.note = o.note;
    if (o.expect != null) {
      if (typeof o.expect !== 'object' || Array.isArray(o.expect)) { problems.push({ line, id, message: 'expect must be an object', severity: 'error' }); return; }
      const ex = o.expect as Record<string, unknown>;
      for (const k of ['contains', 'entities'] as const) {
        if (ex[k] != null) {
          if (!Array.isArray(ex[k])) { problems.push({ line, id, message: `expect.${k} must be an array`, severity: 'error' }); return; }
          for (const item of ex[k] as unknown[]) {
            const ok = typeof item === 'string' ? item.length > 0 : Array.isArray(item) && item.length > 0 && item.every((s) => typeof s === 'string' && s.length > 0);
            if (!ok) { problems.push({ line, id, message: `expect.${k}: items must be non-empty strings or arrays of them`, severity: 'error' }); return; }
          }
        }
      }
      if (ex.abstain != null && typeof ex.abstain !== 'boolean') { problems.push({ line, id, message: 'expect.abstain must be a boolean', severity: 'error' }); return; }
      question.expect = { ...(ex.contains ? { contains: ex.contains as Question['expect'] extends infer _ ? never[] : never } : {}) } as Question['expect'];
      question.expect = {
        ...(ex.contains ? { contains: ex.contains as NonNullable<Question['expect']>['contains'] } : {}),
        ...(ex.entities ? { entities: ex.entities as NonNullable<Question['expect']>['entities'] } : {}),
        ...(ex.abstain != null ? { abstain: ex.abstain as boolean } : {}),
      };
      for (const v of vacuousExpectItems(question)) {
        problems.push({ line, id, message: `expect item ${JSON.stringify(v)} appears in the question text itself — an answer that parrots the question would hit`, severity: 'warn' });
      }
    }
    questions.push(question);
  });
  return { questions, problems };
}

export function serializeQuestions(questions: Question[]): string {
  return questions.map((q) => JSON.stringify(q)).join('\n') + '\n';
}

export function readQuestionsText(): string {
  const p = questionsPath();
  return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : '';
}

export function writeQuestionsText(text: string): void {
  ensureDirs();
  fs.writeFileSync(questionsPath(), text.endsWith('\n') || text === '' ? text : `${text}\n`);
}

// ── Arms (configs.json) ───────────────────────────────────────────────────────

export function parseArms(text: string): { arms: Arm[]; problems: Problem[] } {
  const problems: Problem[] = [];
  let raw: unknown;
  try { raw = JSON.parse(text); } catch (e) { return { arms: [], problems: [{ line: 0, message: `invalid JSON: ${(e as Error).message}`, severity: 'error' }] }; }
  if (!Array.isArray(raw)) return { arms: [], problems: [{ line: 0, message: 'configs.json must be an array of configurations', severity: 'error' }] };
  const arms: Arm[] = [];
  const names = new Set<string>();
  raw.forEach((c, i) => {
    if (!c || typeof c !== 'object' || typeof (c as Arm).name !== 'string' || !(c as Arm).name) { problems.push({ line: i, message: `configuration ${i}: needs a "name"`, severity: 'error' }); return; }
    const o = c as Record<string, unknown>;
    let overrides: Record<string, unknown>;
    if ('overrides' in o) overrides = (o.overrides ?? {}) as Record<string, unknown>;
    else { overrides = { ...o }; delete overrides.name; delete overrides.enabled; }
    if (typeof overrides !== 'object' || Array.isArray(overrides)) { problems.push({ line: i, message: `configuration "${o.name}": overrides must be an object`, severity: 'error' }); return; }
    if (names.has(o.name as string)) { problems.push({ line: i, message: `duplicate configuration name "${o.name}"`, severity: 'error' }); return; }
    names.add(o.name as string);
    arms.push({ name: o.name as string, overrides: overrides as Arm['overrides'], ...(o.enabled === false ? { enabled: false } : {}) });
  });
  return { arms, problems };
}

export function readArmsText(): string {
  const p = configsPath();
  return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : '[]\n';
}

export function writeArms(arms: Arm[]): void {
  ensureDirs();
  fs.writeFileSync(configsPath(), JSON.stringify(arms, null, 2) + '\n');
}

// ── Runs ──────────────────────────────────────────────────────────────────────

export function newRunId(): string {
  return `${new Date().toISOString().replace(/[:.]/g, '-').replace('T', '_').slice(0, 19)}-${crypto.randomBytes(2).toString('hex')}`;
}

export function writeRunMeta(meta: RunMeta): void {
  const dir = runDir(meta.run_id);
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, 'run.json'), JSON.stringify(meta, null, 2) + '\n');
}

export function readRunMeta(runId: string): RunMeta | null {
  const p = path.join(runDir(runId), 'run.json');
  if (!fs.existsSync(p)) return null;
  const raw = JSON.parse(fs.readFileSync(p, 'utf8')) as Partial<RunMeta> & Record<string, unknown>;
  // CLI runs predate some fields — fill what the UI needs.
  return {
    run_id: raw.run_id ?? runId,
    source: raw.source ?? 'cli',
    started_at: raw.started_at ?? '',
    ...(raw.finished_at ? { finished_at: raw.finished_at } : {}),
    status: raw.status ?? 'done',
    stop_reason: raw.stop_reason ?? null,
    engine_id: raw.engine_id ?? '',
    engine: (raw.engine as RunMeta['engine']) ?? null,
    questions_file: raw.questions_file ?? '',
    questions_sha256: raw.questions_sha256 ?? '',
    n_questions: raw.n_questions ?? 0,
    question_ids: raw.question_ids ?? [],
    configs: (raw.configs as Arm[]) ?? [],
    parallel: raw.parallel ?? 0,
    timeout_s: raw.timeout_s ?? 0,
    cooldown_s: raw.cooldown_s ?? 0,
    run_as: raw.run_as ?? 'me',
    gub_subject: raw.gub_subject ?? null,
    node: raw.node ?? '',
  };
}

export function appendRecord(runId: string, record: CellRecord): void {
  fs.appendFileSync(path.join(runDir(runId), 'raw.jsonl'), JSON.stringify(record) + '\n');
}

export function readRecords(runId: string): CellRecord[] {
  const p = path.join(runDir(runId), 'raw.jsonl');
  if (!fs.existsSync(p)) return [];
  return fs.readFileSync(p, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l) as CellRecord);
}

export function writeSummaryCsv(runId: string, csv: string): void {
  fs.writeFileSync(path.join(runDir(runId), 'summary.csv'), csv);
}

export function readSummaryCsv(runId: string): string | null {
  const p = path.join(runDir(runId), 'summary.csv');
  return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : null;
}

export function listRuns(liveIds: Set<string>): RunListItem[] {
  ensureDirs();
  const out: RunListItem[] = [];
  for (const name of fs.readdirSync(runsDir())) {
    if (name.startsWith('.')) continue;
    const meta = readRunMeta(name);
    if (!meta) continue;
    let nCells = 0;
    const rawPath = path.join(runDir(name), 'raw.jsonl');
    if (fs.existsSync(rawPath)) nCells = fs.readFileSync(rawPath, 'utf8').split('\n').filter(Boolean).length;
    out.push({
      run_id: name, source: meta.source, started_at: meta.started_at, status: liveIds.has(name) ? 'running' : meta.status === 'running' ? 'stopped' : meta.status,
      n_cells: nCells, n_questions: meta.n_questions, configs: meta.configs.map((c) => c.name), run_as: meta.run_as ?? null,
      questions_sha256: meta.questions_sha256, live: liveIds.has(name),
    });
  }
  return out.sort((a, b) => b.run_id.localeCompare(a.run_id));
}
