/**
 * batch/metrics.ts — deterministic metrics over one cell's event stream, the
 * expectation judge, per-arm aggregation and the regression list.
 *
 * Pure functions, no I/O. A TypeScript port of the same logic in
 * `scratchpad/batch-run.mjs` (the headless twin) — keep the two in step: the
 * definitions are what make two runs comparable.
 *
 * Iteration split is identical to `trace.ts`: a new iteration starts after
 * every `critic` event; `critic_gate` (deterministic pass) and `sandbox_echo`
 * carry no answer text and are skipped like `loop_escalator`.
 */
import type { SandboxResolved } from '../sandbox';
import {
  KINDS,
  type ArmSummaryRow,
  type CellMetrics,
  type CellRecord,
  type ContainsItem,
  type CriticVerdict,
  type Judgement,
  type Kind,
  type Question,
  type RegressionCell,
  type Regressions,
} from './types';

export const WORD_BUDGET = 250; // blend §4 total budget
export const ABSTAIN_MARKER = 'NO_COMPANY_RECORDS'; // same check as CriticGate / the bot
const NON_EXECUTOR_AUTHORS = new Set(['critic', 'critic_gate', 'loop_escalator', 'sandbox_echo']);

const UUID_RE = /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi;
const DRIVE_ID_RE = /\b[A-Za-z0-9_-]{25,}\b/g;
const BRACKET_CITE_RE = /\[([^\]\s]{6,})\](?!\()/g;

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => !!v && typeof v === 'object' && !Array.isArray(v);

function collectIds(value: unknown, ids: Set<string>, sourceIds: Set<string>, depth = 0): void {
  if (depth > 12 || value == null) return;
  if (typeof value === 'string') {
    for (const m of value.match(UUID_RE) ?? []) ids.add(m.toLowerCase());
    return;
  }
  if (Array.isArray(value)) { for (const v of value) collectIds(v, ids, sourceIds, depth + 1); return; }
  if (isObj(value)) {
    for (const [k, v] of Object.entries(value)) {
      if (typeof v === 'string' && (k === 'fileId' || k === 'id' || /Id$/.test(k))) {
        ids.add(v.toLowerCase());
        if (k === 'fileId') sourceIds.add(v.toLowerCase());
      }
      collectIds(v, ids, sourceIds, depth + 1);
    }
  }
}

export function countWords(text: string): number {
  return text.trim().split(/\s+/).filter((w) => /[\p{L}\p{N}]/u.test(w)).length;
}

export function analyze(events: unknown[]): CellMetrics {
  const iterations: Array<{ text: string; toolCalls: number }> = [];
  let cur = { text: '', toolCalls: 0 };
  const known = new Set<string>();
  const sourceIds = new Set<string>();
  const toolNames: string[] = [];
  const toolErrors: number[] = [];
  const verdicts: CriticVerdict[] = [];
  const modelVersions = new Set<string>();
  const errors: string[] = [];
  let resolved: SandboxResolved | null = null;
  let criticEvents = 0;
  let escalated = false;
  let authError = false;
  let tokens = 0;
  let thoughtTokens = 0;
  let thoughtChars = 0;

  for (const raw of events) {
    if (!isObj(raw)) continue;
    const e = raw;
    const actions = isObj(e.actions) ? e.actions : {};
    const delta = (isObj(actions.state_delta) ? actions.state_delta : isObj(actions.stateDelta) ? actions.stateDelta : {}) as Obj;
    if (isObj(delta.sandbox_resolved)) resolved = delta.sandbox_resolved as unknown as SandboxResolved;
    if (isObj(delta.critic_verdict)) verdicts.push(delta.critic_verdict as CriticVerdict);
    if (actions.escalate) escalated = true;
    const um = (isObj(e.usage_metadata) ? e.usage_metadata : isObj(e.usageMetadata) ? e.usageMetadata : null) as Obj | null;
    if (um) {
      tokens += Number(um.total_token_count ?? um.totalTokenCount ?? 0);
      thoughtTokens += Number(um.thoughts_token_count ?? um.thoughtsTokenCount ?? 0);
    }
    const mv = e.model_version ?? e.modelVersion;
    if (typeof mv === 'string') modelVersions.add(mv);
    const errCode = e.error_code ?? e.errorCode;
    const errMsg = e.error_message ?? e.errorMessage;
    if (errCode || errMsg) errors.push(`${errCode ?? ''} ${errMsg ?? ''}`.trim());

    const author = typeof e.author === 'string' ? e.author : '';
    if (author === 'critic') {
      criticEvents++;
      iterations.push(cur);
      cur = { text: '', toolCalls: 0 };
      continue;
    }
    if (NON_EXECUTOR_AUTHORS.has(author)) continue;

    const content = isObj(e.content) ? e.content : null;
    const parts = content && Array.isArray(content.parts) ? (content.parts as unknown[]) : [];
    for (const p of parts) {
      if (!isObj(p)) continue;
      if (typeof p.text === 'string' && p.text) {
        if (p.thought === true) thoughtChars += p.text.length;
        else cur.text = cur.text ? `${cur.text}\n${p.text}` : p.text;
      }
      const fc = (isObj(p.function_call) ? p.function_call : isObj(p.functionCall) ? p.functionCall : null) as Obj | null;
      if (fc && typeof fc.name === 'string') { cur.toolCalls++; toolNames.push(fc.name); }
      const fr = (isObj(p.function_response) ? p.function_response : isObj(p.functionResponse) ? p.functionResponse : null) as Obj | null;
      if (fr) {
        const r = fr.response;
        collectIds(r, known, sourceIds);
        if (isObj(r) && r.error === true) {
          const st = Number(r.status) || 0;
          toolErrors.push(st);
          if (st === 401) authError = true;
        }
      }
    }
  }
  if (cur.text || cur.toolCalls) iterations.push(cur);

  let answer = '';
  for (let i = iterations.length - 1; i >= 0; i--) if (iterations[i]!.text) { answer = iterations[i]!.text; break; }

  const cited = new Set<string>();
  for (const m of answer.match(UUID_RE) ?? []) cited.add(m.toLowerCase());
  for (const m of answer.match(DRIVE_ID_RE) ?? []) if (!/^[0-9]+$/.test(m) && /[0-9]/.test(m) && /[A-Za-z]/.test(m)) cited.add(m.toLowerCase());
  for (const m of answer.matchAll(BRACKET_CITE_RE)) cited.add(m[1]!.toLowerCase());
  const citationsInvalid = [...cited].filter((id) => !known.has(id)).length;
  const lastVerdict = verdicts.length ? verdicts[verdicts.length - 1]! : null;
  const words = countWords(answer);

  return {
    answer,
    words,
    over_budget: words > WORD_BUDGET,
    citations_total: cited.size,
    citations_invalid: citationsInvalid,
    tool_calls: toolNames.length,
    tool_names: toolNames,
    tool_errors: toolErrors,
    tool_errors_429: toolErrors.filter((s) => s === 429).length,
    rate_limited: toolErrors.includes(429),
    known_ids: known.size,
    source_ids: sourceIds.size,
    iterations: iterations.length,
    critic_events: criticEvents,
    critic_retry: iterations.length > 1,
    critic_sufficient: lastVerdict ? lastVerdict.sufficient === true : null,
    critic_verdicts: verdicts,
    escalated,
    abstained: answer.trim().toUpperCase().startsWith(ABSTAIN_MARKER),
    model_versions: [...modelVersions],
    tokens_total: tokens,
    tokens_thoughts: thoughtTokens,
    thought_chars: thoughtChars,
    auth_error: authError,
    event_errors: errors,
    events: events.length,
    // `resolved` travels on the record as sandbox_resolved; exposed via analyzeResolved.
  };
}

/** The `sandbox_resolved` state delta of a stream, if any (baseline runs have none). */
export function analyzeResolved(events: unknown[]): SandboxResolved | null {
  for (const raw of events) {
    if (!isObj(raw)) continue;
    const actions = isObj(raw.actions) ? raw.actions : {};
    const delta = (isObj(actions.state_delta) ? actions.state_delta : isObj(actions.stateDelta) ? actions.stateDelta : {}) as Obj;
    if (isObj(delta.sandbox_resolved)) return delta.sandbox_resolved as unknown as SandboxResolved;
  }
  return null;
}

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Whole-number match for numeric items: not glued to another digit or thousands group,
 *  but a trailing decimal part is fine ("5,130,085" inside "$5,130,085.67"). */
export function containsExpectation(answer: string, item: ContainsItem): boolean {
  const alternatives = Array.isArray(item) ? item : [item];
  const hay = answer.replace(/\s+/g, ' ');
  return alternatives.some((alt) => {
    const s = String(alt);
    if (/^\d[\d,.]*$/.test(s)) return new RegExp(`(?<!\\d)(?<![.,]\\d{0,2})${escapeRe(s)}(?!\\d|[.,]\\d{3})`).test(hay);
    return hay.toLowerCase().includes(s.toLowerCase());
  });
}

export function judgeExpect(q: Question, m: Pick<CellMetrics, 'answer' | 'abstained'>): Judgement {
  const ex = q.expect;
  if (!ex) return { expect_hit: null, expect_missing: [], entities_hit: null };
  const missing: string[] = (ex.contains ?? []).filter((item) => !containsExpectation(m.answer, item)).map(itemLabel);
  let hit = missing.length === 0;
  if (ex.abstain != null && m.abstained !== ex.abstain) {
    hit = false;
    missing.push(ex.abstain ? '<abstain expected>' : '<answer expected, got abstention>');
  }
  const entitiesHit = ex.entities?.length ? ex.entities.every((item) => containsExpectation(m.answer, item)) : null;
  return { expect_hit: hit, expect_missing: missing, entities_hit: entitiesHit };
}

export const itemLabel = (item: ContainsItem): string => (Array.isArray(item) ? item.join(' | ') : String(item));

/** Items that already sit in the question text — a parroting answer would hit them. */
export function vacuousExpectItems(q: Question): string[] {
  const out: string[] = [];
  for (const item of q.expect?.contains ?? []) {
    for (const alt of Array.isArray(item) ? item : [item]) {
      if (q.q.toLowerCase().includes(String(alt).toLowerCase())) out.push(String(alt));
    }
  }
  return out;
}

// ── Aggregation ───────────────────────────────────────────────────────────────

export function percentile(values: number[], p: number): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.max(0, Math.ceil((p / 100) * sorted.length) - 1)]!;
}
const median = (v: number[]) => percentile(v, 50);
const share = (num: number, den: number) => (den ? num / den : null);

export function summarizeArm(records: CellRecord[], config: string, kind: Kind | null): ArmSummaryRow {
  const cells = records.filter((r) => r.config === config && (kind === null || r.kind === kind));
  const ok = cells.filter((r) => r.status === 'ok');
  const withExpect = ok.filter((r) => r.expect_hit !== null && r.expect_hit !== undefined);
  const n = (f: (r: CellRecord) => number | undefined) => ok.map((r) => f(r) ?? 0);
  return {
    config,
    kind: kind ?? 'ALL',
    n: cells.length,
    n_ok: ok.length,
    n_empty: cells.filter((r) => r.status === 'empty').length,
    n_timeout: cells.filter((r) => r.status === 'timeout').length,
    n_auth: cells.filter((r) => r.status === 'auth').length,
    n_rate_limited: cells.filter((r) => r.status === 'rate_limited').length,
    n_error: cells.filter((r) => r.status === 'error' || r.status === 'transport').length,
    words_median: median(n((r) => r.words)),
    over_budget_share: share(ok.filter((r) => r.over_budget).length, ok.length),
    citations_total: ok.reduce((s, r) => s + (r.citations_total ?? 0), 0),
    citations_invalid: ok.reduce((s, r) => s + (r.citations_invalid ?? 0), 0),
    citations_invalid_share: share(ok.filter((r) => (r.citations_invalid ?? 0) > 0).length, ok.length),
    ids_in_prose_share: share(ok.filter((r) => (r.citations_total ?? 0) > 0).length, ok.length),
    tool_calls_median: median(n((r) => r.tool_calls)),
    iterations_median: median(n((r) => r.iterations)),
    critic_retry_share: share(ok.filter((r) => r.critic_retry).length, ok.length),
    abstained_share: share(ok.filter((r) => r.abstained).length, ok.length),
    expect_n: withExpect.length,
    expect_hit_share: share(withExpect.filter((r) => r.expect_hit).length, withExpect.length),
    latency_p50_ms: percentile(ok.map((r) => r.latency_ms), 50),
    latency_p95_ms: percentile(ok.map((r) => r.latency_ms), 95),
    tokens_median: median(n((r) => r.tokens_total)),
    model_versions: [...new Set(ok.flatMap((r) => r.model_versions ?? []))].join('|'),
    resolved: ok.find((r) => r.sandbox_resolved)?.sandbox_resolved ?? null,
  };
}

export function summarizeRun(records: CellRecord[], configs: string[]): { rows: ArmSummaryRow[]; byKind: ArmSummaryRow[] } {
  const rows = configs.map((c) => summarizeArm(records, c, null));
  const kinds = KINDS.filter((k) => records.some((r) => r.kind === k));
  const byKind = kinds.flatMap((k) => configs.map((c) => summarizeArm(records, c, k)));
  return { rows, byKind };
}

const excerpt = (s: string | undefined, n = 200) => {
  const t = (s ?? '').replace(/\s+/g, ' ');
  return t.length > n ? `${t.slice(0, n)}…` : t;
};

export function regressions(records: CellRecord[], configs: string[], baselineName: string): Regressions | null {
  const baseline = new Map(records.filter((r) => r.config === baselineName).map((r) => [r.question_id, r]));
  if (!baseline.size) return null;
  const out: Regressions = { baseline: baselineName, lost: [], broke: [], won: [] };
  const cell = (r: CellRecord): RegressionCell => ({
    question_id: r.question_id, config: r.config, status: r.status, expect_missing: r.expect_missing ?? [],
    answer_excerpt: excerpt(r.answer), ...(r.error ? { error: r.error } : {}),
  });
  for (const c of configs) {
    if (c === baselineName) continue;
    for (const r of records.filter((x) => x.config === c)) {
      const b = baseline.get(r.question_id);
      if (!b) continue;
      if (b.status === 'ok' && r.status !== 'ok') out.broke.push(cell(r));
      if (b.expect_hit === true && r.expect_hit === false) out.lost.push(cell(r));
      if (b.expect_hit === false && r.expect_hit === true) out.won.push(cell(r));
    }
  }
  return out;
}

// ── summary.csv (same columns as the CLI) ─────────────────────────────────────

export const CSV_COLUMNS = [
  'run_id', 'questions_sha256', 'questions_file', 'engine_id', 'config', 'kind',
  'n', 'n_ok', 'n_empty', 'n_timeout', 'n_auth', 'n_rate_limited', 'n_error',
  'words_median', 'over_budget_share', 'citations_total', 'citations_invalid', 'citations_invalid_share', 'ids_in_prose_share',
  'tool_calls_median', 'iterations_median', 'critic_retry_share', 'abstained_share', 'expect_n', 'expect_hit_share',
  'latency_p50_ms', 'latency_p95_ms', 'tokens_median', 'model_versions',
  'resolved_model', 'resolved_thinking', 'resolved_executor_prompt', 'resolved_critic_enabled',
] as const;

function csvEscape(v: unknown): string {
  if (v === null || v === undefined) return '';
  const s = typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(4)) : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function summaryCsv(rows: ArmSummaryRow[], meta: { run_id: string; questions_sha256: string; questions_file: string; engine_id: string }): string {
  const lines = [CSV_COLUMNS.join(',')];
  for (const r of rows) {
    const full: Record<string, unknown> = {
      ...meta,
      ...r,
      resolved_model: r.resolved?.model ?? '',
      resolved_thinking: r.resolved ? `${r.resolved.thinking_level}/${r.resolved.critic_thinking_level}` : '',
      resolved_executor_prompt: r.resolved ? `${r.resolved.executor_prompt_source}${r.resolved.executor_prompt_sha256 ? ' ' + r.resolved.executor_prompt_sha256 : ''}` : '',
      resolved_critic_enabled: r.resolved ? String(r.resolved.critic_enabled) : '',
    };
    lines.push(CSV_COLUMNS.map((c) => csvEscape(full[c])).join(','));
  }
  return lines.join('\n') + '\n';
}
