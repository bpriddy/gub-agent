/**
 * batch/types.ts — the batch runner's data model (gub-agent#33, epic #29 step 04).
 *
 * One QUESTION × one ARM = one CELL; a RUN is the whole matrix. The shapes
 * here are the same JSON the headless twin (`scratchpad/batch-run.mjs`)
 * writes, so a run started from either side reads back in both.
 */
import type { SandboxOverrides, SandboxResolved } from '../sandbox';

export const KINDS = ['FACT', 'ASSESSMENT', 'AMBIGUOUS', 'PERSONAL', 'MARKET'] as const;
export type Kind = (typeof KINDS)[number];

/** A string, or an array of alternatives any one of which satisfies the item. */
export type ContainsItem = string | string[];

export interface Expect {
  contains?: ContainsItem[];
  /** The answer must (true) / must not (false) be the NO_COMPANY_RECORDS abstention. */
  abstain?: boolean;
  /** Reported separately as `entities_hit`; does not affect `expect_hit`. */
  entities?: ContainsItem[];
}

export interface Question {
  id: string;
  q: string;
  kind: Kind;
  expect?: Expect | null;
  note?: string;
}

/** One configuration of the matrix: a name plus a `state.sandbox` payload. `{}` = ordinary run. */
export interface Arm {
  name: string;
  overrides: SandboxOverrides;
  /** UI-only: an arm can be kept in the file but left out of a run. Absent = enabled. */
  enabled?: boolean;
}

export type CellStatus = 'ok' | 'empty' | 'timeout' | 'auth' | 'rate_limited' | 'transport' | 'error';

export interface CriticVerdict {
  info_sufficient?: boolean;
  answer_satisfies?: boolean;
  sufficient?: boolean;
  reason?: string;
  feedback?: string;
}

/** What `analyze()` derives from one cell's event stream. */
export interface CellMetrics {
  answer: string;
  words: number;
  over_budget: boolean;
  citations_total: number;
  citations_invalid: number;
  tool_calls: number;
  tool_names: string[];
  tool_errors: number[];
  tool_errors_429: number;
  rate_limited: boolean;
  known_ids: number;
  source_ids: number;
  iterations: number;
  critic_events: number;
  critic_retry: boolean;
  critic_sufficient: boolean | null;
  critic_verdicts: CriticVerdict[];
  escalated: boolean;
  abstained: boolean;
  model_versions: string[];
  tokens_total: number;
  tokens_thoughts: number;
  thought_chars: number;
  auth_error: boolean;
  event_errors: string[];
  events: number;
}

export interface Judgement {
  expect_hit: boolean | null;
  expect_missing: string[];
  entities_hit: boolean | null;
}

/** One line of raw.jsonl. Metrics are absent on cells that never produced a stream. */
export interface CellRecord extends Partial<CellMetrics>, Partial<Judgement> {
  run_id: string;
  question_id: string;
  kind: Kind;
  config: string;
  attempt: number;
  started_at: string;
  engine_id: string;
  status: CellStatus;
  latency_ms: number;
  error?: string;
  auth_scope?: 'gtoken' | 'gub_jwt';
  session_id?: string;
  first_event_ms?: number | null;
  sandbox_requested?: SandboxOverrides | null;
  sandbox_resolved?: SandboxResolved | null;
  expect?: Expect | null;
}

export type RunAs = 'me' | 'subject';
export type RunStatus = 'running' | 'done' | 'stopped' | 'failed';

export interface RunMeta {
  run_id: string;
  source: 'ui' | 'cli';
  started_at: string;
  finished_at?: string;
  status: RunStatus;
  stop_reason?: string | null;
  engine_id: string;
  engine: { displayName: string | null; env: Record<string, string> } | null;
  questions_file: string;
  questions_sha256: string;
  n_questions: number;
  question_ids: string[];
  configs: Arm[];
  parallel: number;
  timeout_s: number;
  cooldown_s: number;
  run_as: RunAs;
  /** Email or sub of the JWT the agent's tools run as. */
  gub_subject: string | null;
  node: string;
}

export interface RunProgress {
  total: number;
  completed: number;
  running: string[];
  cooldowns: number;
  cooldown_until: number | null;
  rate_limited: number;
  consecutive_auth: number;
  jwt_seconds_left: number | null;
}

export interface ArmSummaryRow {
  config: string;
  kind: Kind | 'ALL';
  n: number;
  n_ok: number;
  n_empty: number;
  n_timeout: number;
  n_auth: number;
  n_rate_limited: number;
  n_error: number;
  words_median: number | null;
  over_budget_share: number | null;
  citations_total: number;
  citations_invalid: number;
  citations_invalid_share: number | null;
  ids_in_prose_share: number | null;
  tool_calls_median: number | null;
  iterations_median: number | null;
  critic_retry_share: number | null;
  abstained_share: number | null;
  expect_n: number;
  expect_hit_share: number | null;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  tokens_median: number | null;
  model_versions: string;
  resolved: SandboxResolved | null;
}

export interface RegressionCell {
  question_id: string;
  config: string;
  status: CellStatus;
  expect_missing: string[];
  answer_excerpt: string;
  error?: string;
}

export interface Regressions {
  baseline: string;
  /** Arm missed an expect the baseline hit. */
  lost: RegressionCell[];
  /** Arm cell failed where the baseline cell was ok. */
  broke: RegressionCell[];
  /** Arm hit where the baseline missed (for the record). */
  won: RegressionCell[];
}

export interface RunSummary {
  rows: ArmSummaryRow[];
  byKind: ArmSummaryRow[];
  regressions: Regressions | null;
}

/** GET /api/batch/runs/[id] */
export interface RunView {
  meta: RunMeta;
  progress: RunProgress | null;
  records: CellRecord[];
  summary: RunSummary;
  log: string[];
}

export interface RunListItem {
  run_id: string;
  source: RunMeta['source'];
  started_at: string;
  status: RunStatus;
  n_cells: number;
  n_questions: number;
  configs: string[];
  run_as: RunAs | null;
  questions_sha256: string;
  live: boolean;
}
