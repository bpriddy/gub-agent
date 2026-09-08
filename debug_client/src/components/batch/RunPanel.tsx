/**
 * RunPanel.tsx — start a batch (which questions, which arms, parallelism,
 * timeout, cooldown, whose token), watch it (cell grid + counters + log), and
 * read the result: one row per arm, latency by kind, provenance, and the
 * regression lists that are meant to be read by eye. Also opens past runs.
 */
'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { SandboxConfigResponse } from '@/lib/api';
import { ApiError } from '@/lib/api';
import { ResolvedChips } from '@/components/RunColumn';
import { KINDS, type Arm, type ArmSummaryRow, type CellRecord, type Kind, type Question, type RunAs, type RunListItem, type RunView } from '@/lib/batch/types';
import { csvUrl, fetchArms, fetchQuestions, fetchRun, fetchRuns, fetchSubject, startRun, stopRun, type SubjectProbe } from '@/lib/batchApi';
import { B, STATUS_COLOR, num, pct, secs } from './batchStyles';

const POLL_MS = 2500;

export function RunPanel({ options, refreshKey }: { options: SandboxConfigResponse | null; refreshKey: number }) {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [arms, setArms] = useState<Arm[]>([]);
  const [subject, setSubject] = useState<SubjectProbe | null>(null);
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [selectedKinds, setSelectedKinds] = useState<Set<Kind>>(new Set(KINDS));
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [selectedArms, setSelectedArms] = useState<Set<string> | null>(null);
  const [parallel, setParallel] = useState(2);
  const [timeoutS, setTimeoutS] = useState(180);
  const [cooldown, setCooldown] = useState(60);
  const [runAs, setRunAs] = useState<RunAs | null>(null);
  const [showQuestions, setShowQuestions] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [view, setView] = useState<RunView | null>(null);
  const [selectedCell, setSelectedCell] = useState<CellRecord | null>(null);

  useEffect(() => {
    void fetchQuestions().then((v) => setQuestions(v.questions)).catch((e: Error) => setError(e.message));
    void fetchArms().then((v) => { setArms(v.arms); setSelectedArms((s) => s ?? new Set(v.arms.filter((a) => a.enabled !== false).map((a) => a.name))); }).catch((e: Error) => setError(e.message));
    void fetchSubject().then((s) => { setSubject(s); setRunAs((r) => r ?? (s.ok ? 'subject' : 'me')); }).catch(() => setSubject({ configured: false, serviceAccount: null, ok: false, email: null, error: 'probe failed' }));
    void fetchRuns().then((r) => setRuns(r.runs)).catch(() => {});
  }, [refreshKey]);

  const chosenQuestions = useMemo(() => questions.filter((q) => selectedKinds.has(q.kind) && !excluded.has(q.id)), [questions, selectedKinds, excluded]);
  const chosenArms = useMemo(() => arms.filter((a) => selectedArms?.has(a.name)), [arms, selectedArms]);
  const cells = chosenQuestions.length * chosenArms.length;
  const estMin = Math.ceil((cells * 25) / 60 / Math.max(1, parallel));

  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const poll = useCallback(async (id: string) => {
    try {
      const v = await fetchRun(id);
      setView(v);
      if (v.meta.status === 'running') timer.current = setTimeout(() => void poll(id), POLL_MS);
      else void fetchRuns().then((r) => setRuns(r.runs)).catch(() => {});
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);
  const open = (id: string) => { if (timer.current) clearTimeout(timer.current); setRunId(id); setView(null); setSelectedCell(null); void poll(id); };

  const start = async () => {
    if (!cells || !runAs) return;
    setStarting(true); setError(null);
    try {
      const meta = await startRun({ questionIds: chosenQuestions.map((q) => q.id), arms: chosenArms.map((a) => a.name), parallel, timeout: timeoutS, cooldown, runAs });
      open(meta.run_id);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.code}: ${e.message}` : (e as Error).message);
    } finally { setStarting(false); }
  };

  const liveRun = runs.find((r) => r.live);

  return (
    <>
      <section style={B.card}>
        <div style={B.cardHead}>
          <h2 style={B.cardTitle}>new run</h2>
          <span style={B.dim}>{chosenQuestions.length} questions x {chosenArms.length} arms = {cells} cells; roughly {estMin} min at parallel {parallel}, more with cooldowns</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '1rem' }}>
          <div style={B.col}>
            <div style={B.row}>
              <span style={B.lbl}>questions</span>
              {KINDS.map((k) => {
                const n = questions.filter((q) => q.kind === k).length;
                const on = selectedKinds.has(k);
                return (
                  <button key={k} type="button" disabled={!n} style={{ ...B.chip, cursor: n ? 'pointer' : 'default', opacity: n ? 1 : 0.4, ...(on && n ? { background: '#1f4068', color: '#e6edf3' } : {}) }}
                    onClick={() => setSelectedKinds((s) => { const t = new Set(s); if (t.has(k)) t.delete(k); else t.add(k); return t; })}>
                    {k} {n}
                  </button>
                );
              })}
              <button type="button" style={B.btn} onClick={() => setShowQuestions((v) => !v)}>{showQuestions ? 'hide list' : 'pick individually'}</button>
            </div>
            {showQuestions && (
              <div style={{ maxHeight: '14rem', overflowY: 'auto', border: '1px solid #21262d', borderRadius: '4px', padding: '0.25rem 0.5rem' }}>
                {questions.filter((q) => selectedKinds.has(q.kind)).map((q) => (
                  <label key={q.id} style={{ ...B.check, padding: '0.125rem 0' }}>
                    <input type="checkbox" checked={!excluded.has(q.id)} onChange={(e) => setExcluded((s) => { const t = new Set(s); if (e.target.checked) t.delete(q.id); else t.add(q.id); return t; })} />
                    <span style={{ ...B.mono, color: '#7d8590', minWidth: '5.5rem' }}>{q.id}</span>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{q.q}</span>
                    {q.expect && <span style={B.chip}>expect</span>}
                  </label>
                ))}
              </div>
            )}
            <div style={B.row}>
              <span style={B.lbl}>arms</span>
              {arms.map((a) => {
                const on = selectedArms?.has(a.name) ?? false;
                const empty = Object.keys(a.overrides).length === 0;
                return (
                  <button key={a.name} type="button" style={{ ...B.chip, cursor: 'pointer', ...(on ? { background: '#1f4068', color: '#e6edf3' } : {}) }}
                    title={empty ? 'ordinary run: the reference arm' : JSON.stringify(a.overrides)}
                    onClick={() => setSelectedArms((s) => { const t = new Set(s ?? []); if (t.has(a.name)) t.delete(a.name); else t.add(a.name); return t; })}>
                    {a.name}{empty ? ' (base)' : ''}
                  </button>
                );
              })}
              {arms.length === 0 && <span style={B.dim}>no arms yet: add them in the Arms tab</span>}
            </div>
          </div>
          <div style={B.col}>
            <div style={B.row}>
              <label style={B.lbl}>parallel</label>
              <input type="number" min={1} max={6} value={parallel} onChange={(e) => setParallel(Number(e.target.value) || 1)} style={B.inputShort} />
              <label style={B.lbl}>timeout s</label>
              <input type="number" min={30} max={600} value={timeoutS} onChange={(e) => setTimeoutS(Number(e.target.value) || 180)} style={B.inputShort} />
              <label style={B.lbl}>cooldown s</label>
              <input type="number" min={0} max={900} value={cooldown} onChange={(e) => setCooldown(Number(e.target.value) || 0)} style={B.inputShort}
                title="pause for every worker after a cell is rate-limited by GUB (HTTP 429), then retry that cell once" />
            </div>
            <div style={B.row}>
              <span style={B.lbl}>run as</span>
              <label style={{ ...B.check, opacity: subject?.ok ? 1 : 0.5 }}
                title={subject?.configured ? (subject.ok ? `impersonates ${subject.serviceAccount}` : subject.error ?? 'cannot mint') : 'SANDBOX_JWT_SUBJECT_SA is not set in .env.local'}>
                <input type="radio" name="runAs" disabled={!subject?.ok} checked={runAs === 'subject'} onChange={() => setRunAs('subject')} />
                sandbox subject{subject?.email ? `: ${subject.email}` : ''}
              </label>
              <label style={B.check} title="the agent's tools run as you; the page keeps handing the run fresh tokens while it stays open, and the run spends YOUR GUB rate bucket">
                <input type="radio" name="runAs" checked={runAs === 'me'} onChange={() => setRunAs('me')} />
                me
              </label>
            </div>
            {subject && !subject.ok && (
              <div style={B.warn}>
                {subject.configured
                  ? `dedicated subject ${subject.serviceAccount} cannot be minted: ${subject.error ?? 'unknown'}. Your ADC identity needs roles/iam.serviceAccountTokenCreator on it.`
                  : 'no dedicated sandbox subject configured (SANDBOX_JWT_SUBJECT_SA). Runs go as you and spend your GUB rate bucket: 60 /org requests per 15 min, about one cell a minute.'}
              </div>
            )}
            <div style={B.row}>
              <button type="button" style={{ ...B.btnPrimary, opacity: starting || !cells || !!liveRun ? 0.6 : 1 }} disabled={starting || !cells || !!liveRun} onClick={() => void start()}
                title={liveRun ? `run ${liveRun.run_id} is in progress; one at a time` : undefined}>
                {starting ? 'starting...' : liveRun ? 'a run is in progress' : `run ${cells} cells`}
              </button>
              {liveRun && runId !== liveRun.run_id && <button type="button" style={B.btn} onClick={() => open(liveRun.run_id)}>open the live run</button>}
            </div>
            {error && <div style={B.error}>{error}</div>}
          </div>
        </div>
      </section>

      {runId && <RunViewer runId={runId} view={view} options={options} selectedCell={selectedCell} onSelectCell={setSelectedCell} onStopped={() => void poll(runId)} />}

      <section style={B.card}>
        <div style={B.cardHead}>
          <h2 style={B.cardTitle}>runs</h2>
          <span style={B.dim}>{runs.length} under scratchpad/runs, UI and CLI alike</span>
          <button type="button" style={{ ...B.btn, marginLeft: 'auto' }} onClick={() => void fetchRuns().then((r) => setRuns(r.runs))}>refresh</button>
        </div>
        <table style={B.table}>
          <thead><tr><th style={B.th}>run</th><th style={B.th}>started</th><th style={B.th}>status</th><th style={B.th}>cells</th><th style={B.th}>arms</th><th style={B.th}>as</th><th style={B.th}>questions sha</th><th style={B.th}></th></tr></thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run_id} style={{ background: r.run_id === runId ? '#161b22' : undefined }}>
                <td style={{ ...B.td, ...B.mono }}>{r.run_id}</td>
                <td style={{ ...B.td, whiteSpace: 'nowrap' }}>{r.started_at ? new Date(r.started_at).toLocaleString() : '-'}</td>
                <td style={{ ...B.td, color: r.live ? '#58a6ff' : r.status === 'done' ? '#3fb950' : '#e3b341' }}>{r.live ? 'running' : r.status}{r.source === 'cli' ? ' (cli)' : ''}</td>
                <td style={B.tdNum}>{r.n_cells}{r.n_questions ? ` / ${r.n_questions}q` : ''}</td>
                <td style={{ ...B.td, color: '#7d8590' }}>{r.configs.join(', ')}</td>
                <td style={{ ...B.td, color: '#7d8590' }}>{r.run_as ?? '-'}</td>
                <td style={{ ...B.td, ...B.mono, color: '#7d8590' }}>{r.questions_sha256.slice(0, 12)}</td>
                <td style={B.td}><button type="button" style={B.btn} onClick={() => open(r.run_id)}>open</button></td>
              </tr>
            ))}
            {runs.length === 0 && <tr><td colSpan={8} style={{ ...B.td, color: '#7d8590', textAlign: 'center', padding: '1rem' }}>no runs yet</td></tr>}
          </tbody>
        </table>
      </section>
    </>
  );
}

// One run: progress, grid, summary, regressions, detail.

function RunViewer({ runId, view, options, selectedCell, onSelectCell, onStopped }: {
  runId: string; view: RunView | null; options: SandboxConfigResponse | null; selectedCell: CellRecord | null; onSelectCell: (c: CellRecord | null) => void; onStopped: () => void;
}) {
  const [showLog, setShowLog] = useState(false);
  const [showKinds, setShowKinds] = useState(false);
  if (!view) return <section style={B.card}><span style={B.dim}>loading run {runId}...</span></section>;
  const { meta, progress, records, summary } = view;
  const armNames = meta.configs.length ? meta.configs.map((c) => c.name) : [...new Set(records.map((r) => r.config))];
  const questionIds = meta.question_ids.length ? meta.question_ids : [...new Set(records.map((r) => r.question_id))];
  const byKey = new Map(records.map((r) => [`${r.question_id} ${r.config}`, r]));
  const baselineName = summary.regressions?.baseline ?? armNames[0];
  const running = meta.status === 'running';
  const defaults = options?.defaults ?? null;
  const rateLimited = records.filter((r) => r.status === 'rate_limited').length;

  return (
    <section style={B.card}>
      <div style={B.cardHead}>
        <h2 style={B.cardTitle}>run</h2>
        <span style={{ ...B.mono, fontSize: '0.8125rem' }}>{meta.run_id}</span>
        <span style={{ fontSize: '0.75rem', color: running ? '#58a6ff' : meta.status === 'done' ? '#3fb950' : '#e3b341' }}>{meta.status}</span>
        <span style={B.dim}>
          engine ...{meta.engine_id.slice(-6)} | as {meta.run_as}{meta.gub_subject ? ` (${meta.gub_subject})` : ''} | parallel {meta.parallel} | timeout {meta.timeout_s}s | cooldown {meta.cooldown_s}s | questions sha {meta.questions_sha256.slice(0, 12)}
        </span>
        <span style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem' }}>
          <a href={csvUrl(meta.run_id)} style={B.btn}>summary.csv</a>
          <button type="button" style={B.btn} onClick={() => setShowLog((v) => !v)}>log {view.log.length}</button>
          {running && <button type="button" style={B.btnDanger} onClick={() => void stopRun(meta.run_id).then(onStopped)}>stop</button>}
        </span>
      </div>
      {progress && (
        <div style={{ ...B.row, marginBottom: '0.5rem' }}>
          <div style={{ flex: 1, height: 6, background: '#21262d', borderRadius: 3, overflow: 'hidden' }}>
            <div style={{ width: `${(100 * progress.completed) / Math.max(1, progress.total)}%`, height: '100%', background: '#2ea043' }} />
          </div>
          <span style={B.dim}>{progress.completed}/{progress.total}</span>
          {progress.running.length > 0 && <span style={B.dim}>| in flight: {progress.running.join(', ')}</span>}
          {progress.cooldown_until && <span style={{ ...B.dim, color: '#db6d28' }}>| cooling down {Math.max(0, Math.round((progress.cooldown_until - Date.now()) / 1000))}s</span>}
          {progress.jwt_seconds_left !== null && <span style={B.dim}>| jwt {progress.jwt_seconds_left}s</span>}
        </div>
      )}
      {meta.stop_reason && <div style={{ ...B.warn, marginBottom: '0.5rem' }}>{meta.stop_reason}</div>}
      {(rateLimited > 0 || (progress?.cooldowns ?? 0) > 0) && (
        <div style={{ ...B.warn, marginBottom: '0.5rem' }}>
          GUB rate limiting hit this run: {progress?.cooldowns ?? '?'} cooldown(s), {rateLimited} cell(s) still rate-limited after their retry and excluded from every quality column. Lower parallelism or run as the sandbox subject.
        </div>
      )}
      {showLog && <pre style={{ ...B.pre, marginBottom: '0.5rem', maxHeight: '12rem', overflowY: 'auto' }}>{view.log.length ? view.log.join('\n') : '(empty)'}</pre>}

      <div style={{ overflowX: 'auto', marginBottom: '0.75rem' }}>
        <table style={{ ...B.table, width: 'auto' }}>
          <thead><tr><th style={B.th}>question</th>{armNames.map((a) => <th key={a} style={{ ...B.th, textAlign: 'center' }}>{a}</th>)}</tr></thead>
          <tbody>
            {questionIds.map((qid) => (
              <tr key={qid}>
                <td style={{ ...B.td, ...B.mono, whiteSpace: 'nowrap', color: '#7d8590' }}>{qid}</td>
                {armNames.map((a) => {
                  const r = byKey.get(`${qid} ${a}`);
                  const key = `${qid} x ${a}`;
                  const state = r ? (r.status === 'ok' && r.expect_hit === false ? 'miss' : r.status) : progress?.running.includes(key) ? 'running' : 'pending';
                  const sel = selectedCell === r;
                  const tip = r
                    ? [`${r.status}${r.expect_hit === false ? ', missed expect' : ''}`, `${secs(r.latency_ms)}, ${r.words ?? '-'} words, ${r.tool_calls ?? '-'} tools`, r.error ?? ''].filter(Boolean).join(' | ')
                    : state;
                  return (
                    <td key={a} style={{ ...B.td, textAlign: 'center', padding: '0.125rem 0.25rem' }}>
                      <button type="button" disabled={!r} onClick={() => onSelectCell(sel ? null : r ?? null)} title={tip}
                        style={{ width: '4.25rem', height: '1.5rem', border: sel ? '2px solid #e6edf3' : '1px solid #0d1117', borderRadius: 3, background: STATUS_COLOR[state] ?? '#30363d', color: state === 'pending' ? '#7d8590' : '#0d1117', fontSize: '0.6875rem', fontVariantNumeric: 'tabular-nums', cursor: r ? 'pointer' : 'default' }}>
                        {r ? `${(r.latency_ms / 1000).toFixed(0)}s ${r.words ?? ''}w` : state === 'running' ? '...' : ''}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ ...B.row, marginTop: '0.25rem' }}>
          {(['ok', 'miss', 'rate_limited', 'timeout', 'empty', 'auth', 'running'] as const).map((s) => (
            <span key={s} style={{ ...B.dim, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <span style={{ width: 10, height: 10, background: STATUS_COLOR[s], borderRadius: 2, display: 'inline-block' }} />
              {s === 'miss' ? 'ok, missed expect' : s}
            </span>
          ))}
        </div>
      </div>

      {selectedCell && <CellDetail record={selectedCell} defaults={defaults} onClose={() => onSelectCell(null)} />}

      <SummaryTable rows={summary.rows} baseline={baselineName} />
      <div style={{ ...B.col, gap: '0.375rem', margin: '0.5rem 0' }}>
        {summary.rows.map((row) => (
          <div key={row.config} style={B.row}>
            <span style={{ ...B.mono, fontSize: '0.75rem', minWidth: '8rem' }}>{row.config}</span>
            <ResolvedChips resolved={row.resolved} defaults={defaults} />
            {row.model_versions && <span style={B.dim}>served by {row.model_versions}</span>}
            {failedCells(row) && <span style={{ ...B.dim, color: '#ffa198' }}>failed: {failedCells(row)}</span>}
          </div>
        ))}
      </div>
      <button type="button" style={B.btn} onClick={() => setShowKinds((v) => !v)}>{showKinds ? 'hide' : 'show'} by kind</button>
      {showKinds && <SummaryTable rows={summary.byKind} baseline={baselineName} withKind />}

      {summary.regressions && <RegressionLists r={summary.regressions} onPick={(qid, cfg) => onSelectCell(byKey.get(`${qid} ${cfg}`) ?? null)} />}
    </section>
  );
}

const failedCells = (r: ArmSummaryRow) =>
  [r.n_empty && `${r.n_empty} empty`, r.n_timeout && `${r.n_timeout} timeout`, r.n_auth && `${r.n_auth} auth`, r.n_rate_limited && `${r.n_rate_limited} rate-limited`, r.n_error && `${r.n_error} error`].filter(Boolean).join(', ');

function SummaryTable({ rows, baseline, withKind }: { rows: ArmSummaryRow[]; baseline: string | undefined; withKind?: boolean }) {
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={B.table}>
        <thead>
          <tr>
            <th style={B.th}>arm</th>
            {withKind && <th style={B.th}>kind</th>}
            <th style={B.th}>ok/n</th>
            <th style={B.th} title="median answer words over ok cells">words</th>
            <th style={B.th} title="share of answers over 250 words">over 250</th>
            <th style={B.th} title="share of answers citing an id no tool returned">invCit</th>
            <th style={B.th} title="share of answers with any id in prose (the prompt forbids it)">idsIn</th>
            <th style={B.th}>tools</th>
            <th style={B.th} title="critic sent the executor round again">retry</th>
            <th style={B.th} title="NO_COMPANY_RECORDS">abst</th>
            <th style={B.th} title="expect_hit over cells that carry an expect">hit (n)</th>
            <th style={B.th}>p50</th>
            <th style={B.th}>p95</th>
            <th style={B.th}>tokens</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.config}/${r.kind}`} style={{ background: r.config === baseline && !withKind ? '#161b22' : undefined }}>
              <td style={{ ...B.td, ...B.mono }}>{r.config}{r.config === baseline && !withKind ? <span style={B.dim}> (reference)</span> : ''}</td>
              {withKind && <td style={{ ...B.td, color: '#7d8590' }}>{r.kind}</td>}
              <td style={B.tdNum}>{r.n_ok}/{r.n}</td>
              <td style={B.tdNum}>{num(r.words_median)}</td>
              <td style={B.tdNum}>{pct(r.over_budget_share)}</td>
              <td style={B.tdNum}>{pct(r.citations_invalid_share)}</td>
              <td style={B.tdNum}>{pct(r.ids_in_prose_share)}</td>
              <td style={B.tdNum}>{num(r.tool_calls_median)}</td>
              <td style={B.tdNum}>{pct(r.critic_retry_share)}</td>
              <td style={B.tdNum}>{pct(r.abstained_share)}</td>
              <td style={B.tdNum}>{r.expect_n ? `${pct(r.expect_hit_share)} (${r.expect_n})` : '-'}</td>
              <td style={B.tdNum}>{secs(r.latency_p50_ms)}</td>
              <td style={B.tdNum}>{secs(r.latency_p95_ms)}</td>
              <td style={B.tdNum}>{num(r.tokens_median)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RegressionLists({ r, onPick }: { r: NonNullable<RunView['summary']['regressions']>; onPick: (qid: string, cfg: string) => void }) {
  const list = (title: string, items: typeof r.lost, tone: React.CSSProperties) => (
    <div style={{ marginTop: '0.5rem' }}>
      <div style={{ ...B.lbl, marginBottom: '0.25rem' }}>{title} ({items.length})</div>
      {items.length === 0 ? <div style={B.dim}>none</div> : items.map((c, i) => (
        <div key={i} style={{ ...tone, marginBottom: '0.25rem', cursor: 'pointer' }} onClick={() => onPick(c.question_id, c.config)}>
          <b style={B.mono}>{c.question_id}</b> x <b style={B.mono}>{c.config}</b>
          {c.expect_missing.length ? `: missing ${c.expect_missing.join(', ')}` : ''}
          {c.error ? `: ${c.status}, ${c.error}` : ''}
          {c.answer_excerpt && <div style={{ ...B.dim, marginTop: 2 }}>{c.answer_excerpt}</div>}
        </div>
      ))}
    </div>
  );
  return (
    <div style={{ marginTop: '0.75rem' }}>
      {list(`regressions vs ${r.baseline} on expect_hit`, r.lost, B.error)}
      {list(`cells that failed where ${r.baseline} succeeded`, r.broke, B.warn)}
      {r.won.length > 0 && list(`hit where ${r.baseline} missed (for the record)`, r.won, B.notice)}
    </div>
  );
}

function CellDetail({ record: r, defaults, onClose }: { record: CellRecord; defaults: SandboxConfigResponse['defaults'] | null; onClose: () => void }) {
  return (
    <div style={{ border: '1px solid #30363d', borderRadius: '6px', background: '#161b22', padding: '0.75rem', marginBottom: '0.75rem' }}>
      <div style={B.row}>
        <b style={B.mono}>{r.question_id} x {r.config}</b>
        <span style={{ color: STATUS_COLOR[r.status], fontSize: '0.75rem' }}>{r.status}</span>
        <span style={B.dim}>
          {secs(r.latency_ms)} | {r.words ?? '-'} words | {r.tool_calls ?? '-'} tools | {r.iterations ?? '-'} iteration(s) | {r.tokens_total ?? '-'} tokens | attempt {r.attempt}{r.session_id ? ` | session ...${r.session_id.slice(-6)}` : ''}
        </span>
        {r.expect_hit !== null && r.expect_hit !== undefined && (
          <span style={{ ...B.chip, background: r.expect_hit ? '#1f6f1f' : '#9e3a3a', color: '#fff' }}>{r.expect_hit ? 'expect hit' : `missed: ${(r.expect_missing ?? []).join(', ')}`}</span>
        )}
        {r.abstained && <span style={B.chip}>abstained</span>}
        <button type="button" style={{ ...B.btn, marginLeft: 'auto' }} onClick={onClose}>close</button>
      </div>
      <div style={{ marginTop: '0.5rem' }}><ResolvedChips resolved={r.sandbox_resolved ?? null} defaults={defaults} /></div>
      {r.error && <div style={{ ...B.error, marginTop: '0.5rem' }}>{r.error}</div>}
      <div style={{ ...B.lbl, marginTop: '0.5rem' }}>answer</div>
      <div style={{ whiteSpace: 'pre-wrap', fontSize: '0.8125rem', lineHeight: 1.5, maxHeight: '16rem', overflowY: 'auto', background: '#0d1117', border: '1px solid #21262d', borderRadius: 4, padding: '0.5rem' }}>
        {r.answer || <em style={B.dim}>(no answer text)</em>}
      </div>
      {r.tool_names && r.tool_names.length > 0 && (
        <div style={{ ...B.dim, marginTop: '0.5rem' }}>tools: {r.tool_names.join(' > ')}{r.tool_errors && r.tool_errors.length ? ` | tool errors: ${r.tool_errors.join(', ')}` : ''}</div>
      )}
      {r.critic_verdicts && r.critic_verdicts.length > 0 && (
        <div style={{ marginTop: '0.5rem' }}>
          <div style={B.lbl}>critic</div>
          {r.critic_verdicts.map((v, i) => (
            <div key={i} style={{ fontSize: '0.75rem' }}>
              <span style={{ color: v.sufficient ? '#3fb950' : '#ffa198' }}>{v.sufficient ? 'ok' : 'insufficient'}</span> {v.reason}{v.feedback ? <span style={B.dim}> ({v.feedback})</span> : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
