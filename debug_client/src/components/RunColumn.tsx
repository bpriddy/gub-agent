/**
 * RunColumn.tsx — one arm of an A/B: a summary header (what ran, how long,
 * how much) over the unchanged `AgentTrace`. Also exports `ResolvedChips`,
 * the provenance line the single-run view shows above its trace.
 *
 * A run with no `resolved` is a BASELINE run, not a broken one: the echo is
 * silent when no override applies. The header then says "baseline" and
 * describes the deployed defaults `/api/config` reported.
 */
'use client';

import { AgentTrace } from '@/components/AgentTrace';
import type { AgentRunResponse } from '@/lib/api';
import { baselineResolved, promptLabel, type SandboxDefaults, type SandboxResolved } from '@/lib/sandbox';

export type RunState =
  | { status: 'idle' }
  | { status: 'running'; startedAt: number }
  | { status: 'ok'; response: AgentRunResponse }
  | { status: 'error'; code: string; message: string; resolved: SandboxResolved | null };

/** What the summaries and the diff use: provenance, or the synthesised baseline. */
export function effectiveResolved(resolved: SandboxResolved | null, defaults: SandboxDefaults | null): SandboxResolved | null {
  if (resolved) return resolved;
  return defaults ? baselineResolved(defaults) : null;
}

export function RunColumn({ side, run, defaults }: { side: 'A' | 'B'; run: RunState; defaults: SandboxDefaults | null }) {
  return (
    <div style={S.column}>
      <div style={S.head}>
        <div style={S.sideRow}>
          <span style={S.side}>{side}</span>
          {run.status === 'ok' && <RunTitle resolved={run.response.resolved} />}
          {run.status === 'error' && <span style={S.failedTitle}>failed · {run.code}</span>}
          {run.status === 'running' && <span style={S.dim}>running…</span>}
          {run.status === 'idle' && <span style={S.dim}>not run yet</span>}
        </div>

        {run.status === 'ok' && (
          <>
            <ResolvedChips resolved={run.response.resolved} defaults={defaults} />
            <div style={S.numbers}>
              <Num label="duration" value={`${(run.response.durationMs / 1000).toFixed(1)}s`} />
              <Num label="words" value={String(run.response.answerWordCount)} />
              <Num label="tool calls" value={String(run.response.toolCallCount)} />
              <Num label="iterations" value={String(run.response.iterations.length)} />
              <Verdict response={run.response} />
              <Num label="session" value={`…${run.response.sessionId.slice(-6)}`} title={run.response.sessionId} mono />
              <Num label="started" value={new Date(run.response.startedAt).toLocaleTimeString()} />
            </div>
          </>
        )}
        {run.status === 'error' && run.resolved && (
          <ResolvedChips resolved={run.resolved} defaults={defaults} />
        )}
      </div>

      {run.status === 'error' && <div style={S.error}>{run.message}</div>}
      {run.status === 'ok' && <AgentTrace response={run.response} />}
      {run.status === 'running' && <div style={S.placeholder}>Waiting for the engine…</div>}
    </div>
  );
}

function RunTitle({ resolved }: { resolved: SandboxResolved | null }) {
  if (!resolved) {
    return (
      <span style={S.title} title="No sandbox_resolved in the stream — expected: the echo is silent when no override applies.">
        baseline <span style={S.dim}>· deployed defaults</span>
      </span>
    );
  }
  return (
    <span style={S.title}>
      {resolved.label ? `“${resolved.label}”` : 'sandbox run'}
      {resolved.overridden_keys.length > 0 && (
        <span style={S.dim}> · overrides: {resolved.overridden_keys.join(', ')}</span>
      )}
    </span>
  );
}

/** The resolved config as chips. Reads provenance when present, else the deployed defaults. */
export function ResolvedChips({ resolved, defaults }: { resolved: SandboxResolved | null; defaults: SandboxDefaults | null }) {
  const r = effectiveResolved(resolved, defaults);
  if (!r) return null;
  const isBaseline = resolved === null;
  const tone = isBaseline ? S.chipBaseline : S.chip;
  return (
    <div style={S.chips}>
      <span style={tone}>model {r.model}</span>
      <span style={tone}>thinking {r.thinking_level}</span>
      <span style={tone}>temp {r.temperature ?? 'default'}</span>
      <span style={tone}>critic {r.critic_enabled ? `on · ${r.critic_thinking_level}` : 'off'}</span>
      <span style={tone}>executor {promptLabel(r.executor_prompt_source, r.executor_prompt_sha256)}</span>
      <span style={tone}>critic prompt {promptLabel(r.critic_prompt_source, r.critic_prompt_sha256)}</span>
      {isBaseline && <span style={S.chipNote} title="sandbox_echo emits nothing when no override applies">no provenance event — baseline</span>}
    </div>
  );
}

function Verdict({ response }: { response: AgentRunResponse }) {
  let verdict: { sufficient: boolean } | undefined;
  for (let i = response.iterations.length - 1; i >= 0; i--) {
    const c = response.iterations[i]!.critic;
    if (c) { verdict = c; break; }
  }
  if (!verdict) return <Num label="critic" value="no verdict" />;
  return (
    <span style={S.num}>
      <span style={S.numLabel}>critic</span>
      <span style={{ ...S.verdict, ...(verdict.sufficient ? S.verdictGood : S.verdictBad) }}>
        {verdict.sufficient ? '✓ sufficient' : '✗ insufficient'}
      </span>
    </span>
  );
}

function Num({ label, value, title, mono }: { label: string; value: string; title?: string; mono?: boolean }) {
  return (
    <span style={S.num} title={title}>
      <span style={S.numLabel}>{label}</span>
      <span style={{ ...S.numValue, ...(mono ? S.mono : {}) }}>{value}</span>
    </span>
  );
}

const S: Record<string, React.CSSProperties> = {
  column: { minWidth: 0 },
  head: { border: '1px solid #30363d', borderRadius: '6px', background: '#161b22', padding: '0.5rem 0.75rem', marginBottom: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' },
  sideRow: { display: 'flex', alignItems: 'baseline', gap: '0.5rem', flexWrap: 'wrap' },
  side: { fontSize: '0.875rem', fontWeight: 700, background: '#21262d', borderRadius: '4px', padding: '0 0.5rem', color: '#e6edf3' },
  title: { fontSize: '0.875rem', fontWeight: 600 },
  failedTitle: { fontSize: '0.875rem', fontWeight: 600, color: '#ffa198' },
  dim: { color: '#7d8590', fontSize: '0.75rem', fontWeight: 400 },

  chips: { display: 'flex', gap: '0.375rem', flexWrap: 'wrap' },
  chip: { fontSize: '0.6875rem', background: '#1f4068', color: '#e6edf3', padding: '0.125rem 0.5rem', borderRadius: '999px', fontFamily: 'ui-monospace, monospace' },
  chipBaseline: { fontSize: '0.6875rem', background: '#21262d', color: '#c9d1d9', padding: '0.125rem 0.5rem', borderRadius: '999px', fontFamily: 'ui-monospace, monospace' },
  chipNote: { fontSize: '0.6875rem', color: '#7d8590', padding: '0.125rem 0.25rem', fontStyle: 'italic' },

  numbers: { display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'baseline' },
  num: { display: 'inline-flex', flexDirection: 'column', gap: '0.0625rem' },
  numLabel: { fontSize: '0.625rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#7d8590' },
  numValue: { fontSize: '0.875rem', fontVariantNumeric: 'tabular-nums', color: '#e6edf3' },
  mono: { fontFamily: 'ui-monospace, monospace', fontSize: '0.75rem' },
  verdict: { fontSize: '0.6875rem', padding: '0.125rem 0.5rem', borderRadius: '3px', fontWeight: 600, alignSelf: 'flex-start' },
  verdictGood: { background: '#1f6f1f', color: '#fff' },
  verdictBad: { background: '#9e3a3a', color: '#fff' },

  error: { background: '#2d1117', color: '#ffa198', border: '1px solid #56242a', borderRadius: '4px', padding: '0.5rem 0.75rem', fontSize: '0.8125rem', whiteSpace: 'pre-wrap', wordBreak: 'break-word' },
  placeholder: { color: '#7d8590', fontSize: '0.875rem', padding: '2rem 0', textAlign: 'center' },
};
