/**
 * AgentTrace.tsx — renders a structured agent trace: summary badges, final
 * answer, per-iteration cards (tool calls + critic verdict), sources, and
 * optional raw events. Ported from the GUB frontend AgentPage.
 */
'use client';

import { useState } from 'react';
import type { AgentIteration, AgentSource, AgentToolCall } from '@/lib/trace';

export interface AgentResponse {
  text: string;
  sessionId: string;
  durationMs: number;
  iterations: AgentIteration[];
  sources: AgentSource[];
  rawEvents?: unknown[];
}

export function AgentTrace({ response }: { response: AgentResponse }) {
  const toolCallCount = response.iterations.reduce((n, it) => n + it.toolCalls.length, 0);
  return (
    <>
      <div style={S.metaRow}>
        <Badge label={`${Math.round(response.durationMs)}ms`} />
        <Badge label={`${response.iterations.length} iteration${response.iterations.length === 1 ? '' : 's'}`} />
        <Badge label={`${toolCallCount} tool calls`} />
        {response.sources.length > 0 && <Badge label={`${response.sources.length} sources`} />}
      </div>

      <Section title="Final answer">
        <div style={S.answer}>{response.text || <em style={{ color: '#7d8590' }}>(no text)</em>}</div>
      </Section>

      {response.iterations.map((it) => (
        <IterationCard key={it.index} iteration={it} />
      ))}

      {response.sources.length > 0 && (
        <Section title={`Sources (${response.sources.length})`}>
          <ul style={S.sourceList}>
            {response.sources.map((s) => (
              <li key={s.fileId}>
                <a href={`https://drive.google.com/file/d/${s.fileId}/view`} target="_blank" rel="noreferrer" style={S.sourceLink}>
                  {s.name}
                </a>{' '}
                <span style={S.dim}>{s.mimeType ?? 'unknown'}</span>
              </li>
            ))}
          </ul>
        </Section>
      )}

      {response.rawEvents && (
        <Section title={`Raw events (${response.rawEvents.length})`} collapsible defaultOpen={false}>
          <pre style={S.pre}>{JSON.stringify(response.rawEvents, null, 2)}</pre>
        </Section>
      )}
    </>
  );
}

function IterationCard({ iteration }: { iteration: AgentIteration }) {
  const [open, setOpen] = useState(true);
  const verdictColor = iteration.critic?.sufficient ? S.verdictGood : S.verdictBad;
  return (
    <section style={S.section}>
      <button onClick={() => setOpen(!open)} style={S.sectionHeadButton}>
        <span style={S.h2}>Iteration {iteration.index}</span>
        <span style={S.dim}> · {iteration.toolCalls.length} tool calls</span>
        {iteration.critic && (
          <span style={{ ...S.verdictPill, ...verdictColor }}>
            {iteration.critic.sufficient ? '✓ sufficient' : '✗ insufficient'}
          </span>
        )}
        <span style={S.expand}>{open ? '▼' : '▶'}</span>
      </button>
      {open && (
        <div style={S.sectionBody}>
          {iteration.toolCalls.length === 0 && <div style={S.empty}>(no tool calls)</div>}
          {iteration.toolCalls.map((tc, i) => <ToolCallCard key={i} call={tc} />)}

          {iteration.thoughts && <ThoughtsBlock label="Executor thinking" text={iteration.thoughts} />}

          {iteration.text && (
            <div style={{ marginTop: '0.75rem' }}>
              <div style={S.subhead}>Executor text</div>
              <div style={S.executorText}>{iteration.text}</div>
            </div>
          )}

          {iteration.critic && (
            <div style={{ marginTop: '0.75rem' }}>
              <div style={S.subhead}>Critic verdict</div>
              <div style={{ ...S.criticBox, ...verdictColor }}>
                <div style={S.axisRow}>
                  {iteration.critic.infoSufficient !== undefined && (
                    <span style={S.axisChip}>
                      {iteration.critic.infoSufficient ? '✓' : '✗'} retrieved enough
                    </span>
                  )}
                  {iteration.critic.answerSatisfies !== undefined && (
                    <span style={S.axisChip}>
                      {iteration.critic.answerSatisfies ? '✓' : '✗'} answer satisfies
                    </span>
                  )}
                </div>
                <div>
                  <b>{iteration.critic.sufficient ? 'Sufficient' : 'Insufficient'}</b>
                  {iteration.critic.reason && <> — {iteration.critic.reason}</>}
                </div>
                {iteration.critic.feedback && (
                  <div style={S.criticFeedback}>
                    <b>Feedback to executor:</b> {iteration.critic.feedback}
                  </div>
                )}
                {iteration.critic.thoughts && <ThoughtsBlock label="Critic thinking" text={iteration.critic.thoughts} />}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function ToolCallCard({ call }: { call: AgentToolCall }) {
  const [open, setOpen] = useState(false);
  const sourceCount = call.sources?.length ?? 0;
  return (
    <div style={S.toolCard}>
      <button onClick={() => setOpen(!open)} style={S.toolCardHead}>
        <span style={S.toolName}>{call.tool}</span>
        <span style={S.dim}>({summarizeArgs(call.args)})</span>
        {sourceCount > 0 && <span style={S.sourceBadge}>{sourceCount} sources</span>}
        <span style={S.expand}>{open ? '▼' : '▶'}</span>
      </button>
      {open && (
        <div style={S.toolCardBody}>
          <div style={S.subhead}>args</div>
          <pre style={S.preInline}>{JSON.stringify(call.args, null, 2)}</pre>
          <div style={S.subhead}>response</div>
          <pre style={S.preInline}>{JSON.stringify(call.response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

function ThoughtsBlock({ label, text }: { label: string; text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginTop: '0.5rem' }}>
      <button onClick={() => setOpen(!open)} style={S.thoughtsHead}>
        <span>💭 {label}</span>
        <span style={S.expand}>{open ? '▼' : '▶'}</span>
      </button>
      {open && <div style={S.thoughtsBody}>{text}</div>}
    </div>
  );
}

function summarizeArgs(args: Record<string, unknown>): string {
  const keys = Object.keys(args);
  if (keys.length === 0) return '<no args>';
  return keys.map((k) => `${k}: ${shortValue(args[k])}`).join(', ');
}

function shortValue(v: unknown): string {
  if (v === null || v === undefined) return String(v);
  if (typeof v === 'string') return v.length > 30 ? `"${v.slice(0, 28)}…"` : `"${v}"`;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  if (Array.isArray(v)) return `[${v.length}]`;
  if (typeof v === 'object') return '{…}';
  return String(v);
}

function Section({
  title, children, collapsible = false, defaultOpen = true,
}: { title: string; children: React.ReactNode; collapsible?: boolean; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  if (collapsible) {
    return (
      <section style={S.section}>
        <button onClick={() => setOpen(!open)} style={S.sectionHeadButton}>
          <span style={S.h2}>{title}</span>
          <span style={S.expand}>{open ? '▼' : '▶'}</span>
        </button>
        {open && <div style={S.sectionBody}>{children}</div>}
      </section>
    );
  }
  return (
    <section style={S.section}>
      <div style={S.sectionHead}><span style={S.h2}>{title}</span></div>
      <div style={S.sectionBody}>{children}</div>
    </section>
  );
}

function Badge({ label }: { label: string }) {
  return <span style={S.badge}>{label}</span>;
}

const S: Record<string, React.CSSProperties> = {
  metaRow: { display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' },
  badge: { fontSize: '0.75rem', background: '#21262d', color: '#e6edf3', padding: '0.25rem 0.625rem', borderRadius: '999px', fontVariantNumeric: 'tabular-nums' },

  section: { marginBottom: '0.75rem', border: '1px solid #21262d', borderRadius: '6px', overflow: 'hidden' },
  sectionHead: { display: 'flex', alignItems: 'center', gap: '0.5rem', background: '#161b22', padding: '0.5rem 0.75rem' },
  sectionHeadButton: { display: 'flex', alignItems: 'center', gap: '0.5rem', background: '#161b22', padding: '0.5rem 0.75rem', width: '100%', border: 'none', color: 'inherit', font: 'inherit', cursor: 'pointer', textAlign: 'left' },
  sectionBody: { padding: '0.5rem 0.75rem', background: '#0d1117' },
  h2: { fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#7d8590', margin: 0 },
  subhead: { fontSize: '0.6875rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#7d8590', marginTop: '0.5rem', marginBottom: '0.25rem' },
  expand: { marginLeft: 'auto', color: '#7d8590', fontSize: '0.75rem' },
  empty: { color: '#7d8590', fontSize: '0.8125rem', fontStyle: 'italic' },

  answer: { fontSize: '0.9375rem', whiteSpace: 'pre-wrap', lineHeight: 1.5, padding: '0.25rem 0' },

  verdictPill: { fontSize: '0.6875rem', padding: '0.125rem 0.5rem', borderRadius: '3px', fontWeight: 600 },
  verdictGood: { background: '#1f6f1f', color: '#fff' },
  verdictBad: { background: '#9e3a3a', color: '#fff' },
  criticBox: { padding: '0.5rem 0.75rem', borderRadius: '4px', fontSize: '0.8125rem' },
  axisRow: { display: 'flex', gap: '0.5rem', marginBottom: '0.375rem', flexWrap: 'wrap' },
  axisChip: { fontSize: '0.6875rem', background: 'rgba(0,0,0,0.25)', padding: '0.125rem 0.5rem', borderRadius: '999px', fontWeight: 600 },
  criticFeedback: { marginTop: '0.375rem', fontSize: '0.75rem', opacity: 0.92 },

  toolCard: { border: '1px solid #21262d', borderRadius: '4px', marginBottom: '0.375rem' },
  toolCardHead: { display: 'flex', alignItems: 'center', gap: '0.5rem', padding: '0.375rem 0.5rem', background: 'transparent', color: 'inherit', font: 'inherit', border: 'none', width: '100%', cursor: 'pointer', textAlign: 'left' },
  toolName: { fontFamily: 'ui-monospace, monospace', color: '#79c0ff', fontSize: '0.8125rem' },
  toolCardBody: { padding: '0.5rem 0.75rem', borderTop: '1px solid #21262d' },

  preInline: { background: '#161b22', border: '1px solid #21262d', borderRadius: '4px', padding: '0.5rem', fontSize: '0.6875rem', fontFamily: 'ui-monospace, monospace', overflow: 'auto', maxHeight: '20rem', margin: 0 },
  pre: { background: '#161b22', border: '1px solid #30363d', borderRadius: '4px', padding: '0.75rem', fontSize: '0.6875rem', fontFamily: 'ui-monospace, monospace', overflow: 'auto', maxHeight: '60vh' },

  executorText: { background: '#0d1117', padding: '0.5rem 0.75rem', borderRadius: '4px', fontSize: '0.8125rem', whiteSpace: 'pre-wrap', lineHeight: 1.5, color: '#e6edf3' },

  thoughtsHead: { display: 'flex', alignItems: 'center', gap: '0.5rem', width: '100%', background: 'transparent', border: '1px dashed #30363d', borderRadius: '4px', padding: '0.25rem 0.5rem', color: '#a371f7', font: 'inherit', fontSize: '0.75rem', cursor: 'pointer', textAlign: 'left' },
  thoughtsBody: { marginTop: '0.25rem', padding: '0.5rem 0.75rem', background: '#161b22', border: '1px solid #21262d', borderRadius: '4px', fontSize: '0.75rem', whiteSpace: 'pre-wrap', lineHeight: 1.5, color: '#b9a3e3', fontStyle: 'italic' },

  sourceBadge: { fontSize: '0.6875rem', background: '#1f4068', color: '#fff', padding: '0.125rem 0.375rem', borderRadius: '3px' },
  sourceList: { margin: 0, padding: '0 0 0 1.25rem', fontSize: '0.8125rem' },
  sourceLink: { color: '#79c0ff', textDecoration: 'none' },

  dim: { color: '#7d8590', fontSize: '0.8125rem' },
};
