/** batchStyles.ts — shared inline styles for the /batch views (same palette as the console). */
import type React from 'react';

export const B: Record<string, React.CSSProperties> = {
  card: { border: '1px solid #30363d', borderRadius: '6px', background: '#0d1117', padding: '0.75rem', marginBottom: '1rem' },
  cardHead: { display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '0.5rem' },
  cardTitle: { fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#7d8590', margin: 0 },
  row: { display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' },
  col: { display: 'flex', flexDirection: 'column', gap: '0.5rem' },
  lbl: { fontSize: '0.75rem', color: '#7d8590', fontFamily: 'ui-monospace, monospace' },
  dim: { color: '#7d8590', fontSize: '0.75rem' },
  mono: { fontFamily: 'ui-monospace, monospace' },
  input: { background: '#161b22', color: '#e6edf3', border: '1px solid #30363d', borderRadius: '4px', padding: '0.25rem 0.5rem', fontSize: '0.8125rem' },
  inputShort: { width: '4.5rem', background: '#161b22', color: '#e6edf3', border: '1px solid #30363d', borderRadius: '4px', padding: '0.25rem 0.5rem', fontSize: '0.8125rem' },
  select: { background: '#161b22', color: '#e6edf3', border: '1px solid #30363d', borderRadius: '4px', padding: '0.25rem 0.5rem', fontSize: '0.8125rem' },
  textarea: { width: '100%', background: '#161b22', color: '#e6edf3', border: '1px solid #30363d', borderRadius: '4px', padding: '0.5rem', fontSize: '0.75rem', fontFamily: 'ui-monospace, monospace', resize: 'vertical' },
  btn: { background: 'transparent', color: '#79c0ff', border: '1px solid #30363d', borderRadius: '4px', padding: '0.25rem 0.625rem', fontSize: '0.75rem', cursor: 'pointer' },
  btnPrimary: { background: '#238636', color: '#fff', border: 'none', borderRadius: '6px', padding: '0.5rem 1rem', fontSize: '0.875rem', fontWeight: 600, cursor: 'pointer' },
  btnDanger: { background: 'transparent', color: '#ffa198', border: '1px solid #56242a', borderRadius: '4px', padding: '0.25rem 0.625rem', fontSize: '0.75rem', cursor: 'pointer' },
  check: { display: 'flex', alignItems: 'center', gap: '0.375rem', fontSize: '0.75rem', color: '#e6edf3' },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem' },
  th: { textAlign: 'left', padding: '0.25rem 0.5rem', color: '#7d8590', borderBottom: '1px solid #30363d', fontWeight: 500, whiteSpace: 'nowrap' },
  td: { padding: '0.25rem 0.5rem', borderBottom: '1px solid #21262d', verticalAlign: 'top' },
  tdNum: { padding: '0.25rem 0.5rem', borderBottom: '1px solid #21262d', textAlign: 'right', fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' },
  error: { background: '#2d1117', color: '#ffa198', border: '1px solid #56242a', borderRadius: '4px', padding: '0.5rem 0.75rem', fontSize: '0.8125rem', whiteSpace: 'pre-wrap' },
  warn: { background: '#2a2111', color: '#e3b341', border: '1px solid #5a4a1a', borderRadius: '4px', padding: '0.5rem 0.75rem', fontSize: '0.8125rem', whiteSpace: 'pre-wrap' },
  notice: { background: '#1c2a1c', color: '#9be29b', border: '1px solid #2e5a2e', borderRadius: '4px', padding: '0.5rem 0.75rem', fontSize: '0.8125rem' },
  chip: { fontSize: '0.6875rem', background: '#21262d', color: '#c9d1d9', padding: '0.125rem 0.5rem', borderRadius: '999px', fontFamily: 'ui-monospace, monospace' },
  tabs: { display: 'flex', gap: '0.25rem', borderBottom: '1px solid #21262d', marginBottom: '1rem' },
  tab: { background: 'transparent', border: 'none', color: '#7d8590', padding: '0.5rem 0.875rem', fontSize: '0.875rem', cursor: 'pointer', borderBottom: '2px solid transparent' },
  tabActive: { color: '#e6edf3', borderBottom: '2px solid #f78166' },
  pre: { background: '#161b22', border: '1px solid #30363d', borderRadius: '4px', padding: '0.5rem', fontSize: '0.6875rem', overflowX: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word', margin: 0 },
};

export const STATUS_COLOR: Record<string, string> = {
  ok: '#2ea043', miss: '#d29922', empty: '#f85149', error: '#f85149', transport: '#f85149', timeout: '#a371f7', auth: '#db4d4d', rate_limited: '#db6d28', running: '#58a6ff', pending: '#30363d',
};

export const pct = (x: number | null | undefined) => (x === null || x === undefined ? '–' : `${Math.round(x * 100)}%`);
export const secs = (ms: number | null | undefined) => (ms === null || ms === undefined ? '–' : `${(ms / 1000).toFixed(1)}s`);
export const num = (x: number | null | undefined) => (x === null || x === undefined ? '–' : String(x));
