/**
 * QuestionsEditor.tsx — edit the question set (`scratchpad/questions.jsonl`)
 * as a table with a per-row form, or as raw JSONL. Saving goes through
 * PUT /api/batch/questions, which refuses invalid input with per-line problems
 * — the file on disk is always one the runner accepts.
 *
 * Expectations are never generated here: the form only records what a person
 * (or a documented data check) confirmed, and the `note` field is where the
 * provenance of an `expect` belongs.
 */
'use client';

import { useEffect, useMemo, useState } from 'react';
import { KINDS, type ContainsItem, type Expect, type Kind, type Question } from '@/lib/batch/types';
import { fetchQuestions, saveQuestions, saveQuestionsText, type Problem, type QuestionsView } from '@/lib/batchApi';
import { ApiError } from '@/lib/api';
import { B } from './batchStyles';

const itemsToText = (items: ContainsItem[] | undefined) => (items ?? []).map((i) => (Array.isArray(i) ? i.join(' | ') : i)).join('\n');
const textToItems = (text: string): ContainsItem[] | undefined => {
  const items = text.split('\n').map((l) => l.trim()).filter(Boolean).map((l) => {
    const alts = l.split('|').map((s) => s.trim()).filter(Boolean);
    return alts.length > 1 ? alts : alts[0]!;
  });
  return items.length ? items : undefined;
};

export function expectSummary(ex: Expect | null | undefined): string {
  if (!ex) return '';
  const parts: string[] = [];
  if (ex.contains?.length) parts.push(`contains ${ex.contains.length}`);
  if (ex.abstain != null) parts.push(ex.abstain ? 'abstain' : 'no abstain');
  if (ex.entities?.length) parts.push(`entities ${ex.entities.length}`);
  return parts.join(' · ');
}

export function QuestionsEditor({ onChanged }: { onChanged?: (questions: Question[]) => void }) {
  const [view, setView] = useState<QuestionsView | null>(null);
  const [draft, setDraft] = useState<Question[]>([]);
  const [dirty, setDirty] = useState(false);
  const [mode, setMode] = useState<'table' | 'raw'>('table');
  const [rawText, setRawText] = useState('');
  const [openId, setOpenId] = useState<string | null>(null);
  const [filter, setFilter] = useState<Kind | 'ALL'>('ALL');
  const [problems, setProblems] = useState<Problem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const load = () => fetchQuestions().then((v) => { setView(v); setDraft(v.questions); setRawText(v.text); setProblems(v.problems); setDirty(false); onChanged?.(v.questions); }).catch((e: Error) => setError(e.message));
  useEffect(() => { void load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const counts = useMemo(() => Object.fromEntries(KINDS.map((k) => [k, draft.filter((q) => q.kind === k).length])) as Record<Kind, number>, [draft]);
  const shown = filter === 'ALL' ? draft : draft.filter((q) => q.kind === filter);

  const update = (id: string, patch: Partial<Question>) => { setDraft((d) => d.map((q) => (q.id === id ? { ...q, ...patch } : q))); setDirty(true); };
  const remove = (id: string) => { setDraft((d) => d.filter((q) => q.id !== id)); setDirty(true); if (openId === id) setOpenId(null); };
  const duplicate = (id: string) => {
    const src = draft.find((q) => q.id === id); if (!src) return;
    const copy: Question = { ...src, id: nextId(draft, src.kind), note: src.note ? `${src.note} (copy)` : undefined };
    setDraft((d) => { const i = d.findIndex((q) => q.id === id); return [...d.slice(0, i + 1), copy, ...d.slice(i + 1)]; }); setDirty(true); setOpenId(copy.id);
  };
  const add = () => { const q: Question = { id: nextId(draft, 'FACT'), q: '', kind: 'FACT' }; setDraft((d) => [...d, q]); setDirty(true); setOpenId(q.id); setFilter('ALL'); };

  const save = async () => {
    setSaving(true); setError(null); setNotice(null);
    try {
      const v = mode === 'raw' ? await saveQuestionsText(rawText) : await saveQuestions(draft);
      setView(v); setDraft(v.questions); setRawText(v.text); setProblems(v.problems); setDirty(false); onChanged?.(v.questions);
      setNotice(`saved ${v.questions.length} question(s) → ${v.path} · sha256 ${v.sha256.slice(0, 12)}`);
      if (mode === 'raw') setMode('table');
    } catch (e) {
      const err = e as ApiError & { problems?: Problem[] };
      setError(`${err.code ?? 'Error'}: ${err.message}`);
      if (err.problems) setProblems(err.problems);
    } finally { setSaving(false); }
  };

  const switchMode = (m: 'table' | 'raw') => {
    if (m === mode) return;
    if (m === 'raw') { setRawText(draft.map((q) => JSON.stringify(q)).join('\n') + '\n'); setMode('raw'); return; }
    // raw → table: parse locally so a typo does not lose the text
    try {
      const qs = rawText.split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l) as Question);
      setDraft(qs); setDirty(true); setMode('table');
    } catch (e) { setError(`raw JSONL does not parse: ${(e as Error).message} — fix it or save (the server reports the line)`); }
  };

  return (
    <section style={B.card}>
      <div style={B.cardHead}>
        <h2 style={B.cardTitle}>question set</h2>
        <span style={B.dim}>{draft.length} questions · {draft.filter((q) => q.expect).length} with an expect{view ? ` · sha256 ${view.sha256.slice(0, 12)}` : ''}{dirty ? ' · unsaved changes' : ''}</span>
        <span style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem' }}>
          <button type="button" style={{ ...B.btn, ...(mode === 'table' ? { color: '#e6edf3' } : {}) }} onClick={() => switchMode('table')}>table</button>
          <button type="button" style={{ ...B.btn, ...(mode === 'raw' ? { color: '#e6edf3' } : {}) }} onClick={() => switchMode('raw')}>raw JSONL</button>
          <button type="button" style={B.btn} onClick={() => void load()} disabled={saving}>reload</button>
          <button type="button" style={{ ...B.btnPrimary, padding: '0.25rem 0.75rem', fontSize: '0.75rem', opacity: saving ? 0.6 : 1 }} onClick={() => void save()} disabled={saving}>{saving ? 'saving…' : 'save'}</button>
        </span>
      </div>
      {view && <div style={B.dim}>{view.path}</div>}
      {notice && <div style={{ ...B.notice, marginTop: '0.5rem' }}>{notice}</div>}
      {error && <div style={{ ...B.error, marginTop: '0.5rem' }}>{error}</div>}
      {problems.length > 0 && (
        <div style={{ ...(problems.some((p) => p.severity === 'error') ? B.error : B.warn), marginTop: '0.5rem' }}>
          {problems.map((p, i) => <div key={i}>{p.severity === 'error' ? '✗' : '⚠'} {p.line ? `line ${p.line}` : ''}{p.id ? ` ${p.id}` : ''}: {p.message}</div>)}
        </div>
      )}

      {mode === 'raw' ? (
        <textarea value={rawText} onChange={(e) => { setRawText(e.target.value); setDirty(true); }} style={{ ...B.textarea, minHeight: '24rem', marginTop: '0.5rem' }} spellCheck={false} />
      ) : (
        <>
          <div style={{ ...B.row, marginTop: '0.5rem' }}>
            <button type="button" style={{ ...B.chip, cursor: 'pointer', ...(filter === 'ALL' ? { background: '#1f4068', color: '#e6edf3' } : {}) }} onClick={() => setFilter('ALL')}>all {draft.length}</button>
            {KINDS.map((k) => (
              <button key={k} type="button" style={{ ...B.chip, cursor: 'pointer', ...(filter === k ? { background: '#1f4068', color: '#e6edf3' } : {}) }} onClick={() => setFilter(k)}>{k} {counts[k]}</button>
            ))}
            <button type="button" style={{ ...B.btn, marginLeft: 'auto' }} onClick={add}>+ question</button>
          </div>
          <table style={{ ...B.table, marginTop: '0.5rem' }}>
            <thead><tr><th style={B.th}>id</th><th style={B.th}>kind</th><th style={B.th}>question</th><th style={B.th}>expect</th><th style={B.th}>note</th><th style={B.th}></th></tr></thead>
            <tbody>
              {shown.map((q) => (
                <RowAndEditor key={q.id} q={q} open={openId === q.id} onToggle={() => setOpenId(openId === q.id ? null : q.id)} onChange={(patch) => update(q.id, patch)} onRemove={() => remove(q.id)} onDuplicate={() => duplicate(q.id)} allIds={draft.map((x) => x.id)} />
              ))}
              {shown.length === 0 && <tr><td colSpan={6} style={{ ...B.td, color: '#7d8590', textAlign: 'center', padding: '1rem' }}>no questions{filter !== 'ALL' ? ` of kind ${filter}` : ''} — add one</td></tr>}
            </tbody>
          </table>
        </>
      )}
      <div style={{ ...B.dim, marginTop: '0.5rem', lineHeight: 1.5 }}>
        <b>Do not invent expectations.</b> Fill <code>contains</code> only with values confirmed from real data or by a person, and say in <code>note</code> where the confirmation came from. One item per line; alternatives on one line separated by <code>|</code> (any one suffices); purely numeric items match as whole numbers. An item that already appears in the question is flagged — a refusal that parrots the question would hit it.
      </div>
    </section>
  );
}

function nextId(all: Question[], kind: Kind): string {
  const prefix = kind === 'ASSESSMENT' ? 'assess' : kind === 'AMBIGUOUS' ? 'ambig' : kind === 'PERSONAL' ? 'pers' : kind.toLowerCase();
  const used = new Set(all.map((q) => q.id));
  for (let n = 1; ; n++) { const id = `${prefix}-${String(n).padStart(2, '0')}`; if (!used.has(id)) return id; }
}

function RowAndEditor({ q, open, onToggle, onChange, onRemove, onDuplicate, allIds }: {
  q: Question; open: boolean; onToggle: () => void; onChange: (patch: Partial<Question>) => void; onRemove: () => void; onDuplicate: () => void; allIds: string[];
}) {
  const dupId = allIds.filter((id) => id === q.id).length > 1;
  return (
    <>
      <tr onClick={onToggle} style={{ cursor: 'pointer', background: open ? '#161b22' : undefined }}>
        <td style={{ ...B.td, ...B.mono, whiteSpace: 'nowrap', color: dupId ? '#ffa198' : undefined }}>{q.id}{dupId ? ' (dup)' : ''}</td>
        <td style={{ ...B.td, whiteSpace: 'nowrap' }}>{q.kind}</td>
        <td style={B.td}>{q.q || <em style={{ color: '#7d8590' }}>(empty)</em>}</td>
        <td style={{ ...B.td, whiteSpace: 'nowrap', color: q.expect ? '#e6edf3' : '#7d8590' }}>{expectSummary(q.expect) || '—'}</td>
        <td style={{ ...B.td, color: '#7d8590', maxWidth: '18rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={q.note}>{q.note}</td>
        <td style={{ ...B.td, whiteSpace: 'nowrap' }} onClick={(e) => e.stopPropagation()}>
          <button type="button" style={B.btn} onClick={onDuplicate} title="duplicate">⧉</button>{' '}
          <button type="button" style={B.btnDanger} onClick={onRemove} title="delete">✕</button>
        </td>
      </tr>
      {open && (
        <tr>
          <td colSpan={6} style={{ ...B.td, background: '#161b22', padding: '0.75rem' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '0.75rem' }}>
              <div style={B.col}>
                <div style={B.row}>
                  <label style={B.lbl}>id</label>
                  <input value={q.id} onChange={(e) => onChange({ id: e.target.value.trim() })} style={{ ...B.input, ...B.mono, width: '10rem' }} />
                  <label style={B.lbl}>kind</label>
                  <select value={q.kind} onChange={(e) => onChange({ kind: e.target.value as Kind })} style={B.select}>
                    {KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
                  </select>
                </div>
                <label style={B.lbl}>question</label>
                <textarea value={q.q} onChange={(e) => onChange({ q: e.target.value })} style={{ ...B.textarea, minHeight: '4rem', fontFamily: 'inherit', fontSize: '0.875rem' }} />
                <label style={B.lbl}>note — where the expectation was confirmed, data quirks, volatility</label>
                <textarea value={q.note ?? ''} onChange={(e) => onChange({ note: e.target.value || undefined })} style={{ ...B.textarea, minHeight: '3.5rem', fontFamily: 'inherit' }} />
              </div>
              <div style={B.col}>
                <label style={B.check}>
                  <input type="checkbox" checked={!!q.expect} onChange={(e) => onChange({ expect: e.target.checked ? (q.expect ?? {}) : undefined })} />
                  has an expect (deterministic check)
                </label>
                {q.expect && (
                  <>
                    <label style={B.lbl}>contains — one per line; alternatives with |</label>
                    <textarea value={itemsToText(q.expect.contains)} onChange={(e) => onChange({ expect: { ...q.expect, contains: textToItems(e.target.value) } })} style={{ ...B.textarea, minHeight: '4rem' }} spellCheck={false} />
                    <div style={B.row}>
                      <label style={B.lbl}>abstain</label>
                      <select value={q.expect.abstain === undefined ? '' : String(q.expect.abstain)} onChange={(e) => onChange({ expect: { ...q.expect, ...(e.target.value === '' ? { abstain: undefined } : { abstain: e.target.value === 'true' }) } })} style={B.select}>
                        <option value="">not checked</option>
                        <option value="true">must be NO_COMPANY_RECORDS</option>
                        <option value="false">must NOT abstain</option>
                      </select>
                    </div>
                    <label style={B.lbl}>entities — reported separately, does not affect expect_hit</label>
                    <textarea value={itemsToText(q.expect.entities)} onChange={(e) => onChange({ expect: { ...q.expect, entities: textToItems(e.target.value) } })} style={{ ...B.textarea, minHeight: '2.5rem' }} spellCheck={false} />
                  </>
                )}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
