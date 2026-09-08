/**
 * ArmsEditor.tsx — edit the matrix (`scratchpad/configs.json`): each arm is a
 * name plus the same sandbox config form the single-run console uses
 * (`ConfigPanel`), validated against the active engine's lists. An arm with no
 * overrides is the ordinary-run baseline; it stays empty on purpose.
 */
'use client';

import { useEffect, useMemo, useState } from 'react';
import { ConfigPanel, emptyForm, formToOverrides, type SandboxForm } from '@/components/ConfigPanel';
import type { SandboxConfigResponse } from '@/lib/api';
import { ApiError } from '@/lib/api';
import { validateOverrides, type SandboxLists, type SandboxOverrides, type ThinkingLevel } from '@/lib/sandbox';
import type { Arm } from '@/lib/batch/types';
import { fetchArms, saveArms, type Problem } from '@/lib/batchApi';
import { B } from './batchStyles';

interface ArmDraft { key: number; name: string; enabled: boolean; form: SandboxForm }

/** Inverse of formToOverrides for what the file holds. */
export function overridesToForm(o: SandboxOverrides): SandboxForm {
  const f = emptyForm();
  f.model = o.model ?? '';
  f.thinking_level = (o.thinking_level ?? '') as SandboxForm['thinking_level'];
  f.critic_thinking_level = (o.critic_thinking_level ?? '') as SandboxForm['critic_thinking_level'];
  f.temperature = o.temperature != null ? String(o.temperature) : '';
  if (o.executor_instruction != null) { f.executor_mode = 'inline'; f.executor_inline = o.executor_instruction; } else f.executor_variant = o.executor_variant ?? '';
  if (o.critic_instruction != null) { f.critic_mode = 'inline'; f.critic_inline = o.critic_instruction; } else f.critic_variant = o.critic_variant ?? '';
  f.critic_enabled = o.critic_enabled !== false;
  f.label = o.label ?? '';
  return f;
}

let keySeq = 1;

export function ArmsEditor({ options, onChanged }: { options: SandboxConfigResponse | null; onChanged?: (arms: Arm[]) => void }) {
  const [arms, setArms] = useState<ArmDraft[]>([]);
  const [path, setPath] = useState('');
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [problems, setProblems] = useState<Problem[]>([]);

  const lists: SandboxLists | null = useMemo(
    () => (options ? { models: options.models, thinkingLevelModels: options.thinkingLevelModels, variants: options.variants } : null),
    [options],
  );

  const load = () => fetchArms().then((v) => {
    setArms(v.arms.map((a) => ({ key: keySeq++, name: a.name, enabled: a.enabled !== false, form: overridesToForm(a.overrides) })));
    setPath(v.path); setProblems(v.problems); setDirty(false); onChanged?.(v.arms);
  }).catch((e: Error) => setError(e.message));
  useEffect(() => { void load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const checks = useMemo(() => arms.map((a) => (lists ? validateOverrides(formToOverrides(a.form), lists) : null)), [arms, lists]);
  const names = arms.map((a) => a.name.trim());
  const nameProblems = arms.map((a, i) => (!a.name.trim() ? 'name required' : names.indexOf(a.name.trim()) !== i ? 'duplicate name' : /\s/.test(a.name) ? 'no spaces in names' : null));
  const blocking = checks.some((c) => c && !c.ok) || nameProblems.some(Boolean);

  const patch = (key: number, p: Partial<ArmDraft>) => { setArms((d) => d.map((a) => (a.key === key ? { ...a, ...p } : a))); setDirty(true); };
  const remove = (key: number) => { setArms((d) => d.filter((a) => a.key !== key)); setDirty(true); };
  const add = () => { setArms((d) => [...d, { key: keySeq++, name: `arm_${d.length + 1}`, enabled: true, form: emptyForm() }]); setDirty(true); };
  const duplicate = (key: number) => { setArms((d) => { const i = d.findIndex((a) => a.key === key); const src = d[i]!; return [...d.slice(0, i + 1), { key: keySeq++, name: `${src.name}_copy`, enabled: src.enabled, form: { ...src.form } }, ...d.slice(i + 1)]; }); setDirty(true); };

  const save = async () => {
    if (!lists || blocking) return;
    setSaving(true); setError(null); setNotice(null);
    try {
      const payload: Arm[] = arms.map((a, i) => ({ name: a.name.trim(), overrides: (checks[i]!.ok ? (checks[i] as { ok: true; value: SandboxOverrides | null }).value : null) ?? {}, ...(a.enabled ? {} : { enabled: false }) }));
      const v = await saveArms(payload);
      setArms(v.arms.map((a) => ({ key: keySeq++, name: a.name, enabled: a.enabled !== false, form: overridesToForm(a.overrides) })));
      setProblems(v.problems); setDirty(false); onChanged?.(v.arms);
      setNotice(`saved ${v.arms.length} arm(s) → ${v.path}`);
    } catch (e) {
      const err = e as ApiError & { problems?: Problem[] };
      setError(`${err.code ?? 'Error'}: ${err.message}`);
      if (err.problems) setProblems(err.problems);
    } finally { setSaving(false); }
  };

  return (
    <section style={B.card}>
      <div style={B.cardHead}>
        <h2 style={B.cardTitle}>arms (configurations)</h2>
        <span style={B.dim}>{arms.length} arms · {arms.filter((a) => a.enabled).length} enabled{dirty ? ' · unsaved changes' : ''}</span>
        <span style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem' }}>
          <button type="button" style={B.btn} onClick={add}>+ arm</button>
          <button type="button" style={B.btn} onClick={() => void load()} disabled={saving}>reload</button>
          <button type="button" style={{ ...B.btnPrimary, padding: '0.25rem 0.75rem', fontSize: '0.75rem', opacity: saving || blocking || !lists ? 0.6 : 1 }} onClick={() => void save()} disabled={saving || blocking || !lists} title={blocking ? 'fix the highlighted arms first' : undefined}>{saving ? 'saving…' : 'save'}</button>
        </span>
      </div>
      {path && <div style={B.dim}>{path}</div>}
      {notice && <div style={{ ...B.notice, marginTop: '0.5rem' }}>{notice}</div>}
      {error && <div style={{ ...B.error, marginTop: '0.5rem' }}>{error}</div>}
      {problems.length > 0 && <div style={{ ...B.error, marginTop: '0.5rem' }}>{problems.map((p, i) => <div key={i}>{p.id ? `${p.id}: ` : ''}{p.message}</div>)}</div>}

      <div style={{ ...B.col, marginTop: '0.75rem' }}>
        {arms.map((a, i) => (
          <div key={a.key} style={{ border: `1px solid ${nameProblems[i] ? '#8b3a3a' : '#21262d'}`, borderRadius: '6px', padding: '0.5rem' }}>
            <div style={{ ...B.row, marginBottom: '0.5rem' }}>
              <label style={B.check} title="Disabled arms stay in the file but are left out of a run">
                <input type="checkbox" checked={a.enabled} onChange={(e) => patch(a.key, { enabled: e.target.checked })} />
              </label>
              <label style={B.lbl}>name</label>
              <input value={a.name} onChange={(e) => patch(a.key, { name: e.target.value })} style={{ ...B.input, ...B.mono, width: '12rem' }} />
              {nameProblems[i] && <span style={{ color: '#ffa198', fontSize: '0.75rem' }}>{nameProblems[i]}</span>}
              {Object.keys(formToOverridesClean(a.form)).length === 0 && <span style={B.chip} title="No overrides: an ordinary run, no sandbox_resolved. The reference arm.">ordinary run · baseline</span>}
              <span style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem' }}>
                <button type="button" style={B.btn} onClick={() => duplicate(a.key)}>⧉ duplicate</button>
                <button type="button" style={B.btnDanger} onClick={() => remove(a.key)}>✕ remove</button>
              </span>
            </div>
            <ConfigPanel title={a.name || '?'} form={a.form} onChange={(f) => patch(a.key, { form: f })} options={options} validation={checks[i] && !checks[i]!.ok ? (checks[i] as { ok: false; message: string }).message : null} defaultOpen={false} />
          </div>
        ))}
        {arms.length === 0 && <div style={{ ...B.dim, textAlign: 'center', padding: '1rem' }}>no arms yet — add one; an arm with no overrides is the baseline</div>}
      </div>
      <div style={{ ...B.dim, marginTop: '0.5rem', lineHeight: 1.5 }}>
        Each arm becomes <code>state.sandbox</code> for its cells; the runner adds <code>label: batch:&lt;run&gt;:&lt;arm&gt;</code> to every non-empty arm. Leave the label empty unless you want a fixed one. A model other than the default needs <code>DYNAMIC</code> thinking on both roles.
      </div>
    </section>
  );
}

function formToOverridesClean(form: SandboxForm): Record<string, unknown> {
  const raw = formToOverrides(form);
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(raw)) {
    if (v === undefined || v === null || v === '') continue;
    if (k === 'critic_enabled' && v === true) continue;
    out[k] = v;
  }
  return out;
}

export type { ThinkingLevel };
