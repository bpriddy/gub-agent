/**
 * ConfigPanel.tsx — the sandbox config for one run (or one A/B arm), plus
 * the form model behind it and the engine badge.
 *
 * Every option list comes from `GET /api/config`; nothing here knows a model
 * or variant name. The form is what is ASKED FOR — `sandbox_resolved` in the
 * trace is what ran, and the summaries/diff read that, never this.
 */
'use client';

import { useState } from 'react';
import type { SandboxConfigResponse } from '@/lib/api';
import { BASELINE_VARIANT, utf8Bytes, type ThinkingLevel } from '@/lib/sandbox';

// ── Form model ─────────────────────────────────────────────────────────────

export type PromptMode = 'variant' | 'inline';

/** UI state for one config. `''` everywhere means "leave the deployed default". */
export interface SandboxForm {
  model: string;
  thinking_level: '' | ThinkingLevel;
  critic_thinking_level: '' | ThinkingLevel;
  /** Text, parsed by the validator so a typo says "must be a number" rather than vanishing. */
  temperature: string;
  executor_mode: PromptMode;
  /** '' = no override (deployed prompt); `baseline` = the registry's copy, which gets a hash. */
  executor_variant: string;
  executor_inline: string;
  critic_mode: PromptMode;
  critic_variant: string;
  critic_inline: string;
  critic_enabled: boolean;
  label: string;
}

export function emptyForm(): SandboxForm {
  return {
    model: '',
    thinking_level: '',
    critic_thinking_level: '',
    temperature: '',
    executor_mode: 'variant',
    executor_variant: '',
    executor_inline: '',
    critic_mode: 'variant',
    critic_variant: '',
    critic_inline: '',
    critic_enabled: true,
    label: '',
  };
}

/** Raw `config` for the request body — `validateOverrides` normalises it. */
export function formToOverrides(form: SandboxForm): Record<string, unknown> {
  return {
    model: form.model || undefined,
    thinking_level: form.thinking_level || undefined,
    critic_thinking_level: form.critic_thinking_level || undefined,
    temperature: form.temperature.trim() || undefined,
    executor_instruction: form.executor_mode === 'inline' ? form.executor_inline : undefined,
    executor_variant: form.executor_mode === 'variant' ? form.executor_variant || undefined : undefined,
    critic_instruction: form.critic_mode === 'inline' ? form.critic_inline : undefined,
    critic_variant: form.critic_mode === 'variant' ? form.critic_variant || undefined : undefined,
    critic_enabled: form.critic_enabled,
    label: form.label.trim() || undefined,
  };
}

/** Restore from localStorage without trusting the stored shape. */
export function coerceForm(raw: unknown): SandboxForm {
  const base = emptyForm();
  if (!raw || typeof raw !== 'object') return base;
  const r = raw as Record<string, unknown>;
  const str = (k: keyof SandboxForm) => (typeof r[k] === 'string' ? (r[k] as string) : (base[k] as string));
  return {
    model: str('model'),
    thinking_level: str('thinking_level') as SandboxForm['thinking_level'],
    critic_thinking_level: str('critic_thinking_level') as SandboxForm['critic_thinking_level'],
    temperature: str('temperature'),
    executor_mode: r.executor_mode === 'inline' ? 'inline' : 'variant',
    executor_variant: str('executor_variant'),
    executor_inline: str('executor_inline'),
    critic_mode: r.critic_mode === 'inline' ? 'inline' : 'variant',
    critic_variant: str('critic_variant'),
    critic_inline: str('critic_inline'),
    critic_enabled: r.critic_enabled !== false,
    label: str('label'),
  };
}

/** One line for the collapsed panel header. */
export function summarizeForm(form: SandboxForm): string {
  const parts: string[] = [];
  if (form.label.trim()) parts.push(`“${form.label.trim()}”`);
  if (form.model) parts.push(form.model);
  if (form.thinking_level) parts.push(`thinking ${form.thinking_level}`);
  if (form.temperature.trim()) parts.push(`t=${form.temperature.trim()}`);
  if (form.executor_mode === 'inline') parts.push(form.executor_inline.trim() ? 'executor inline' : 'executor inline (empty)');
  else if (form.executor_variant) parts.push(`executor ${form.executor_variant}`);
  if (form.critic_mode === 'inline') parts.push(form.critic_inline.trim() ? 'critic inline' : 'critic inline (empty)');
  else if (form.critic_variant) parts.push(`critic ${form.critic_variant}`);
  if (!form.critic_enabled) parts.push('critic off');
  else if (form.critic_thinking_level) parts.push(`critic thinking ${form.critic_thinking_level}`);
  return parts.length > 0 ? parts.join(' · ') : 'no overrides — deployed defaults';
}

// ── Panel ──────────────────────────────────────────────────────────────────

export function ConfigPanel({
  title,
  form,
  onChange,
  options,
  validation,
  disabled = false,
  actions,
  defaultOpen = true,
}: {
  title?: string;
  form: SandboxForm;
  onChange: (next: SandboxForm) => void;
  options: SandboxConfigResponse | null;
  /** The validator's message for the current form, or null when it is runnable. */
  validation: string | null;
  disabled?: boolean;
  actions?: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const set = <K extends keyof SandboxForm>(key: K, value: SandboxForm[K]) => onChange({ ...form, [key]: value });

  const defaults = options?.defaults;
  const modelNeedsDynamic = !!(form.model && options && !options.thinkingLevelModels.includes(form.model));
  const maxBytes = options?.maxPromptBytes ?? 64 * 1024;

  return (
    <section style={{ ...S.panel, ...(validation ? S.panelInvalid : {}) }}>
      <button onClick={() => setOpen(!open)} style={S.head} type="button">
        <span style={S.headTitle}>{title ? `Config ${title}` : 'Sandbox config'}</span>
        {!open && <span style={S.headSummary}>{summarizeForm(form)}</span>}
        {actions && <span style={S.headActions} onClick={(e) => e.stopPropagation()}>{actions}</span>}
        <span style={S.expand}>{open ? '▼' : '▶'}</span>
      </button>

      {open && (
        <div style={S.body}>
          {!options && <div style={S.dim}>Loading options from /api/config…</div>}

          <div style={S.row}>
            <label style={S.lbl}>model</label>
            <select value={form.model} onChange={(e) => set('model', e.target.value)} style={S.select} disabled={disabled || !options}>
              <option value="">deployed default{defaults ? ` (${defaults.model})` : ''}</option>
              {options?.models.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <label style={S.lbl}>thinking</label>
            <select value={form.thinking_level} onChange={(e) => set('thinking_level', e.target.value as SandboxForm['thinking_level'])} style={S.select} disabled={disabled || !options}>
              <option value="">default{defaults ? ` (${defaults.thinkingLevel})` : ''}</option>
              {options?.thinkingLevels.map((l) => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
            <label style={S.lbl}>temperature</label>
            <input
              value={form.temperature}
              onChange={(e) => set('temperature', e.target.value)}
              placeholder="default"
              inputMode="decimal"
              style={S.inputShort}
              disabled={disabled}
              title="0.0–2.0 — an out-of-range value is refused, never clamped"
            />
          </div>
          {modelNeedsDynamic && (
            <div style={S.hint}>
              {form.model} does not accept a named thinking level — set thinking and critic thinking to DYNAMIC (or turn the critic off).{' '}
              <button
                type="button"
                style={S.linkBtn}
                onClick={() => onChange({ ...form, thinking_level: 'DYNAMIC', critic_thinking_level: form.critic_enabled ? 'DYNAMIC' : form.critic_thinking_level })}
              >
                set DYNAMIC
              </button>
            </div>
          )}

          <PromptRow
            role="executor"
            mode={form.executor_mode}
            variant={form.executor_variant}
            inline={form.executor_inline}
            variants={options?.variants.executor ?? []}
            maxBytes={maxBytes}
            disabled={disabled || !options}
            onMode={(m) => set('executor_mode', m)}
            onVariant={(v) => set('executor_variant', v)}
            onInline={(t) => set('executor_inline', t)}
          />

          <PromptRow
            role="critic"
            mode={form.critic_mode}
            variant={form.critic_variant}
            inline={form.critic_inline}
            variants={options?.variants.critic ?? []}
            maxBytes={maxBytes}
            disabled={disabled || !options}
            onMode={(m) => set('critic_mode', m)}
            onVariant={(v) => set('critic_variant', v)}
            onInline={(t) => set('critic_inline', t)}
            trailing={
              <>
                <label style={S.check}>
                  <input type="checkbox" checked={form.critic_enabled} onChange={(e) => set('critic_enabled', e.target.checked)} disabled={disabled} />
                  critic enabled
                </label>
                <label style={S.lbl}>critic thinking</label>
                <select
                  value={form.critic_thinking_level}
                  onChange={(e) => set('critic_thinking_level', e.target.value as SandboxForm['critic_thinking_level'])}
                  style={S.select}
                  disabled={disabled || !options || !form.critic_enabled}
                >
                  <option value="">default{defaults ? ` (${defaults.criticThinkingLevel})` : ''}</option>
                  {options?.thinkingLevels.map((l) => (
                    <option key={l} value={l}>{l}</option>
                  ))}
                </select>
              </>
            }
          />

          <div style={S.row}>
            <label style={S.lbl}>label</label>
            <input
              value={form.label}
              onChange={(e) => set('label', e.target.value)}
              placeholder="free-form, provenance only — e.g. concise-v2"
              style={S.input}
              disabled={disabled}
            />
          </div>

          <div style={S.footnote}>
            “deployed default” sends no override for that knob. The registry’s <code>{BASELINE_VARIANT}</code> variant is the
            same prompt text but runs through the sandbox path and gets a hash — pick it for a deliberate baseline arm so the
            two arms are labelled alike.
          </div>

          {validation && <div style={S.invalid}>{validation}</div>}
        </div>
      )}
    </section>
  );
}

function PromptRow({
  role, mode, variant, inline, variants, maxBytes, disabled, onMode, onVariant, onInline, trailing,
}: {
  role: 'executor' | 'critic';
  mode: PromptMode;
  variant: string;
  inline: string;
  variants: string[];
  maxBytes: number;
  disabled: boolean;
  onMode: (m: PromptMode) => void;
  onVariant: (v: string) => void;
  onInline: (t: string) => void;
  trailing?: React.ReactNode;
}) {
  const bytes = utf8Bytes(inline);
  const over = bytes > maxBytes;
  return (
    <>
      <div style={S.row}>
        <label style={S.lbl}>{role}</label>
        <select value={mode === 'inline' ? '__inline__' : variant} onChange={(e) => {
          if (e.target.value === '__inline__') onMode('inline');
          else { onMode('variant'); onVariant(e.target.value); }
        }} style={S.select} disabled={disabled}>
          <option value="">deployed prompt (no override)</option>
          {variants.map((v) => (
            <option key={v} value={v}>{v}</option>
          ))}
          <option value="__inline__">inline text…</option>
        </select>
        {trailing}
      </div>
      {mode === 'inline' && (
        <div style={S.inlineWrap}>
          <textarea
            value={inline}
            onChange={(e) => onInline(e.target.value)}
            placeholder={`Full ${role} prompt. Literal {braces} are fine — the instruction is a callable, so ADK's state substitution is bypassed. The current-date block is appended by the agent; do not include one.`}
            style={{ ...S.textarea, ...(over ? S.textareaOver : {}) }}
            disabled={disabled}
            spellCheck={false}
          />
          <div style={{ ...S.bytes, ...(over ? S.bytesOver : {}) }}>
            {bytes.toLocaleString()} / {maxBytes.toLocaleString()} bytes
            {inline.trim() === '' && <span style={S.dim}> · empty — no override will be sent</span>}
          </div>
        </div>
      )}
    </>
  );
}

// ── Engine badge ───────────────────────────────────────────────────────────

export function EngineBadge({ config, error }: { config: SandboxConfigResponse | null; error: string | null }) {
  if (error) {
    return <div style={{ ...S.badge, ...S.badgeUnknown }} title={error}>engine: /api/config failed — {error}</div>;
  }
  if (!config) return <div style={{ ...S.badge, ...S.badgeUnknown }}>engine: …</div>;
  const { engine } = config;
  const isProd = engine.isProd;
  const tail = engine.id.slice(-6);
  const sandboxWord = engine.sandboxEnabled === null ? 'SANDBOX_ENABLED unread' : engine.sandboxEnabled ? 'SANDBOX_ENABLED=1' : 'SANDBOX_ENABLED=0';
  return (
    <div style={{ ...S.badge, ...(isProd ? S.badgeProd : S.badgeDev) }} title={`${engine.id}${engine.displayName ? ` · ${engine.displayName}` : ''} · lists from ${engine.listsSource}`}>
      <div style={S.badgeLine}>
        <b>{isProd ? 'PROD' : engine.target}</b> · {engine.displayName ?? 'engine'} · …{tail} · <span style={S.mono}>{sandboxWord}</span>
      </div>
      {isProd && (
        <div style={S.badgeWarn}>
          ⚠ This is the production engine serving the Chat bot. It ignores <code>state.sandbox</code> outright, so every
          config below is inert here and an A/B will compare the baseline against itself. Set{' '}
          <code>SANDBOX_AGENT_ENGINE_ID</code> (or <code>AGENT_ENGINE_TARGET=sandbox</code>) in <code>.env.local</code>.
        </div>
      )}
      {!isProd && engine.sandboxEnabled === null && (
        <div style={S.badgeNote}>
          could not read the engine’s baked-in env (aiplatform.reasoningEngines.get) — prod detection relies on the id only.
        </div>
      )}
    </div>
  );
}

const S: Record<string, React.CSSProperties> = {
  panel: { border: '1px solid #30363d', borderRadius: '6px', marginBottom: '1rem', background: '#0d1117', overflow: 'hidden' },
  panelInvalid: { borderColor: '#8b3a3a' },
  head: { display: 'flex', alignItems: 'center', gap: '0.75rem', width: '100%', background: '#161b22', border: 'none', color: 'inherit', font: 'inherit', padding: '0.5rem 0.75rem', cursor: 'pointer', textAlign: 'left' },
  headTitle: { fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#7d8590', whiteSpace: 'nowrap' },
  headSummary: { fontSize: '0.75rem', color: '#e6edf3', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 },
  headActions: { marginLeft: 'auto', display: 'flex', gap: '0.5rem' },
  expand: { color: '#7d8590', fontSize: '0.75rem' },
  body: { padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' },

  row: { display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' },
  lbl: { fontSize: '0.75rem', color: '#7d8590', minWidth: '4.5rem', fontFamily: 'ui-monospace, monospace' },
  select: { background: '#161b22', color: '#e6edf3', border: '1px solid #30363d', borderRadius: '4px', padding: '0.25rem 0.5rem', fontSize: '0.8125rem', minWidth: '10rem' },
  input: { flex: 1, minWidth: '12rem', background: '#161b22', color: '#e6edf3', border: '1px solid #30363d', borderRadius: '4px', padding: '0.25rem 0.5rem', fontSize: '0.8125rem' },
  inputShort: { width: '5rem', background: '#161b22', color: '#e6edf3', border: '1px solid #30363d', borderRadius: '4px', padding: '0.25rem 0.5rem', fontSize: '0.8125rem' },
  check: { display: 'flex', alignItems: 'center', gap: '0.375rem', fontSize: '0.75rem', color: '#e6edf3' },

  inlineWrap: { marginLeft: '5rem' },
  textarea: { width: '100%', minHeight: '9rem', background: '#161b22', color: '#e6edf3', border: '1px solid #30363d', borderRadius: '4px', padding: '0.5rem', fontSize: '0.75rem', fontFamily: 'ui-monospace, monospace', resize: 'vertical' },
  textareaOver: { borderColor: '#f85149' },
  bytes: { fontSize: '0.6875rem', color: '#7d8590', marginTop: '0.125rem', fontVariantNumeric: 'tabular-nums' },
  bytesOver: { color: '#ffa198' },

  hint: { fontSize: '0.75rem', color: '#d29922', marginLeft: '5rem' },
  linkBtn: { background: 'transparent', border: 'none', color: '#79c0ff', cursor: 'pointer', font: 'inherit', fontSize: '0.75rem', padding: 0, textDecoration: 'underline' },
  footnote: { fontSize: '0.6875rem', color: '#7d8590', lineHeight: 1.5 },
  invalid: { background: '#2d1117', color: '#ffa198', border: '1px solid #56242a', borderRadius: '4px', padding: '0.5rem 0.75rem', fontSize: '0.75rem', whiteSpace: 'pre-wrap' },
  dim: { color: '#7d8590', fontSize: '0.75rem' },

  badge: { fontSize: '0.75rem', borderRadius: '6px', padding: '0.375rem 0.625rem', border: '1px solid #30363d', maxWidth: '100%' },
  badgeDev: { borderColor: '#2ea043', color: '#e6edf3', background: 'rgba(46,160,67,0.08)' },
  badgeProd: { border: '2px solid #f85149', color: '#ffa198', background: 'rgba(248,81,73,0.12)' },
  badgeUnknown: { color: '#7d8590' },
  badgeLine: { whiteSpace: 'nowrap' },
  badgeWarn: { marginTop: '0.25rem', fontSize: '0.75rem', lineHeight: 1.5, whiteSpace: 'normal', fontWeight: 600 },
  badgeNote: { marginTop: '0.25rem', fontSize: '0.6875rem', color: '#7d8590', whiteSpace: 'normal' },
  mono: { fontFamily: 'ui-monospace, monospace' },
};
