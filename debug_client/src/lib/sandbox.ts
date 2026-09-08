/**
 * sandbox.ts — TypeScript mirror of the `state["sandbox"]` contract
 * (`gub_agent/sandbox.py`) and of the prompt-variant registry
 * (`gub_agent/prompts/variants/__init__.py`).
 *
 * This file is ISOMORPHIC: it runs in the API routes (validation before the
 * Vertex call, so a bad form value is a legible 400 instead of an empty 200
 * stream) and in the browser (pre-flight, so the same message shows before
 * the run). It therefore reads no env — the lists it validates against come
 * in as an argument (`SandboxLists`), resolved server-side by
 * `sandboxServer.ts` and shipped to the client by `GET /api/config`.
 *
 * DUPLICATION SEAM. There is no Python process to ask, so the following are
 * hand-mirrored and will drift first when the agent changes:
 *   - `THINKING_LEVELS`            ← sandbox.ThinkingLevel
 *   - `EXECUTOR_THINKING_LEVEL`,
 *     `CRITIC_THINKING_LEVEL`      ← sandbox.EXECUTOR_/CRITIC_THINKING_LEVEL
 *   - `MAX_PROMPT_BYTES`           ← sandbox.MAX_PROMPT_BYTES
 *   - `DEFAULT_MODEL_ALLOWLIST`,
 *     `DEFAULT_THINKING_LEVEL_MODELS`,
 *     `DEFAULT_MODEL`              ← config.py defaults (the deployed engine's
 *                                    env is preferred when it can be read —
 *                                    see sandboxServer.ts)
 *   - `VARIANT_NAMES`              ← prompts/variants VARIANTS keys (no env
 *                                    fallback exists: this one is mirror-only)
 *   - the validation rules + messages in `validateOverrides` ← sandbox._validate
 */

export const THINKING_LEVELS = ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'DYNAMIC'] as const;
export type ThinkingLevel = (typeof THINKING_LEVELS)[number];

export const ROLES = ['executor', 'critic'] as const;
export type Role = (typeof ROLES)[number];

/** Baseline thinking levels the planners are built with (sandbox.py:84-85). */
export const EXECUTOR_THINKING_LEVEL: ThinkingLevel = 'MEDIUM';
export const CRITIC_THINKING_LEVEL: ThinkingLevel = 'LOW';

/** sandbox.MAX_PROMPT_BYTES — inline prompts above this must become variants. */
export const MAX_PROMPT_BYTES = 64 * 1024;

/** config.py defaults, used only when neither the engine nor .env.local says otherwise. */
export const DEFAULT_MODEL = 'gemini-3.5-flash';
export const DEFAULT_MODEL_ALLOWLIST = ['gemini-3.5-flash', 'gemini-2.5-pro'];
export const DEFAULT_THINKING_LEVEL_MODELS = ['gemini-3.5-flash'];

/** prompts/variants/__init__.py VARIANTS, split per role (variant_names(role)). */
export const VARIANT_NAMES: Record<Role, string[]> = {
  executor: ['baseline', 'v2_concise', 'v3_grounding'],
  critic: ['baseline'],
};

/** The name every role reserves for "what production runs right now". */
export const BASELINE_VARIANT = 'baseline';

/** `state["sandbox"]` — one-to-one with pydantic `SandboxOverrides`. */
export interface SandboxOverrides {
  executor_instruction?: string | null;
  executor_variant?: string | null;
  critic_instruction?: string | null;
  critic_variant?: string | null;
  model?: string | null;
  thinking_level?: ThinkingLevel | null;
  critic_thinking_level?: ThinkingLevel | null;
  temperature?: number | null;
  critic_enabled?: boolean;
  label?: string | null;
}

export const OVERRIDE_KEYS: ReadonlyArray<keyof SandboxOverrides> = [
  'executor_instruction',
  'executor_variant',
  'critic_instruction',
  'critic_variant',
  'model',
  'thinking_level',
  'critic_thinking_level',
  'temperature',
  'critic_enabled',
  'label',
];

/** `sandbox_resolved` — the flat provenance `resolved_config()` emits. */
export interface SandboxResolved {
  label: string | null;
  model: string;
  temperature: number | null;
  thinking_level: ThinkingLevel;
  critic_thinking_level: ThinkingLevel;
  critic_enabled: boolean;
  executor_prompt_source: string;
  executor_prompt_sha256: string | null;
  critic_prompt_source: string;
  critic_prompt_sha256: string | null;
  overridden_keys: string[];
}

/** The lists a validation runs against — the engine's, not this file's. */
export interface SandboxLists {
  models: string[];
  thinkingLevelModels: string[];
  variants: Record<Role, string[]>;
}

export type ValidationResult =
  | { ok: true; value: SandboxOverrides | null }
  | { ok: false; message: string };

/** What `validateOverrides` builds: only keys that are set, never null. */
type Normalised = { [K in keyof SandboxOverrides]?: NonNullable<SandboxOverrides[K]> };

/**
 * Validate a raw `config` and normalise it to what the engine should see.
 *
 * Returns `value: null` when nothing differs from the baseline — the caller
 * then omits the `sandbox` key entirely, so the request is byte-identical to
 * one made before the sandbox existed (and `sandbox_echo` stays silent).
 *
 * Mirrors `read_overrides` + `_validate`: unknown keys are dropped (the engine
 * would warn and drop them too), wrong types and invalid values are refused
 * with the engine's own wording where it has one.
 */
export function validateOverrides(raw: unknown, lists: SandboxLists): ValidationResult {
  if (raw === undefined || raw === null) return { ok: true, value: null };
  if (typeof raw !== 'object' || Array.isArray(raw)) {
    return { ok: false, message: 'sandbox: config must be an object.' };
  }
  const input = raw as Record<string, unknown>;
  const out: Normalised = {};

  for (const key of ['executor_instruction', 'executor_variant', 'critic_instruction', 'critic_variant', 'model', 'label'] as const) {
    const v = input[key];
    if (v === undefined || v === null || v === '') continue;
    if (typeof v !== 'string') return { ok: false, message: `sandbox: invalid override — ${key}: must be a string` };
    out[key] = v;
  }
  for (const key of ['thinking_level', 'critic_thinking_level'] as const) {
    const v = input[key];
    if (v === undefined || v === null || v === '') continue;
    if (typeof v !== 'string' || !(THINKING_LEVELS as readonly string[]).includes(v)) {
      return { ok: false, message: `sandbox: invalid override — ${key}: must be one of ${THINKING_LEVELS.join(', ')}` };
    }
    out[key] = v as ThinkingLevel;
  }
  if (input.temperature !== undefined && input.temperature !== null && input.temperature !== '') {
    const t = typeof input.temperature === 'string' ? Number(input.temperature) : input.temperature;
    if (typeof t !== 'number' || Number.isNaN(t)) {
      return { ok: false, message: 'sandbox: invalid override — temperature: must be a number' };
    }
    out.temperature = t;
  }
  if (input.critic_enabled !== undefined && input.critic_enabled !== null) {
    if (typeof input.critic_enabled !== 'boolean') {
      return { ok: false, message: 'sandbox: invalid override — critic_enabled: must be a boolean' };
    }
    if (input.critic_enabled === false) out.critic_enabled = false;
  }

  // ── semantic checks: sandbox._validate ──────────────────────────────────
  if (out.model !== undefined && !lists.models.includes(out.model)) {
    const allowed = lists.models.join(', ') || '(empty allowlist)';
    return {
      ok: false,
      message: `sandbox: model '${out.model}' is not allowed. Allowed models: ${allowed} (set SANDBOX_MODEL_ALLOWLIST to widen).`,
    };
  }
  if (out.model !== undefined && !lists.thinkingLevelModels.includes(out.model)) {
    const executorLevel = out.thinking_level ?? EXECUTOR_THINKING_LEVEL;
    const criticLevel = out.critic_thinking_level ?? CRITIC_THINKING_LEVEL;
    const criticEnabled = out.critic_enabled !== false;
    const offending: string[] = [];
    if (executorLevel !== 'DYNAMIC') offending.push(`thinking_level=${executorLevel}`);
    if (criticEnabled && criticLevel !== 'DYNAMIC') offending.push(`critic_thinking_level=${criticLevel}`);
    if (offending.length > 0) {
      return {
        ok: false,
        message:
          `sandbox: model '${out.model}' does not accept a named thinking level ` +
          `(Vertex answers 400 INVALID_ARGUMENT), but ${offending.join(', ')} would apply ` +
          'to this run. Set thinking_level (and critic_thinking_level, unless ' +
          'critic_enabled is false) to DYNAMIC for this model — or add the model to ' +
          'SANDBOX_THINKING_LEVEL_MODELS if it does accept named levels.',
      };
    }
  }
  if (out.temperature !== undefined && !(out.temperature >= 0 && out.temperature <= 2)) {
    return {
      ok: false,
      message: `sandbox: temperature ${out.temperature} is outside 0.0..2.0. The value is NOT clamped — a clamp would silently distort an A/B.`,
    };
  }
  for (const field of ['executor_instruction', 'critic_instruction'] as const) {
    const text = out[field];
    if (text === undefined) continue;
    const size = utf8Bytes(text);
    if (size > MAX_PROMPT_BYTES) {
      return {
        ok: false,
        message: `sandbox: ${field} is ${size} bytes, over the ${MAX_PROMPT_BYTES} byte cap — register the text as a prompt variant and pass the variant name instead of inline text.`,
      };
    }
  }
  for (const role of ROLES) {
    const name = out[`${role}_variant`];
    if (name === undefined) continue;
    const known = lists.variants[role];
    if (!known.includes(name)) {
      return {
        ok: false,
        message: `unknown ${role} prompt variant '${name}'. Available ${role} variants: ${known.join(', ')}. Register it as a module in gub_agent/prompts/variants/ or pass the text inline as ${role}_instruction — there is deliberately no fallback to the baseline prompt.`,
      };
    }
  }

  return { ok: true, value: Object.keys(out).length === 0 ? null : out };
}

export function utf8Bytes(text: string): number {
  if (typeof TextEncoder !== 'undefined') return new TextEncoder().encode(text).length;
  return Buffer.byteLength(text, 'utf-8');
}

/** Stable JSON for "did the config change since this session started". */
export function overridesKey(value: SandboxOverrides | null): string {
  if (!value) return '';
  const sorted: Record<string, unknown> = {};
  for (const key of OVERRIDE_KEYS) {
    if (value[key] !== undefined && value[key] !== null) sorted[key] = value[key];
  }
  return JSON.stringify(sorted);
}

// ── Baseline + diff ────────────────────────────────────────────────────────

/** Deployed defaults the UI uses to label a run that emitted no provenance. */
export interface SandboxDefaults {
  model: string;
  thinkingLevel: ThinkingLevel;
  criticThinkingLevel: ThinkingLevel;
}

/**
 * What a run with NO overrides resolved to. `sandbox_echo` emits nothing for
 * such a run (an extra event would change the trace shape), so the client
 * synthesises this from the engine's defaults to label the arm and to diff
 * against. `overridden_keys` is empty by definition.
 */
export function baselineResolved(defaults: SandboxDefaults): SandboxResolved {
  return {
    label: null,
    model: defaults.model,
    temperature: null,
    thinking_level: defaults.thinkingLevel,
    critic_thinking_level: defaults.criticThinkingLevel,
    critic_enabled: true,
    executor_prompt_source: 'baseline',
    executor_prompt_sha256: null,
    critic_prompt_source: 'baseline',
    critic_prompt_sha256: null,
    overridden_keys: [],
  };
}

export interface ResolvedDiffLine {
  key: string;
  a: string;
  b: string;
}

/**
 * How B differs from A, from the two PROVENANCE payloads (what ran), never
 * from the form (what was asked). `label` is provenance, not configuration,
 * and is left out; a prompt counts as different when its source differs or
 * when both sides carry a hash and the hashes differ — a `baseline` variant
 * (hashed) and the deployed baseline (unhashed) are the same text.
 */
export function diffResolved(a: SandboxResolved, b: SandboxResolved): ResolvedDiffLine[] {
  const lines: ResolvedDiffLine[] = [];
  const scalar = (key: 'model' | 'temperature' | 'thinking_level' | 'critic_thinking_level' | 'critic_enabled', name: string) => {
    if (a[key] !== b[key]) lines.push({ key: name, a: fmt(a[key]), b: fmt(b[key]) });
  };
  scalar('model', 'model');
  scalar('thinking_level', 'thinking');
  scalar('temperature', 'temperature');
  scalar('critic_enabled', 'critic');
  scalar('critic_thinking_level', 'critic thinking');
  for (const role of ROLES) {
    const srcA = a[`${role}_prompt_source`];
    const srcB = b[`${role}_prompt_source`];
    const shaA = a[`${role}_prompt_sha256`];
    const shaB = b[`${role}_prompt_sha256`];
    const differs = srcA !== srcB || (shaA !== null && shaB !== null && shaA !== shaB);
    if (differs) lines.push({ key: `${role} prompt`, a: promptLabel(srcA, shaA), b: promptLabel(srcB, shaB) });
  }
  return lines;
}

export function promptLabel(source: string, sha: string | null): string {
  return sha ? `${source} · ${sha}` : source;
}

function fmt(v: unknown): string {
  if (v === null || v === undefined) return 'default';
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  return String(v);
}
