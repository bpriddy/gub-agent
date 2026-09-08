/**
 * sandboxServer.ts — server-only: what the active engine accepts.
 *
 * The lists the UI offers and the server validates against are resolved in
 * this order, per key:
 *
 *   1. the engine's own deploy-time env (`describeEngine`) — ground truth,
 *      this is the value `config.py` reads inside the engine;
 *   2. the same variable in this process's env (`.env.local`), for when the
 *      engine resource cannot be read (no aiplatform.reasoningEngines.get);
 *   3. the mirrored `config.py` default (`sandbox.ts`).
 *
 * Variant names have no env: they are mirror-only (`VARIANT_NAMES`).
 * Cached briefly so an A/B (two requests) and the page load share one GET.
 */
import {
  DEFAULT_MODEL,
  DEFAULT_MODEL_ALLOWLIST,
  DEFAULT_THINKING_LEVEL_MODELS,
  CRITIC_THINKING_LEVEL,
  EXECUTOR_THINKING_LEVEL,
  MAX_PROMPT_BYTES,
  THINKING_LEVELS,
  VARIANT_NAMES,
  type Role,
  type SandboxDefaults,
  type SandboxLists,
  type ThinkingLevel,
} from './sandbox';
import { KNOWN_PROD_ENGINE_ID, activeEngine, describeEngine, type ActiveEngine, type EngineDescription } from './vertex';

export interface EngineInfo extends ActiveEngine {
  /** Red badge. True for the prod target, the known prod id, or an engine whose baked-in env disables the sandbox. */
  isProd: boolean;
  displayName: string | null;
  /** `SANDBOX_ENABLED` as baked into the engine; null when the resource could not be read. */
  sandboxEnabled: boolean | null;
  /** Where the model allowlist came from: the engine's baked-in env, this process's env, or the mirrored config.py default. */
  listsSource: 'engine' | 'local' | 'default';
}

/** Shape of `GET /api/config`. */
export interface SandboxConfigResponse {
  engine: EngineInfo;
  models: string[];
  thinkingLevelModels: string[];
  thinkingLevels: readonly ThinkingLevel[];
  variants: Record<Role, string[]>;
  defaults: SandboxDefaults;
  maxPromptBytes: number;
}

const CACHE_TTL_MS = 60_000;
let cache: { at: number; key: string; value: SandboxConfigResponse } | null = null;

export async function sandboxConfig(): Promise<SandboxConfigResponse> {
  const engine = activeEngine();
  const key = `${engine.target}:${engine.id}`;
  if (cache && cache.key === key && Date.now() - cache.at < CACHE_TTL_MS) return cache.value;

  const described = await describeEngine(engine);
  const value = buildConfig(engine, described);
  cache = { at: Date.now(), key, value };
  return value;
}

/** The lists the agent route validates against — same source as the UI's options. */
export async function sandboxLists(): Promise<SandboxLists> {
  const cfg = await sandboxConfig();
  return { models: cfg.models, thinkingLevelModels: cfg.thinkingLevelModels, variants: cfg.variants };
}

function buildConfig(engine: ActiveEngine, described: EngineDescription | null): SandboxConfigResponse {
  const engineEnv = described?.env ?? {};
  const pick = (name: string): string | undefined => engineEnv[name] ?? process.env[name]?.trim() ?? undefined;

  const models = csv(pick('SANDBOX_MODEL_ALLOWLIST')) ?? DEFAULT_MODEL_ALLOWLIST;
  const listsSource: EngineInfo['listsSource'] =
    csv(engineEnv.SANDBOX_MODEL_ALLOWLIST) ? 'engine' : csv(process.env.SANDBOX_MODEL_ALLOWLIST?.trim()) ? 'local' : 'default';
  const thinkingLevelModels = csv(pick('SANDBOX_THINKING_LEVEL_MODELS')) ?? DEFAULT_THINKING_LEVEL_MODELS;
  const model = pick('GEMINI_MODEL') || DEFAULT_MODEL;

  const enabledRaw = engineEnv.SANDBOX_ENABLED;
  const sandboxEnabled = enabledRaw === undefined ? null : ['1', 'true', 'yes'].includes(enabledRaw.toLowerCase());

  return {
    engine: {
      ...engine,
      isProd: engine.target === 'prod' || engine.id === KNOWN_PROD_ENGINE_ID || sandboxEnabled === false,
      displayName: described?.displayName ?? null,
      sandboxEnabled,
      listsSource,
    },
    models,
    thinkingLevelModels,
    thinkingLevels: THINKING_LEVELS,
    variants: VARIANT_NAMES,
    defaults: { model, thinkingLevel: EXECUTOR_THINKING_LEVEL, criticThinkingLevel: CRITIC_THINKING_LEVEL },
    maxPromptBytes: MAX_PROMPT_BYTES,
  };
}

function csv(value: string | undefined): string[] | undefined {
  if (value === undefined) return undefined;
  const items = value.split(',').map((s) => s.trim()).filter(Boolean);
  return items.length > 0 ? items : undefined;
}
