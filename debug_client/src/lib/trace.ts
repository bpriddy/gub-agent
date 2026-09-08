/**
 * trace.ts — types + parser that turn the Vertex AI Agent Engine event
 * stream into a structured trace. Shared by the server route (parsing) and
 * the client components (rendering).
 *
 * The agent is a LoopAgent(max_iterations=2) emitting events per iteration:
 *   sandbox_echo state_delta.sandbox_resolved (sandbox runs ONLY)  author=sandbox_echo
 *   executor function_call(s) / function_response(s) / text   author=gub_agent
 *   critic structured verdict                                 author=critic
 *   loop_escalator (no payload)                               author=loop_escalator
 * We split iterations on each critic event.
 *
 * The echo event is emitted once, first, and only when `state["sandbox"]`
 * changed something (gub_agent/sandbox.py `SandboxEcho`). A run with no
 * overrides has NO such event — `resolved: null` means "baseline", not
 * "broken".
 */
import type { SandboxResolved } from './sandbox';

const CRITIC_AUTHOR = 'critic';
const ESCALATOR_AUTHOR = 'loop_escalator';
const SANDBOX_ECHO_AUTHOR = 'sandbox_echo';
const RESOLVED_STATE_KEY = 'sandbox_resolved';

export interface AgentSource {
  fileId: string;
  name: string;
  mimeType: string | null;
}

export interface AgentToolCall {
  tool: string;
  args: Record<string, unknown>;
  response?: unknown;
  sources?: AgentSource[];
}

export interface AgentCriticVerdict {
  sufficient: boolean;
  /** Axis 1 — did the tool calls gather enough to answer the question? */
  infoSufficient?: boolean;
  /** Axis 2 — does the synthesized answer satisfy the question? */
  answerSatisfies?: boolean;
  reason?: string;
  feedback?: string;
  /** Thought-summary text (only when the agent runs with EMIT_THINKING). */
  thoughts?: string;
}

export interface AgentIteration {
  index: number;
  toolCalls: AgentToolCall[];
  text: string;
  critic?: AgentCriticVerdict;
  /** Executor thought-summary text (only when EMIT_THINKING is set). */
  thoughts?: string;
}

export interface AgentTrace {
  text: string;
  iterations: AgentIteration[];
  sources: AgentSource[];
  /** `sandbox_resolved` provenance; null for a run with no overrides. */
  resolved: SandboxResolved | null;
  /** Whitespace-separated words in the final answer. */
  answerWordCount: number;
  /** Tool calls across all iterations. */
  toolCallCount: number;
}

interface Accumulator {
  iterations: AgentIteration[];
  current: AgentIteration;
  sourcesByFileId: Map<string, AgentSource>;
  resolved: SandboxResolved | null;
}

function newIteration(index: number): AgentIteration {
  return { index, toolCalls: [], text: '' };
}

export function buildTrace(events: unknown[]): AgentTrace {
  const acc: Accumulator = {
    iterations: [],
    current: newIteration(1),
    sourcesByFileId: new Map(),
    resolved: null,
  };

  for (const evt of events) consumeEvent(evt, acc);

  if (acc.current.toolCalls.length > 0 || acc.current.text.length > 0 || acc.current.critic) {
    acc.iterations.push(acc.current);
  }

  const text = finalAnswer(acc.iterations);
  return {
    text,
    iterations: acc.iterations,
    sources: Array.from(acc.sourcesByFileId.values()),
    resolved: acc.resolved,
    answerWordCount: wordCount(text),
    toolCallCount: acc.iterations.reduce((n, it) => n + it.toolCalls.length, 0),
  };
}

export function wordCount(text: string): number {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

function consumeEvent(evt: unknown, acc: Accumulator): void {
  if (!evt || typeof evt !== 'object') return;
  const e = evt as Record<string, unknown>;
  const author = typeof e.author === 'string' ? e.author : null;

  if (author === ESCALATOR_AUTHOR) return;

  const resolved = extractResolved(e);
  if (resolved) acc.resolved = resolved;
  if (author === SANDBOX_ECHO_AUTHOR) return;

  if (author === CRITIC_AUTHOR) {
    const verdict = extractCriticVerdict(e);
    if (verdict) acc.current.critic = verdict;
    acc.iterations.push(acc.current);
    acc.current = newIteration(acc.iterations.length + 1);
    return;
  }

  const content = e.content as { parts?: Array<Record<string, unknown>> } | undefined;
  if (!content?.parts) return;

  for (const part of content.parts) {
    const txt = part.text;
    if (typeof txt === 'string' && txt.length > 0) {
      if (part.thought === true) {
        // Thought-summary part (EMIT_THINKING) — keep it OUT of the answer text.
        acc.current.thoughts = acc.current.thoughts ? `${acc.current.thoughts}\n${txt}` : txt;
      } else {
        acc.current.text = acc.current.text ? `${acc.current.text}\n${txt}` : txt;
      }
    }

    const fnCall = part.function_call as { name?: string; args?: Record<string, unknown> } | undefined;
    if (fnCall && typeof fnCall.name === 'string') {
      acc.current.toolCalls.push({ tool: fnCall.name, args: fnCall.args ?? {} });
      continue;
    }

    const fnResp = part.function_response as
      | { name?: string; response?: Record<string, unknown> }
      | undefined;
    if (fnResp && typeof fnResp.name === 'string') {
      for (let i = acc.current.toolCalls.length - 1; i >= 0; i--) {
        const tc = acc.current.toolCalls[i]!;
        if (tc.tool === fnResp.name && tc.response === undefined) {
          tc.response = fnResp.response;
          const sources = extractSourcesFromValue(fnResp.response);
          if (sources.length > 0) {
            tc.sources = sources;
            for (const s of sources) {
              if (!acc.sourcesByFileId.has(s.fileId)) acc.sourcesByFileId.set(s.fileId, s);
            }
          }
          break;
        }
      }
    }
  }
}

/** `actions.state_delta.sandbox_resolved` (ADK dumps snake_case; camelCase accepted defensively). */
function extractResolved(evt: Record<string, unknown>): SandboxResolved | null {
  const actions = evt.actions as Record<string, unknown> | undefined;
  const delta = (actions?.state_delta ?? actions?.stateDelta) as Record<string, unknown> | undefined;
  const raw = delta?.[RESOLVED_STATE_KEY];
  if (!raw || typeof raw !== 'object') return null;
  const r = raw as Record<string, unknown>;
  if (typeof r.model !== 'string') return null;
  return {
    label: typeof r.label === 'string' ? r.label : null,
    model: r.model,
    temperature: typeof r.temperature === 'number' ? r.temperature : null,
    thinking_level: r.thinking_level as SandboxResolved['thinking_level'],
    critic_thinking_level: r.critic_thinking_level as SandboxResolved['critic_thinking_level'],
    critic_enabled: r.critic_enabled !== false,
    executor_prompt_source: typeof r.executor_prompt_source === 'string' ? r.executor_prompt_source : 'baseline',
    executor_prompt_sha256: typeof r.executor_prompt_sha256 === 'string' ? r.executor_prompt_sha256 : null,
    critic_prompt_source: typeof r.critic_prompt_source === 'string' ? r.critic_prompt_source : 'baseline',
    critic_prompt_sha256: typeof r.critic_prompt_sha256 === 'string' ? r.critic_prompt_sha256 : null,
    overridden_keys: Array.isArray(r.overridden_keys) ? r.overridden_keys.filter((k): k is string => typeof k === 'string') : [],
  };
}

function extractCriticVerdict(evt: Record<string, unknown>): AgentCriticVerdict | null {
  const content = evt.content as { parts?: Array<Record<string, unknown>> } | undefined;
  if (!content?.parts) return null;
  let verdict: AgentCriticVerdict | null = null;
  let thoughts = '';
  for (const part of content.parts) {
    // Thought-summary part (EMIT_THINKING) — collect, don't treat as the verdict.
    if (part.thought === true && typeof part.text === 'string') {
      thoughts = thoughts ? `${thoughts}\n${part.text}` : part.text;
      continue;
    }
    if (verdict) continue;
    const fnResp = part.function_response as { response?: Record<string, unknown> } | undefined;
    const obj = fnResp?.response;
    if (obj && typeof obj.sufficient === 'boolean') {
      verdict = shapeCritic(obj);
      continue;
    }
    const txt = part.text;
    if (typeof txt === 'string') {
      try {
        const parsed = JSON.parse(txt) as Record<string, unknown>;
        if (typeof parsed.sufficient === 'boolean') verdict = shapeCritic(parsed);
      } catch {
        /* not json */
      }
    }
  }
  if (verdict && thoughts) verdict.thoughts = thoughts;
  return verdict;
}

function shapeCritic(obj: Record<string, unknown>): AgentCriticVerdict {
  const out: AgentCriticVerdict = { sufficient: obj.sufficient === true };
  if (typeof obj.info_sufficient === 'boolean') out.infoSufficient = obj.info_sufficient;
  if (typeof obj.answer_satisfies === 'boolean') out.answerSatisfies = obj.answer_satisfies;
  if (typeof obj.reason === 'string') out.reason = obj.reason;
  if (typeof obj.feedback === 'string') out.feedback = obj.feedback;
  return out;
}

function extractSourcesFromValue(value: unknown): AgentSource[] {
  if (!value || typeof value !== 'object') return [];
  const arr = (value as Record<string, unknown>)._sources;
  if (!Array.isArray(arr)) return [];
  const out: AgentSource[] = [];
  for (const src of arr as Array<Record<string, unknown>>) {
    if (typeof src.fileId === 'string' && typeof src.name === 'string') {
      out.push({
        fileId: src.fileId,
        name: src.name,
        mimeType: typeof src.mimeType === 'string' ? src.mimeType : null,
      });
    }
  }
  return out;
}

function finalAnswer(iterations: AgentIteration[]): string {
  for (let i = iterations.length - 1; i >= 0; i--) {
    if (iterations[i]!.text) return iterations[i]!.text;
  }
  return '';
}
