/**
 * trace.ts — types + parser that turn the Vertex AI Agent Engine event
 * stream into a structured trace. Shared by the server route (parsing) and
 * the client components (rendering).
 *
 * The agent is a LoopAgent(max_iterations=2) emitting events per iteration:
 *   executor function_call(s) / function_response(s) / text   author=gub_agent
 *   critic structured verdict                                 author=critic
 *   loop_escalator (no payload)                               author=loop_escalator
 * We split iterations on each critic event.
 */

const CRITIC_AUTHOR = 'critic';
const ESCALATOR_AUTHOR = 'loop_escalator';

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
}

export interface AgentIteration {
  index: number;
  toolCalls: AgentToolCall[];
  text: string;
  critic?: AgentCriticVerdict;
}

export interface AgentTrace {
  text: string;
  iterations: AgentIteration[];
  sources: AgentSource[];
}

interface Accumulator {
  iterations: AgentIteration[];
  current: AgentIteration;
  sourcesByFileId: Map<string, AgentSource>;
}

function newIteration(index: number): AgentIteration {
  return { index, toolCalls: [], text: '' };
}

export function buildTrace(events: unknown[]): AgentTrace {
  const acc: Accumulator = {
    iterations: [],
    current: newIteration(1),
    sourcesByFileId: new Map(),
  };

  for (const evt of events) consumeEvent(evt, acc);

  if (acc.current.toolCalls.length > 0 || acc.current.text.length > 0 || acc.current.critic) {
    acc.iterations.push(acc.current);
  }

  return {
    text: finalAnswer(acc.iterations),
    iterations: acc.iterations,
    sources: Array.from(acc.sourcesByFileId.values()),
  };
}

function consumeEvent(evt: unknown, acc: Accumulator): void {
  if (!evt || typeof evt !== 'object') return;
  const e = evt as Record<string, unknown>;
  const author = typeof e.author === 'string' ? e.author : null;

  if (author === ESCALATOR_AUTHOR) return;

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
      acc.current.text = acc.current.text ? `${acc.current.text}\n${txt}` : txt;
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

function extractCriticVerdict(evt: Record<string, unknown>): AgentCriticVerdict | null {
  const content = evt.content as { parts?: Array<Record<string, unknown>> } | undefined;
  if (!content?.parts) return null;
  for (const part of content.parts) {
    const fnResp = part.function_response as { response?: Record<string, unknown> } | undefined;
    const obj = fnResp?.response;
    if (obj && typeof obj.sufficient === 'boolean') return shapeCritic(obj);
    const txt = part.text;
    if (typeof txt === 'string') {
      try {
        const parsed = JSON.parse(txt) as Record<string, unknown>;
        if (typeof parsed.sufficient === 'boolean') return shapeCritic(parsed);
      } catch {
        /* not json */
      }
    }
  }
  return null;
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
