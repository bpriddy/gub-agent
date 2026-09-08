/**
 * page.tsx — main debug UI.
 *
 *   - not signed in → Google Sign-In gate
 *   - signed in     → sandbox config + question box + agent trace
 *                     (or, with A/B on, two configs and two traces side by side)
 *
 * The agent runs AS the signed-in user: we pass the GUB access token in the
 * Authorization header to /api/agent, which seeds it into the Vertex AI
 * session state as gub_jwt. The sandbox config rides the same state object
 * (`state.sandbox`, gub_agent/sandbox.py) and is therefore fixed at session
 * creation: a single run keeps its session for follow-ups until the config
 * changes; an A/B always opens two fresh sessions, one per arm.
 */
'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useAuth } from '@/lib/auth/useAuth';
import { AgentTrace } from '@/components/AgentTrace';
import {
  ConfigPanel,
  EngineBadge,
  coerceForm,
  emptyForm,
  formToOverrides,
  type SandboxForm,
} from '@/components/ConfigPanel';
import { RunColumn, ResolvedChips, effectiveResolved, type RunState } from '@/components/RunColumn';
import { ApiError, fetchConfig, runAgent, type AgentRunResponse, type SandboxConfigResponse } from '@/lib/api';
import { diffResolved, overridesKey, validateOverrides, type SandboxLists, type SandboxOverrides } from '@/lib/sandbox';

const LS_FORM_A = 'gub-debug.sandbox.formA';
const LS_FORM_B = 'gub-debug.sandbox.formB';

export default function Page() {
  const auth = useAuth();

  if (auth.status === 'loading') {
    return <Centered>Restoring session…</Centered>;
  }
  if (auth.status !== 'authenticated') {
    return <LoginGate auth={auth} />;
  }
  return <Console auth={auth} />;
}

// ── Login gate ────────────────────────────────────────────────────────────

function LoginGate({ auth }: { auth: ReturnType<typeof useAuth> }) {
  const btnRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (btnRef.current) auth.renderGoogleButton(btnRef.current);
  }, [auth]);

  return (
    <Centered>
      <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'center' }}>
        <h1 style={{ fontSize: '1.25rem', fontWeight: 600 }}>gub-agent debug</h1>
        <p style={{ color: '#7d8590', fontSize: '0.875rem', maxWidth: 360 }}>
          Sign in with your work Google account. The agent runs as you — its
          tools query GUB with your access.
        </p>
        <div ref={btnRef} />
        {auth.status === 'error' && auth.error && (
          <div style={{ color: '#ffa198', fontSize: '0.8125rem', maxWidth: 360 }}>{auth.error}</div>
        )}
      </div>
    </Centered>
  );
}

// ── Console ──────────────────────────────────────────────────────────────

function Console({ auth }: { auth: ReturnType<typeof useAuth> }) {
  // Engine + option lists — the UI hardcodes none of them.
  const [options, setOptions] = useState<SandboxConfigResponse | null>(null);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  useEffect(() => {
    fetchConfig().then(setOptions).catch((e: Error) => setOptionsError(e.message));
  }, []);
  const lists: SandboxLists | null = useMemo(
    () => (options ? { models: options.models, thinkingLevelModels: options.thinkingLevelModels, variants: options.variants } : null),
    [options],
  );
  const defaults = options?.defaults ?? null;

  // Configs, persisted so they survive a reload. Loaded after mount (SSR
  // renders the empty form; writing back only once loaded, so the stored
  // value is never clobbered by the default).
  const [formA, setFormA] = useState<SandboxForm>(emptyForm);
  const [formB, setFormB] = useState<SandboxForm>(emptyForm);
  const [formsLoaded, setFormsLoaded] = useState(false);
  useEffect(() => {
    setFormA(coerceForm(readJson(LS_FORM_A)));
    setFormB(coerceForm(readJson(LS_FORM_B)));
    setFormsLoaded(true);
  }, []);
  useEffect(() => { if (formsLoaded) writeJson(LS_FORM_A, formA); }, [formA, formsLoaded]);
  useEffect(() => { if (formsLoaded) writeJson(LS_FORM_B, formB); }, [formB, formsLoaded]);

  // Pre-flight: the same validator the server runs, on the same lists, so an
  // invalid model/thinking pair reads as the engine's message BEFORE the run.
  const checkA = useMemo(() => (lists ? validateOverrides(formToOverrides(formA), lists) : null), [formA, lists]);
  const checkB = useMemo(() => (lists ? validateOverrides(formToOverrides(formB), lists) : null), [formB, lists]);
  const invalidA = checkA && !checkA.ok ? checkA.message : null;
  const invalidB = checkB && !checkB.ok ? checkB.message : null;

  const [abMode, setAbMode] = useState(false);
  const [message, setMessage] = useState('');
  const [includeRaw, setIncludeRaw] = useState(false);

  // ── single run ──
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionConfigKey, setSessionConfigKey] = useState('');
  const [response, setResponse] = useState<AgentRunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const resetSession = useCallback(() => {
    setSessionId(null);
    setSessionConfigKey('');
    setResponse(null);
    setNotice(null);
  }, []);

  const run = async () => {
    if (!message.trim() || running || !checkA) return;
    if (!checkA.ok) { setError(checkA.message); return; }
    const config: SandboxOverrides | null = checkA.value;
    const key = overridesKey(config);

    // Overrides are seeded at session creation: a changed config cannot apply
    // to the open session, so start a new one and say so.
    let sid = sessionId;
    if (sid && key !== sessionConfigKey) {
      sid = null;
      setNotice('Config changed since this session started — a new session was opened for this run.');
    } else {
      setNotice(null);
    }

    setRunning(true);
    setError(null);
    try {
      const data = await runAgent({ message, sessionId: sid, includeRaw, config });
      setResponse(data);
      setSessionId(data.sessionId);
      if (!sid) setSessionConfigKey(key);
    } catch (e) {
      setError(e instanceof ApiError ? `${e.code}: ${e.message}` : (e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  // ── A/B ──
  const [runA, setRunA] = useState<RunState>({ status: 'idle' });
  const [runB, setRunB] = useState<RunState>({ status: 'idle' });
  const abRunning = runA.status === 'running' || runB.status === 'running';

  const runAB = async () => {
    if (!message.trim() || abRunning || !checkA || !checkB) return;
    if (!checkA.ok || !checkB.ok) {
      setRunA(checkA.ok ? { status: 'idle' } : { status: 'error', code: 'BAD_CONFIG', message: checkA.message, resolved: null });
      setRunB(checkB.ok ? { status: 'idle' } : { status: 'error', code: 'BAD_CONFIG', message: checkB.message, resolved: null });
      return;
    }
    const startedAt = Date.now();
    setRunA({ status: 'running', startedAt });
    setRunB({ status: 'running', startedAt });
    // Two FRESH sessions (no sessionId), fired together so GUB data cannot
    // shift between the arms.
    const [a, b] = await Promise.allSettled([
      runAgent({ message, includeRaw, config: checkA.value }),
      runAgent({ message, includeRaw, config: checkB.value }),
    ]);
    setRunA(settle(a));
    setRunB(settle(b));
  };

  const diff = useMemo(() => {
    if (runA.status !== 'ok' || runB.status !== 'ok') return null;
    const a = effectiveResolved(runA.response.resolved, defaults);
    const b = effectiveResolved(runB.response.resolved, defaults);
    if (!a || !b) return null;
    return diffResolved(a, b);
  }, [runA, runB, defaults]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) void (abMode ? runAB() : run());
  };

  const busy = abMode ? abRunning : running;

  return (
    <div style={{ ...S.page, maxWidth: abMode ? 1500 : 900 }}>
      <header style={S.header}>
        <div style={S.headerLeft}>
          <h1 style={S.title}>gub-agent debug</h1>
          <EngineBadge config={options} error={optionsError} />
        </div>
        <div style={S.headerRight}>
          <label style={S.checkRow} title="Compare two configs on one question, each in its own fresh session">
            <input type="checkbox" checked={abMode} onChange={(e) => setAbMode(e.target.checked)} />
            A/B
          </label>
          <a href="/batch" style={S.navLink} title="Batch runs: many questions × many configs → numbers (gub-agent#33)">batch</a>
          <span style={S.userEmail}>{auth.user?.email}</span>
          <button onClick={() => void auth.logout()} style={S.signOut}>sign out</button>
        </div>
      </header>

      {abMode ? (
        <div style={S.twoCol}>
          <ConfigPanel title="A" form={formA} onChange={setFormA} options={options} validation={invalidA} disabled={abRunning} />
          <ConfigPanel
            title="B"
            form={formB}
            onChange={setFormB}
            options={options}
            validation={invalidB}
            disabled={abRunning}
            actions={
              <button type="button" onClick={() => setFormB({ ...formA })} style={S.smallBtn} title="Make B identical to A, then change one thing">
                copy A → B
              </button>
            }
          />
        </div>
      ) : (
        <ConfigPanel form={formA} onChange={setFormA} options={options} validation={invalidA} disabled={running} defaultOpen={false} />
      )}

      <div style={S.inputRow}>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder='Ask a business question — "how is the chevy account?" · "most expensive campaign last year" · "who led the Q3 nike work?"'
          style={S.input}
          onKeyDown={onKeyDown}
        />
        <div style={S.inputAside}>
          <button
            onClick={() => void (abMode ? runAB() : run())}
            disabled={busy || !options || !!invalidA || (abMode && !!invalidB)}
            style={{ ...S.runBtn, opacity: busy || !options || invalidA || (abMode && invalidB) ? 0.6 : 1 }}
            title={invalidA ?? (abMode ? invalidB ?? undefined : undefined)}
          >
            {busy ? 'Running…' : abMode ? 'Run A/B (⌘↵)' : 'Run (⌘↵)'}
          </button>
          <label style={S.checkRow}>
            <input type="checkbox" checked={includeRaw} onChange={(e) => setIncludeRaw(e.target.checked)} />
            include raw events
          </label>
          {abMode ? (
            <div style={S.sessionPill} title="Overrides live in session state, so each arm gets its own fresh session on every run. Turn A/B off to ask a follow-up.">
              2 fresh sessions per run · no follow-ups
            </div>
          ) : (
            <>
              <button onClick={resetSession} style={S.resetBtn} title="Start a fresh agent session">
                new session
              </button>
              {sessionId && <div style={S.sessionPill} title={sessionId}>session …{sessionId.slice(-6)}</div>}
            </>
          )}
        </div>
      </div>

      {abMode ? (
        <>
          <DiffLine diff={diff} runA={runA} runB={runB} />
          <div style={S.twoCol}>
            <RunColumn side="A" run={runA} defaults={defaults} />
            <RunColumn side="B" run={runB} defaults={defaults} />
          </div>
          {runA.status === 'idle' && runB.status === 'idle' && (
            <div style={S.placeholder}>One question, two configs, two fresh sessions. Traces render side by side.</div>
          )}
        </>
      ) : (
        <>
          {notice && <div style={S.notice}>{notice}</div>}
          {error && <div style={S.error}>{error}</div>}
          {response && (response.resolved || response.sandbox) && (
            <div style={S.provenanceRow}>
              <ResolvedChips resolved={response.resolved} defaults={defaults} />
            </div>
          )}
          {response && <AgentTrace response={response} />}
          {!response && !error && !running && (
            <div style={S.placeholder}>Ask a question and watch how the agent decomposes it into queries.</div>
          )}
        </>
      )}
    </div>
  );
}

function DiffLine({ diff, runA, runB }: { diff: ReturnType<typeof diffResolved> | null; runA: RunState; runB: RunState }) {
  if (runA.status === 'idle' && runB.status === 'idle') return null;
  if (runA.status === 'running' || runB.status === 'running') {
    return <div style={S.diff}><span style={S.diffLabel}>B vs A</span> running…</div>;
  }
  if (!diff) {
    const failed = [runA.status === 'error' ? 'A' : null, runB.status === 'error' ? 'B' : null].filter(Boolean).join(' and ');
    return <div style={S.diff}><span style={S.diffLabel}>B vs A</span> no comparison — {failed || 'a side'} failed.</div>;
  }
  if (diff.length === 0) {
    return (
      <div style={{ ...S.diff, ...S.diffSame }}>
        <span style={S.diffLabel}>B vs A</span> identical configurations ran on both sides (from provenance) — any difference below is model variance, not the config.
      </div>
    );
  }
  return (
    <div style={S.diff}>
      <span style={S.diffLabel}>B differs from A</span>
      {diff.map((d) => (
        <span key={d.key} style={S.diffItem}>
          <span style={S.diffKey}>{d.key}</span> {d.a} → <b>{d.b}</b>
        </span>
      ))}
    </div>
  );
}

function settle(result: PromiseSettledResult<AgentRunResponse>): RunState {
  if (result.status === 'fulfilled') return { status: 'ok', response: result.value };
  const e = result.reason as unknown;
  if (e instanceof ApiError) return { status: 'error', code: e.code, message: e.message, resolved: e.resolved };
  return { status: 'error', code: 'Error', message: (e as Error)?.message ?? String(e), resolved: null };
}

function readJson(key: string): unknown {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as unknown) : null;
  } catch {
    return null;
  }
}

function writeJson(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable — configs just don't persist */
  }
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#7d8590', fontSize: '0.875rem' }}>
      {children}
    </div>
  );
}

const S: Record<string, React.CSSProperties> = {
  page: { minHeight: '100vh', padding: '1.5rem 2rem', margin: '0 auto' },
  header: { display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem', borderBottom: '1px solid #21262d', paddingBottom: '1rem', marginBottom: '1.5rem' },
  headerLeft: { display: 'flex', alignItems: 'flex-start', gap: '1rem', flexWrap: 'wrap', minWidth: 0 },
  title: { fontSize: '1.25rem', margin: 0, whiteSpace: 'nowrap' },
  headerRight: { display: 'flex', alignItems: 'center', gap: '0.75rem', flexShrink: 0 },
  userEmail: { fontSize: '0.8125rem', color: '#7d8590' },
  signOut: { background: 'transparent', color: '#7d8590', border: '1px solid #30363d', borderRadius: '4px', padding: '0.25rem 0.625rem', fontSize: '0.75rem', cursor: 'pointer' },
  navLink: { color: '#79c0ff', fontSize: '0.8125rem', textDecoration: 'none', border: '1px solid #30363d', borderRadius: '4px', padding: '0.25rem 0.625rem' },
  smallBtn: { background: 'transparent', color: '#79c0ff', border: '1px solid #30363d', borderRadius: '4px', padding: '0.125rem 0.5rem', fontSize: '0.6875rem', cursor: 'pointer' },

  twoCol: { display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)', gap: '1rem', alignItems: 'start' },

  inputRow: { display: 'grid', gridTemplateColumns: '1fr auto', gap: '1rem', marginBottom: '1rem' },
  input: { width: '100%', minHeight: '4rem', background: '#161b22', color: '#e6edf3', border: '1px solid #30363d', borderRadius: '6px', padding: '0.625rem 0.75rem', fontSize: '0.9375rem', resize: 'vertical' },
  inputAside: { display: 'flex', flexDirection: 'column', gap: '0.5rem', alignItems: 'stretch', maxWidth: '14rem' },
  runBtn: { background: '#238636', color: '#fff', border: 'none', borderRadius: '6px', padding: '0.5rem 1rem', fontSize: '0.875rem', fontWeight: 600, cursor: 'pointer' },
  resetBtn: { background: 'transparent', color: '#7d8590', border: '1px solid #30363d', borderRadius: '4px', padding: '0.25rem 0.5rem', fontSize: '0.75rem', cursor: 'pointer' },
  checkRow: { display: 'flex', alignItems: 'center', gap: '0.375rem', fontSize: '0.75rem', color: '#7d8590' },
  sessionPill: { fontSize: '0.6875rem', color: '#7d8590', textAlign: 'center', fontFamily: 'ui-monospace, monospace' },

  provenanceRow: { marginBottom: '0.75rem' },
  notice: { background: '#1c2a1c', color: '#9be29b', border: '1px solid #2e5a2e', borderRadius: '4px', padding: '0.5rem 0.75rem', marginBottom: '0.5rem', fontSize: '0.8125rem' },
  error: { background: '#2d1117', color: '#ffa198', border: '1px solid #56242a', borderRadius: '4px', padding: '0.5rem 0.75rem', marginBottom: '0.5rem', fontSize: '0.8125rem', whiteSpace: 'pre-wrap' },
  placeholder: { color: '#7d8590', fontSize: '0.875rem', padding: '2rem 0', textAlign: 'center' },

  diff: { display: 'flex', alignItems: 'baseline', gap: '0.75rem', flexWrap: 'wrap', background: '#161b22', border: '1px solid #30363d', borderRadius: '6px', padding: '0.5rem 0.75rem', marginBottom: '1rem', fontSize: '0.8125rem' },
  diffSame: { borderColor: '#9e6a03', color: '#e3b341' },
  diffLabel: { fontSize: '0.6875rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: '#7d8590' },
  diffItem: { fontFamily: 'ui-monospace, monospace', fontSize: '0.75rem' },
  diffKey: { color: '#7d8590' },
};
