/**
 * page.tsx — main debug UI.
 *
 *   - not signed in → Google Sign-In gate
 *   - signed in     → question box + agent trace
 *
 * The agent runs AS the signed-in user: we pass the GUB access token in the
 * Authorization header to /api/agent, which seeds it into the Vertex AI
 * session state as gub_jwt.
 */
'use client';

import { useEffect, useRef, useState } from 'react';
import { useAuth } from '@/lib/auth/useAuth';
import { getValidAccessToken } from '@/lib/auth/tokenStore';
import { AgentTrace, type AgentResponse } from '@/components/AgentTrace';

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
  const [message, setMessage] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [includeRaw, setIncludeRaw] = useState(false);
  const [response, setResponse] = useState<AgentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  const run = async () => {
    if (!message.trim() || running) return;
    setRunning(true);
    setError(null);
    try {
      const accessToken = await getValidAccessToken();
      const res = await fetch('/api/agent', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ message, ...(sessionId ? { sessionId } : {}), includeRaw }),
      });
      const data = (await res.json()) as AgentResponse & { code?: string; message?: string };
      if (!res.ok) {
        setError(`${data.code ?? 'Error'}: ${data.message ?? res.statusText}`);
        return;
      }
      setResponse(data);
      setSessionId(data.sessionId);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div style={S.page}>
      <header style={S.header}>
        <h1 style={S.title}>gub-agent debug</h1>
        <div style={S.headerRight}>
          <span style={S.userEmail}>{auth.user?.email}</span>
          <button onClick={() => void auth.logout()} style={S.signOut}>sign out</button>
        </div>
      </header>

      <div style={S.inputRow}>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder='Ask a business question — "how is the chevy account?" · "most expensive campaign last year" · "who led the Q3 nike work?"'
          style={S.input}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) void run();
          }}
        />
        <div style={S.inputAside}>
          <button onClick={() => void run()} disabled={running} style={{ ...S.runBtn, opacity: running ? 0.6 : 1 }}>
            {running ? 'Running…' : 'Run (⌘↵)'}
          </button>
          <label style={S.checkRow}>
            <input type="checkbox" checked={includeRaw} onChange={(e) => setIncludeRaw(e.target.checked)} />
            include raw events
          </label>
          <button onClick={() => { setSessionId(null); setResponse(null); }} style={S.resetBtn} title="Start a fresh agent session">
            new session
          </button>
          {sessionId && <div style={S.sessionPill} title={sessionId}>session …{sessionId.slice(-6)}</div>}
        </div>
      </div>

      {error && <div style={S.error}>{error}</div>}
      {response && <AgentTrace response={response} />}
      {!response && !error && !running && (
        <div style={S.placeholder}>Ask a question and watch how the agent decomposes it into queries.</div>
      )}
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#7d8590', fontSize: '0.875rem' }}>
      {children}
    </div>
  );
}

const S: Record<string, React.CSSProperties> = {
  page: { minHeight: '100vh', padding: '1.5rem 2rem', maxWidth: 900, margin: '0 auto' },
  header: { display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', borderBottom: '1px solid #21262d', paddingBottom: '1rem', marginBottom: '1.5rem' },
  title: { fontSize: '1.25rem', margin: 0 },
  headerRight: { display: 'flex', alignItems: 'center', gap: '0.75rem' },
  userEmail: { fontSize: '0.8125rem', color: '#7d8590' },
  signOut: { background: 'transparent', color: '#7d8590', border: '1px solid #30363d', borderRadius: '4px', padding: '0.25rem 0.625rem', fontSize: '0.75rem', cursor: 'pointer' },

  inputRow: { display: 'grid', gridTemplateColumns: '1fr auto', gap: '1rem', marginBottom: '1rem' },
  input: { width: '100%', minHeight: '4rem', background: '#161b22', color: '#e6edf3', border: '1px solid #30363d', borderRadius: '6px', padding: '0.625rem 0.75rem', fontSize: '0.9375rem', resize: 'vertical' },
  inputAside: { display: 'flex', flexDirection: 'column', gap: '0.5rem', alignItems: 'stretch' },
  runBtn: { background: '#238636', color: '#fff', border: 'none', borderRadius: '6px', padding: '0.5rem 1rem', fontSize: '0.875rem', fontWeight: 600, cursor: 'pointer' },
  resetBtn: { background: 'transparent', color: '#7d8590', border: '1px solid #30363d', borderRadius: '4px', padding: '0.25rem 0.5rem', fontSize: '0.75rem', cursor: 'pointer' },
  checkRow: { display: 'flex', alignItems: 'center', gap: '0.375rem', fontSize: '0.75rem', color: '#7d8590' },
  sessionPill: { fontSize: '0.6875rem', color: '#7d8590', textAlign: 'center', fontFamily: 'ui-monospace, monospace' },

  error: { background: '#2d1117', color: '#ffa198', border: '1px solid #56242a', borderRadius: '4px', padding: '0.5rem 0.75rem', marginBottom: '0.5rem', fontSize: '0.8125rem' },
  placeholder: { color: '#7d8590', fontSize: '0.875rem', padding: '2rem 0', textAlign: 'center' },
};
