/**
 * /batch — the batch runner inside the sandbox UI (gub-agent#33, epic #29
 * step 04): edit the question set and the arms, start a matrix on the sandbox
 * engine, watch it, read the numbers. Same sign-in, same engine badge, same
 * config form as the single-run console; the files it edits are the ones the
 * headless `scratchpad/batch-run.mjs` reads.
 */
'use client';

import { useEffect, useRef, useState } from 'react';
import { useAuth } from '@/lib/auth/useAuth';
import { EngineBadge } from '@/components/ConfigPanel';
import { fetchConfig, type SandboxConfigResponse } from '@/lib/api';
import { QuestionsEditor } from '@/components/batch/QuestionsEditor';
import { ArmsEditor } from '@/components/batch/ArmsEditor';
import { RunPanel } from '@/components/batch/RunPanel';
import { B } from '@/components/batch/batchStyles';

type Tab = 'run' | 'questions' | 'arms';

export default function BatchPage() {
  const auth = useAuth();
  if (auth.status === 'loading') return <Centered>Restoring session…</Centered>;
  if (auth.status !== 'authenticated') return <LoginGate auth={auth} />;
  return <Batch auth={auth} />;
}

function LoginGate({ auth }: { auth: ReturnType<typeof useAuth> }) {
  const btnRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (btnRef.current) auth.renderGoogleButton(btnRef.current); }, [auth]);
  return (
    <Centered>
      <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'center' }}>
        <h1 style={{ fontSize: '1.25rem', fontWeight: 600 }}>gub-agent debug · batch</h1>
        <p style={{ color: '#7d8590', fontSize: '0.875rem', maxWidth: 380 }}>Sign in with your work Google account. Batch runs go as the dedicated sandbox subject when one is configured, otherwise as you.</p>
        <div ref={btnRef} />
        {auth.status === 'error' && auth.error && <div style={{ color: '#ffa198', fontSize: '0.8125rem', maxWidth: 360 }}>{auth.error}</div>}
      </div>
    </Centered>
  );
}

function Batch({ auth }: { auth: ReturnType<typeof useAuth> }) {
  const [options, setOptions] = useState<SandboxConfigResponse | null>(null);
  const [optionsError, setOptionsError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('run');
  const [refreshKey, setRefreshKey] = useState(0);
  useEffect(() => { fetchConfig().then(setOptions).catch((e: Error) => setOptionsError(e.message)); }, []);

  const goRun = () => { setRefreshKey((k) => k + 1); setTab('run'); };

  return (
    <div style={{ minHeight: '100vh', padding: '1.5rem 2rem', margin: '0 auto', maxWidth: 1500 }}>
      <header style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '1rem', borderBottom: '1px solid #21262d', paddingBottom: '1rem', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', flexWrap: 'wrap', minWidth: 0 }}>
          <h1 style={{ fontSize: '1.25rem', margin: 0, whiteSpace: 'nowrap' }}>gub-agent debug <span style={{ color: '#7d8590' }}>· batch</span></h1>
          <EngineBadge config={options} error={optionsError} />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexShrink: 0 }}>
          <a href="/" style={{ color: '#79c0ff', fontSize: '0.8125rem', textDecoration: 'none', border: '1px solid #30363d', borderRadius: '4px', padding: '0.25rem 0.625rem' }}>← single run</a>
          <span style={{ fontSize: '0.8125rem', color: '#7d8590' }}>{auth.user?.email}</span>
          <button onClick={() => void auth.logout()} style={B.btn}>sign out</button>
        </div>
      </header>

      <nav style={B.tabs}>
        {(['run', 'questions', 'arms'] as Tab[]).map((t) => (
          <button key={t} type="button" onClick={() => (t === 'run' ? goRun() : setTab(t))} style={{ ...B.tab, ...(tab === t ? B.tabActive : {}) }}>{t}</button>
        ))}
        <span style={{ ...B.dim, marginLeft: 'auto', alignSelf: 'center' }}>every question × every arm, a fresh session per cell, deterministic metrics — no LLM judge</span>
      </nav>

      {tab === 'run' && <RunPanel options={options} refreshKey={refreshKey} />}
      {tab === 'questions' && <QuestionsEditor />}
      {tab === 'arms' && <ArmsEditor options={options} />}
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#7d8590', fontSize: '0.875rem' }}>{children}</div>;
}
