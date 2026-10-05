import React, {useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {LoaderCircle, Settings} from 'lucide-react';
import './style.css';
import {api, BUSY, type Health, type Project} from './api';
import {Home} from './Home';
import {KeySetup} from './KeySetup';
import {ProjectView} from './Project';
import {ErrorBanner} from './ui';

function App() {
  const [ready, setReady] = useState(false);
  const [checking, setChecking] = useState(true);
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [aiReady, setAiReady] = useState(false);
  const [showKeySetup, setShowKeySetup] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api('/local-session', {method: 'POST'})
      .then(() => setReady(true))
      .catch(e => setError((e as Error).message))
      .finally(() => setChecking(false));
  }, []);

  useEffect(() => {
    if (!ready) return;
    Promise.all([api<Project[]>('/projects'), api<Health>('/health')])
      .then(([p, h]) => {
        setProjects(p);
        setAiReady(h.ai_configured);
      })
      .catch(e => setError((e as Error).message));
  }, [ready]);

  // Keep "Working" badges on the home screen up to date.
  const anyWorking = !project && projects.some(p => BUSY.includes(p.status));
  useEffect(() => {
    if (!anyWorking) return;
    const timer = setInterval(() => {
      api<Project[]>('/projects').then(setProjects).catch(() => undefined);
    }, 5000);
    return () => clearInterval(timer);
  }, [anyWorking]);

  async function act(fn: () => Promise<void>) {
    setError('');
    setBusy(true);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function open(id: string) {
    const p = await api<Project>('/projects/' + id);
    setProject(p);
    window.scrollTo(0, 0);
  }

  async function goHome() {
    setProject(null);
    setError('');
    setProjects(await api<Project[]>('/projects'));
    window.scrollTo(0, 0);
  }

  async function create(name: string) {
    const fallback = `My video · ${new Date().toLocaleDateString(undefined, {month: 'short', day: 'numeric'})}`;
    const p = await api<{id: string}>('/projects', {method: 'POST', body: JSON.stringify({name: name.trim() || fallback})});
    await open(p.id);
  }

  const working = !!project?.job && !['COMPLETED', 'FAILED'].includes(project.job.status);

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <header className="app-header">
        <button className="brand" onClick={() => ready && act(goHome)} aria-label="Vinteo, go to all videos">
          <img className="brand-logo" src="/vinteo-logo.png" alt="" width={34} height={34} />
          Vinteo
        </button>
        {ready && aiReady && !working && !showKeySetup && (
          <button className="header-action" onClick={() => setShowKeySetup(true)}>
            <Settings size={15} aria-hidden="true" />
            AI settings
          </button>
        )}
      </header>

      <main id="main">
        {error && <ErrorBanner message={error} onDismiss={() => setError('')} />}

        {ready && (!aiReady || showKeySetup) && (
          <KeySetup
            replacing={aiReady}
            onConnected={() => {
              setAiReady(true);
              setShowKeySetup(false);
            }}
            onClose={aiReady ? () => setShowKeySetup(false) : undefined}
          />
        )}

        {checking ? (
          <div className="loading" role="status">
            <LoaderCircle className="spin" aria-hidden="true" />
            Opening your workspace…
          </div>
        ) : !ready ? null : project ? (
          <ProjectView
            key={project.id}
            project={project}
            setProject={setProject}
            aiReady={aiReady}
            onHome={() => act(goHome)}
            onOpenKeySetup={() => {
              setShowKeySetup(true);
              window.scrollTo(0, 0);
            }}
          />
        ) : (
          <Home projects={projects} busy={busy} onCreate={name => act(() => create(name))} onOpen={id => act(() => open(id))} />
        )}
      </main>
    </>
  );
}

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
