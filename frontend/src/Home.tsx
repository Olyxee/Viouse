import {ChevronRight, Clapperboard, Film, LoaderCircle, Plus} from 'lucide-react';
import {useState} from 'react';
import {formatDate, type Project} from './api';
import {STEPS, StatusBadge} from './ui';

type Props = {
  projects: Project[];
  busy: boolean;
  onCreate: (name: string) => void;
  onOpen: (id: string) => void;
};

export function Home({projects, busy, onCreate, onOpen}: Props) {
  const [name, setName] = useState('');

  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <h1>
            Raw clips in.
            <br />
            <span>A finished reel out.</span>
          </h1>
          <p>Add your footage, pick a style, and the editor cuts a vertical video for you. No editing skills needed.</p>
          <ol className="how-it-works" aria-label="How it works">
            {STEPS.map((step, i) => (
              <li key={step}>
                <span aria-hidden="true">{i + 1}</span>
                {step}
              </li>
            ))}
          </ol>
        </div>
        <form
          className="new-video"
          onSubmit={e => {
            e.preventDefault();
            onCreate(name);
          }}
        >
          <h2>Start a new video</h2>
          <label htmlFor="project-name">Name (optional)</label>
          <input
            id="project-name"
            placeholder="e.g. Leg day vlog"
            value={name}
            maxLength={120}
            onChange={e => setName(e.target.value)}
          />
          <button className="primary wide" disabled={busy}>
            {busy ? <LoaderCircle size={20} className="spin" /> : <Plus size={20} />}
            New video
          </button>
        </form>
      </section>

      <section className="projects" aria-labelledby="projects-title">
        <div className="section-title">
          <h2 id="projects-title">Your videos</h2>
          {projects.length > 0 && <span>{projects.length === 1 ? '1 video' : `${projects.length} videos`}</span>}
        </div>
        {projects.length === 0 ? (
          <div className="empty">
            <div className="empty-icon" aria-hidden="true">
              <Clapperboard size={30} />
            </div>
            <h3>No videos yet</h3>
            <p>Press “New video” to add your first clips. It only takes a minute to get started.</p>
          </div>
        ) : (
          <ul className="project-grid">
            {projects.map(p => (
              <li key={p.id}>
                <button className="project-card" onClick={() => onOpen(p.id)} disabled={busy}>
                  <span className={`project-art tone-${p.status === 'COMPLETED' ? 'ready' : 'plain'}`} aria-hidden="true">
                    <Film size={30} />
                  </span>
                  <span className="project-card-info">
                    <span className="project-card-text">
                      <span className="project-name">{p.name}</span>
                      <span className="project-meta">
                        <StatusBadge status={p.status} />
                        <small>{formatDate(p.created_at)}</small>
                      </span>
                    </span>
                    <ChevronRight size={20} aria-hidden="true" />
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}
