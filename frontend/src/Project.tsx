import {
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle2,
  Download,
  Film,
  LoaderCircle,
  Music2,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Trash2,
  Upload,
  X,
} from 'lucide-react';
import {useEffect, useRef, useState, type DragEvent, type RefObject} from 'react';
import {api, formatDuration, uploadFile, type Project as ProjectData} from './api';
import {CaptionStudio, captionProblem, defaultCaptions, type CaptionSettings} from './CaptionStudio';
import {ErrorBanner, Stepper} from './ui';

const VIDEO_TYPES = ['.mp4', '.mov', '.webm'];
const MUSIC_TYPES = ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'];

const EDIT_STAGES: [string, string][] = [
  ['QUEUED', 'Getting ready'],
  ['PREPROCESSING', 'Looking through your clips'],
  ['ANALYZING', 'Finding the best moments'],
  ['PLANNING', 'Planning the story'],
  ['EDITING', 'Cutting, timing and framing'],
  ['RENDERING', 'Making the video'],
  ['REVIEWING', 'Watching it back and polishing'],
];
const CAPTION_STAGES: [string, string][] = [
  ['QUEUED', 'Getting ready'],
  ['RENDERING', 'Adding your subtitles'],
];

const LENGTHS = [
  [0, 'Auto'],
  [30, '30 sec'],
  [45, '45 sec'],
  [60, '60 sec'],
] as const;

const PACING = [
  ['automatic', 'Let the editor decide', 'Picks what suits your footage'],
  ['high-energy', 'Fast and punchy', 'Quick cuts, strong moments'],
  ['cinematic', 'Calm and cinematic', 'Slower, more room to breathe'],
] as const;

type UploadItem = {key: string; name: string; progress: number; state: 'waiting' | 'uploading' | 'error'; error?: string};

type Props = {
  project: ProjectData;
  setProject: (p: ProjectData) => void;
  aiReady: boolean;
  onHome: () => void;
  onOpenKeySetup: () => void;
};

const extension = (name: string) => name.slice(name.lastIndexOf('.')).toLowerCase();

export function ProjectView({project, setProject, aiReady, onHome, onOpenKeySetup}: Props) {
  const clips = project.assets?.filter(a => a.kind === 'video') || [];
  const music = project.assets?.filter(a => a.kind === 'music').at(-1);
  const references = project.assets?.filter(a => a.kind === 'reference') || [];
  const totalFootage = clips.reduce((n, a) => n + a.duration, 0);
  const job = project.job;
  const working = !!job && !['COMPLETED', 'FAILED'].includes(job.status);
  const failed = job?.status === 'FAILED';
  const version = project.versions?.find(v => v.id === project.best_version_id);
  const locked = !!project.versions?.length;

  const [step, setStep] = useState(clips.length ? 1 : 0);
  // New projects start on Auto; a fixed length only sticks if the footage is long enough for it.
  const [length, setLength] = useState<number>(job ? project.target_duration || 0 : 0);
  const usableLength = length && totalFootage < length ? 0 : length;
  const [style, setStyle] = useState(PACING.some(([id]) => id === project.style) ? project.style : 'automatic');
  const [captions, setCaptions] = useState<CaptionSettings>({...defaultCaptions, ...(version?.captions || project.captions)});
  const [consent, setConsent] = useState(false);
  const [consentNudge, setConsentNudge] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [drag, setDrag] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [feedbackDone, setFeedbackDone] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(false);
  const uploading = uploads.some(u => u.state !== 'error');

  const fileInput = useRef<HTMLInputElement>(null);
  const musicInput = useRef<HTMLInputElement>(null);
  const refInput = useRef<HTMLInputElement>(null);
  const consentBox = useRef<HTMLInputElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const mounted = useRef(true);
  useEffect(() => () => void (mounted.current = false), []);

  const view = working ? 2 : version ? 3 : step;

  // Move focus to the new screen's heading so keyboard and screen-reader users follow the flow.
  const firstView = useRef(true);
  useEffect(() => {
    if (firstView.current) {
      firstView.current = false;
      return;
    }
    heading.current?.focus();
  }, [view]);

  // Live progress while the editor is working.
  useEffect(() => {
    if (!working) return;
    const timer = setInterval(() => {
      api<ProjectData>('/projects/' + project.id)
        .then(p => mounted.current && setProject(p))
        .catch(e => mounted.current && setError((e as Error).message));
    }, 2000);
    return () => clearInterval(timer);
  }, [project.id, working]);

  // When a new version arrives, show the subtitles it was made with.
  useEffect(() => {
    if (version) setCaptions({...defaultCaptions, ...(version.captions || project.captions)});
  }, [version?.id]);

  async function refresh() {
    const p = await api<ProjectData>('/projects/' + project.id);
    if (mounted.current) setProject(p);
  }

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

  async function upload(list: FileList | File[] | null, kind: 'video' | 'music' | 'reference') {
    if (!list || working || uploading) return;
    const allowed = kind === 'music' ? MUSIC_TYPES : VIDEO_TYPES;
    const files = Array.from(list);
    const rejected = files.filter(f => !allowed.includes(extension(f.name)));
    const accepted = files.filter(f => allowed.includes(extension(f.name)));
    if (rejected.length) {
      setError(
        `${rejected.map(f => f.name).join(', ')} can't be used. Please add ${kind === 'music' ? 'MP3, WAV, M4A, AAC, OGG or FLAC audio' : 'MP4, MOV or WebM video'} files.`,
      );
    } else {
      setError('');
    }
    if (!accepted.length) return;

    const items: UploadItem[] = accepted.map((f, i) => ({key: `${Date.now()}-${i}`, name: f.name, progress: 0, state: 'waiting'}));
    setUploads(items);
    const patch = (key: string, change: Partial<UploadItem>) =>
      setUploads(current => current.map(u => (u.key === key ? {...u, ...change} : u)));

    for (const [i, file] of accepted.entries()) {
      const key = items[i].key;
      patch(key, {state: 'uploading'});
      try {
        await uploadFile(project.id, file, kind, progress => patch(key, {progress}));
        setUploads(current => current.filter(u => u.key !== key));
        await refresh().catch(() => undefined);
      } catch (e) {
        patch(key, {state: 'error', error: (e as Error).message});
      }
    }
    for (const input of [fileInput, musicInput, refInput]) if (input.current) input.current.value = '';
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDrag(false);
    upload(e.dataTransfer.files, 'video');
  }

  async function removeAsset(assetId: string) {
    await api(`/projects/${project.id}/assets/${assetId}`, {method: 'DELETE'});
    await refresh();
  }

  async function start() {
    await api(`/projects/${project.id}/jobs`, {
      method: 'POST',
      body: JSON.stringify({style, target_duration: usableLength, process_consent: consent, captions}),
    });
    setConsent(false);
    await refresh();
  }

  function requestCreate() {
    if (!consent) {
      setConsentNudge(true);
      consentBox.current?.focus();
      return;
    }
    act(start);
  }

  async function restyle() {
    if (!version) return;
    await api(`/projects/${project.id}/captions`, {method: 'POST', body: JSON.stringify({version_id: version.id, captions})});
    await refresh();
  }

  async function rate(rating: 'good' | 'needs-work') {
    if (!version) return;
    await api(`/projects/${project.id}/feedback`, {
      method: 'POST',
      body: JSON.stringify({version_id: version.id, rating, text: feedback, research_consent: false}),
    });
    setFeedbackDone(true);
  }

  async function deleteProject() {
    await api('/projects/' + project.id, {method: 'DELETE'});
    onHome();
  }

  const reachable = version || working ? [] : clips.length ? [0, 1] : [0];
  const previewDuration = usableLength || Math.min(60, Math.max(30, totalFootage * 0.2), totalFootage) || 30;
  const jobError = failed ? job?.error || 'Something went wrong while making your video. Your footage is safe.' : '';

  const consentBlock = (
    <div className={`consent ${consentNudge && !consent ? 'nudge' : ''}`}>
      <label className="check">
        <input
          ref={consentBox}
          type="checkbox"
          checked={consent}
          onChange={e => {
            setConsent(e.target.checked);
            setConsentNudge(false);
          }}
          aria-describedby={`consent-detail-${project.id}`}
        />
        <span>
          <strong>Let AI watch my clips</strong>
        </span>
      </label>
      <p id={`consent-detail-${project.id}`} className="hint">
        To plan the edit, short samples of your footage are sent to Google Gemini. Your original files stay on this computer.
      </p>
      {consentNudge && !consent && (
        <p className="field-error" role="alert">
          Tick “Let AI watch my clips” to continue.
        </p>
      )}
    </div>
  );

  return (
    <div className="project-view">
      <div className="project-top">
        <button type="button" className="back-link" onClick={onHome}>
          <ArrowLeft size={16} aria-hidden="true" />
          All videos
        </button>
        <span className="project-title" title={project.name}>
          {project.name}
        </span>
      </div>

      <Stepper current={view} reachable={reachable} onSelect={i => setStep(i)} />

      {error && <ErrorBanner message={error} onDismiss={() => setError('')} />}

      {/* Step 1: footage */}
      {view === 0 && (
        <section className="panel step-panel" aria-labelledby="step-title">
          <header className="step-head">
            <h1 id="step-title" ref={heading} tabIndex={-1}>
              Add your clips
            </h1>
            <p>Add the raw videos you filmed. Add as many as you like, up to 30 clips. The editor picks the best parts.</p>
          </header>

          {jobError && !locked && (
            <ErrorBanner
              message={jobError}
              action={<p className="hint">Your clips are saved. Go to “Choose style” and press Create to try again.</p>}
            />
          )}

          <div
            className={`dropzone ${drag ? 'drag' : ''}`}
            onDragOver={e => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={onDrop}
          >
            <span className="upload-icon" aria-hidden="true">
              <Upload size={28} />
            </span>
            <h2>Drag your videos here</h2>
            <p>or</p>
            <button type="button" className="secondary" disabled={uploading || working} onClick={() => fileInput.current?.click()}>
              Choose videos from your computer
            </button>
            <p className="hint">MP4, MOV or WebM</p>
            <input
              ref={fileInput}
              type="file"
              multiple
              accept={VIDEO_TYPES.join(',')}
              hidden
              onChange={e => upload(e.target.files, 'video')}
            />
          </div>

          <UploadList items={uploads} onDismiss={key => setUploads(u => u.filter(x => x.key !== key))} />

          {clips.length > 0 && (
            <div className="clip-block">
              <div className="clip-count">
                <strong>{clips.length === 1 ? '1 clip added' : `${clips.length} clips added`}</strong>
                <span>{formatDuration(totalFootage)} of footage</span>
              </div>
              <ul className="clip-list">
                {clips.map(a => (
                  <li className="clip-row" key={a.id}>
                    <Film size={18} aria-hidden="true" />
                    <div>
                      <strong>{a.filename}</strong>
                      <small>
                        {formatDuration(a.duration)} · {a.metadata.width} × {a.metadata.height}
                      </small>
                    </div>
                    {!locked && (
                      <button
                        type="button"
                        className="icon-button"
                        aria-label={`Remove ${a.filename}`}
                        disabled={busy || uploading}
                        onClick={() => act(() => removeAsset(a.id))}
                      >
                        <X size={16} />
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <footer className="step-actions">
            <p className="privacy">
              <ShieldCheck size={16} aria-hidden="true" />
              Your original files stay on this computer.
            </p>
            <button type="button" className="primary" disabled={!clips.length || uploading} onClick={() => setStep(1)}>
              Next: choose style
              <ArrowRight size={18} aria-hidden="true" />
            </button>
          </footer>
        </section>
      )}

      {/* Step 2: style */}
      {view === 1 && (
        <section className="panel step-panel" aria-labelledby="step-title">
          <header className="step-head">
            <h1 id="step-title" ref={heading} tabIndex={-1}>
              Choose a style
            </h1>
            <p>The defaults work well. Change anything you like, then press Create.</p>
          </header>

          {jobError && <ErrorBanner message={jobError} action={<p className="hint">Your clips are saved. Press Create to try again.</p>} />}

          <div className="style-grid">
            <div className="style-main">
              <fieldset>
                <legend className="field-label">Video length</legend>
                <div className="segmented four" role="radiogroup" aria-label="Video length">
                  {LENGTHS.map(([n, label]) => {
                    const tooLong = n > 0 && totalFootage < n;
                    return (
                      <button
                        type="button"
                        role="radio"
                        key={n}
                        aria-checked={usableLength === n}
                        className={usableLength === n ? 'selected' : ''}
                        disabled={tooLong}
                        title={tooLong ? 'Add more footage to make a video this long' : undefined}
                        onClick={() => setLength(n)}
                      >
                        {label}
                      </button>
                    );
                  })}
                </div>
                <p className="hint">Auto picks a length that fits your footage (usually 30 to 60 seconds).</p>
              </fieldset>

              <fieldset>
                <legend className="field-label">Pace</legend>
                <div className="choice-list" role="radiogroup" aria-label="Pace">
                  {PACING.map(([id, title, desc]) => (
                    <button
                      type="button"
                      role="radio"
                      key={id}
                      aria-checked={style === id}
                      className={`choice ${style === id ? 'selected' : ''}`}
                      onClick={() => setStyle(id)}
                    >
                      <span className="radio" aria-hidden="true" />
                      <span>
                        <strong>{title}</strong>
                        <small>{desc}</small>
                      </span>
                    </button>
                  ))}
                </div>
              </fieldset>

              <fieldset>
                <legend className="field-label">
                  Music <span className="optional">optional</span>
                </legend>
                <input
                  ref={musicInput}
                  type="file"
                  accept={MUSIC_TYPES.join(',')}
                  hidden
                  onChange={e => upload(e.target.files, 'music')}
                />
                <div className="file-pick">
                  <Music2 size={18} aria-hidden="true" />
                  <span className="file-pick-name">{music ? music.filename : 'No song added. Your clips keep their own sound.'}</span>
                  <button type="button" className="secondary small" disabled={busy || uploading} onClick={() => musicInput.current?.click()}>
                    {music ? 'Change' : 'Add a song'}
                  </button>
                  {music && (
                    <button
                      type="button"
                      className="link-button"
                      disabled={busy || uploading}
                      onClick={() =>
                        act(async () => {
                          for (const a of project.assets?.filter(x => x.kind === 'music') || []) await removeAsset(a.id);
                        })
                      }
                    >
                      Remove
                    </button>
                  )}
                </div>
              </fieldset>

              <details className="disclosure">
                <summary>More options</summary>
                <div className="disclosure-body">
                  <p className="field-label">Match my editing style</p>
                  <p className="hint">Add a few videos you edited before and the editor will copy their feel: pace, cuts and framing.</p>
                  <input
                    ref={refInput}
                    type="file"
                    multiple
                    accept={VIDEO_TYPES.join(',')}
                    hidden
                    onChange={e => upload(e.target.files, 'reference')}
                  />
                  <div className="file-pick">
                    <Film size={18} aria-hidden="true" />
                    <span className="file-pick-name">
                      {references.length
                        ? `${references.length} past edit${references.length > 1 ? 's' : ''} added`
                        : 'No past edits added'}
                    </span>
                    <button type="button" className="secondary small" disabled={busy || uploading} onClick={() => refInput.current?.click()}>
                      Add videos
                    </button>
                  </div>
                </div>
              </details>

              <UploadList items={uploads} onDismiss={key => setUploads(u => u.filter(x => x.key !== key))} />
            </div>

            <div className="style-side">
              <CaptionStudio value={captions} onChange={setCaptions} duration={previewDuration} disabled={busy} />
            </div>
          </div>

          <footer className="create-bar">
            {consentBlock}
            <div className="create-actions">
              {!aiReady && (
                <p className="field-error">
                  Connect the AI first.{' '}
                  <button type="button" className="link-button" onClick={onOpenKeySetup}>
                    Set it up
                  </button>
                </p>
              )}
              <button
                type="button"
                className="primary big"
                disabled={!clips.length || busy || uploading || !aiReady || !!captionProblem(captions, previewDuration)}
                onClick={requestCreate}
              >
                {busy ? <LoaderCircle size={20} className="spin" /> : <Sparkles size={20} aria-hidden="true" />}
                {failed ? 'Try again' : 'Create my video'}
              </button>
              <button type="button" className="link-button" onClick={() => setStep(0)}>
                <ArrowLeft size={14} aria-hidden="true" /> Back to clips
              </button>
            </div>
          </footer>
        </section>
      )}

      {/* Step 3: progress */}
      {view === 2 && job && <ProgressView job={job} headingRef={heading} />}

      {/* Step 4: review */}
      {view === 3 && version && (
        <section className="review" aria-labelledby="step-title">
          <div className="player-wrap">
            <video key={version.id} controls playsInline preload="metadata" src={`/api/projects/${project.id}/versions/${version.id}/video`} />
            <span>
              {Math.round(version.duration)} sec · vertical 1080 × 1920
            </span>
          </div>

          <div className="review-side">
            {jobError && (
              <ErrorBanner
                message={jobError}
                action={<p className="hint">Your earlier video below is still fine to download.</p>}
              />
            )}
            <p className="ready-tag">
              <CheckCircle2 size={16} aria-hidden="true" /> Ready
            </p>
            <h1 id="step-title" ref={heading} tabIndex={-1}>
              Your video is ready
            </h1>
            {version.title && <p className="video-title">{version.title}</p>}

            <a className="primary big wide" href={`/api/projects/${project.id}/versions/${version.id}/video?download=true`} download>
              <Download size={20} aria-hidden="true" />
              Download video (MP4)
            </a>

            {(project.versions?.length || 0) > 1 && (
              <div className="version-choice">
                <label htmlFor="export-version">Showing</label>
                <select
                  id="export-version"
                  value={version.id}
                  onChange={e => setProject({...project, best_version_id: e.target.value})}
                >
                  {project.versions?.map((v, i) => (
                    <option key={v.id} value={v.id}>
                      Version {i + 1}
                      {v.captions?.mode === 'off' ? ' · no subtitles' : v.captions?.mode === 'custom' ? ' · your words' : ''}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="feedback">
              <h2>How did it turn out?</h2>
              {feedbackDone ? (
                <p role="status" className="thanks">
                  <Check size={16} aria-hidden="true" /> Thanks, your feedback is saved.
                </p>
              ) : (
                <>
                  <label htmlFor="feedback-text" className="sr-only">
                    Anything to change? (optional)
                  </label>
                  <textarea
                    id="feedback-text"
                    placeholder="Anything to change? (optional)"
                    maxLength={2000}
                    value={feedback}
                    onChange={e => setFeedback(e.target.value)}
                  />
                  <div className="feedback-buttons">
                    <button type="button" className="secondary small" disabled={busy} onClick={() => act(() => rate('good'))}>
                      👍 Looks good
                    </button>
                    <button type="button" className="secondary small" disabled={busy} onClick={() => act(() => rate('needs-work'))}>
                      👎 Needs work
                    </button>
                  </div>
                </>
              )}
            </div>

            <details className="disclosure">
              <summary>Change subtitles</summary>
              <div className="disclosure-body">
                <CaptionStudio value={captions} onChange={setCaptions} duration={version.duration} disabled={busy} onExport={() => act(restyle)} />
              </div>
            </details>

            <details className="disclosure">
              <summary>More options</summary>
              <div className="disclosure-body">
                <h3 className="field-label">Try a different cut</h3>
                <p className="hint">The editor makes a fresh edit from the same clips. Your current video is kept.</p>
                {consentBlock}
                <button
                  type="button"
                  className="secondary wide"
                  disabled={busy || !aiReady || !!captionProblem(captions, previewDuration)}
                  onClick={requestCreate}
                >
                  <RefreshCw size={17} aria-hidden="true" />
                  Make a different cut
                </button>

                {version.decisions?.length ? (
                  <details className="disclosure nested">
                    <summary>Why it was edited this way</summary>
                    <ol className="decisions">
                      {version.decisions.map((d, i) => (
                        <li key={i}>
                          <strong>{d.purpose}</strong> {d.why.slice(0, 3).join(' · ')}
                        </li>
                      ))}
                    </ol>
                  </details>
                ) : null}

                <details className="disclosure nested">
                  <summary>Video details</summary>
                  <dl className="details-list">
                    <div>
                      <dt>Length</dt>
                      <dd>{Math.round(version.duration)} sec</dd>
                    </div>
                    <div>
                      <dt>Clips used</dt>
                      <dd>
                        {version.source_clips_used} of {version.total_clips}
                      </dd>
                    </div>
                    <div>
                      <dt>Pace</dt>
                      <dd>{version.style.replace('-', ' ')}</dd>
                    </div>
                  </dl>
                </details>
              </div>
            </details>
          </div>
        </section>
      )}

      {!working && (
        <div className="delete-area">
          {deleteConfirm ? (
            <div className="delete-confirm" role="group" aria-label="Confirm delete">
              <span>Delete this video project, its clips and every version? This can't be undone.</span>
              <button type="button" className="danger" disabled={busy} onClick={() => act(deleteProject)}>
                Yes, delete it
              </button>
              <button type="button" className="link-button" onClick={() => setDeleteConfirm(false)}>
                Keep it
              </button>
            </div>
          ) : (
            <button type="button" className="link-button muted" onClick={() => setDeleteConfirm(true)}>
              <Trash2 size={14} aria-hidden="true" />
              Delete project
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function UploadList({items, onDismiss}: {items: UploadItem[]; onDismiss: (key: string) => void}) {
  if (!items.length) return null;
  return (
    <ul className="upload-list" aria-label="Uploads">
      {items.map(item => (
        <li key={item.key} className={item.state}>
          <div className="upload-row">
            <span className="upload-name">{item.name}</span>
            {item.state === 'error' ? (
              <button type="button" className="icon-button" aria-label={`Dismiss ${item.name}`} onClick={() => onDismiss(item.key)}>
                <X size={15} />
              </button>
            ) : (
              <span className="upload-pct">{item.state === 'waiting' ? 'Waiting' : `${Math.round(item.progress * 100)}%`}</span>
            )}
          </div>
          {item.state === 'error' ? (
            <p className="field-error">{item.error} Try adding this file again.</p>
          ) : (
            <div
              className="bar"
              role="progressbar"
              aria-label={`Uploading ${item.name}`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(item.progress * 100)}
            >
              <span style={{width: `${item.progress * 100}%`}} />
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}

function ProgressView({job, headingRef}: {job: NonNullable<ProjectData['job']>; headingRef: RefObject<HTMLHeadingElement | null>}) {
  const stages = job.kind === 'captions' ? CAPTION_STAGES : EDIT_STAGES;
  const current = Math.max(
    0,
    stages.findIndex(([id]) => id === job.stage),
  );
  const percent = Math.round(((current + 0.5) / stages.length) * 100);

  return (
    <section className="panel progress" aria-labelledby="step-title">
      <span className="progress-icon" aria-hidden="true">
        <Sparkles size={28} />
      </span>
      <h1 id="step-title" ref={headingRef} tabIndex={-1}>
        {job.kind === 'captions' ? 'Adding your subtitles…' : 'Making your video…'}
      </h1>
      <p>
        This usually takes a few minutes, longer for lots of footage. <strong>You can close this page</strong>; the editor keeps working and
        your video will be here when you come back.
      </p>

      <div className="bar bar-lg" role="progressbar" aria-label="Overall progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
        <span style={{width: `${percent}%`}} />
      </div>

      <ol className="stage-list" aria-live="polite">
        {stages.map(([id, label], i) => {
          const state = i < current ? 'done' : i === current ? 'active' : 'todo';
          return (
            <li key={id} className={state}>
              <span className="stage-dot" aria-hidden="true">
                {state === 'done' ? <Check size={15} /> : state === 'active' ? <LoaderCircle size={15} className="spin" /> : null}
              </span>
              <div>
                {label}
                {state === 'active' && job.detail && <small>{job.detail}</small>}
                <span className="sr-only">{state === 'done' ? ' (done)' : state === 'active' ? ' (in progress)' : ''}</span>
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
