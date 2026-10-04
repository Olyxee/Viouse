"""Upload → Analyze → Understand → Select → Plan → Edit → Render → Evaluate → Improve.

Each responsibility lives in its own module:
  perception     deterministic per-frame signals          (perception.py)
  understanding  multimodal clip logging + moment scoring  (understanding.py)
  editorial      the edit concept / story                  (editorial.py)
  timeline       precise, explainable edit operations      (timeline.py)
  execution      frame-accurate renderer + audio mixer     (engine.py)
  critic         technical + watched review → revisions    (critic.py)
  preferences    creator style profile                     (style.py)
"""
import time
from pathlib import Path
from .config import settings
from .db import session, Project, Asset, Job, docs, save_doc, DOCUMENTS
from .storage import publish_private

MAX_RENDERS = 3


def update(job_id, stage, detail):
    if job_id is None:
        print(f'[{stage}] {detail}', flush=True)
        return
    with session() as db:
        job = db.get(Job, job_id)
        if job is None: raise RuntimeError('Job deleted')
        job.stage, job.detail, job.heartbeat = stage, detail, time.time()
        project = db.get(Project, job.project_id)
        project.status = stage


def load_project(project_id):
    with session() as db:
        project = db.get(Project, project_id)
        assets = db.query(Asset).filter_by(project_id=project_id).order_by(Asset.created_at).all()
    videos = [a for a in assets if a.kind == 'video']
    music = next((a for a in reversed(assets) if a.kind == 'music'), None)
    references = [a for a in assets if a.kind == 'reference']
    directory = settings.data_dir / 'projects' / project_id / 'work'
    directory.mkdir(parents=True, exist_ok=True)
    return project, assets, videos, music, references, directory


def perceive_all(project, videos, directory, progress):
    from .perception import perceive
    out = {}
    for i, asset in enumerate(videos):
        progress(f'Analysing motion, framing and quality in clip {i + 1} of {len(videos)}')
        out[asset.id] = perceive(asset, directory)
    return out


def understand_all(ai, project, videos, directory, progress):
    """One multimodal pass per clip (cached by asset and model)."""
    from .perception import make_proxy
    from .understanding import understand_batch, batches, coverage
    model = settings.gemini_model
    cached = {d.payload['asset_id']: d.payload for d in docs('clip_understanding', project.id)
              if d.payload.get('model') == model and coverage(d.payload['clip'], next((a.duration for a in videos if a.id == d.payload['asset_id']), 1)) >= .5}
    out = {}
    pending = []
    for asset in videos:
        if asset.id in cached: out[asset.id] = cached[asset.id]['clip']
        elif asset.duration < 1.5:
            out[asset.id] = {'summary': 'too short to use', 'location': '', 'people': [], 'main_subject': '', 'moments': []}
            save_doc('clip_understanding', project.id, {'asset_id': asset.id, 'model': model, 'clip': out[asset.id]})
        else:
            pending.append((asset, make_proxy(asset.path, Path(directory) / f'{asset.id}-watch.mp4')))
    groups = batches(pending)
    for gi, group in enumerate(groups):
        progress(f'Watching footage: {len(out)} of {len(videos)} clips understood')
        logged = understand_batch(ai, group)
        # Clips a batch could not log reliably are watched again on their own.
        for asset, proxy in group:
            if asset.id not in logged: logged.update(understand_batch(ai, [(asset, proxy)]))
        for asset, _ in group:
            out[asset.id] = logged[asset.id]
            save_doc('clip_understanding', project.id, {'asset_id': asset.id, 'model': model, 'clip': out[asset.id]})
    progress(f'Watched all {len(videos)} clips')
    return out


def build_moments(videos, perceived, understood):
    from .understanding import moment_records, score_moments
    records = []
    for asset in videos:
        signals, summary = perceived[asset.id]
        records += moment_records(asset, understood[asset.id], signals, summary)
    return score_moments(records)


def process(job_id, project_id=None, captions=None):
    from .ai import AI
    from .editorial import create_concept
    from .timeline import plan_timeline, revise
    from .engine import render_timeline
    from .critic import evaluate
    from .style import profile_for
    from .music import analyze_music
    from .understanding import select_candidates
    started = time.monotonic()
    ai = AI()
    if job_id:
        with session() as db:
            job = db.get(Job, job_id)
            project_id, captions = job.project_id, job.metrics.get('captions', {})
    captions = captions or {'mode': 'auto'}
    project, assets, videos, music, references, directory = load_project(project_id)
    timings = {}

    def stage(name, detail):
        update(job_id, name, detail)

    t = time.monotonic()
    perceived = perceive_all(project, videos, directory, lambda m: stage('PREPROCESSING', m))
    timings['perception'] = time.monotonic() - t
    t = time.monotonic()
    understood = understand_all(ai, project, videos, directory, lambda m: stage('ANALYZING', m))
    stage('ANALYZING', 'Ranking every moment')
    moments = build_moments(videos, perceived, understood)
    save_doc('moments', project.id, {'moments': [{k: v for k, v in m.items() if k != 'embedding'} for m in moments]}, job_id)
    candidates = select_candidates(moments)
    timings['understanding'] = time.monotonic() - t
    profile = profile_for(ai, project, references, directory, lambda m: stage('ANALYZING', m))
    music_data = analyze_music(music.path, directory) if music else None

    t = time.monotonic()
    stage('PLANNING', 'Developing the editorial concept')
    concept = create_concept(ai, candidates, understood, videos, project, profile, music_data)
    concept_id = save_doc('concepts', project.id, concept, job_id)
    timings['concept'] = time.monotonic() - t

    versions = []
    timeline = None
    review = None
    for attempt in range(MAX_RENDERS):
        stage('EDITING', 'Planning cuts, speed, framing and sound' if attempt == 0 else f'Applying review fixes (pass {attempt + 1})')
        rethink = review is not None and attempt == 1 and (review['review']['story'] < .5 or review['review']['human_editor_believability'] < .45)
        if rethink:
            # The story itself did not work: rethink the concept with the critic's notes, not just patch shots.
            stage('PLANNING', 'Rethinking the edit after review')
            notes = {'previous_title': concept['title'], 'previous_shots': [s['clip_id'] for s in concept['shots']],
                     'critic_problems': [i['detail'] for i in review['issues'] if i['severity'] >= .5][:12],
                     'instruction': 'Build a materially better, tighter and more coherent edit that fixes these problems.'}
            concept = create_concept(ai, candidates, understood, videos, project, profile, music_data, feedback=notes)
            concept_id = save_doc('concepts', project.id, concept, job_id)
            timeline = plan_timeline(concept, moments, perceived, music, music_data, profile, captions)
        elif timeline is None:
            timeline = plan_timeline(concept, moments, perceived, music, music_data, profile, captions)
        else:
            timeline = revise(timeline, review, concept, moments, perceived, music, music_data, profile, captions)
        timeline_id = save_doc('timelines', project.id, timeline, job_id)
        outdir = settings.data_dir / 'projects' / project.id / 'renders' / (job_id or f'bench-{int(time.time())}') / str(attempt + 1)
        t = time.monotonic()
        output, metadata = render_timeline(timeline, assets, outdir, lambda n, total: stage('RENDERING', f'Rendering version {attempt + 1} · {n} of {total} shots'))
        timings[f'render_{attempt + 1}'] = time.monotonic() - t
        stage('REVIEWING', f'Watching version {attempt + 1} back')
        review = evaluate(ai, output, timeline, perceived, outdir)
        save_doc('critic_results', project.id, review, job_id)
        publish_private(output)
        payload = {'timeline_id': timeline_id, 'edl_id': timeline_id, 'concept_id': concept_id, 'path': output,
                   'duration': metadata['duration'], 'critic': review, 'revision': attempt + 1, 'style': concept.get('pacing', ''),
                   'title': concept['title'], 'source_clips_used': len({s['asset_id'] for s in timeline['segments']}),
                   'total_clips': len(videos), 'detected_moments': len(moments), 'captions': captions,
                   'decisions': [{'purpose': s['purpose'], 'why': s['reasons']} for s in timeline['segments']]}
        versions.append((save_doc('video_versions', project.id, payload, job_id), payload))
        if review['passed']: break
    best_id, best = max(versions, key=lambda v: v[1]['critic']['overall'])
    metrics = {'clips': len(videos), 'footage_seconds': sum(a.duration for a in videos), 'moments': len(moments),
               'candidates': len(candidates), 'renders': len(versions), 'critic_score': best['critic']['overall'],
               'best_version_id': best_id, 'timings_seconds': timings, 'total_seconds': time.monotonic() - started,
               'ai_calls': ai.calls, 'captions': captions}
    if job_id:
        with session() as db:
            job = db.get(Job, job_id)
            job.status, job.stage, job.detail, job.metrics = 'COMPLETED', 'COMPLETED', 'Your video is ready', {**job.metrics, **metrics}
            db.get(Project, project.id).status = 'COMPLETED'
    return best_id, best, metrics


def process_caption_job(job_id):
    """Re-render the saved timeline with new subtitle settings; never burn text over text."""
    from copy import deepcopy
    from .captions import checked_settings
    from .engine import render_timeline
    with session() as db:
        job = db.get(Job, job_id)
        source = db.get(DOCUMENTS['video_versions'], job.metrics['source_version_id'])
        if not source or source.project_id != job.project_id: raise ValueError('Source version is unavailable')
        decision = db.get(DOCUMENTS['timelines'], source.payload.get('timeline_id', ''))
        if not decision or decision.project_id != job.project_id: raise ValueError('Source timeline is unavailable')
        timeline = deepcopy(decision.payload)
        assets = db.query(Asset).filter_by(project_id=job.project_id).order_by(Asset.created_at).all()
    timeline['captions'] = checked_settings(job.metrics['captions'], timeline['duration']).model_dump()
    update(job_id, 'RENDERING', 'Styling subtitles on your existing edit')
    outdir = settings.data_dir / 'projects' / job.project_id / 'renders' / job_id / '1'
    output, metadata = render_timeline(timeline, assets, outdir, lambda n, total: update(job_id, 'RENDERING', f'Prepared {n} of {total} shots for subtitles'))
    publish_private(output)
    timeline_id = save_doc('timelines', job.project_id, timeline, job_id)
    payload = {**source.payload, 'path': output, 'duration': metadata['duration'], 'timeline_id': timeline_id, 'edl_id': timeline_id,
               'captions': timeline['captions'], 'source_version_id': source.id, 'subtitle_only': True}
    version_id = save_doc('video_versions', job.project_id, payload, job_id)
    with session() as db:
        current = db.get(Job, job_id)
        current.metrics = {**job.metrics, 'best_version_id': version_id, 'ai_calls': []}
        current.status, current.stage, current.detail = 'COMPLETED', 'COMPLETED', 'Your subtitled video is ready'
        db.get(Project, job.project_id).status = 'COMPLETED'
