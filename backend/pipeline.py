import json
import time
from pathlib import Path
from .config import settings, active_vision_model
from .db import session, Project, Asset, Job, docs, save_doc
from .preprocess import segment_video
from .ai import AI
from .audio import transcribe, audio_features
from .planner import rank_segments, understand, create_plan, build_edl
from .render import render
from .media import frame
from .schemas import Critic
from .storage import publish_private

def update(job_id, stage, detail):
    with session() as db:
        job = db.get(Job,job_id)
        if job is None: raise RuntimeError('Job deleted')
        job.stage,job.detail,job.heartbeat = stage,detail,time.time()
        project = db.get(Project,job.project_id)
        project.status = stage

def review(ai, output, plan, edl, directory):
    paths,times = [],[]
    for i,shot in enumerate(edl['timeline'][:30]):
        at = shot['timeline_start']+shot['duration']*.5
        p = Path(directory)/f'review-{i}.jpg'
        frame(output,at,p)
        paths.append(p)
        times.append(at)
    return ai.structured(Critic,'Review this actual rendered video using chronologically sampled frames (one per shot), exact shot timings, and the story. Acknowledge limitations of frame sampling; do not claim to have heard audio. Only recommend actionable changes supported by evidence. Request revision confidence >=0.8 only for clear improvements. '+json.dumps({'frame_times':times,'story':plan.model_dump(),'edl':edl}),paths).model_dump()

def process(job_id):
    started = time.monotonic()
    ai = AI()
    with session() as db:
        job = db.get(Job,job_id)
        project = db.get(Project,job.project_id)
        assets = db.query(Asset).filter_by(project_id=project.id).order_by(Asset.created_at).all()
    videos = [a for a in assets if a.kind=='video']
    music = next((a for a in reversed(assets) if a.kind=='music'),None)
    directory = settings.data_dir/'projects'/project.id/'work'
    directory.mkdir(parents=True,exist_ok=True)
    metrics = {'clips':len(videos),'footage_seconds':sum(a.duration for a in videos),'cost_usd':None,'captions':job.metrics.get('captions',{})}
    timings = {}
    stage_started = time.monotonic()
    update(job_id,'PREPROCESSING','Reading and segmenting all footage')
    cached_segments = docs('segments',project.id)
    segments = []
    for i,asset in enumerate(videos):
        existing = [d.payload for d in cached_segments if d.payload['asset_id']==asset.id]
        if existing:
            segments.extend(existing)
        else:
            produced = segment_video(asset,directory)
            # Save whole asset segmentation atomically, avoiding partial cache hits.
            from .db import DOCUMENTS
            with session() as db:
                for s in produced: db.add(DOCUMENTS['segments'](project_id=project.id,payload=s))
            segments.extend(produced)
        update(job_id,'PREPROCESSING',f'Prepared {i+1} of {len(videos)} clips')
    timings['preprocessing'] = time.monotonic()-stage_started
    stage_started = time.monotonic()
    update(job_id,'ANALYZING','Understanding your footage')
    analyses = {d.payload['segment_id']:d.payload for d in docs('analysis_results',project.id) if d.payload.get('model')==active_vision_model() and d.payload.get('provider','openai')==ai.provider}
    for i,s in enumerate(segments):
        cached = analyses.get(s['segment_id'])
        if cached: s['analysis'] = cached['analysis']
        else:
            s['analysis'] = ai.analyze(s)
            save_doc('analysis_results',project.id,{'segment_id':s['segment_id'],'analysis':s['analysis'],'model':active_vision_model(),'provider':ai.provider},job_id)
        update(job_id,'ANALYZING',f'Understood {i+1} of {len(segments)} moments')
    transcripts = []
    speech_model=settings.gemini_model if ai.provider=='gemini' else 'whisper-1'
    cached_transcripts = {d.payload['asset_id']:d.payload for d in docs('transcripts',project.id) if d.payload.get('provider','openai')==ai.provider and d.payload.get('model','whisper-1')==speech_model}
    for i,a in enumerate(videos):
        value = cached_transcripts.get(a.id)
        if value is None:
            value = transcribe(a,directory,ai)
            value.update({'provider':ai.provider,'model':speech_model})
            save_doc('transcripts',project.id,value,job_id)
        transcripts.append(value)
        update(job_id,'ANALYZING',f'Checked audio in {i+1} of {len(videos)} clips')
    music_data = audio_features(music.path,directory) if music else {'beats':[],'drops':[],'bpm':None}
    segments = rank_segments(segments)
    workout = understand(ai,segments)
    save_doc('workouts',project.id,workout,job_id)
    for event in workout['important_events']: save_doc('event_groups',project.id,event,job_id)
    timings['analysis'] = time.monotonic()-stage_started
    update(job_id,'PLANNING','Choosing highlights and planning your story')
    target = project.target_duration or min(60,max(30,sum(a.duration for a in videos)*.20))
    target = round(min(target,sum(a.duration for a in videos)) * 30)/30
    prior = [d.payload['story'] for d in docs('edit_plans',project.id) if d.job_id != job_id][-4:]
    stage_started = time.monotonic()
    plan = create_plan(ai,segments,workout,transcripts,target,project.style,music_data,prior)
    timings['planning'] = time.monotonic()-stage_started
    candidates = []
    for revision in range(2):
        plan_id = save_doc('edit_plans',project.id,plan.model_dump(),job_id)
        update(job_id,'EDITING','Tracking the subject and building the timeline')
        edl = build_edl(plan,segments,target,music_data['beats'])
        edl['music'] = music_data
        edl['captions'] = metrics['captions']
        edl_id = save_doc('edit_decisions',project.id,edl,job_id)
        outdir = settings.data_dir/'projects'/project.id/'renders'/job_id/str(revision+1)
        update(job_id,'RENDERING',f'Rendering version {revision+1}')
        stage_started = time.monotonic()
        output, metadata = render(edl,assets,outdir,transcripts,music,lambda n,total:update(job_id,'RENDERING',f'Rendered {n} of {total} shots · version {revision+1}'))
        timings[f'render_{revision+1}'] = time.monotonic()-stage_started
        update(job_id,'REVIEWING',f'Reviewing version {revision+1}')
        critic = review(ai,output,plan,edl,outdir)
        save_doc('critic_results',project.id,critic,job_id)
        publish_private(output)
        payload = {'edit_plan_id':plan_id,'edl_id':edl_id,'path':output,'duration':metadata['duration'],
                   'critic':critic,'revision':revision+1,'style':plan.style,'source_clips_used':len({s['asset_id'] for s in edl['timeline']}),
                   'total_clips':len(videos),'detected_moments':len(segments),'title':plan.title,'captions':metrics['captions']}
        version_id = save_doc('video_versions',project.id,payload,job_id)
        candidates.append((version_id,payload))
        if revision==0 and critic['revision_confidence'] >= .8 and critic['recommended_changes']:
            update(job_id,'PLANNING','Improving the edit after review')
            try:
                plan = create_plan(ai,segments,workout,transcripts,target,project.style,music_data,[],{'critic':critic,'previous_plan':plan.model_dump(),'previous_edl':edl})
            except Exception as exc:
                # A failed optional revision must not discard a valid, reviewed draft.
                metrics['revision_skipped']=type(exc).__name__
                break
        else: break
    best_id,best = max(candidates,key=lambda v:v[1]['critic']['overall'])
    best_edl=next(d.payload for d in docs('edit_decisions',project.id) if d.id==best['edl_id'])
    metrics.update({'segments':len(segments),'selected_segments':[s['segment_id'] for s in best_edl['timeline']],
                    'rejected_segments':[s['segment_id'] for s in segments if s['segment_id'] not in {x['segment_id'] for x in best_edl['timeline']}],
                    'critic_score':best['critic']['overall'],'revision_performed':len(candidates)>1,
                    'best_version_id':best_id,'timings_seconds':timings,'total_seconds':time.monotonic()-started,'ai_calls':ai.calls})
    with session() as db:
        job = db.get(Job,job_id)
        job.status,job.stage,job.detail,job.metrics = 'COMPLETED','COMPLETED','Your video is ready',metrics
        db.get(Project,project.id).status = 'COMPLETED'

def process_caption_job(job_id):
    """Rebuild the existing timeline from originals: never burn new text over old text."""
    from copy import deepcopy
    from .db import DOCUMENTS
    from .captions import checked_settings
    with session() as db:
        job=db.get(Job,job_id)
        source=db.get(DOCUMENTS['video_versions'],job.metrics['source_version_id'])
        if not source or source.project_id!=job.project_id: raise ValueError('Source version is unavailable')
        decision=db.get(DOCUMENTS['edit_decisions'],source.payload['edl_id'])
        if not decision or decision.project_id!=job.project_id: raise ValueError('Source timeline is unavailable')
        edl=deepcopy(decision.payload)
        assets=db.query(Asset).filter_by(project_id=job.project_id).order_by(Asset.created_at).all()
    captions=checked_settings(job.metrics['captions'],edl['duration']).model_dump()
    edl['captions']=captions
    transcripts=list({d.payload['asset_id']:d.payload for d in docs('transcripts',job.project_id)}.values())
    music=next((a for a in reversed(assets) if a.kind=='music'),None)
    update(job_id,'RENDERING','Styling subtitles on your existing edit')
    outdir=settings.data_dir/'projects'/job.project_id/'renders'/job_id/'1'
    output,metadata=render(edl,assets,outdir,transcripts,music,lambda n,total:update(job_id,'RENDERING',f'Prepared {n} of {total} shots for subtitles'))
    publish_private(output)
    edl_id=save_doc('edit_decisions',job.project_id,edl,job_id)
    payload={**source.payload,'path':output,'duration':metadata['duration'],'edl_id':edl_id,'captions':captions,'source_version_id':source.id,'subtitle_only':True}
    version_id=save_doc('video_versions',job.project_id,payload,job_id)
    with session() as db:
        current=db.get(Job,job_id)
        current.metrics={**job.metrics,'best_version_id':version_id,'ai_calls':[]}
        current.status,current.stage,current.detail='COMPLETED','COMPLETED','Your subtitled video is ready'
        db.get(Project,job.project_id).status='COMPLETED'
