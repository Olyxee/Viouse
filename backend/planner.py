import json
import math
import numpy as np
from .schemas import Plan, Workout
from .preprocess import track_subject

def rank_segments(segments):
    for s in segments:
        a = s['analysis']
        similarities = []
        for other in segments:
            if other['segment_id'] == s['segment_id']: continue
            if other['analysis']['activity'].lower() != a['activity'].lower(): continue
            x,y = np.array(s['embedding']),np.array(other['embedding'])
            similarities.append(float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-8)))
        similarity = max(similarities,default=0)
        uniqueness = 1 - max(0,similarity-.75)/.25
        relevance = min(1,len(set(a['possible_role']) & {'hook','climax','progression','reaction'})/2)
        base = .35*a['interestingness'] + .25*a['visual_quality'] + .25*relevance + .15*uniqueness
        penalties = .04*len(a['problems']) + (.15 if a['visual_quality'] < .2 else 0)
        s['highlight_score'] = round(max(0,base-penalties),4)
        s['score_factors'] = {'uniqueness':uniqueness,'story_relevance':relevance,'penalties':penalties}
    return segments

def public_segments(segments):
    return [{k:v for k,v in s.items() if k not in ('frames','embedding','proxy','track')} for s in segments]

def understand(ai, segments):
    result = ai.structured(Workout, 'Summarize the content and its observed sequence for a creator reel. Infer the topic from the actual footage: lifestyle, travel, food, products, education, fitness, or other topics. Group related actions, subjects, or angles, but never claim synchronized cameras from similar appearance alone. Set synchronization confidence low unless there is actual temporal evidence. All IDs must come from the supplied segments. '+json.dumps(public_segments(segments)))
    valid = {s['segment_id'] for s in segments}
    for event in result.important_events:
        if not set(event.segment_ids) <= valid: raise ValueError('Workout referenced nonexistent footage')
    return result.model_dump()

def validate_plan(plan, segments, target):
    available = {s['segment_id']:s for s in segments}
    used = {}
    total = 0
    for shot in plan.story:
        s = available.get(shot.segment_id)
        if s is None: raise ValueError(f'Unknown segment: {shot.segment_id}')
        a,b = shot.source_start,shot.source_end
        if not all(math.isfinite(v) for v in (a,b)) or a < s['source_start']-.001 or b > s['source_end']+.001 or b-a < .4:
            raise ValueError(f'Invalid source range for {shot.segment_id}: {a}–{b}; allowed {s["source_start"]}–{s["source_end"]}')
        for x,y in used.get(s['asset_id'],[]):
            if min(b,y)-max(a,x) > .01: raise ValueError('Overlapping source footage repeats in this edit')
        used.setdefault(s['asset_id'],[]).append((a,b))
        total += b-a
    if abs(total-target) > .20: raise ValueError(f'Shot durations sum to {total:.2f}s; must equal {target:.2f}s')
    if abs(plan.target_duration-target) > .20: raise ValueError('Incorrect target duration')
    return plan

def signature(story):
    return [(s['segment_id'],round(s['source_start'],1),round(s['source_end'],1)) for s in story]

def create_plan(ai, segments, workout, transcripts, target, style, music, previous=(), feedback=None):
    prompt = ('Create a coherent creator reel from the real footage below. Adapt the story to its actual topic and available speech: strong hook, clear context, meaningful progression or demonstration, payoff, clean ending. Preserve explanations and speech meaning. Do not force a workout or physical-effort narrative. Quiet, static shots may be valuable. '
              'Do not invent events. Choose strongest single angle for each event; no presumed multi-camera synchronization. Use 1–3s shots for high-energy, 3–6s for cinematic. '
              'Source times are ABSOLUTE within original files. Only real IDs, ranges inside segment bounds, no overlapping/repeated source ranges. '
              f'The sum of all shot durations MUST equal {target:.3f} seconds (within 0.2s). Return target_duration={target:.3f}. '
              'Explain every selection, angle choice, and major rejected candidate. If previous edits supplied, create a materially different sequence or different selections, not just slightly changed cut times. '
              'Treat music beats as suggestions, favor climax near a drop. Do not select only by scores. '
              + json.dumps({'workout':workout,'segments':public_segments(segments),'transcripts':transcripts,'style':style,'music':music,'previous_edits':previous,'revision_feedback':feedback}))
    errors = []
    for _ in range(3):
        plan = ai.structured(Plan,prompt+'\nValidation errors to correct: '+json.dumps(errors))
        try:
            validate_plan(plan,segments,target)
            if style != 'automatic' and plan.style != style: raise ValueError('Keep the requested style')
            ids = [s.segment_id for s in plan.story]
            if previous and any(ids == [s['segment_id'] for s in p] for p in previous):
                raise ValueError('Alternative must change shot selections or ordering, not merely cut points')
            if feedback and 'previous_plan' in feedback and signature(plan.model_dump()['story'])==signature(feedback['previous_plan']['story']):
                raise ValueError('The proposed revision is unchanged; apply the actionable critic feedback')
            return plan
        except ValueError as exc: errors.append(str(exc))
    raise ValueError('Planner could not produce a valid edit: '+'; '.join(errors))

def build_edl(plan, segments, target, beats=()):
    available = {s['segment_id']:s for s in segments}
    shots = [s.model_dump() for s in plan.story]
    # Snap cuts to nearby measured onsets only when both neighboring source bounds permit it.
    cursor = 0.0
    for i in range(len(shots)-1):
        shot,nxt = shots[i:i+2]
        end = cursor+shot['source_end']-shot['source_start']
        nearby = min(beats,key=lambda b:abs(b-end),default=end)
        shift = nearby-end
        if abs(shift) <= (.18 if plan.style=='high-energy' else .09):
            if shot['source_end']+shift <= available[shot['segment_id']]['source_end'] and nxt['source_start']+shift >= available[nxt['segment_id']]['source_start'] and min(shot['source_end']+shift-shot['source_start'],nxt['source_end']-nxt['source_start']-shift) >= .4:
                shot['source_end'] += shift
                nxt['source_start'] += shift
        cursor += shot['source_end']-shot['source_start']
    # Resolve to exact 30fps boundaries without inventing footage or freezing frames.
    desired_frames = round(target*30)
    lengths = [round((s['source_end']-s['source_start'])*30) for s in shots]
    delta = desired_frames-sum(lengths)
    for i,s in enumerate(shots):
        if delta == 0: break
        capacity = int((available[s['segment_id']]['source_end']-s['source_start'])*30+1e-6)
        adjustment = min(delta, capacity-lengths[i]) if delta > 0 else max(delta,12-lengths[i])
        lengths[i] += adjustment
        delta -= adjustment
    if delta: raise ValueError('Insufficient selected footage to meet exact target duration')
    timeline, cursor = [], 0
    for shot,nframes in zip(shots,lengths):
        seg = available[shot['segment_id']]
        duration = nframes/30
        end = shot['source_start']+duration
        if end > seg['source_end']+.035: raise ValueError('Frame rounding exceeds source range')
        if 'track' not in seg: seg['track'] = track_subject(seg,seg['analysis']['subject_box'])
        timeline.append({**shot,'asset_id':seg['asset_id'],'source_end':end,'timeline_start':cursor/30,
                         'duration':duration,'frames':nframes,'speed':1.0,'crop':seg['track'],
                         'track_offset':shot['source_start']-seg['source_start'],
                         'brightness':seg['brightness'],'highlight_score':seg['highlight_score']})
        cursor += nframes
    return {'fps':30,'width':1080,'height':1920,'duration':cursor/30,'style':plan.style,'timeline':timeline}
