"""Understanding: the multimodal model watches each whole clip once; local signals
then refine its coarse timestamps and score every candidate moment.

The model describes *what happens*; deterministic measurements decide *exactly
where* (cut points, dead time, framing, sharpness, duplicates).
"""
import json
from typing import Literal
import numpy as np
from pydantic import Field
from .schemas import Strict
from .perception import window, ANALYSIS_FPS

State = Literal['performing', 'setup', 'resting', 'walking', 'talking', 'reacting', 'spotting', 'camera_only', 'other']
RoleName = Literal['hook', 'establishing', 'build', 'intensity', 'climax', 'payoff', 'ending', 'transition', 'broll', 'humor']


class ObservedMoment(Strict):
    start: float = Field(ge=0, description='Seconds from clip start')
    end: float = Field(ge=0)
    activity: str = Field(description='What happens, e.g. "barbell back squat", "treadmill sprint", "walking to rack", "talking to camera"')
    exercise: str = Field(description='Exercise name if an exercise is visible, else "none"')
    state: State
    subject: str = Field(description='Short visual description of the main person (clothing), never a name or identity')
    shot_type: Literal['wide', 'medium', 'closeup', 'extreme_closeup']
    camera_angle: Literal['front', 'side', 'rear', 'three_quarter', 'low', 'high', 'unknown']
    camera_motion: Literal['static', 'handheld', 'pan', 'moving', 'chaotic']
    action_complete: bool = Field(description='True if the full movement or event is visible from start to finish')
    importance: float = Field(ge=0, le=1, description='How much this moment matters to a gym video')
    effort: float = Field(ge=0, le=1, description='Visible physical effort or intensity')
    expressive: float = Field(ge=0, le=1, description='Emotion, humour, celebration or personality on display')
    peak_time: float = Field(ge=0, description='The single strongest instant, e.g. lockout of the hardest rep')
    best_start: float = Field(ge=0, description='Start of the tightest worthwhile 1.5-5 s excerpt')
    best_end: float = Field(ge=0)
    speech: str = Field(description='Verbatim intelligible speech in this moment, or empty')
    speech_useful: bool = Field(description='True only if the speech adds meaning or humour to an edit')
    possible_roles: list[RoleName]
    problems: list[str] = Field(description='e.g. "camera pointed at floor", "subject cut off", "blurry", "obstructed", "dark"')
    notes: str


class ClipUnderstanding(Strict):
    summary: str
    location: str
    people: list[str]
    main_subject: str
    moments: list[ObservedMoment]


PROMPT = ('You are an experienced short-form fitness video editor logging raw gym footage before an edit. '
          'Watch the entire clip, including its audio. Split it into consecutive, non-overlapping moments that cover the whole clip; '
          'start a new moment whenever the state changes (setup, performing a set, resting, walking, talking, reacting). '
          'Timestamps are seconds from the clip start, precise to 0.5 s. Long idle stretches may be one moment. '
          'For every moment give the tightest worthwhile excerpt (best_start/best_end) and the strongest instant (peak_time). '
          'Be honest about weak footage: floor shots, people walking through frame, obstructions, blur and dead time. '
          'Humour, pranks, celebrations and reactions are valuable story material. Do not identify people. '
          'The clip duration is {duration:.2f} s; never exceed it.')


def understand_clip(ai, proxy, duration):
    from .gemini import generate
    from .config import settings
    if ai.provider != 'gemini':
        raise RuntimeError('Clip understanding requires Gemini video input')
    fps = 2 if duration <= 150 else 1
    result = generate(settings.gemini_api_key, settings.gemini_model, ClipUnderstanding, PROMPT.format(duration=duration),
                      video=proxy, video_fps=fps, media_resolution='MEDIA_RESOLUTION_LOW', calls=ai.calls, max_tokens=32768)
    return sanitize(result.model_dump(), duration)


def sanitize(clip, duration):
    """Clamp model timestamps into the real clip and make moments contiguous and ordered."""
    moments = []
    for m in sorted(clip['moments'], key=lambda m: m['start']):
        m['start'] = float(np.clip(m['start'], 0, duration))
        m['end'] = float(np.clip(max(m['end'], m['start']), 0, duration))
        if moments and m['start'] < moments[-1]['end']: m['start'] = moments[-1]['end']
        if m['end'] - m['start'] < .3: continue
        m['best_start'] = float(np.clip(m['best_start'], m['start'], m['end']))
        m['best_end'] = float(np.clip(m['best_end'], m['best_start'], m['end']))
        if m['best_end'] - m['best_start'] < .8:
            m['best_start'], m['best_end'] = m['start'], m['end']
        m['peak_time'] = float(np.clip(m['peak_time'], m['best_start'], m['best_end']))
        moments.append(m)
    clip['moments'] = moments
    return clip


def _norm(value, lo, hi):
    return float(np.clip((value - lo) / (hi - lo), 0, 1))


def local_metrics(signals, summary, start, end):
    """Measured quality and motion of a time range (all values 0-1)."""
    subj = window(signals, start, end, 'subj_mag')
    sharp = window(signals, start, end, 'sharpness')
    luma = window(signals, start, end, 'luma')
    hi = window(signals, start, end, 'clip_hi')
    lo = window(signals, start, end, 'clip_lo')
    present = window(signals, start, end, 'subject_present')
    area = window(signals, start, end, 'sarea')
    y1 = window(signals, start, end, 'sy1')
    x1, x2 = window(signals, start, end, 'sx1'), window(signals, start, end, 'sx2')
    camx, camy = window(signals, start, end, 'cam_dx'), window(signals, start, end, 'cam_dy')
    shake = float(np.hypot(np.diff(camx, prepend=camx[0]), np.diff(camy, prepend=camy[0])).mean())
    pan = float(np.hypot(camx.mean(), camy.mean()))
    exposure = 1 - min(1, abs(float(np.median(luma)) - .45) * 2.2) - min(.5, float(hi.mean()) * 3) - min(.5, float(lo.mean()) * 2)
    sharpness = _norm(float(np.median(sharp)), 60, 450)
    framing = 0.0
    if present.mean() > .2:
        size = _norm(float(np.median(area)), .01, .12)
        head_room = 1.0 if float(np.median(y1)) > .02 else .55
        centred = 1 - min(1, abs(float(np.median((x1 + x2) / 2)) - .5) * 2.2)
        framing = float(present.mean()) * (.45 * size + .3 * centred + .25 * head_room)
    return {'motion': _norm(float(np.percentile(subj, 80)), .003, .035),
            'motion_mean': float(subj.mean()),
            'visual_quality': float(np.clip(.55 * sharpness + .45 * max(0, exposure) - min(.4, shake * 120), 0, 1)),
            'sharpness': sharpness, 'exposure': float(max(0, exposure)), 'framing': float(np.clip(framing, 0, 1)),
            'subject_presence': float(present.mean()), 'shake': shake, 'camera_pan': pan,
            'brightness': float(np.median(luma))}


def refine_cut(signals, start, end, peak, bounds):
    """Snap excerpt boundaries to motion lulls so cuts land on natural action beats."""
    t = signals['t']
    energy = np.convolve(signals['subj_mag'], np.ones(3) / 3, mode='same')
    lo, hi = bounds

    def lull(target, radius, side):
        sel = (t >= max(lo, target - radius)) & (t <= min(hi, target + radius))
        if side == 'before': sel &= t <= peak
        else: sel &= t >= peak
        if not sel.any(): return target
        idx = np.where(sel)[0]
        return float(t[idx[np.argmin(energy[idx])]])

    s = lull(start, .6, 'before')
    e = lull(end, .6, 'after') + .1
    if e - s < 1.0: s, e = start, end
    return float(np.clip(s, lo, hi)), float(np.clip(e, lo, hi))


def dead_time(signals, start, end, speech_useful):
    """Fraction of a range with neither subject movement nor useful speech."""
    subj = window(signals, start, end, 'subj_mag')
    if speech_useful or len(subj) == 0: return 0.0
    return float((np.convolve(subj, np.ones(5) / 5, mode='same') < .0025).mean())


def moment_records(asset, clip, signals, summary):
    """Structured per-segment metadata (the spec's segment record) for one clip."""
    records = []
    reps = summary.get('reps', [])
    impacts = summary.get('impacts', [])
    for i, m in enumerate(clip['moments']):
        cut_start, cut_end = refine_cut(signals, m['best_start'], m['best_end'], m['peak_time'], (m['start'], m['end']))
        local = local_metrics(signals, summary, cut_start, cut_end)
        inside_reps = [r for r in reps if m['start'] <= r['t'] <= m['end']]
        inside_impacts = [x for x in impacts if cut_start - .2 <= x['t'] <= cut_end + .2]
        dead = dead_time(signals, m['start'], m['end'], m['speech_useful'])
        emb_t = signals['embed_t']
        sel = (emb_t >= m['start'] - 1) & (emb_t <= m['end'] + 1)
        emb = signals['embeddings'][sel].mean(axis=0) if sel.any() else signals['embeddings'][np.argmin(abs(emb_t - m['peak_time']))]
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        records.append({
            'clip_id': f'{asset.id[:6]}_{i:02d}', 'asset_id': asset.id, 'source': asset.filename,
            'start': round(m['start'], 3), 'end': round(m['end'], 3), 'cut_start': round(cut_start, 3), 'cut_end': round(cut_end, 3),
            'peak_time': round(m['peak_time'], 3), 'activity': m['activity'], 'exercise': m['exercise'], 'state': m['state'],
            'subject': m['subject'], 'shot_type': f"{m['shot_type']} {m['camera_angle']} angle", 'shot_size': m['shot_type'],
            'camera_angle': m['camera_angle'], 'camera_motion': m['camera_motion'],
            'motion_intensity': round(max(local['motion'], .7 * m['effort']) if m['state'] == 'performing' else local['motion'], 3),
            'visual_quality': round(local['visual_quality'], 3), 'framing': round(local['framing'], 3),
            'importance': round(m['importance'], 3), 'effort': round(m['effort'], 3), 'expressive': round(m['expressive'], 3),
            'action_complete': m['action_complete'], 'contains_dead_time': dead > .35, 'dead_fraction': round(dead, 3),
            'speech': m['speech'], 'speech_useful': m['speech_useful'], 'possible_role': m['possible_roles'],
            'problems': m['problems'], 'notes': m['notes'], 'reps': [round(r['t'], 2) for r in inside_reps],
            'impacts': [round(x['t'], 2) for x in inside_impacts], 'local': local,
            'source_fps': summary.get('fps', 30), 'embedding': emb.astype(float).round(5).tolist()})
    return records


def score_moments(records):
    """Multi-factor moment scores; uniqueness is relative to the whole collection."""
    if not records: return records
    E = np.array([r['embedding'] for r in records])
    sims = E @ E.T
    np.fill_diagonal(sims, -1)
    for i, r in enumerate(records):
        # Near-identical footage elsewhere (another take/angle of the same thing) lowers uniqueness.
        same_act = [j for j in range(len(records)) if j != i and records[j]['exercise'] == r['exercise'] and r['exercise'] != 'none']
        dup = max([sims[i, j] for j in same_act] + [float(sims[i].max()) * .92])
        uniqueness = float(np.clip(1 - (dup - .78) / .2, 0, 1))
        action = .6 * r['importance'] + .4 * r['effort']
        roles = set(r['possible_role'])
        story = min(1, .35 * len(roles & {'hook', 'climax', 'payoff', 'humor'}) + .2 * len(roles & {'build', 'intensity', 'establishing', 'ending'}))
        factors = {
            'action_importance': round(action, 3), 'image_quality': r['visual_quality'], 'motion': r['motion_intensity'],
            'composition': r['framing'], 'uniqueness': round(uniqueness, 3), 'emotional_value': r['expressive'],
            'completeness': 1.0 if r['action_complete'] else .45, 'story_usefulness': round(story, 3)}
        weights = {'action_importance': .27, 'image_quality': .13, 'motion': .1, 'composition': .1, 'uniqueness': .1,
                   'emotional_value': .12, 'completeness': .06, 'story_usefulness': .12}
        base = sum(factors[k] * w for k, w in weights.items())
        penalty = .05 * len(r['problems']) + .25 * r['dead_fraction'] + (.2 if r['state'] in ('camera_only', 'walking') and r['expressive'] < .5 else 0)
        if r['local']['subject_presence'] < .25 and r['state'] != 'camera_only': penalty += .1
        r['score_factors'] = factors
        r['score'] = round(float(np.clip(base - penalty, 0, 1)), 4)
    return records


def select_candidates(records, limit=36, diversity=.55):
    """Maximal-marginal-relevance selection: strong moments that are not near-copies."""
    pool = [r for r in records if r['cut_end'] - r['cut_start'] >= .8 and r['score'] > .18]
    pool.sort(key=lambda r: r['score'], reverse=True)
    chosen = []
    while pool and len(chosen) < limit:
        def mmr(r):
            if not chosen: return r['score']
            sim = max(float(np.dot(r['embedding'], c['embedding'])) + (.08 if c['asset_id'] == r['asset_id'] else 0) for c in chosen)
            return r['score'] - diversity * max(0, sim - .7)
        best = max(pool, key=mmr)
        best['selection_value'] = round(mmr(best), 4)
        chosen.append(best)
        pool.remove(best)
    return chosen


def public(record):
    """Model-facing view of a moment (no embeddings or dense data)."""
    keep = ('clip_id', 'source', 'cut_start', 'cut_end', 'peak_time', 'activity', 'exercise', 'state', 'subject', 'shot_type',
            'camera_motion', 'motion_intensity', 'visual_quality', 'importance', 'effort', 'expressive', 'action_complete',
            'contains_dead_time', 'speech', 'speech_useful', 'possible_role', 'problems', 'notes', 'score', 'reps', 'impacts')
    out = {k: record[k] for k in keep if k in record}
    out['duration'] = round(record['cut_end'] - record['cut_start'], 2)
    return out
