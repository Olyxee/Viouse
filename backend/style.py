"""Preference learner: a creator style profile from reference edits and feedback.

Reference videos are measured (shot lengths, transitions, motion, colour, cut/beat
relationship) and watched once by the multimodal model for structure and taste.
The resulting profile biases planning (pacing, effect frequency, text, look)
without copying any reference.
"""
import json
import time
from pathlib import Path
from typing import Literal
import numpy as np
import cv2
from pydantic import Field
from .schemas import Strict
from .db import session, CreatorProfile, docs
from .perception import decode, analysis_size

DEFAULT_SHOT = 1.7


class StyleObservation(Strict):
    content_type: str
    intro_structure: str
    ending_structure: str
    pacing: Literal['fast', 'medium', 'slow']
    uses_slow_motion: Literal['never', 'rarely', 'sometimes', 'often']
    uses_text: Literal['never', 'hook_only', 'frequent']
    text_style: str
    transition_style: str
    effects: list[str]
    music_relationship: str
    colour_style: str
    signature_traits: list[str] = Field(description='What makes this edit feel like this creator, in short phrases')


def measure_reference(path, meta):
    width, height = analysis_size(meta['width'], meta['height'])
    hist_prev, cuts, diffs, lumas, sats, warm, contrast = None, [], [], [], [], [], []
    for i, img in enumerate(decode(path, width, height, fps=10)):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
        h /= np.linalg.norm(h) + 1e-8
        d = float(1 - np.dot(h, hist_prev)) if hist_prev is not None else 0
        diffs.append(d)
        hist_prev = h
        if i % 5 == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lumas.append(gray.mean() / 255); contrast.append(gray.std() / 255)
            sats.append(hsv[..., 1].mean() / 255); warm.append(lab[..., 2].mean() - 128)
    d = np.array(diffs)
    if len(d) < 10: return None
    thr = max(.25, float(np.percentile(d, 97)))
    hard = [i / 10 for i in range(1, len(d)) if d[i] > thr and d[i] > 2.5 * np.median(d[max(0, i - 5):i + 5])]
    gradual = 0
    for i in range(3, len(d) - 3):
        if .08 < d[i - 2:i + 2].mean() < thr and d[i - 3:i + 3].min() > .05: gradual += 1
    duration = len(d) / 10
    bounds = [0] + hard + [duration]
    shots = np.diff(bounds)
    shots = shots[shots > .2]
    return {'duration': duration, 'shots': len(shots), 'mean_shot': float(shots.mean()) if len(shots) else duration,
            'median_shot': float(np.median(shots)) if len(shots) else duration, 'cut_times': hard,
            'gradual_transition_frames': gradual, 'luma': float(np.mean(lumas)), 'contrast': float(np.mean(contrast)),
            'saturation': float(np.mean(sats)), 'warmth': float(np.mean(warm))}


def cut_beat_alignment(path, cut_times, directory):
    from .music import load_mono, onset_envelope, SR, HOP
    from scipy.signal import find_peaks
    try: y = load_mono(path, directory)
    except Exception: return None
    env = onset_envelope(y)
    peaks, _ = find_peaks(env, height=.25, distance=8)
    onsets = peaks * HOP / SR
    if not len(onsets) or not cut_times: return None
    return float(np.mean([np.min(np.abs(onsets - c)) < .09 for c in cut_times]))


def analyse_reference(ai, asset, directory):
    from .perception import make_proxy
    from .gemini import generate
    from .config import settings
    measured = measure_reference(asset.path, asset.meta)
    if measured is None: return None
    measured['beat_alignment'] = cut_beat_alignment(asset.path, measured['cut_times'], directory) if asset.meta.get('has_audio') else None
    proxy = make_proxy(asset.path, Path(directory) / f'{asset.id}-style.mp4', fps=6)
    observed = generate(settings.gemini_api_key, settings.gemini_model, StyleObservation,
                        'This is a finished video the creator edited and likes. Describe its editing style precisely so a different '
                        'video can be edited in the same spirit (not copied). Focus on structure, pacing, effects, text, transitions, '
                        'colour and how cuts relate to the music.', video=proxy, video_fps=4, media_resolution='MEDIA_RESOLUTION_LOW',
                        calls=ai.calls).model_dump()
    return {'asset_id': asset.id, 'measured': {k: v for k, v in measured.items() if k != 'cut_times'}, 'observed': observed}


def build_profile(references, feedback):
    if not references and not feedback: return None
    m = [r['measured'] for r in references]
    o = [r['observed'] for r in references]
    profile = {'sources': len(references), 'updated_at': time.time()}
    if m:
        median_shot = float(np.median([x['median_shot'] for x in m]))
        profile['average_shot_duration'] = round(float(np.mean([x['mean_shot'] for x in m])), 3)
        profile['shot_length_scale'] = round(float(np.clip(median_shot / DEFAULT_SHOT, .6, 1.8)), 3)
        profile['typical_duration'] = round(float(np.median([x['duration'] for x in m])), 2)
        rate = float(np.mean([min(1, x['gradual_transition_frames'] / max(1, x['shots']) / 4) for x in m]))
        profile['transition_rate'] = round(float(np.clip(rate, .05, .35)), 3)
        profile['look'] = {'contrast': round(float(np.clip(1 + (np.mean([x['contrast'] for x in m]) - .22) * 1.5, .95, 1.18)), 3),
                           'saturation': round(float(np.clip(1 + (np.mean([x['saturation'] for x in m]) - .3) * 1.2, .85, 1.25)), 3)}
        aligned = [x['beat_alignment'] for x in m if x.get('beat_alignment') is not None]
        if aligned: profile['beat_sync'] = round(float(np.mean(aligned)), 3)
    if o:
        slow = {'never': 0, 'rarely': .1, 'sometimes': .25, 'often': .45}
        profile['slow_motion_rate'] = round(float(np.mean([slow[x['uses_slow_motion']] for x in o])), 3)
        profile['text_usage'] = max((x['uses_text'] for x in o), key=lambda v: ['never', 'hook_only', 'frequent'].index(v))
    profile['editorial'] = {
        'reference_styles': [{k: x[k] for k in ('intro_structure', 'ending_structure', 'pacing', 'transition_style', 'colour_style', 'signature_traits', 'uses_text')} for x in o],
        'average_shot_seconds': profile.get('average_shot_duration'),
        'creator_feedback': [f"{f['rating']}: {f['text']}" for f in feedback if f.get('text')][-8:],
        'instruction': 'Edit in the spirit of these preferences; never copy a reference shot-for-shot.'}
    return profile


def profile_for(ai, project, references, directory, progress):
    """Load the creator profile, folding in any new reference videos and feedback."""
    with session() as db:
        stored = db.query(CreatorProfile).filter_by(user_id=project.user_id).order_by(CreatorProfile.created_at.desc()).first()
        payload = dict(stored.payload) if stored else {}
    analysed = {r['asset_id']: r for r in payload.get('references', [])}
    for i, ref in enumerate(references):
        if ref.id in analysed: continue
        progress(f'Learning your style from reference {i + 1} of {len(references)}')
        result = analyse_reference(ai, ref, directory)
        if result: analysed[ref.id] = result
    feedback = [d.payload for d in docs('feedback', project.id)]
    profile = build_profile(list(analysed.values()), feedback)
    if profile is None: return None
    profile['references'] = list(analysed.values())
    with session() as db:
        db.add(CreatorProfile(user_id=project.user_id, payload=profile))
    return profile
