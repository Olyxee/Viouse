"""Editorial reasoning: decide what the video should become before any cut is made."""
import json
from pathlib import Path
from typing import Literal
import cv2
import numpy as np
from pydantic import Field
from .schemas import Strict
from .understanding import public
from .music import summary as music_summary

Section = Literal['hook', 'establishing', 'build', 'intensity', 'climax', 'payoff', 'ending']
Treatment = Literal['none', 'slow_motion_peak', 'speed_ramp', 'punch_in', 'push_in', 'pull_out', 'keep_natural_audio',
                    'keep_speech', 'freeze_frame', 'impact_accent', 'match_cut_from_previous', 'motion_transition']


class PlannedShot(Strict):
    clip_id: str
    section: Section
    purpose: str = Field(description='What this shot does for the viewer at this point')
    treatment: list[Treatment] = Field(description='Only what this shot needs; usually ["none"]')
    reason: str = Field(description='Why this shot, here, with this treatment')


class Concept(Strict):
    video_type: str = Field(description='e.g. "leg day highlight", "gym day vlog", "gym prank / funny moments", "PR attempt"')
    title: str
    logline: str
    hook_text: str = Field(description='Optional 2-6 word on-screen hook for the opening, or empty if the visuals speak for themselves')
    structure: list[Section]
    energy_progression: str
    hero_clip_id: str
    pacing: Literal['fast', 'medium', 'slow_build']
    target_duration: float = Field(ge=12, le=60)
    editing_style: str
    transition_strategy: str
    music_strategy: str
    ending: str
    shots: list[PlannedShot]
    rejected: list[str] = Field(description='Notable moments left out and why (short)')


PROMPT = ('You are the lead editor of a fitness creator channel. Below are the strongest candidate moments found in a creator\'s '
          'raw gym footage (already logged and scored by your assistant), the clip summaries, and a contact sheet showing each '
          'candidate at its peak (labelled with its clip_id). Develop an editorial concept for ONE short-form vertical video '
          '(TikTok/Reels/Shorts) that a creator would genuinely post.\n'
          'Decide what kind of gym video this footage really supports, the opening hook (the first 1.5 s must grab attention: '
          'strongest action, a funny beat, or an intriguing moment, never setup or walking), how energy progresses, which moment '
          'is the hero/climax and where it lands (roughly 60-80% through), and how it ends (a payoff: reaction, final rep, '
          'celebration, or a clean button). Order shots for story and energy, NOT chronologically or by clip. '
          'Every shot must earn its place; avoid near-identical shots (same exercise, same angle) unless deliberately showing progression. '
          'Keep the edit tight: {lo:.0f}-{hi:.0f} seconds, typically 8-18 shots of 1.2-3.5 s each (the hero and meaningful speech may run longer). Treatments are optional and must serve the action: '
          'slow motion only on a genuinely strong peak (usually only the hero), punch-in for static shots that need energy, keep_speech only '
          'when the words matter, impact_accent when a weight lands. Most shots should be ["none"]. '
          'Humour and personality are welcome when the footage has them. Use only clip_ids from the list. '
          'Creator preferences (may be empty): {profile}. Music: {music}.')


def contact_sheet(candidates, videos, path, cols=6, tile=(150, 266)):
    by_id = {a.id: a for a in videos}
    rows = int(np.ceil(len(candidates) / cols))
    sheet = np.full((rows * (tile[1] + 22), cols * tile[0], 3), 18, np.uint8)
    for i, c in enumerate(candidates):
        asset = by_id[c['asset_id']]
        cap = cv2.VideoCapture(asset.path)
        cap.set(cv2.CAP_PROP_POS_MSEC, c['peak_time'] * 1000)
        ok, img = cap.read()
        cap.release()
        r, k = divmod(i, cols)
        x, y = k * tile[0], r * (tile[1] + 22)
        if ok:
            # OpenCV does not apply rotation metadata; match the probed orientation.
            rot = asset.meta.get('rotation', 0)
            if abs(rot) % 180 == 90 and img.shape[1] > img.shape[0]:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if rot < 0 else cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w = img.shape[:2]
            scale = min(tile[0] / w, tile[1] / h)
            small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            oy, ox = (tile[1] - small.shape[0]) // 2, (tile[0] - small.shape[1]) // 2
            sheet[y + 22 + oy:y + 22 + oy + small.shape[0], x + ox:x + ox + small.shape[1]] = small
        cv2.putText(sheet, c['clip_id'], (x + 4, y + 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def duration_range(candidates, profile):
    usable = sum(min(4, c['cut_end'] - c['cut_start']) for c in candidates)
    # Short-form rewards tight edits: aim well below the usable total.
    hi = float(np.clip(usable * .3, 18, 45))
    lo = float(np.clip(hi * .6, 15, 30))
    if profile and profile.get('typical_duration'):
        mid = float(np.clip(profile['typical_duration'], 15, 60))
        lo, hi = max(15, mid * .75), min(60, mid * 1.25)
    return lo, hi


def validate(concept, candidates):
    ids = {c['clip_id'] for c in candidates}
    seen = set()
    errors = []
    for s in concept.shots:
        if s.clip_id not in ids: errors.append(f'Unknown clip_id {s.clip_id}')
        if s.clip_id in seen: errors.append(f'{s.clip_id} used twice')
        seen.add(s.clip_id)
    if concept.hero_clip_id not in seen: errors.append('hero_clip_id must be one of the planned shots')
    if not 4 <= len(concept.shots) <= 26: errors.append('Plan between 4 and 26 shots')
    if concept.shots and concept.shots[0].section != 'hook': errors.append('The first shot must be the hook')
    if errors: raise ValueError('; '.join(errors))


def create_concept(ai, candidates, understood, videos, project, profile, music, feedback=None):
    directory = Path(videos[0].path).parent.parent / 'work'
    sheet = contact_sheet(candidates, videos, directory / 'candidates.jpg')
    lo, hi = duration_range(candidates, profile)
    if project.target_duration: lo, hi = project.target_duration * .9, project.target_duration * 1.1
    clips = [{'source': a.filename, 'summary': understood[a.id]['summary'], 'people': understood[a.id]['people']} for a in videos if a.id in understood]
    payload = {'candidates': [public(c) for c in candidates], 'clips': clips}
    if feedback: payload['revision_request'] = feedback
    requested = {'high-energy': 'The creator asked for fast pacing (pacing="fast").',
                 'cinematic': 'The creator asked for a cinematic, more spacious edit (pacing="slow_build" or "medium").'}.get(project.style, '')
    prompt = PROMPT.format(lo=lo, hi=hi, profile=json.dumps((profile or {}).get('editorial', {})),
                           music=json.dumps(music_summary(music)) if music else 'none supplied; natural gym audio and subtle sound design') + ' ' + requested + '\n' + json.dumps(payload)
    errors = []
    for _ in range(3):
        concept = ai.structured(Concept, prompt + ('\nFix these problems from your previous answer: ' + json.dumps(errors) if errors else ''), [sheet])
        try:
            validate(concept, candidates)
            result = concept.model_dump()
            result['target_duration'] = float(np.clip(result['target_duration'], lo, hi))
            return result
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError('Editorial concept failed validation: ' + '; '.join(errors))
