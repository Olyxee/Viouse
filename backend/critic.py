"""Critic: measures the finished render, then has the multimodal model watch it.

Problems become concrete timeline fixes (replace, trim, shorten, remove effect,
reframe...) that the planner applies before rendering again.
"""
import json
from pathlib import Path
from typing import Literal
import numpy as np
import cv2
from pydantic import Field
from .schemas import Strict
from .perception import decode, PersonSegmenter, subject_from_mask
from .engine import decode_audio
from .media import ffmpeg

Fix = Literal['replace_shot', 'remove_shot', 'trim_head', 'trim_tail', 'shorten', 'remove_effect', 'reframe_wider',
              'move_earlier', 'drop_hook_text', 'drop_captions', 'shorter_overall', 'none']
Category = Literal['awkward_cut', 'repeated_shot', 'dead_time', 'broken_transition', 'bad_crop', 'subject_out_of_frame',
                   'excessive_effect', 'inconsistent_color', 'weak_opening', 'pacing', 'too_long_section', 'audio_problem',
                   'text_problem', 'render_error', 'weak_shot', 'other']


class ReviewIssue(Strict):
    shot_index: int = Field(description='Index from the shot list, or -1 for the whole video')
    category: Category
    severity: float = Field(ge=0, le=1)
    fix: Fix
    seconds: float = Field(ge=0, description='Amount for trims/shortening, else 0')
    detail: str


class Review(Strict):
    human_editor_believability: float = Field(ge=0, le=1, description='Would a viewer believe a competent human editor made this on purpose?')
    hook_strength: float = Field(ge=0, le=1)
    pacing: float = Field(ge=0, le=1)
    story: float = Field(ge=0, le=1)
    visual_polish: float = Field(ge=0, le=1)
    audio: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)
    would_post: bool
    strengths: list[str]
    issues: list[ReviewIssue]


PROMPT = ('You are a demanding senior short-form editor reviewing a finished vertical gym video before it is posted. '
          'Watch and listen to the whole video. The shot list below gives each shot\'s index, timing, intended purpose and the '
          'edit operations applied. Judge it the way a creator would: Would someone reasonably believe a competent human editor '
          'intentionally made this? Check for awkward cuts, repeated or near-identical shots, dead time, broken or excessive '
          'transitions, bad crops (cut-off heads/bodies, subject leaving frame), effects that do not serve the action, '
          'inconsistent colour, a weak opening, pacing problems, sections that drag, audio problems (peaks, accidental silence, '
          'harsh sound effects), on-screen text problems and render glitches. Only report issues you actually observe; '
          'for each, choose the single most effective fix. Prefer replace_shot for weak/irrelevant shots, trim_head/trim_tail '
          'for bad starts/ends, shorten for shots that drag, remove_effect for unnecessary effects, reframe_wider for bad crops, '
          'drop_hook_text for a bad opening title, drop_captions for unreadable or distracting subtitles. '
          'Automated measurements are included; treat them as evidence. Shot list and measurements:\n')


def measure(output, timeline, perceived, directory):
    """Deterministic checks on the rendered file."""
    meta = {'width': 270, 'height': 480}
    seg = PersonSegmenter()
    rows = []
    prev = None
    prev_subject = None
    for i, img in enumerate(decode(output, 270, 480, fps=10)):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        subject = None
        if i % 2 == 0:
            found = subject_from_mask(seg(img), prev_subject)
            subject = found[0] if found else None
            prev_subject = subject[5:7] if subject else prev_subject
        rows.append({'t': i / 10, 'luma': float(gray.mean() / 255), 'diff': float(np.mean(cv2.absdiff(gray, prev)) / 255) if prev is not None else 1.0,
                     'a': float(lab[..., 1].mean() - 128), 'b': float(lab[..., 2].mean() - 128), 'subject': subject})
        prev = gray
    issues = []
    segments = timeline['segments']

    def seg_index(t):
        for k, s in enumerate(segments):
            if s['timeline_start'] <= t < s['timeline_end']: return k
        return len(segments) - 1

    # Black frames and freezes.
    black = [r['t'] for r in rows if r['luma'] < .035]
    if len(black) >= 2:
        issues.append({'shot_index': seg_index(black[0]), 'category': 'render_error', 'severity': .9, 'fix': 'replace_shot', 'seconds': 0,
                       'detail': f'{len(black)} black frames from {black[0]:.1f}s', 'source': 'measured'})
    run = 0
    for r in rows:
        k = seg_index(r['t'])
        frozen_ok = segments[k].get('freeze') and r['t'] > segments[k]['timeline_end'] - segments[k]['freeze']['hold'] - .1
        run = run + 1 if r['diff'] < .0008 and not frozen_ok else 0
        if run == 12:
            issues.append({'shot_index': k, 'category': 'dead_time', 'severity': .6, 'fix': 'trim_tail', 'seconds': 1.0,
                           'detail': f'picture frozen for over a second near {r["t"]:.1f}s', 'source': 'measured'})
    # Per-shot colour consistency and framing.
    per = []
    for k, s in enumerate(segments):
        rs = [r for r in rows if s['timeline_start'] + .1 <= r['t'] < s['timeline_end'] - .1] or [rows[min(len(rows) - 1, int(s['timeline_start'] * 10))]]
        luma = float(np.median([r['luma'] for r in rs]))
        cast = float(np.median([np.hypot(r['a'], r['b'] - 2.5) for r in rs]))
        subs = [r['subject'] for r in rs if r['subject'] is not None]
        sampled = max(1, len([r for r in rs if int(r['t'] * 10) % 2 == 0]))
        sig = perceived[s['asset_id']][0]
        inside = (sig['t'] >= s['source_start']) & (sig['t'] <= s['source_end'])
        source_presence = float(sig['subject_present'][inside].mean()) if inside.any() else 0.0
        head_cut = sum(1 for b in subs if b[1] < .005 and b[4] > .08) / max(1, len(subs))
        edge = sum(1 for b in subs if (b[0] < .003 or b[2] > .997) and b[4] > .05) / max(1, len(subs))
        presence = len(subs) / sampled
        per.append({'index': k, 'luma': round(luma, 3), 'cast': round(cast, 2), 'subject_presence': round(presence, 2),
                    'source_subject_presence': round(source_presence, 2), 'head_cut': round(head_cut, 2), 'edge_touch': round(edge, 2)})
        if source_presence > .6 and presence < .35:
            issues.append({'shot_index': k, 'category': 'subject_out_of_frame', 'severity': .55, 'fix': 'reframe_wider', 'seconds': 0,
                           'detail': 'subject visible in the source is mostly missing from the reframed shot', 'source': 'measured'})
        elif head_cut > .5 and s['reframe']['keys'][0]['z'] > 1.05:
            issues.append({'shot_index': k, 'category': 'bad_crop', 'severity': .5, 'fix': 'reframe_wider', 'seconds': 0,
                           'detail': 'the zoomed crop cuts into the top of the subject', 'source': 'measured'})
    lumas = np.array([p['luma'] for p in per])
    if len(lumas) > 2:
        mid = float(np.median(lumas))
        for p in per:
            if abs(p['luma'] - mid) > .14:
                issues.append({'shot_index': p['index'], 'category': 'inconsistent_color', 'severity': .35, 'fix': 'none', 'seconds': 0,
                               'detail': f"shot brightness {p['luma']:.2f} vs edit median {mid:.2f}", 'source': 'measured'})
    # Pacing and repetition.
    for k, s in enumerate(segments):
        limit = 6.5 if s['hero'] or s['audio']['keep_speech'] else 4.2
        if s['duration'] > limit:
            issues.append({'shot_index': k, 'category': 'too_long_section', 'severity': .45, 'fix': 'shorten', 'seconds': round(s['duration'] - limit + .3, 2),
                           'detail': f"{s['duration']:.1f}s shot is long for its purpose", 'source': 'measured'})
    emb = []
    for s in segments:
        sig = perceived[s['asset_id']][0]
        k = int(np.argmin(np.abs(sig['embed_t'] - (s['source_start'] + s['source_end']) / 2)))
        emb.append(sig['embeddings'][k])
    for i in range(len(emb)):
        for j in range(i + 1, len(emb)):
            sim = float(np.dot(emb[i], emb[j]))
            if sim > .95 and segments[j]['exercise'] == segments[i]['exercise']:
                issues.append({'shot_index': j, 'category': 'repeated_shot', 'severity': .55, 'fix': 'replace_shot', 'seconds': 0,
                               'detail': f'shot {j} looks nearly identical to shot {i}', 'source': 'measured'})
    effects = sum(1 for s in segments if s['speed_keys']) + sum(1 for s in segments if s['transition_in']['type'] in ('whip', 'zoom'))
    if effects > max(2, len(segments) * .4):
        issues.append({'shot_index': -1, 'category': 'excessive_effect', 'severity': .5, 'fix': 'none', 'seconds': 0,
                       'detail': f'{effects} speed/transition effects across {len(segments)} shots', 'source': 'measured'})
    # Audio: clipping and accidental silence.
    audio = decode_audio(output, 0, timeline['duration'] + .1)
    audio_stats = {}
    if len(audio):
        peak = float(20 * np.log10(np.abs(audio).max() + 1e-9))
        hop = 4800
        n = len(audio) // hop
        rms_db = 20 * np.log10(np.sqrt((audio[:n * hop].reshape(n, hop, 2) ** 2).mean(axis=(1, 2))) + 1e-9)
        silent = rms_db < -48
        longest, cur = 0, 0
        for v in silent[2:-2]:
            cur = cur + 1 if v else 0
            longest = max(longest, cur)
        audio_stats = {'peak_dbfs': round(peak, 2), 'longest_silence_s': round(longest * .1, 2), 'mean_db': round(float(rms_db.mean()), 1)}
        if peak > -.3:
            issues.append({'shot_index': -1, 'category': 'audio_problem', 'severity': .4, 'fix': 'none', 'seconds': 0, 'detail': f'audio peaks at {peak:.1f} dBFS', 'source': 'measured'})
        if longest >= 6:
            issues.append({'shot_index': -1, 'category': 'audio_problem', 'severity': .45, 'fix': 'none', 'seconds': 0,
                           'detail': f'{longest * .1:.1f}s of near-silence mid-video', 'source': 'measured'})
    if not 12 <= timeline['duration'] <= 62:
        issues.append({'shot_index': -1, 'category': 'pacing', 'severity': .6, 'fix': 'shorter_overall' if timeline['duration'] > 62 else 'none',
                       'seconds': 0, 'detail': f"duration {timeline['duration']:.1f}s outside 15-60s", 'source': 'measured'})
    return {'per_shot': per, 'audio': audio_stats, 'effects': effects, 'issues': issues}


def shot_list(timeline):
    out = []
    for i, s in enumerate(timeline['segments']):
        ops = []
        if s['speed_keys']: ops.append('speed ramp/slow motion')
        if s['transition_in']['type'] != 'cut': ops.append(f"{s['transition_in']['type']} transition in")
        zs = [k['z'] for k in s['reframe']['keys']]
        if max(zs) - min(zs) > .03: ops.append('animated zoom')
        elif zs[0] > 1.04: ops.append(f'reframed {zs[0]:.2f}x')
        if s.get('stabilize'): ops.append('stabilised')
        if s.get('freeze'): ops.append('freeze frame')
        out.append({'index': i, 'start': round(s['timeline_start'], 2), 'end': round(s['timeline_end'], 2), 'purpose': s['purpose'],
                    'intent': s['intent'], 'activity': s['exercise'], 'operations': ops})
    return out


def evaluate(ai, output, timeline, perceived, directory):
    measured = measure(output, timeline, perceived, directory)
    proxy = Path(directory) / 'review.mp4'
    ffmpeg('-i', output, '-vf', 'scale=540:960', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '30', '-c:a', 'aac', '-b:a', '96k', proxy)
    from .gemini import generate
    from .config import settings
    context = {'title': timeline['title'], 'duration': timeline['duration'], 'shots': shot_list(timeline),
               'overlays': timeline['overlays'], 'sound_effects': timeline['sfx'],
               'music': bool(timeline.get('music')), 'measurements': {k: v for k, v in measured.items() if k != 'issues'},
               'measured_issues': measured['issues']}
    review = generate(settings.gemini_api_key, settings.gemini_model, Review, PROMPT + json.dumps(context), video=proxy, video_fps=4,
                      calls=ai.calls, max_tokens=24576).model_dump()
    proxy.unlink(missing_ok=True)
    issues = [dict(i, source='watched') for i in review['issues']] + measured['issues']
    hook_end = max([o['end'] for o in timeline['overlays']] + [0])
    for i in issues:
        # Text complaints after the opening title are about subtitles, not the hook text.
        if i['fix'] == 'drop_hook_text' and 0 <= i['shot_index'] < len(timeline['segments']) and timeline['segments'][i['shot_index']]['timeline_start'] >= hook_end:
            i['fix'] = 'drop_captions'
            i['shot_index'] = -1
    n = len(timeline['segments'])
    technical_penalty = min(.3, sum(i['severity'] for i in measured['issues']) * .05)
    overall = round(.7 * review['overall'] + .3 * review['human_editor_believability'] - technical_penalty, 4)
    fixes, seen = [], set()
    for issue in sorted(issues, key=lambda i: i['severity'], reverse=True):
        if issue['fix'] == 'none' or issue['severity'] < .4: continue
        index = issue['shot_index'] if 0 <= issue['shot_index'] < n else None
        if index is None and issue['fix'] not in ('drop_hook_text', 'drop_captions', 'shorter_overall'): continue
        key = (index, issue['fix'])
        if key in seen or (index is not None and any(f['index'] == index and f['op'] in ('replace_shot', 'remove_shot') for f in fixes)): continue
        seen.add(key)
        fixes.append({'op': issue['fix'], 'index': index, 'seconds': issue['seconds'], 'detail': issue['detail'], 'source': issue['source']})
        if len(fixes) >= 6: break
    severe = [i for i in issues if i['severity'] >= .7 and i['fix'] != 'none']
    passed = overall >= .8 and review['would_post'] and not severe
    return {'overall': overall, 'passed': passed, 'review': review, 'measured': measured, 'issues': issues, 'fixes': fixes,
            'technical_penalty': round(technical_penalty, 4), 'problems': [i['detail'] for i in issues],
            'recommended_changes': [f"{f['op']} shot {f['index']}: {f['detail']}" for f in fixes]}
