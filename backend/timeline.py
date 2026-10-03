"""Timeline planner: turns the editorial concept into precise, explainable operations.

The planner is deterministic. It never invents effects; every operation below is
justified by a measured signal (motion, framing, shake, light, beats) or by an
editorial intent, and records the reason. The execution engine only executes.
"""
import copy
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .perception import window, ANALYSIS_FPS

FPS = 30
WIDTH, HEIGHT = 1080, 1920
MAX_ZOOM = 1.25
ZOOM_CEILING = 1.38   # absolute limit including camera moves (576-px sources upscale 1.9× already)
SECTION_BASE = {'hook': 1.7, 'establishing': 1.8, 'build': 1.6, 'intensity': 1.35, 'climax': 3.0, 'payoff': 2.2, 'ending': 2.6}
PACING = {'fast': .9, 'medium': 1.15, 'slow_build': 1.3}
TARGET_LUMA = .46
TARGET_CAST_B = 2.5      # keep a touch of warmth so skin stays natural


# ── helpers ──────────────────────────────────────────────────────────────────────

def base_crop(meta):
    """Largest 9:16 window inside the (rotated) source, normalised to source size."""
    w, h = meta['width'], meta['height']
    if w / h >= 9 / 16: return (h * 9 / 16) / w, 1.0
    return 1.0, (w * 16 / 9) / h


def clamp_center(cx, cy, bw, bh, z):
    w, h = bw / z, bh / z
    return float(np.clip(cx, w / 2, 1 - w / 2)), float(np.clip(cy, h / 2, 1 - h / 2))


def output_duration(speed_keys, a, b):
    """Duration on the timeline of source range [a,b] under a piecewise-linear speed map."""
    s = np.linspace(a, b, max(2, int((b - a) * 600)))
    v = speed_at(speed_keys, s)
    return float(np.trapezoid(1 / v, s))


def speed_at(keys, s):
    if not keys: return np.ones_like(np.asarray(s, float))
    ts = [k['t'] for k in keys]
    vs = [k['speed'] for k in keys]
    return np.interp(s, ts, vs)


def motion_curve(signals, a, b):
    t = signals['t']
    sel = (t >= a - .05) & (t <= b + .05)
    return t[sel], gaussian_filter1d(signals['subj_mag'], 1.2)[sel]


# ── per-shot decisions ───────────────────────────────────────────────────────────

def choose_window(m, length, section, signals, hero):
    """Source in/out points. Cut-on-action for build shots, completion for hero/ending."""
    lo, hi = m['start'], m['end']
    peak = m['peak_time']
    if hi - lo <= length: return lo, hi, ['used the whole usable moment']
    reasons = []
    if hero or section in ('climax', 'payoff', 'ending'):
        # Let the strongest instant complete, with a beat of follow-through.
        end = min(hi, max(m['cut_end'], peak + .7))
        start = end - length
        reasons.append('ends just after the peak so the action completes')
    elif section == 'hook':
        start = max(lo, peak - length * .45)
        end = start + length
        reasons.append('opens straight into the action, no setup')
    else:
        start = peak - length * .58
        end = start + length
        reasons.append('cut on action: leaves mid-movement to carry energy into the next shot')
    if start < lo: start, end = lo, lo + length
    if end > hi: start, end = hi - length, hi
    # Never begin inside dead time when a livelier in-point exists nearby.
    t, mc = motion_curve(signals, start, min(hi, start + .6))
    if len(mc) and mc.max() < .003 and section not in ('ending', 'establishing') and m['cut_start'] > start:
        shift = min(m['cut_start'] - start, hi - end)
        if shift > .1:
            start, end = start + shift, end + shift
            reasons.append(f'trimmed {shift:.1f}s of dead time from the head')
    # Fine-tune the out-point onto motion for cut-on-action shots.
    if section in ('build', 'intensity', 'hook') and not hero:
        t, mc = motion_curve(signals, end - .25, min(hi, end + .25))
        if len(mc):
            best = float(t[np.argmax(mc)])
            if lo + .8 < best <= hi and abs(best - end) <= .25:
                start += best - end
                end = best
    return float(max(lo, start)), float(min(hi, end)), reasons


def speed_plan(m, start, end, section, treatments, hero, signals, fps):
    """Speed keys (source time → speed). Only around a genuine peak; recovery to normal."""
    keys, reasons = [], []
    peak = float(np.clip(m['peak_time'], start + .2, end - .2))
    t, mc = motion_curve(signals, peak - .5, peak + .5)
    strong = len(mc) > 0 and float(mc.max()) > .008
    wants_slow = 'slow_motion_peak' in treatments or (hero and 'none' not in treatments)
    if wants_slow and strong and end - start > 1.2:
        slow = .5 if fps >= 29 else .6
        a, b = max(start, peak - .55), min(end, peak + .45)
        keys = [{'t': start, 'speed': 1.0}, {'t': max(start, a - .2), 'speed': 1.0}, {'t': a, 'speed': slow},
                {'t': b, 'speed': slow}, {'t': min(end, b + .25), 'speed': 1.0}, {'t': end, 'speed': 1.0}]
        reasons.append(f'speed ramp: normal → {slow:.2f}× through the strongest instant → back to normal')
    elif wants_slow and not strong:
        reasons.append('slow motion requested but the peak has little movement, so it plays at normal speed')
    if 'speed_ramp' in treatments and not keys:
        t, mc = motion_curve(signals, start, end)
        if len(mc) > 6:
            onset_idx = np.argmax(mc > max(.006, mc.max() * .45))
            onset = float(t[onset_idx])
            if onset - start > .8:
                keys = [{'t': start, 'speed': 1.8}, {'t': onset - .15, 'speed': 1.8}, {'t': onset + .1, 'speed': 1.0}, {'t': end, 'speed': 1.0}]
                reasons.append('accelerated through the approach, normal speed once the movement starts')
    return [k for i, k in enumerate(keys) if i == 0 or k['t'] > keys[i - 1]['t'] + 1e-3], reasons


def reframe_plan(meta, signals, summary, start, end, section, treatments, hero, m):
    """Subject-aware 9:16 framing: static when the subject is still, smooth tracking when not."""
    bw, bh = base_crop(meta)
    t = signals['t']
    sel = (t >= start) & (t <= end)
    reasons = []
    if sel.sum() < 2:
        return {'mode': 'full', 'base': [bw, bh], 'keys': [{'t': start, 'cx': .5, 'cy': .5, 'z': 1.0}]}, reasons
    present = signals['subject_present'][sel].mean()
    x1, y1, x2, y2 = (signals[k][sel].astype(float) for k in ('sx1', 'sy1', 'sx2', 'sy2'))
    landscape = meta['width'] > meta['height']
    sub_w = np.median(x2 - x1)
    if landscape and present > .3 and sub_w > bw * .95:
        reasons.append('subject too wide for a vertical crop: full frame over a soft background')
        return {'mode': 'fit', 'base': [bw, bh], 'keys': [{'t': start, 'cx': .5, 'cy': .5, 'z': 1.0}]}, reasons
    cx = gaussian_filter1d((x1 + x2) / 2, 7, mode='nearest')
    top = gaussian_filter1d(y1, 7, mode='nearest')
    bottom = gaussian_filter1d(y2, 7, mode='nearest')
    z = 1.0
    # Only tighten a shot when the whole subject is inside the source frame with room to
    # spare; otherwise the camera operator's framing is kept (never crop into a body).
    contained = float(np.mean((y1 > .03) & (y2 < .97) & (x1 > .02) & (x2 < .98)))
    if present > .6 and contained > .8 and m['shot_size'] in ('wide', 'medium') and section != 'establishing':
        height = float(np.percentile(bottom - top, 80))
        if .05 < height < .6:
            z = float(np.clip(bh * .7 / height, 1.0, MAX_ZOOM))
            if z > 1.05: reasons.append(f'reframed {z:.2f}× on the subject so the action reads on a phone')
            else: z = 1.0
    if landscape and z == 1.0 and present > .3: reasons.append('tracks the subject inside the horizontal frame instead of a centre crop')
    keys = []
    times = t[sel]
    moving = float(np.ptp(cx)) > .06 or float(np.ptp(top)) > .08
    sample = range(0, len(times), 3) if moving else [0, len(times) - 1]
    for i in sample:
        w, h = bw / z, bh / z
        if present > .3:
            ccx = cx[i] if moving else float(np.median(cx))
            head = top[i] if moving else float(np.percentile(top, 20))
            foot = bottom[i] if moving else float(np.percentile(bottom, 80))
            ccy = (head + foot) / 2
            if head < ccy - h / 2 + .07 * h: ccy = head + h / 2 - .07 * h     # never cut off the head
        else:
            ccx, ccy = .5, .5
        ccx, ccy = clamp_center(ccx, ccy, bw, bh, z)
        keys.append({'t': float(times[i]), 'cx': round(ccx, 4), 'cy': round(ccy, 4), 'z': round(z, 4)})
    if moving and present > .3: reasons.append('smoothly follows the subject as they move')
    # Camera-style motion: only where it adds intent.
    if hero and ('push_in' in treatments or 'slow_motion_peak' in treatments):
        for k in keys: k['z'] = round(min(ZOOM_CEILING, k['z'] * (1 + .1 * (k['t'] - start) / max(.1, end - start))), 4)
        keys[-1]['t'] = end
        reasons.append('slow push-in builds weight on the hero moment')
    elif 'pull_out' in treatments or (section == 'ending' and 'freeze_frame' not in treatments and m['motion_intensity'] < .3):
        for k in keys: k['z'] = round(k['z'] * (1.08 - .08 * (k['t'] - start) / max(.1, end - start)), 4)
        keys[-1]['t'] = end
        reasons.append('gentle pull-out lets the ending breathe')
    elif 'punch_in' in treatments or 'push_in' in treatments or (m['camera_motion'] == 'static' and m['motion_intensity'] < .35 and end - start > 2.0):
        if 'punch_in' in treatments and end - start > 1.6:
            mid = start + (end - start) * .5
            step = []
            for k in keys:
                if k['t'] < mid: step.append(k)
            last = dict(keys[-1] if not step else step[-1])
            step.append({**last, 't': mid - 1 / FPS})
            step.append({**last, 't': mid, 'z': round(min(ZOOM_CEILING, last['z'] * 1.16), 4)})
            step += [{**k, 'z': round(min(ZOOM_CEILING, k['z'] * 1.16), 4)} for k in keys if k['t'] > mid]
            if step[-1]['t'] < end: step.append({**step[-1], 't': end})
            keys = step
            reasons.append('punch-in halfway through adds energy to a static shot')
        else:
            for k in keys: k['z'] = round(k['z'] * (1 + .06 * (k['t'] - start) / max(.1, end - start)), 4)
            keys[-1]['t'] = end
            reasons.append('subtle push-in keeps a static shot alive')
    for k in keys: k['cx'], k['cy'] = clamp_center(k['cx'], k['cy'], bw, bh, k['z'])
    if keys[0]['t'] > start: keys.insert(0, {**keys[0], 't': start})
    if keys[-1]['t'] < end: keys.append({**keys[-1], 't': end})
    return {'mode': 'track', 'base': [bw, bh], 'keys': keys}, reasons


def stabilization(signals, start, end, camera_motion, zoom):
    """Crop-based stabilisation offsets: follow the content's high-frequency shake only."""
    t = signals['t']
    sel = (t >= start - .5) & (t <= end + .5)
    if sel.sum() < 6: return None, []
    dx, dy = signals['cam_dx'][sel].astype(float), signals['cam_dy'][sel].astype(float)
    shake = float(np.hypot(np.diff(dx), np.diff(dy)).mean())
    if camera_motion not in ('handheld', 'moving', 'chaotic') and shake < .0018: return None, []
    cx, cy = np.cumsum(dx), np.cumsum(dy)
    ox, oy = cx - gaussian_filter1d(cx, ANALYSIS_FPS * .6, mode='nearest'), cy - gaussian_filter1d(cy, ANALYSIS_FPS * .6, mode='nearest')
    margin = (1 - 1 / zoom) / 2 if zoom > 1 else 0
    limit = max(.012, margin * .8)
    ox, oy = np.clip(ox, -limit, limit), np.clip(oy, -limit, limit)
    keys = [{'t': float(tt), 'dx': round(float(a), 5), 'dy': round(float(b), 5)} for tt, a, b in zip(t[sel], ox, oy)]
    return {'keys': keys, 'min_zoom': round(1 + 2.2 * limit, 4)}, ['stabilised handheld shake']


def colour_plan(signals, start, end, look):
    luma = float(np.median(window(signals, start, end, 'luma')))
    a = float(np.median(window(signals, start, end, 'cast_a')))
    b = float(np.median(window(signals, start, end, 'cast_b')))
    sat = float(np.median(window(signals, start, end, 'saturation')))
    gain = float(np.clip((TARGET_LUMA / max(.05, luma)) ** .65, .82, 1.4))
    reasons = []
    if abs(gain - 1) > .06: reasons.append(f'exposure {"lifted" if gain > 1 else "pulled down"} to match the other shots')
    corr_a, corr_b = -.65 * a, -.65 * (b - TARGET_CAST_B)
    if abs(corr_a) + abs(corr_b) > 2.5: reasons.append('white balance normalised')
    sat_gain = float(np.clip(look['saturation'] * (.32 / max(.08, sat)) ** .25, .85, 1.25))
    return {'exposure': round(gain, 4), 'shift_a': round(corr_a, 3), 'shift_b': round(corr_b, 3),
            'contrast': look['contrast'], 'saturation': round(sat_gain, 4), 'measured_luma': round(luma, 4)}, reasons


# ── transitions, music, sound ────────────────────────────────────────────────────

def boundary_motion(signals, a, b):
    t = signals['t']
    sel = (t >= a) & (t <= b)
    if not sel.any(): return 0.0, 0.0
    return float(signals['cam_dx'][sel].mean() * -1 + signals['subj_dx'][sel].mean()), float(np.abs(signals['subj_mag'][sel]).mean())


def choose_transitions(segments, perceived, profile):
    cap = max(1, round(len(segments) * (profile or {}).get('transition_rate', .15)))
    used = 0
    last_special = -3
    for i in range(1, len(segments)):
        prev, cur = segments[i - 1], segments[i]
        ps, _ = perceived[prev['asset_id']]
        cs, _ = perceived[cur['asset_id']]
        out_dx, out_mag = boundary_motion(ps, prev['source_end'] - .3, prev['source_end'])
        in_dx, in_mag = boundary_motion(cs, cur['source_start'], cur['source_start'] + .3)
        tr = {'type': 'cut', 'duration': 0.0}
        why = 'hard cut'
        if out_mag > .008 or in_mag > .008: why = 'cut on action'
        if used < cap and i - last_special >= 3:
            if cur['purpose'] == 'climax' and prev['purpose'] != 'climax':
                tr, why = {'type': 'zoom', 'duration': .2}, 'zoom transition punches into the hero moment'
            elif abs(out_dx) > .006 and abs(in_dx) > .004 and np.sign(out_dx) == np.sign(in_dx):
                tr = {'type': 'whip', 'duration': .2, 'direction': 'left' if out_dx < 0 else 'right'}
                why = 'both shots move the same way, so a whip carries the motion across the cut'
            elif 'motion_transition' in cur.get('treatment', []) and (out_mag > .01 or abs(out_dx) > .004):
                tr = {'type': 'whip', 'duration': .2, 'direction': 'left' if out_dx <= 0 else 'right'}
                why = 'motion transition rides the movement out of the previous shot'
        if tr['type'] != 'cut':
            used += 1
            last_special = i
        elif 'match_cut_from_previous' in cur.get('treatment', []) or (
                prev['exercise'] == cur['exercise'] and cur['exercise'] != 'none' and prev['asset_id'] != cur['asset_id']):
            tr, why = {'type': 'match_cut', 'duration': 0.0}, 'match cut: same movement from another angle, subject position aligned'
            align_subject(prev, cur)
        cur['transition_in'] = tr
        cur['reasons'].append(why)
    segments[0]['transition_in'] = {'type': 'cut', 'duration': 0.0}


def align_subject(prev, cur):
    """Match cut: start the incoming frame where the subject sat at the end of the outgoing one."""
    keys_prev, keys_cur = prev['reframe']['keys'], cur['reframe']['keys']
    if not keys_prev or not keys_cur: return
    dx = keys_prev[-1]['cx'] - keys_cur[0]['cx']
    bw, bh = cur['reframe']['base']
    for k in keys_cur:
        k['cx'], k['cy'] = clamp_center(k['cx'] + dx * .5, k['cy'], bw, bh, k['z'])


def place_music(segments, music_data, hero_index, duration):
    """Music offset so the hero peak lands on the strongest musically meaningful point."""
    hero = segments[hero_index]
    hero_peak = hero['timeline_start'] + hero.get('peak_offset', hero['duration'] * .5)
    targets = [d['t'] for d in music_data['drops'] if d['strength'] > .08] + music_data['downbeats']
    best, best_score = 0.0, -1e9
    for t in targets:
        offset = t - hero_peak
        if offset < 0 or offset + duration > music_data['duration'] - .5: continue
        energy = music_data['energy']
        e0 = energy[min(len(energy) - 1, int(offset * 2))]
        is_drop = any(abs(t - d['t']) < .01 for d in music_data['drops'])
        score = (2 if is_drop else 0) + e0 * .5 - offset / 600
        if score > best_score: best, best_score = offset, score
    return round(best, 3), ('hero moment lands on the music drop' if best_score >= 2 else 'hero moment lands on a downbeat')


def snap_to_beats(segments, music_data, offset, slack):
    """Pull some cuts onto beats (downbeats first). Never every beat; only when the source allows."""
    beats = np.array(music_data['beats'])
    downs = np.array(music_data['downbeats'])
    shift_total = 0.0
    snapped = 0
    for i in range(len(segments) - 1):
        seg = segments[i]
        cut = seg['timeline_start'] + shift_total + seg['duration']
        music_t = cut + offset
        cands = []
        if len(downs): cands.append((float(downs[np.argmin(abs(downs - music_t))]), .16))
        if len(beats): cands.append((float(beats[np.argmin(abs(beats - music_t))]), .08))
        for target, tol in cands:
            delta = target - music_t
            room = slack[i]
            if abs(delta) <= tol and room[0] <= delta <= room[1] and seg['duration'] + delta > .7 and seg.get('speed_keys') == []:
                seg['source_end'] += delta
                seg['duration'] += delta
                shift_total += delta
                seg['reasons'].append('out-point nudged onto the beat')
                snapped += 1
                break
        segments[i + 1]['timeline_start'] = segments[i]['timeline_start'] + segments[i]['duration']
    return snapped


def sound_design(segments, music_data):
    sfx = []
    for seg in segments:
        tr = seg['transition_in']
        if tr['type'] in ('whip', 'zoom'):
            sfx.append({'type': 'whoosh', 'at': round(seg['timeline_start'] - .12, 3), 'gain': .45, 'reason': f"accent for the {tr['type']} transition"})
    hero = next((s for s in segments if s.get('hero')), None)
    if hero:
        peak_t = hero['timeline_start'] + hero.get('peak_offset', 0)
        if not music_data:
            sfx.append({'type': 'riser', 'at': round(max(0, hero['timeline_start'] - 1.2), 3), 'gain': .3, 'reason': 'builds tension into the hero moment (no music supplied)'})
        if not hero.get('natural_impact'):
            sfx.append({'type': 'impact', 'at': round(peak_t, 3), 'gain': .5 if music_data else .6, 'reason': 'weight on the hero peak'})
    for seg in segments:
        if 'impact_accent' in seg.get('treatment', []) and not seg.get('hero') and not seg.get('natural_impact'):
            sfx.append({'type': 'hit', 'at': round(seg['timeline_start'] + seg.get('peak_offset', 0), 3), 'gain': .35, 'reason': 'subtle accent where the weight lands'})
    sfx.sort(key=lambda s: s['at'])
    # Restraint: at most one effect sound every ~4 seconds.
    kept = []
    for s in sfx:
        if s['at'] < 0: continue
        if kept and s['at'] - kept[-1]['at'] < 1.5 and s['type'] != 'impact': continue
        kept.append(s)
    return kept


# ── main entry points ────────────────────────────────────────────────────────────

def plan_timeline(concept, moments, perceived, music_asset, music_data, profile, captions, overrides=None):
    overrides = overrides or {}
    by_id = {m['clip_id']: m for m in moments}
    shots = overrides.get('shots') or concept['shots']
    look = {'contrast': 1.07, 'saturation': 1.05, **((profile or {}).get('look') or {})}
    pace = PACING.get(concept.get('pacing'), 1.0) * (profile or {}).get('shot_length_scale', 1.0)
    target = float(overrides.get('target_duration', concept['target_duration']))
    hero_id = overrides.get('hero_clip_id', concept['hero_clip_id'])
    # Desired lengths, scaled so the whole edit approaches the target duration.
    base = []
    for s in shots:
        m = by_id[s['clip_id']]
        want = SECTION_BASE.get(s['section'], 1.6) * pace
        if s['clip_id'] == hero_id: want = max(want, 3.0)
        if 'keep_speech' in s['treatment'] and m['speech']: want = max(want, min(6.5, .4 * len(m['speech'].split()) + 1))
        base.append(want)
    avail = [by_id[s['clip_id']]['end'] - by_id[s['clip_id']]['start'] for s in shots]
    scale = 1.0
    for _ in range(30):
        total = sum(min(a, b * scale) for a, b in zip(avail, base))
        if abs(total - target) < .05: break
        scale *= target / max(total, .1)
        scale = float(np.clip(scale, .45, 2.2))
    lengths = [max(.7, min(a, b * scale)) for a, b in zip(avail, base)]
    lengths = [overrides.get('lengths', {}).get(str(i), L) for i, L in enumerate(lengths)]
    segments = []
    cursor = 0.0
    for i, s in enumerate(shots):
        m = by_id[s['clip_id']]
        asset_id = m['asset_id']
        signals, summary = perceived[asset_id]
        hero = s['clip_id'] == hero_id
        treatments = [t for t in s['treatment'] if t not in overrides.get('removed_effects', {}).get(str(i), [])]
        start, end, reasons = choose_window(m, lengths[i], s['section'], signals, hero)
        trim = overrides.get('trims', {}).get(str(i))
        if trim:
            start = float(np.clip(start + trim.get('head', 0), m['start'], end - .7))
            end = float(np.clip(end - trim.get('tail', 0), start + .7, m['end']))
            reasons.append('re-trimmed after review')
        speed_keys, why = speed_plan(m, start, end, s['section'], treatments, hero, signals, m['source_fps'])
        reasons += why
        if speed_keys:
            # Slow motion stretches time; give the time back by tightening the source range.
            for _ in range(6):
                out = output_duration(speed_keys, start, end)
                excess = out - lengths[i] * 1.25
                if excess < .05: break
                start = min(start + excess * .4, m['peak_time'] - .8)
                end = max(end - excess * .3, m['peak_time'] + .5)
                speed_keys[0]['t'] = start
                speed_keys[-1]['t'] = end
                speed_keys = [k for k in speed_keys if start <= k['t'] <= end]
                if speed_keys[0]['t'] > start: speed_keys.insert(0, {'t': start, 'speed': speed_keys[0]['speed']})
        meta = {'width': summary['width'], 'height': summary['height']}
        reframe, why = reframe_plan(meta, signals, summary, start, end, s['section'], treatments, hero, m)
        if overrides.get('fit', {}).get(str(i)): reframe['keys'] = [{**k, 'z': 1.0} for k in reframe['keys']]; why = ['reframed wider after review']
        reasons += why
        stab, why = stabilization(signals, start, end, m['camera_motion'], max(k['z'] for k in reframe['keys']))
        if stab:
            for k in reframe['keys']: k['z'] = round(max(k['z'], stab['min_zoom']), 4)
            reasons += why
        color, why = colour_plan(signals, start, end, look)
        reasons += why
        duration = output_duration(speed_keys, start, end) if speed_keys else end - start
        peak_offset = (output_duration(speed_keys, start, min(end, max(start + .01, m['peak_time']))) if speed_keys
                       else float(np.clip(m['peak_time'], start, end) - start))
        freeze = None
        if 'freeze_frame' in treatments and s['section'] == 'ending':
            freeze = {'hold': .6}
            duration += .6
            reasons.append('holds the final frame as a button')
        speech = 'keep_speech' in treatments and bool(m['speech'])
        natural = 1.0 if speech else .8 if ('keep_natural_audio' in treatments or m['impacts']) else .55
        segments.append({
            'index': i, 'clip_id': s['clip_id'], 'asset_id': asset_id, 'source': m['source'], 'exercise': m['exercise'],
            'source_start': round(start, 4), 'source_end': round(end, 4), 'timeline_start': round(cursor, 4),
            'duration': round(duration, 4), 'purpose': s['section'], 'intent': s['purpose'], 'treatment': treatments,
            'hero': hero, 'peak_offset': round(peak_offset, 4), 'speed_keys': speed_keys, 'reframe': reframe,
            'stabilize': stab, 'color': color, 'freeze': freeze, 'natural_impact': bool(m['impacts']),
            'audio': {'natural_gain': natural, 'keep_speech': speech, 'speech': m['speech'] if speech else ''},
            'reasons': [s['reason']] + reasons})
        cursor += duration
    choose_transitions(segments, perceived, profile)
    decisions = []
    music = None
    if music_data and music_asset:
        slack = []
        for seg in segments:
            m = by_id[seg['clip_id']]
            slack.append((max(-.2, -(seg['source_end'] - seg['source_start'] - .8)), max(0.0, min(.2, m['end'] - seg['source_end']))))
        hero_index = next((i for i, s in enumerate(segments) if s['hero']), len(segments) // 2)
        offset, why = place_music(segments, music_data, hero_index, cursor)
        snapped = snap_to_beats(segments, music_data, offset, slack)
        cursor = segments[-1]['timeline_start'] + segments[-1]['duration']
        music = {'asset_id': music_asset.id, 'offset': offset, 'gain': .9, 'duck_under_speech': .28,
                 'fade_in': .05 if offset > 0 else 0, 'fade_out': 1.2}
        decisions += [why, f'{snapped} of {len(segments) - 1} cuts aligned to beats; the rest follow the action']
    for i, seg in enumerate(segments):
        seg['timeline_start'] = round(sum(s['duration'] for s in segments[:i]), 4)
        seg['timeline_end'] = round(seg['timeline_start'] + seg['duration'], 4)
    total = segments[-1]['timeline_end']
    frames = int(round(total * FPS))
    overlays = []
    hook_text = overrides.get('hook_text', concept.get('hook_text', '')).strip()
    if hook_text and len(hook_text.split()) <= 7:
        overlays.append({'type': 'text', 'text': hook_text, 'start': .1, 'end': round(min(2.8, max(1.6, segments[0]['duration'] + .4)), 3),
                         'position': 'upper_third', 'style': 'hook', 'reason': 'states the hook while the opening shot plays'})
    speech_events = [{'start': s['timeline_start'] + .05, 'end': s['timeline_end'] - .05, 'text': s['audio']['speech']}
                     for s in segments if s['audio']['keep_speech'] and s['audio']['speech']]
    return {
        'version': 2, 'fps': FPS, 'width': WIDTH, 'height': HEIGHT, 'aspect_ratio': '9:16',
        'duration_target': round(target, 3), 'duration': round(frames / FPS, 4), 'frames': frames,
        'title': concept['title'], 'video_type': concept['video_type'], 'look': look,
        'segments': segments, 'overlays': overlays, 'captions': captions, 'speech_events': speech_events,
        'music': music, 'sfx': sound_design(segments, music_data), 'decisions': decisions,
        'plan_state': {'shots': copy.deepcopy(shots), 'overrides': overrides, 'hero_clip_id': hero_id}}


def revise(timeline, review, concept, moments, perceived, music_asset, music_data, profile, captions):
    """Apply the critic's concrete fixes to the plan, then re-plan deterministically."""
    state = copy.deepcopy(timeline['plan_state'])
    overrides = state['overrides']
    shots = state['shots']
    used = {s['clip_id'] for s in shots}
    by_id = {m['clip_id']: m for m in moments}
    ranked = sorted(moments, key=lambda m: m['score'], reverse=True)
    removed = set()
    for fix in review.get('fixes', []):
        i = fix.get('index')
        op = fix['op']
        if i is not None and not 0 <= i < len(shots): continue
        key = str(i)
        if op == 'replace_shot':
            old = by_id[shots[i]['clip_id']]
            def fit(m):
                same = m['exercise'] == old['exercise'] and m['exercise'] != 'none'
                return m['score'] + (.08 if same else 0) + (.1 if shots[i]['section'] in m['possible_role'] else 0)
            pool = [m for m in ranked if m['clip_id'] not in used and m['cut_end'] - m['cut_start'] >= 1 and not m['contains_dead_time']
                    and max(float(np.dot(m['embedding'], by_id[s['clip_id']]['embedding'])) for s in shots) < .93]
            if pool:
                new = max(pool, key=fit)
                used.add(new['clip_id'])
                shots[i] = {**shots[i], 'clip_id': new['clip_id'], 'treatment': ['none'],
                            'reason': f"replaced after review: {fix.get('detail', '')}"[:200]}
                if timeline['plan_state']['hero_clip_id'] == old['clip_id']: overrides['hero_clip_id'] = new['clip_id']
        elif op == 'remove_shot' and len(shots) - len(removed) > 5 and shots[i]['clip_id'] != state['hero_clip_id']:
            removed.add(i)
        elif op in ('trim_head', 'trim_tail'):
            t = overrides.setdefault('trims', {}).setdefault(key, {'head': 0, 'tail': 0})
            t['head' if op == 'trim_head' else 'tail'] += float(np.clip(fix.get('seconds', .5), .2, 2))
        elif op == 'shorten':
            seg = timeline['segments'][i]
            overrides.setdefault('lengths', {})[key] = max(.8, seg['source_end'] - seg['source_start'] - float(np.clip(fix.get('seconds', .6), .2, 3)))
        elif op == 'remove_effect':
            overrides.setdefault('removed_effects', {})[key] = [t for t in shots[i]['treatment'] if t != 'none'] + ['slow_motion_peak', 'punch_in', 'push_in', 'speed_ramp']
        elif op == 'reframe_wider':
            overrides.setdefault('fit', {})[key] = True
        elif op == 'move_earlier' and i > 1:
            shots.insert(i - 1, shots.pop(i))
        elif op == 'drop_hook_text':
            overrides['hook_text'] = ''
        elif op == 'shorter_overall':
            overrides['target_duration'] = max(15, timeline['duration'] * .85)
    if removed:
        shots = [s for j, s in enumerate(shots) if j not in removed]
        for name in ('trims', 'lengths', 'removed_effects', 'fit'):
            overrides.pop(name, None)
    if shots[0]['section'] != 'hook': shots[0] = {**shots[0], 'section': 'hook'}
    overrides['shots'] = shots
    return plan_timeline(concept, moments, perceived, music_asset, music_data, profile, captions, overrides)
