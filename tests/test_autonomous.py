import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.io import wavfile

TEST_DIR = Path(os.environ['DATA_DIR'])

from backend.media import ffmpeg, probe
from backend.perception import perceive
from backend.timeline import plan_timeline, revise, output_duration
from backend.engine import render_timeline, time_map
from backend.music import analyze_music
from backend.understanding import sanitize, as_seconds, score_moments, select_candidates


@pytest.fixture(scope='module')
def clip():
    path = TEST_DIR / 'moving.mp4'
    # A moving pattern with a tone: deterministic motion for the perception pass.
    ffmpeg('-f', 'lavfi', '-i', 'testsrc2=size=360x640:rate=30', '-f', 'lavfi', '-i', 'sine=frequency=220:sample_rate=48000',
           '-t', '8', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', path)
    meta = probe(path)
    asset = SimpleNamespace(id='clip', path=str(path), meta=meta, duration=meta['duration'], filename='moving.mp4', kind='video')
    signals, summary = perceive(asset, TEST_DIR / 'work')
    return asset, signals, summary


def moment(cid, start, end, peak, **extra):
    base = {'clip_id': cid, 'asset_id': 'clip', 'source': 'moving.mp4', 'start': start, 'end': end, 'cut_start': start, 'cut_end': end,
            'peak_time': peak, 'shot_size': 'medium', 'camera_motion': 'static', 'motion_intensity': .6, 'source_fps': 30,
            'impacts': [], 'speech': '', 'exercise': 'squat', 'possible_role': ['build'], 'score': .6,
            'embedding': list(np.eye(4)[len(cid) % 4]) , 'contains_dead_time': False}
    return {**base, **extra}


def concept(shots, hero):
    return {'title': 'Test', 'video_type': 'test', 'hook_text': 'Leg day', 'pacing': 'fast', 'target_duration': 6,
            'hero_clip_id': hero, 'shots': shots}


def shot(cid, section, treatment=('none',)):
    return {'clip_id': cid, 'section': section, 'purpose': section, 'treatment': list(treatment), 'reason': 'test'}


def test_perception_measures_every_frame(clip):
    asset, signals, summary = clip
    assert abs(len(signals['t']) - 80) <= 2
    assert summary['frames'] == len(signals['t'])
    assert signals['embeddings'].shape[0] >= 3
    assert np.isfinite(signals['subj_mag']).all()


def test_timeline_is_explainable_and_ramps_only_the_hero(clip):
    asset, signals, summary = clip
    moments = [moment('a', 0, 2.5, 1.2), moment('bb', 2.5, 5.5, 4.0), moment('ccc', 5.5, 8, 6.8)]
    c = concept([shot('a', 'hook'), shot('bb', 'build'), shot('ccc', 'climax', ['slow_motion_peak'])], 'ccc')
    tl = plan_timeline(c, moments, {'clip': (signals, summary)}, None, None, None, {'mode': 'off'})
    segs = tl['segments']
    assert [s['purpose'] for s in segs] == ['hook', 'build', 'climax']
    assert all(s['reasons'] for s in segs)
    assert segs[0]['speed_keys'] == [] and segs[1]['speed_keys'] == []
    for s in segs:
        assert s['source_start'] >= 0 and s['source_end'] <= 8
        expected = output_duration(s['speed_keys'], s['source_start'], s['source_end']) if s['speed_keys'] else s['source_end'] - s['source_start']
        assert abs(s['duration'] - expected) < .02
    assert abs(tl['duration'] - sum(s['duration'] for s in segs)) < .05
    assert tl['overlays'][0]['text'] == 'Leg day'


def test_time_map_slows_and_recovers():
    seg = {'source_start': 0, 'source_end': 2, 'duration': output_duration([{'t': 0, 'speed': 1}, {'t': .5, 'speed': 1}, {'t': .7, 'speed': .5},
                                                                             {'t': 1.3, 'speed': .5}, {'t': 1.5, 'speed': 1}, {'t': 2, 'speed': 1}], 0, 2),
           'speed_keys': [{'t': 0, 'speed': 1}, {'t': .5, 'speed': 1}, {'t': .7, 'speed': .5}, {'t': 1.3, 'speed': .5}, {'t': 1.5, 'speed': 1}, {'t': 2, 'speed': 1}]}
    src, speeds = time_map(seg, 30)
    assert np.all(np.diff(src) > 0)
    assert speeds.min() == pytest.approx(.5, abs=.02)
    assert src[-1] <= 2


def test_render_executes_timeline_with_transitions_and_audio(clip):
    asset, signals, summary = clip
    moments = [moment('a', 0, 2.5, 1.2), moment('bb', 2.5, 5.5, 4.0), moment('ccc', 5.5, 8, 6.8)]
    c = concept([shot('a', 'hook'), shot('bb', 'build'), shot('ccc', 'climax', ['slow_motion_peak'])], 'ccc')
    tl = plan_timeline(c, moments, {'clip': (signals, summary)}, None, None, None, {'mode': 'off'})
    tl['width'], tl['height'] = 270, 480
    tl['segments'][1]['transition_in'] = {'type': 'whip', 'duration': .2, 'direction': 'left'}
    out, meta = render_timeline(tl, [asset], TEST_DIR / 'render')
    assert (meta['width'], meta['height']) == (270, 480)
    assert meta['has_audio'] and abs(meta['duration'] - tl['duration']) < .2


def test_revision_replaces_flagged_shot(clip):
    asset, signals, summary = clip
    moments = [moment('a', 0, 2, 1), moment('bb', 2, 4, 3), moment('ccc', 4, 6, 5), moment('dddd', 6, 8, 7, score=.9)]
    c = concept([shot('a', 'hook'), shot('bb', 'build'), shot('ccc', 'climax')], 'ccc')
    tl = plan_timeline(c, moments, {'clip': (signals, summary)}, None, None, None, {'mode': 'off'})
    revised = revise(tl, {'fixes': [{'op': 'replace_shot', 'index': 1, 'detail': 'weak'}]}, c, moments, {'clip': (signals, summary)}, None, None, None, {'mode': 'off'})
    assert revised['segments'][1]['clip_id'] == 'dddd'
    assert revised['segments'][0]['clip_id'] == 'a'


def test_music_analysis_finds_tempo():
    sr = 22050
    t = np.arange(sr * 12) / sr
    y = np.zeros_like(t)
    for beat in np.arange(0, 12, .5):          # 120 BPM clicks
        i = int(beat * sr)
        y[i:i + 600] += np.sin(2 * np.pi * 1000 * t[:600]) * np.exp(-np.arange(600) / 120)
    path = TEST_DIR / 'clicks.wav'
    wavfile.write(path, sr, (y * 20000).astype(np.int16))
    music = analyze_music(path, TEST_DIR)
    assert 115 <= music['bpm'] <= 125 or 58 <= music['bpm'] <= 62 or 235 <= music['bpm'] <= 245
    gaps = np.diff(music['beats'])
    assert abs(np.median(gaps) - .5) < .05


def test_model_timestamps_are_repaired_and_scored():
    clip = {'summary': '', 'location': '', 'people': [], 'main_subject': '', 'moments': [
        {'start': 0, 'end': .31, 'best_start': 0, 'best_end': .2, 'peak_time': .15}]}
    fixed = sanitize(as_seconds(clip, 36.6), 36.6)
    assert fixed['moments'][0]['end'] == pytest.approx(31)
    records = [moment('a', 0, 3, 1, importance=.9, effort=.8, visual_quality=.8, framing=.7, expressive=.2, action_complete=True,
                      dead_fraction=0, problems=[], state='performing', local={'subject_presence': 1}, embedding=[1, 0]),
               moment('b', 0, 3, 1, importance=.9, effort=.8, visual_quality=.8, framing=.7, expressive=.2, action_complete=True,
                      dead_fraction=0, problems=[], state='performing', local={'subject_presence': 1}, embedding=[1, 0]),
               moment('c', 0, 3, 1, importance=.5, effort=.4, visual_quality=.8, framing=.7, expressive=.2, action_complete=True,
                      dead_fraction=0, problems=[], state='performing', local={'subject_presence': 1}, embedding=[0, 1], exercise='bench')]
    scored = score_moments(records)
    assert scored[0]['score_factors']['uniqueness'] < scored[2]['score_factors']['uniqueness']
    chosen = select_candidates(scored, limit=2)
    assert {c['clip_id'] for c in chosen} == {'a', 'c'} or {c['clip_id'] for c in chosen} == {'b', 'c'}
