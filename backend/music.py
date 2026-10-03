"""Music analysis: tempo, beat grid, downbeats, sections, drops and energy."""
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, stft
from .media import ffmpeg

SR = 22050
HOP = 512


def load_mono(path, directory, sr=SR):
    wav = Path(directory) / 'music-analysis.wav'
    try:
        ffmpeg('-i', path, '-vn', '-ac', '1', '-ar', str(sr), '-c:a', 'pcm_s16le', wav)
        _, raw = wavfile.read(wav)
    finally:
        wav.unlink(missing_ok=True)
    return raw.astype(np.float32) / 32768


def onset_envelope(y):
    _, _, Z = stft(y, SR, nperseg=2048, noverlap=2048 - HOP, boundary=None, padded=False)
    mag = np.log1p(np.abs(Z) * 100)
    flux = np.maximum(0, np.diff(mag, axis=1)).sum(axis=0)
    flux = np.concatenate([[0], flux])
    flux -= uniform_filter1d(flux, 16)
    return np.maximum(flux, 0) / (flux.max() + 1e-9)


def tempo(env):
    fps = SR / HOP
    ac = np.correlate(env, env, 'full')[len(env) - 1:]
    lags = np.arange(len(ac))
    bpm = 60 * fps / np.maximum(lags, 1)
    valid = (bpm >= 70) & (bpm <= 180)
    # Prefer tempi near 120 BPM (log-Gaussian weighting), as in standard beat trackers.
    weight = np.exp(-.5 * (np.log2(bpm / 120) / .9) ** 2)
    score = np.where(valid, ac * weight, 0)
    lag = int(np.argmax(score))
    return float(60 * fps / lag), lag


def track_beats(env, period, tightness=100):
    """Dynamic-programming beat tracker (Ellis 2007)."""
    n = len(env)
    score = env.copy()
    back = np.full(n, -1)
    window = np.arange(-2 * period, -period // 2 + 1)
    penalty = -tightness * np.log(-window / period) ** 2
    for i in range(n):
        idx = i + window
        ok = idx >= 0
        if not ok.any(): continue
        cand = score[idx[ok]] + penalty[ok]
        j = int(np.argmax(cand))
        score[i] = env[i] + cand[j]
        back[i] = idx[ok][j]
    tail = n - 2 * period + int(np.argmax(score[max(0, n - 2 * period):]))
    beats = [max(0, tail)]
    while back[beats[-1]] >= 0: beats.append(int(back[beats[-1]]))
    return np.array(sorted(beats))


def analyze_music(path, directory):
    y = load_mono(path, directory)
    duration = len(y) / SR
    env = onset_envelope(y)
    fps = SR / HOP
    bpm, period = tempo(env)
    beat_frames = track_beats(env, period)
    beats = beat_frames / fps
    # Downbeat phase: the bar position (of 4) with the strongest onsets.
    strengths = env[beat_frames]
    phase = int(np.argmax([strengths[p::4].mean() if len(strengths[p::4]) else 0 for p in range(4)]))
    downbeats = beats[phase::4]
    # Loudness curve and sections (energy plateaus separated by sustained changes).
    frame = SR // 2
    n = len(y) // frame
    rms = np.sqrt((y[:n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    smooth = uniform_filter1d(db, 8)
    energy = np.clip((smooth - np.percentile(smooth, 5)) / (np.percentile(smooth, 98) - np.percentile(smooth, 5) + 1e-9), 0, 1)
    delta = np.concatenate([[0], smooth[8:] - smooth[:-8], np.zeros(7)])
    changes, _ = find_peaks(np.abs(delta), height=3.0, distance=16)
    bounds = [0.0] + [float(c * .5) for c in changes] + [duration]
    sections = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < 2: continue
        sel = slice(int(a * 2), max(int(a * 2) + 1, int(b * 2)))
        sections.append({'start': round(a, 2), 'end': round(b, 2), 'energy': round(float(energy[sel].mean()), 3)})
    # Drops: a sustained energy rise landing on a downbeat.
    drops = []
    for c in changes:
        if delta[c] <= 0: continue
        t = c * .5
        snapped = float(downbeats[np.argmin(abs(downbeats - t))]) if len(downbeats) else t
        if abs(snapped - t) > 1.2: snapped = t
        before = energy[max(0, c - 16):c].mean() if c > 0 else 0
        after = energy[c:c + 16].mean()
        drops.append({'t': round(snapped, 3), 'strength': round(float(max(0, after - before)), 3), 'energy_after': round(float(after), 3)})
    drops.sort(key=lambda d: d['strength'], reverse=True)
    accents = beats[find_peaks(env[beat_frames], prominence=.1)[0]] if len(beat_frames) > 2 else beats
    return {'duration': round(duration, 3), 'bpm': round(bpm, 2), 'beats': [round(float(b), 3) for b in beats],
            'downbeats': [round(float(b), 3) for b in downbeats], 'accents': [round(float(b), 3) for b in accents],
            'sections': sections, 'drops': drops[:6], 'energy': [round(float(e), 3) for e in energy],
            'energy_rate': 2, 'method': 'spectral-flux onsets, autocorrelation tempo, DP beat tracking, energy-change sections'}


def summary(music):
    """Compact model-facing description."""
    if not music: return None
    return {'bpm': music['bpm'], 'duration': music['duration'], 'sections': music['sections'][:12], 'drops': music['drops'][:4]}
