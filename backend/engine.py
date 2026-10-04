"""Execution engine: renders a timeline frame by frame. It makes no creative choices.

Video: every output frame maps to an exact source time through the segment's speed
map, then crop/zoom/stabilisation keyframes, colour correction, transition shaping
and overlays are applied before frames are piped to the H.264 encoder.
Audio: natural sound (speed-mapped), music with ducking, synthesized SFX, limiter,
and loudness normalisation.
"""
import subprocess
from pathlib import Path
import numpy as np
import cv2
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
from .config import binary
from .media import ffmpeg, probe

SR = 48000
WORK = (720, 1280)      # compositing resolution; FFmpeg upscales with Lanczos + sharpening
FONT = Path('C:/Windows/Fonts/segoeuib.ttf')


# ── timing ───────────────────────────────────────────────────────────────────────

def time_map(seg, fps):
    """Source time for each output frame of a segment (handles ramps and freezes)."""
    a, b = seg['source_start'], seg['source_end']
    keys = seg.get('speed_keys') or []
    hold = (seg.get('freeze') or {}).get('hold', 0)
    n = int(round(seg['duration'] * fps))
    if keys:
        s = np.linspace(a, b, max(2, int((b - a) * 1200)))
        v = np.interp(s, [k['t'] for k in keys], [k['speed'] for k in keys])
        out = np.concatenate([[0], np.cumsum((1 / v[1:] + 1 / v[:-1]) / 2 * np.diff(s))])
        tau = np.arange(n) / fps
        src = np.interp(tau, out, s)
    else:
        src = a + np.arange(n) / fps
    if hold:
        src = np.minimum(src, b - 1 / fps)
    return np.clip(src, a, b - 1e-3), (np.interp(src, [k['t'] for k in keys], [k['speed'] for k in keys]) if keys else np.ones(n))


def interp_keys(keys, t, names):
    ts = [k['t'] for k in keys]
    return [float(np.interp(t, ts, [k[n] for k in keys])) for n in names]


# ── frame sources ────────────────────────────────────────────────────────────────

class SourceReader:
    """Sequential constant-rate frame reader with a two-frame window for blending."""
    def __init__(self, path, meta, start, end, rate=30):
        self.w, self.h, self.rate = meta['width'], meta['height'], rate
        self.t0 = max(0.0, start - .05)
        self.proc = subprocess.Popen(
            [binary('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin', '-ss', f'{self.t0:.4f}', '-i', str(path),
             '-t', f'{end - self.t0 + .3:.4f}', '-an', '-vf', f'fps={rate},scale={self.w}:{self.h},setsar=1',
             '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.index = -1
        self.frames = {}

    def _read(self):
        raw = self.proc.stdout.read(self.w * self.h * 3)
        if len(raw) < self.w * self.h * 3: return None
        self.index += 1
        return np.frombuffer(raw, np.uint8).reshape(self.h, self.w, 3)

    def get(self, k):
        k = max(0, k)
        while self.index < k:
            f = self._read()
            if f is None: break
            self.frames[self.index] = f
            for old in [i for i in self.frames if i < self.index - 2]: del self.frames[old]
        return self.frames.get(min(k, self.index))

    def at(self, t, blend):
        x = (t - self.t0) * self.rate
        k = int(np.floor(x))
        a = self.get(k)
        if a is None: raise ValueError('Source ended before the requested frame')
        frac = x - k
        if blend and frac > .08:
            b = self.get(k + 1)
            if b is not None and b is not a: return cv2.addWeighted(a, 1 - frac, b, frac, 0)
        return a

    def close(self):
        self.proc.stdout.close()
        self.proc.kill()
        self.proc.wait()


# ── look ─────────────────────────────────────────────────────────────────────────

def colour_lut(color):
    x = np.arange(256) / 255.0
    gamma = 1 / max(.5, color['exposure']) ** .85
    y = x ** gamma
    c = color.get('contrast', 1.0)
    y = y + (c - 1) * (y - .5) * (1 - np.abs(2 * y - 1) ** 2) * 1.6     # S-curve that protects whites and blacks
    L = np.clip(y * 255, 0, 255)
    s = color.get('saturation', 1.0)
    a = np.clip((np.arange(256) - 128 + color.get('shift_a', 0)) * s + 128, 0, 255)
    b = np.clip((np.arange(256) - 128 + color.get('shift_b', 0)) * s + 128, 0, 255)
    return np.stack([L, a, b], axis=1).astype(np.uint8).reshape(256, 1, 3)


def grade(img, lut):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return cv2.cvtColor(cv2.LUT(lab, lut), cv2.COLOR_LAB2BGR)


# ── geometry ─────────────────────────────────────────────────────────────────────

def crop_rect(seg, t, w, h):
    rf = seg['reframe']
    bw, bh = rf['base']
    cx, cy, z = interp_keys(rf['keys'], t, ('cx', 'cy', 'z'))
    if seg.get('stabilize'):
        dx, dy = interp_keys(seg['stabilize']['keys'], t, ('dx', 'dy'))
        cx += dx
        cy += dy * w / h            # perception measures both axes in frame widths
    cw, ch = bw * w / z, bh * h / z
    x0 = float(np.clip(cx * w - cw / 2, 0, w - cw))
    y0 = float(np.clip(cy * h - ch / 2, 0, h - ch))
    return x0, y0, cw, ch


def frame_for(img, seg, t, lut, W, H):
    h, w = img.shape[:2]
    if seg['reframe']['mode'] == 'fit':
        bg = cv2.resize(cv2.GaussianBlur(cv2.resize(img, (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), 6), (W, H))
        scale = min(W / w, H / h)
        fg = cv2.resize(grade(img, lut), (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2), interpolation=cv2.INTER_CUBIC)
        out = (grade(bg, lut) * .55).astype(np.uint8)
        y, x = (H - fg.shape[0]) // 2, (W - fg.shape[1]) // 2
        out[y:y + fg.shape[0], x:x + fg.shape[1]] = fg
        return out
    x0, y0, cw, ch = crop_rect(seg, t, w, h)
    # Sub-pixel crop + scale in one affine warp keeps slow push-ins perfectly smooth.
    pad = 2
    xi, yi = int(max(0, np.floor(x0) - pad)), int(max(0, np.floor(y0) - pad))
    xe, ye = int(min(w, np.ceil(x0 + cw) + pad)), int(min(h, np.ceil(y0 + ch) + pad))
    region = grade(np.ascontiguousarray(img[yi:ye, xi:xe]), lut)
    sx, sy = W / cw, H / ch
    M = np.array([[sx, 0, -(x0 - xi) * sx], [0, sy, -(y0 - yi) * sy]], np.float32)
    return cv2.warpAffine(region, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def motion_blur(img, length):
    """Horizontal directional blur (computed at half resolution for speed)."""
    taps = int(length) // 2
    if taps < 2: return img
    small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    small = cv2.filter2D(small, -1, np.full((1, taps), 1 / taps, np.float32))
    return cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def shape_transition(img, kind, progress, direction, outgoing):
    """progress 0→1 toward the cut on the outgoing side; 1→0 away from it on the incoming side."""
    H, W = img.shape[:2]
    p = float(np.clip(progress, 0, 1)) ** 1.6
    if kind == 'whip':
        sign = -1 if direction == 'left' else 1
        shift = sign * p * W * .35 * (1 if outgoing else -1)
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        moved = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        return motion_blur(moved, (8 + p * 110) * W / 1080)
    if kind == 'zoom':
        s = 1 + .18 * p
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
        zoomed = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
        if p > .15:
            layers = [zoomed] + [cv2.warpAffine(zoomed, cv2.getRotationMatrix2D((W / 2, H / 2), 0, 1 + .03 * i * p), (W, H), borderMode=cv2.BORDER_REFLECT) for i in (1, 2, 3)]
            zoomed = np.mean(layers, axis=0).astype(np.uint8)
        return zoomed
    return img


# ── overlays ─────────────────────────────────────────────────────────────────────

def text_layer(text, W, size=86):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(str(FONT), size) if FONT.exists() else ImageFont.load_default()
    words = text.upper().split()
    lines, cur = [], ''
    for word in words:
        trial = (cur + ' ' + word).strip()
        if font.getlength(trial) > W * .82 and cur: lines.append(cur); cur = word
        else: cur = trial
    lines.append(cur)
    line_h = int(size * 1.12)
    img = Image.new('RGBA', (W, line_h * len(lines) + 40), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        x = (W - font.getlength(line)) / 2
        y = 20 + i * line_h
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255), stroke_width=7, stroke_fill=(10, 10, 10, 235))
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)


def apply_overlay(frame, layer, y, alpha, scale=1.0):
    if alpha <= 0: return frame
    if scale != 1.0:
        layer = cv2.resize(layer, (int(layer.shape[1] * scale), int(layer.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
    H, W = frame.shape[:2]
    h, w = layer.shape[:2]
    x = (W - w) // 2
    y = int(y - h / 2)
    ys, ye = max(0, y), min(H, y + h)
    xs, xe = max(0, x), min(W, x + w)
    if ys >= ye or xs >= xe: return frame
    part = layer[ys - y:ye - y, xs - x:xe - x]
    a = part[..., 3:4].astype(np.float32) / 255 * alpha
    region = frame[ys:ye, xs:xe].astype(np.float32)
    frame[ys:ye, xs:xe] = (region * (1 - a) + part[..., :3] * a).astype(np.uint8)
    return frame


# ── audio ────────────────────────────────────────────────────────────────────────

def decode_audio(path, start, duration):
    proc = subprocess.run([binary('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin', '-ss', f'{max(0, start):.4f}', '-i', str(path),
                           '-t', f'{duration:.4f}', '-vn', '-ac', '2', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True)
    data = np.frombuffer(proc.stdout, np.float32)
    return data[:len(data) // 2 * 2].reshape(-1, 2).copy()


def synth(kind, seconds=None):
    rng = np.random.default_rng({'whoosh': 1, 'impact': 2, 'riser': 3, 'hit': 4}[kind])
    if kind == 'whoosh':
        n = int(SR * .42)
        t = np.linspace(0, 1, n)
        noise = rng.standard_normal(n)
        env = np.sin(np.pi * t) ** 2.2
        out = np.zeros(n)
        for i, f in enumerate(np.geomspace(500, 3500, 8)):
            sos = butter(2, [f * .7, f * 1.3], 'bandpass', fs=SR, output='sos')
            part = sosfilt(sos, noise)
            centre = .2 + .6 * i / 7
            out += part * np.exp(-((t - centre) / .18) ** 2)
        out *= env
    elif kind in ('impact', 'hit'):
        n = int(SR * (1.1 if kind == 'impact' else .35))
        t = np.arange(n) / SR
        boom = np.sin(2 * np.pi * (48 * t + 30 * (1 - np.exp(-t * 18)) / 18)) * np.exp(-t * (4.5 if kind == 'impact' else 14))
        click = sosfilt(butter(2, 2500, 'lowpass', fs=SR, output='sos'), rng.standard_normal(n)) * np.exp(-t * 60)
        out = boom + .35 * click
    else:  # riser
        n = int(SR * (seconds or 1.2))
        t = np.arange(n) / SR
        sweep = np.sin(2 * np.pi * (200 * t + 900 * t ** 2 / (2 * t[-1])))
        hiss = sosfilt(butter(2, 3000, 'highpass', fs=SR, output='sos'), rng.standard_normal(n))
        out = (.4 * sweep + .6 * hiss) * (t / t[-1]) ** 2.5
    out = out / (np.abs(out).max() + 1e-9)
    return np.stack([out, out], axis=1).astype(np.float32)


def mix_audio(timeline, by_id, music_asset, directory):
    total = int(round(timeline['duration'] * SR))
    natural = np.zeros((total, 2), np.float32)
    speech_mask = np.zeros(total, np.float32)
    has_music = bool(timeline.get('music') and music_asset)
    for seg in timeline['segments']:
        asset = by_id[seg['asset_id']]
        n = int(round(seg['duration'] * SR))
        start = int(round(seg['timeline_start'] * SR))
        if not asset.meta.get('has_audio'): continue
        src = decode_audio(asset.path, seg['source_start'], seg['source_end'] - seg['source_start'] + .1)
        if len(src) < 2: continue
        tau = np.arange(n) / SR
        times, speeds = time_map(seg, SR) if n else (np.zeros(0), np.zeros(0))
        idx = np.clip((times - seg['source_start']) * SR, 0, len(src) - 1)
        part = np.stack([np.interp(idx, np.arange(len(src)), src[:, c]) for c in range(2)], axis=1).astype(np.float32)
        gain = seg['audio']['natural_gain'] * (.5 if has_music and not seg['audio']['keep_speech'] else 1.0)
        g = np.where(np.abs(speeds - 1) < .05, gain, gain * .3).astype(np.float32)
        if seg.get('freeze'):
            held = tau > seg['duration'] - seg['freeze']['hold']
            g[held] *= np.linspace(1, 0, held.sum()) if held.any() else 1
        fade = min(n // 2, int(SR * .012))
        if fade > 1:
            g[:fade] *= np.linspace(0, 1, fade)
            g[-fade:] *= np.linspace(1, 0, fade)
        end = min(total, start + n)
        natural[start:end] += part[:end - start] * g[:end - start, None]
        if seg['audio']['keep_speech']: speech_mask[start:end] = 1
    mix = natural
    if has_music:
        m = timeline['music']
        music = decode_audio(music_asset.path, m['offset'], timeline['duration'] + .2)[:total]
        music = np.pad(music, ((0, total - len(music)), (0, 0)))
        duck = np.convolve(speech_mask, np.ones(int(SR * .25)) / int(SR * .25), mode='same')
        g = m['gain'] * (1 - (1 - m['duck_under_speech']) * np.clip(duck, 0, 1))
        fo = int(SR * m['fade_out'])
        if fo: g[-fo:] *= np.linspace(1, 0, fo) ** 1.5
        mix = mix + music * g[:, None].astype(np.float32)
    for fx in timeline.get('sfx', []):
        clip = synth(fx['type']) * fx['gain']
        at = int(fx['at'] * SR)
        end = min(total, at + len(clip))
        if at < total: mix[at:end] += clip[:end - at]
    # Every edit ends cleanly: a short fade on the whole mix.
    tail = min(total, int(SR * .7))
    if tail: mix[-tail:] *= np.linspace(1, 0, tail)[:, None] ** 1.3
    # Gentle limiter: unity below -3 dBFS, smooth knee above.
    peak = np.abs(mix).max() + 1e-9
    mix = mix / max(1.0, peak / .95)
    mix = np.tanh(mix * 1.2) / np.tanh(1.2)
    path = Path(directory) / 'mix.wav'
    wavfile.write(path, SR, (np.clip(mix, -1, 1) * 32767).astype(np.int16))
    return path


# ── main ─────────────────────────────────────────────────────────────────────────

def render_timeline(timeline, assets, directory, progress=lambda *a: None):
    from .captions import selected_events, write_captions
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    by_id = {a.id: a for a in assets}
    OW, OH, fps = timeline['width'], timeline['height'], timeline['fps']
    W, H = WORK if OW / OH == WORK[0] / WORK[1] and OW > WORK[0] else (OW, OH)
    total_frames = int(round(timeline['duration'] * fps))
    vf = [f'scale={OW}:{OH}:flags=lanczos', 'unsharp=5:5:0.45:5:5:0.0']
    events, caption_settings = selected_events(timeline.get('captions'), timeline.get('speech_events', []), timeline['duration'])
    if events:
        write_captions(events, directory / 'captions.ass', caption_settings.model_dump())
        vf.append('subtitles=filename=captions.ass')
    video = directory / 'video.mp4'
    encoder = subprocess.Popen([binary('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                                '-s', f'{W}x{H}', '-r', str(fps), '-i', '-', '-vf', ','.join(vf), '-c:v', 'libx264', '-preset', 'fast',
                                '-crf', '18', '-pix_fmt', 'yuv420p', '-r', str(fps), str(video)], stdin=subprocess.PIPE, stderr=subprocess.PIPE, cwd=directory)
    overlays = [(o, text_layer(o['text'], W, size=int(86 * W / 1080))) for o in timeline.get('overlays', []) if o['type'] == 'text']
    written = 0
    segments = timeline['segments']
    try:
        for si, seg in enumerate(segments):
            asset = by_id.get(seg['asset_id'])
            if asset is None: raise ValueError('Unknown source asset in timeline')
            times, speeds = time_map(seg, fps)
            n = len(times)
            if si == len(segments) - 1: n = total_frames - written
            reader = SourceReader(asset.path, asset.meta, seg['source_start'], seg['source_end'])
            lut = colour_lut(seg['color'])
            nxt = segments[si + 1]['transition_in'] if si + 1 < len(segments) else {'type': 'cut'}
            tin = seg['transition_in']
            try:
                for j in range(n):
                    t = float(times[min(j, len(times) - 1)])
                    slow = speeds[min(j, len(speeds) - 1)] < .95
                    img = frame_for(reader.at(t, slow), seg, t, lut, W, H)
                    k_out = int(round(nxt.get('duration', 0) * fps / 2))
                    k_in = int(round(tin.get('duration', 0) * fps / 2))
                    if nxt['type'] in ('whip', 'zoom') and j >= n - k_out:
                        img = shape_transition(img, nxt['type'], (j - (n - k_out) + 1) / k_out, nxt.get('direction'), True)
                    if tin['type'] in ('whip', 'zoom') and j < k_in:
                        img = shape_transition(img, tin['type'], 1 - j / k_in, tin.get('direction'), False)
                    tl = (written + j) / fps
                    for o, layer in overlays:
                        if o['start'] <= tl <= o['end']:
                            a = min(1, (tl - o['start']) / .15, (o['end'] - tl) / .2)
                            pop = 1 + .08 * max(0, 1 - (tl - o['start']) / .18)
                            img = apply_overlay(img, layer, int(H * .22), a, pop)
                    encoder.stdin.write(np.ascontiguousarray(img).tobytes())
            finally:
                reader.close()
            written += n
            progress(si + 1, len(segments))
        encoder.stdin.close()
        if encoder.wait() != 0: raise RuntimeError(encoder.stderr.read().decode('utf-8', 'replace')[-3000:])
    except BrokenPipeError:
        raise RuntimeError(encoder.stderr.read().decode('utf-8', 'replace')[-3000:])
    finally:
        if encoder.poll() is None: encoder.kill()
    music_asset = by_id.get((timeline.get('music') or {}).get('asset_id'))
    audio = mix_audio(timeline, by_id, music_asset, directory)
    output = directory / 'final.mp4'
    ffmpeg('-i', video, '-i', audio, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11',
           '-c:a', 'aac', '-b:a', '192k', '-ar', str(SR), '-t', f"{timeline['duration']:.4f}", '-movflags', '+faststart', output)
    metadata = probe(output)
    if (metadata['width'], metadata['height']) != (OW, OH) or metadata['codec'] != 'h264' or not metadata['has_audio'] or abs(metadata['duration'] - timeline['duration']) > .2:
        raise ValueError(f'Render validation failed: {metadata}')
    for f in (video, audio): f.unlink(missing_ok=True)
    return str(output), metadata
