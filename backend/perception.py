"""Perception: deterministic frame-level signals for every source clip.

Nothing here calls an AI service. Every frame is decoded at a fixed analysis rate
and measured for motion (camera vs subject), quality, exposure, colour cast, the
primary person and their movement. Audio is measured for loudness and impacts.
The results are cached per asset so re-edits never re-scan footage.
"""
import json
import subprocess
from pathlib import Path
import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from scipy.io import wavfile
from .config import binary
from .captions import MODEL_PATH
from .media import pcm

ANALYSIS_FPS = 10
LONG_SIDE = 384
SEG_EVERY = 4          # person segmentation at 2.5 fps, interpolated between
CLIP_EVERY = 20        # one visual embedding every two seconds
COLOR_EVERY = 5
VERSION = 3


def analysis_size(width, height):
    scale = LONG_SIDE / max(width, height)
    return max(2, int(round(width * scale / 2)) * 2), max(2, int(round(height * scale / 2)) * 2)


def decode(path, width, height, fps=ANALYSIS_FPS, start=None, duration=None):
    """Yield BGR frames at a constant rate; FFmpeg applies rotation metadata."""
    args = [binary('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin']
    if start is not None: args += ['-ss', f'{max(0, start):.4f}']
    args += ['-i', str(path)]
    if duration is not None: args += ['-t', f'{duration:.4f}']
    args += ['-an', '-vf', f'fps={fps},scale={width}:{height}:flags=area,setsar=1', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=width * height * 3 * 8)
    size = width * height * 3
    try:
        while True:
            raw = proc.stdout.read(size)
            if len(raw) < size: break
            yield np.frombuffer(raw, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


class PersonSegmenter:
    """PP-HumanSeg (local ONNX). Returns a soft person probability map."""
    def __init__(self):
        self.net = cv2.dnn.readNet(str(MODEL_PATH)) if MODEL_PATH.is_file() else None

    def __call__(self, frame):
        if self.net is None: return None
        blob = cv2.dnn.blobFromImage(frame, 1 / 127.5, (192, 192), (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)
        scores = self.net.forward()[0]
        e = np.exp(scores - scores.max(axis=0))
        return (e[1] / e.sum(axis=0)).astype(np.float32)


def subject_from_mask(prob, previous=None):
    """Pick the primary person: large, central and continuous with the previous subject."""
    if prob is None: return None
    mask = (prob > .5).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best, best_score = None, 0
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        frac = area / mask.size
        if frac < .008: continue
        cx, cy = centroids[i] / 192
        score = frac * (1.2 - abs(cx - .5))
        if previous is not None:
            score *= 1.6 - min(1, np.hypot(cx - previous[0], cy - previous[1]) * 2.5)
        if score > best_score:
            best_score = score
            best = [x / 192, y / 192, (x + w) / 192, (y + h) / 192, frac, cx, cy]
    people = int(sum(1 for i in range(1, count) if stats[i][4] / mask.size >= .02))
    return best, people


class Embedder:
    """CLIP ViT-B/32 image embeddings for duplicate/similarity reasoning (loaded lazily)."""
    _model = None

    def __call__(self, frames):
        if not frames: return np.zeros((0, 512), np.float32)
        try:
            import torch
            if Embedder._model is None:
                import os
                os.environ.setdefault('HF_HUB_OFFLINE', '1')
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
                model.eval()
                torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
                Embedder._model = (model, preprocess)
            model, preprocess = Embedder._model
            from PIL import Image
            batch = torch.stack([preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames])
            with torch.no_grad():
                out = model.encode_image(batch).float()
            out = out / out.norm(dim=-1, keepdim=True)
            return out.numpy().astype(np.float32)
        except Exception:
            # Fallback descriptor: normalised HSV histogram, still useful for near-duplicates.
            vectors = []
            for f in frames:
                hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
                h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
                vectors.append(h / (np.linalg.norm(h) + 1e-8))
            return np.array(vectors, np.float32)


def analyze_video(path, meta):
    width, height = analysis_size(meta['width'], meta['height'])
    seg = PersonSegmenter()
    rows, embed_frames, embed_times = [], [], []
    prev_gray = prev_small = None
    subject_prev = None
    last_subject = None
    for i, img in enumerate(decode(path, width, height)):
        t = i / ANALYSIS_FPS
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        if i % COLOR_EVERY == 0:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            colour = (float(lab[..., 1].mean() - 128), float(lab[..., 2].mean() - 128),
                      float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255))
        if i % SEG_EVERY == 0:
            found = subject_from_mask(seg(img), subject_prev)
            if found is not None:
                last_subject, people = found
                subject_prev = last_subject[5:7] if last_subject else subject_prev
            else:
                last_subject, people = None, 0
            people_now = people
        cam = (0.0, 0.0)
        subj_flow = (0.0, 0.0)
        subj_mag = 0.0
        diff = 0.0
        if prev_small is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_small, small, None, .5, 3, 13, 3, 5, 1.1, 0)
            fh, fw = small.shape
            inside = np.zeros((fh, fw), bool)
            if last_subject:
                x1, y1, x2, y2 = last_subject[:4]
                inside[int(y1 * fh):int(np.ceil(y2 * fh)), int(x1 * fw):int(np.ceil(x2 * fw))] = True
            background = flow[~inside] if (~inside).sum() > fh * fw * .15 else flow.reshape(-1, 2)
            cam = tuple(float(v) for v in np.median(background, axis=0) / fw)
            if inside.any():
                rel = flow[inside] - np.array(cam) * fw
                subj_flow = tuple(float(v) for v in rel.mean(axis=0) / fw)
                subj_mag = float(np.linalg.norm(rel, axis=1).mean() / fw)
            diff = float(np.mean(cv2.absdiff(gray, prev_gray)) / 255)
        rows.append([t, diff, cam[0], cam[1], subj_flow[0], subj_flow[1], subj_mag,
                     float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.mean() / 255),
                     float((gray > 245).mean()), float((gray < 12).mean()),
                     *colour,
                     *(last_subject[:5] if last_subject else [np.nan] * 5), people_now])
        if i % CLIP_EVERY == 0:
            embed_frames.append(img.copy())
            embed_times.append(t)
        prev_gray, prev_small = gray, small
    if not rows: raise ValueError('No decodable video frames')
    columns = ['t', 'diff', 'cam_dx', 'cam_dy', 'subj_dx', 'subj_dy', 'subj_mag', 'sharpness', 'luma', 'clip_hi', 'clip_lo',
               'cast_a', 'cast_b', 'saturation', 'sx1', 'sy1', 'sx2', 'sy2', 'sarea', 'people']
    data = {name: np.array([r[j] for r in rows], np.float32) for j, name in enumerate(columns)}
    # Interpolate subject boxes between segmentation samples and short dropouts.
    data['subject_present'] = (~np.isnan(data['sx1'])).astype(np.float32)
    for name in ('sx1', 'sy1', 'sx2', 'sy2', 'sarea'):
        v = data[name]
        ok = ~np.isnan(v)
        if ok.sum() >= 2: data[name] = np.interp(data['t'], data['t'][ok], v[ok]).astype(np.float32)
        elif ok.sum() == 1: data[name] = np.full_like(v, v[ok][0])
        else: data[name] = np.full_like(v, {'sx1': 0, 'sy1': 0, 'sx2': 1, 'sy2': 1, 'sarea': 0}[name])
    data['embeddings'] = Embedder()(embed_frames)
    data['embed_t'] = np.array(embed_times, np.float32)
    return data


def audio_signals(path, directory, has_audio):
    if not has_audio: return {'rms_t': np.zeros(0, np.float32), 'rms': np.zeros(0, np.float32), 'impacts': []}
    wav = Path(directory) / f'{Path(path).stem}-perception.wav'
    try:
        pcm(path, wav)
        sr, raw = wavfile.read(wav)
    finally:
        wav.unlink(missing_ok=True)
    y = raw.astype(np.float32) / 32768
    hop = sr // 50
    n = len(y) // hop
    if n < 4: return {'rms_t': np.zeros(0, np.float32), 'rms': np.zeros(0, np.float32), 'impacts': []}
    frames = y[:n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    # Percussive impacts (plates, drops, slams): sudden rise well above local loudness.
    rise = np.maximum(0, np.diff(db, prepend=db[0]))
    local = uniform_filter1d(db, 50)
    peaks, props = find_peaks(rise, height=9, distance=15)
    impacts = [{'t': float(p / 50), 'strength': float(min(1, rise[p] / 25)), 'level_db': float(db[p])}
               for p in peaks if db[min(n - 1, p + 1)] > local[p] + 6 and db[min(n - 1, p + 1)] > -40]
    return {'rms_t': (np.arange(n) / 50).astype(np.float32), 'rms': rms.astype(np.float32), 'impacts': impacts}


def detect_reps(data):
    """Repetitions from the primary subject's vertical oscillation and motion energy."""
    if data['subject_present'].mean() < .3 or len(data['t']) < 20: return []
    t = data['t']
    top = uniform_filter1d(data['sy1'].astype(float), 3)
    center = uniform_filter1d(((data['sy1'] + data['sy2']) / 2).astype(float), 3)
    signal = .6 * top + .4 * center
    trend = uniform_filter1d(signal, 41)
    osc = signal - trend
    energy = uniform_filter1d(data['subj_mag'].astype(float), 5)
    reps = []
    for sign in (1, -1):
        peaks, props = find_peaks(sign * osc, prominence=.018, distance=int(ANALYSIS_FPS * .9), width=2)
        found = []
        for k, p in enumerate(peaks):
            left, right = int(props['left_bases'][k]), int(props['right_bases'][k])
            if right - left > ANALYSIS_FPS * 8 or energy[left:right + 1].max() < .004: continue
            found.append({'t': float(t[p]), 'start': float(t[left]), 'end': float(t[right]),
                          'amplitude': float(props['prominences'][k]), 'energy': float(energy[left:right + 1].mean())})
        if len(found) > len(reps): reps = found
    # Keep only rhythmic clusters (2+ reps); isolated wobbles are not reps.
    clustered = [r for i, r in enumerate(reps) if any(abs(r['t'] - o['t']) < 6 for j, o in enumerate(reps) if j != i)]
    return clustered


def summarize(data, audio, reps, meta):
    t = data['t']
    shake = np.hypot(np.diff(data['cam_dx'], prepend=0), np.diff(data['cam_dy'], prepend=0))
    return {'version': VERSION, 'frames': int(len(t)), 'duration': float(t[-1] + 1 / ANALYSIS_FPS),
            'sharpness_median': float(np.median(data['sharpness'])), 'luma_median': float(np.median(data['luma'])),
            'subject_presence': float(data['subject_present'].mean()), 'motion_median': float(np.median(data['subj_mag'])),
            'shake_median': float(np.median(shake)), 'reps': reps, 'impacts': audio['impacts'],
            'people_max': int(np.nanmax(data['people'])) if len(t) else 0, 'width': meta['width'], 'height': meta['height'],
            'fps': meta['fps'], 'orientation': meta['orientation']}


def perceive(asset, directory):
    """Analyse one asset, caching the dense arrays next to the project work files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / f'{asset.id}-signals.npz'
    summary_path = directory / f'{asset.id}-signals.json'
    if cache.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get('version') == VERSION: return load_signals(asset.id, directory), summary
    data = analyze_video(asset.path, asset.meta)
    audio = audio_signals(asset.path, directory, asset.meta.get('has_audio'))
    reps = detect_reps(data)
    summary = summarize(data, audio, reps, asset.meta)
    np.savez_compressed(cache, **data, rms_t=audio['rms_t'], rms=audio['rms'])
    summary_path.write_text(json.dumps(summary))
    return load_signals(asset.id, directory), summary


def load_signals(asset_id, directory):
    with np.load(Path(directory) / f'{asset_id}-signals.npz') as z:
        return {k: z[k] for k in z.files}


def window(data, start, end, key):
    """Values of one per-frame signal inside [start, end)."""
    t = data['t']
    sel = (t >= start) & (t < end)
    if not sel.any():
        idx = int(np.clip(np.searchsorted(t, start), 0, len(t) - 1))
        sel = np.zeros_like(t, bool)
        sel[idx] = True
    return data[key][sel]


def make_proxy(path, target, height=512, fps=4):
    """Small video+audio proxy the multimodal model watches (keeps native aspect)."""
    target = Path(target)
    if not target.exists():
        from .media import ffmpeg
        ffmpeg('-i', path, '-vf', f'scale=-2:{height},fps={fps}', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '33',
               '-c:a', 'aac', '-ac', '1', '-b:a', '32k', '-movflags', '+faststart', target)
    return target
