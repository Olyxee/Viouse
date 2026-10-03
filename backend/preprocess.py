from pathlib import Path
import cv2
import numpy as np
from .media import make_proxy, frame

def segment_video(asset, directory):
    """Scan every proxy frame locally; sample only three images per segment for AI."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    proxy = directory / f'{asset.id}-proxy.mp4'
    make_proxy(asset.path, proxy)
    cap = cv2.VideoCapture(str(proxy))
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    last_gray, last_hist = None, None
    boundaries, measurements, index = [0.0], [], 0
    while True:
        ok, img = cap.read()
        if not ok: break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([img], [0, 1, 2], None, [4, 4, 4], [0, 256]*3).flatten()
        hist /= np.linalg.norm(hist) + 1e-8
        t = index / fps
        motion = float(np.mean(cv2.absdiff(gray, last_gray)) / 255) if last_gray is not None else 0
        change = float(1 - np.dot(hist, last_hist)) if last_hist is not None else 0
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        measurements.append((t, motion, sharpness, float(gray.mean()/255)))
        elapsed = t - boundaries[-1]
        # Scene changes, action/rest transitions, and a bounded six-second window.
        recent_motion = np.mean([x[1] for x in measurements[-8:]])
        transition = len(measurements) > 16 and abs(recent_motion - np.mean([x[1] for x in measurements[-16:-8]])) > .10
        if elapsed >= 6 or (elapsed >= 1.5 and (change > .40 or transition)):
            boundaries.append(t)
        last_gray, last_hist = gray, hist
        index += 1
    cap.release()
    if asset.duration - boundaries[-1] < .5 and len(boundaries) > 1: boundaries.pop()
    boundaries.append(asset.duration)
    segments = []
    for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        sid = f'{asset.id}_{i:04d}'
        times = [start + (end-start)*f for f in (.15, .5, .85)]
        frames = []
        embeddings = []
        for j, at in enumerate(times):
            p = directory / f'{sid}-{j}.jpg'
            frame(proxy, at, p)
            frames.append(str(p))
            img = cv2.imread(str(p))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0,180,0,256]).flatten()
            hist /= np.linalg.norm(hist) + 1e-8
            embeddings.append(hist)
        stats = [m for m in measurements if start <= m[0] < end]
        segments.append({'segment_id': sid, 'asset_id': asset.id, 'source_start': start, 'source_end': end,
                         'duration': end-start, 'frames': frames, 'frame_times': times,
                         'embedding': np.mean(embeddings, axis=0).tolist(),
                         'motion': float(np.mean([m[1] for m in stats])) if stats else 0,
                         'sharpness': float(np.mean([m[2] for m in stats])) if stats else 0,
                         'brightness': float(np.mean([m[3] for m in stats])) if stats else .5,
                         'proxy': str(proxy)})
    return segments

def track_subject(segment, box):
    """KLT feature tracking anchored to the semantic subject region; reset on drift."""
    cap = cv2.VideoCapture(segment['proxy'])
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    cap.set(cv2.CAP_PROP_POS_MSEC, segment['source_start']*1000)
    ok, img = cap.read()
    if not ok:
        cap.release()
        return {'mode': 'fit', 'points': []}
    h, w = img.shape[:2]
    x1,y1,x2,y2 = np.clip(box, 0, 1)
    # If anatomy/equipment cannot fit, preserve the entire image over a blurred fill.
    crop_width = h*9/16
    if x2 <= x1 or y2 <= y1 or (x2-x1)*w*1.2 > crop_width or w/h <= 9/16:
        cap.release()
        return {'mode': 'fit', 'points': []}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    mask[int(y1*h):int(y2*h), int(x1*w):int(x2*w)] = 255
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=.02, minDistance=7, mask=mask)
    center = float((x1+x2)*.5*w)
    smooth = center
    points = [[0.0, float(np.clip(smooth/w, 0, 1))]]
    total = int(segment['duration']*fps)
    for i in range(1, total):
        ok, img = cap.read()
        if not ok: break
        next_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if pts is None or len(pts) < 8:
            cap.release()
            return {'mode': 'fit', 'points': [], 'reason': 'Tracking confidence lost; preserve complete frame'}
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(gray, next_gray, pts, None)
        if nxt is None:
            cap.release()
            return {'mode': 'fit', 'points': []}
        valid = status.reshape(-1) == 1
        delta = np.median(nxt[valid] - pts[valid], axis=0).reshape(-1) if valid.sum() >= 8 else [0,0]
        center = float(np.clip(center + delta[0], crop_width/2, w-crop_width/2))
        smooth += .12*(center-smooth)
        pts, gray = nxt[valid].reshape(-1,1,2), next_gray
        if i % 4 == 0: points.append([round(i/fps,3), float(smooth/w)])
    cap.release()
    return {'mode': 'track', 'points': points}
