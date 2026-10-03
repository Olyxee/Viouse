import json
import subprocess
from fractions import Fraction
from pathlib import Path
from .config import binary

def run(args, timeout=1800):
    result = subprocess.run([str(a) for a in args], capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', 'replace')[-5000:])
    return result.stdout

def ffmpeg(*args, timeout=1800):
    return run([binary('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', *args], timeout)

def probe(path):
    raw = json.loads(run([binary('ffprobe'), '-v', 'error', '-show_format', '-show_streams', '-of', 'json', path], 45))
    video = next((s for s in raw['streams'] if s['codec_type'] == 'video' and not s.get('disposition', {}).get('attached_pic')), None)
    audio = next((s for s in raw['streams'] if s['codec_type'] == 'audio'), None)
    duration = float(raw.get('format', {}).get('duration', 0))
    rotation = 0
    if video:
        rotation = next((s.get('rotation', 0) for s in video.get('side_data_list', []) if 'rotation' in s), 0)
    w, h = (video['width'], video['height']) if video else (0, 0)
    if abs(rotation) % 180 == 90: w, h = h, w
    return {'duration': duration, 'width': w, 'height': h,
            'fps': float(Fraction(video.get('avg_frame_rate', '0/1'))) if video and video.get('avg_frame_rate') != '0/0' else 0,
            'codec': video['codec_name'] if video else audio['codec_name'] if audio else 'unknown',
            'orientation': 'portrait' if h > w else 'landscape', 'has_audio': audio is not None,
            'has_video': video is not None, 'rotation': rotation,
            'creation_time': raw.get('format', {}).get('tags', {}).get('creation_time')}

def make_proxy(source, target):
    if not Path(target).exists():
        ffmpeg('-i', source, '-map', '0:v:0', '-an', '-vf', "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2,fps=8", '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28', target)

def frame(source, at, target):
    ffmpeg('-ss', max(0, at), '-i', source, '-frames:v', '1', '-vf', 'scale=640:640:force_original_aspect_ratio=decrease', target, timeout=60)
    if not Path(target).exists():
        ffmpeg('-ss', max(0, at-.25), '-i', source, '-frames:v', '1', '-vf', 'scale=640:640:force_original_aspect_ratio=decrease', target, timeout=60)
    if not Path(target).exists(): raise ValueError('Could not decode a representative frame')

def pcm(source, target):
    ffmpeg('-i', source, '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', target)
