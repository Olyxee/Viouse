from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
from .media import pcm, ffmpeg
from .config import settings

def gemini_transcribe(asset,directory,ai,raw,sr,rms):
    from .gemini import generate
    from .schemas import SpeechTranscript
    segments,labels=[],[]
    # One-minute inline WAV chunks remain small; timestamps are relative to each chunk.
    for start in range(0,int(asset.duration)+1,60):
        chunk_raw=raw[start*sr:(start+60)*sr]
        duration=len(chunk_raw)/sr
        if duration<.25: continue
        chunk=Path(directory)/f'{asset.id}-gemini-speech-{start}.wav'
        wavfile.write(chunk,sr,chunk_raw)
        try:
            result=generate(settings.gemini_api_key,settings.gemini_model,SpeechTranscript,
                f'Transcribe only clearly intelligible spoken words in this {duration:.3f}-second recording. '
                'Do not invent speech from music, breaths, grunts, equipment noise or silence. '
                'Use the original language, short phrases of at most six words, and accurate start/end seconds relative to this audio chunk. '
                'Return empty segments when speech is absent or uncertain. Confidence must reflect intelligibility. Classify the observed audio.',
                audio=chunk,calls=ai.calls)
            last_end=0.0
            for s in result.segments:
                if s.end<=s.start or s.start>=duration or s.end>duration+.05 or s.start<last_end-.05:
                    raise ValueError('Gemini returned invalid or overlapping speech timestamps')
                last_end=s.end
                if s.confidence>=.8 and s.text.strip():
                    segments.append({'start':start+s.start,'end':start+min(s.end,duration),'text':s.text.strip()})
            labels.append(result.classification)
        finally: chunk.unlink(missing_ok=True)
    return {'asset_id':asset.id,'segments':segments,'classification':'speech' if segments else ', '.join(sorted(set(labels))),
            'rms':rms,'provider':'gemini','model':settings.gemini_model,'timestamp_method':'model-estimated phrase timestamps'}

def audio_features(path,directory):
    try: return _audio_features(path,directory)
    finally: (Path(directory)/'music.wav').unlink(missing_ok=True)

def _audio_features(path, directory):
    wav = Path(directory)/'music.wav'
    pcm(path, wav)
    sr, data = wavfile.read(wav)
    y = data.astype(float)/32768
    hop = 512
    count = len(y)//hop
    if count < 4: return {'bpm': None, 'beats': [], 'drops': []}
    energy = np.sqrt(np.mean(y[:count*hop].reshape(count,hop)**2,axis=1))
    onset = np.maximum(0, np.diff(energy, prepend=energy[0]))
    peaks, _ = find_peaks(onset, distance=int(.25*sr/hop), prominence=max(.005,float(np.std(onset)*.7)))
    times = peaks*hop/sr
    gaps = np.diff(times)
    gaps = gaps[(gaps > .3)&(gaps < 1)]
    bpm = float(60/np.median(gaps)) if len(gaps) else None
    strongest = sorted(peaks, key=lambda i: float(onset[i]), reverse=True)[:10]
    return {'bpm': bpm, 'beats': times.tolist(), 'drops': sorted([float(i*hop/sr) for i in strongest]), 'method': 'RMS onset peaks; approximate beat/drop detection'}

def transcribe(asset,directory,ai):
    try: return _transcribe(asset,directory,ai)
    finally:
        (Path(directory)/f'{asset.id}-audio.wav').unlink(missing_ok=True)
        for chunk in Path(directory).glob(f'{asset.id}-speech-*.wav'): chunk.unlink(missing_ok=True)

def _transcribe(asset, directory, ai):
    if not asset.meta['has_audio']: return {'asset_id':asset.id,'segments':[], 'classification':'no audio'}
    wav = Path(directory)/f'{asset.id}-audio.wav'
    pcm(asset.path, wav)
    sr, raw = wavfile.read(wav)
    y = raw.astype(float)/32768
    rms = float(np.sqrt(np.mean(y*y)))
    if rms < .002: return {'asset_id':asset.id,'segments':[], 'classification':'silence', 'rms':rms}
    if ai.provider=='gemini': return gemini_transcribe(asset,directory,ai,raw,sr,rms)
    segments = []
    # Five-minute chunks stay below transcription's upload limit.
    for start in range(0, int(asset.duration)+1, 300):
        if asset.duration-start < .25: continue
        chunk = Path(directory)/f'{asset.id}-speech-{start}.wav'
        wavfile.write(chunk, sr, raw[start*sr:(start+300)*sr])
        with chunk.open('rb') as f:
            result = ai.client.audio.transcriptions.create(model='whisper-1', file=f, response_format='verbose_json', timestamp_granularities=['segment','word'])
        obj = result.model_dump()
        for seg in obj.get('segments', []):
            text = seg['text'].strip()
            if seg.get('no_speech_prob',1) > .45 or seg.get('avg_logprob',-10) < -1 or len(text.split()) < 2: continue
            # Word groups keep subtitles short and accurately synchronized.
            words = [w for w in obj.get('words',[]) if w['start'] >= seg['start']-.05 and w['end'] <= seg['end']+.15]
            if words:
                for i in range(0,len(words),6):
                    group = words[i:i+6]
                    segments.append({'start':start+group[0]['start'],'end':start+group[-1]['end'],'text':' '.join(w['word'].strip() for w in group)})
            else:
                segments.append({'start':start+seg['start'],'end':start+seg['end'],'text':text})
        chunk.unlink(missing_ok=True)
    spectrum = np.abs(np.fft.rfft(y[:min(len(y),sr*10)])) + 1e-10
    flatness = float(np.exp(np.mean(np.log(spectrum)))/np.mean(spectrum))
    classification = 'speech' if segments else 'likely broadband noise' if flatness > .35 else 'non-speech / possible music'
    return {'asset_id':asset.id,'segments':segments,'classification':classification,'rms':rms,'spectral_flatness':flatness}
