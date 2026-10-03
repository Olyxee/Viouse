from pathlib import Path
from .captions import write_captions, selected_events, person_mask
from .media import ffmpeg, probe, run
from .config import binary

def crop_expression(points, offset):
    # A continuous piecewise linear center avoids jumps at tracking samples.
    if not points: return '0.5'
    expr = str(round(points[-1][1],6))
    for (t0,x0),(t1,x1) in reversed(list(zip(points,points[1:]))):
        if t1 <= t0: continue
        value = f'({x0:.6f}+({x1-x0:.6f})*clip((t+{offset:.5f}-{t0:.5f})/{t1-t0:.5f},0,1))'
        expr = f'if(lt(t+{offset:.5f},{t1:.5f}),{value},{expr})'
    return expr

def caption_events(edl, transcripts):
    by_asset = {t['asset_id']:t['segments'] for t in transcripts}
    events = []
    for shot in edl['timeline']:
        for text in by_asset.get(shot['asset_id'],[]):
            start,end = max(text['start'],shot['source_start']),min(text['end'],shot['source_end'])
            if end-start < .15: continue
            events.append({'start':shot['timeline_start']+(start-shot['source_start'])/shot['speed'],
                           'end':shot['timeline_start']+(end-shot['source_start'])/shot['speed'], 'text':text['text']})
    return events

def render(edl, assets, directory, transcripts=(), music=None, progress=lambda *args:None, width=1080, height=1920):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    by_id = {a.id:a for a in assets}
    parts = []
    for i,shot in enumerate(edl['timeline']):
        asset = by_id.get(shot['asset_id'])
        if asset is None: raise ValueError('Unknown source asset in EDL')
        if shot['source_start'] < 0 or shot['source_end'] > asset.duration+.035: raise ValueError('Source range outside asset')
        path = directory/f'part-{i:03}.mp4'
        d = shot['duration']
        inputs = ['-ss',str(shot['source_start']),'-t',str(d),'-i',asset.path]
        if not asset.meta['has_audio']: inputs += ['-f','lavfi','-i','anullsrc=r=48000:cl=stereo']
        bright = max(-.03,min(.03,(.45-shot.get('brightness',.45))*.08))
        common = f'setsar=1,fps=30,eq=brightness={bright:.4f}:contrast=1.025'
        if shot['crop']['mode'] == 'track':
            center = crop_expression(shot['crop']['points'],shot.get('track_offset',0))
            visual = f"[0:v]setpts=PTS-STARTPTS,crop=w=ih*9/16:h=ih:x='clip(iw*({center})-ow/2,0,iw-ow)':y=0,scale={width}:{height},{common}[v]"
        else:
            visual = f'[0:v]setpts=PTS-STARTPTS,split[bg][fg];[bg]scale={width//4}:{height//4}:force_original_aspect_ratio=increase,crop={width//4}:{height//4},boxblur=12:2,scale={width}:{height}[b];[fg]scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2[f];[b][f]overlay=(W-w)/2:(H-h)/2,{common}[v]'
        audio = '0:a:0' if asset.meta['has_audio'] else '1:a:0'
        ffmpeg(*inputs,'-filter_complex',visual,'-map','[v]','-map',audio,'-af',f'aresample=48000,apad,atrim=duration={d},asetpts=PTS-STARTPTS','-t',d,'-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-ar','48000','-ac','2','-video_track_timescale','15360',path)
        parts.append(path)
        progress(i+1,len(edl['timeline']))
    concat = directory/'concat.txt'
    concat.write_text('\n'.join(f"file '{p.name}'" for p in parts),encoding='utf-8')
    joined = directory/'joined.mp4'
    ffmpeg('-f','concat','-safe','0','-i',concat,'-c','copy',joined)
    speech_events = caption_events(edl,transcripts)
    events, caption_settings = selected_events(edl.get("captions"),speech_events,edl["duration"])
    args = ['-i',str(joined)]
    filters = []
    if music:
        args += ['-stream_loop','-1','-i',music.path]
        # Time-aware music ducking uses actual selected speech, not gym noise.
        speech = '+'.join(f'between(t,{e["start"]:.3f},{e["end"]:.3f})' for e in speech_events) or '0'
        filters += [f"[1:a]atrim=duration={edl['duration']},asetpts=PTS-STARTPTS,volume='if(gt({speech},0),0.09,0.30)':eval=frame,afade=t=out:st={max(0,edl['duration']-1)}:d=1[m]",
                    '[0:a][m]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=9[a]']
    elif any(by_id[shot['asset_id']].meta['has_audio'] for shot in edl['timeline']):
        filters += [f"[0:a]apad=pad_dur=3,loudnorm=I=-14:TP=-1.5:LRA=9,atrim=duration={edl['duration']}[a]"]
    else:
        # Loudness is undefined for generated silence; leave the silent track intact.
        filters += ['[0:a]anull[a]']
    if events:
        captions = directory/'captions.ass'
        write_captions(events,captions,caption_settings.model_dump())
        # Run from render directory so Windows drive colons never enter filters.
        if caption_settings.style=='behind':
            mask=directory/'person-mask.mp4'
            person_mask(joined,mask)
            mask_index=2 if music else 1
            args += ['-i',str(mask)]
            filters += [f'[0:v]split[clean][letters];[letters]subtitles=filename=captions.ass[text];[{mask_index}:v]scale={width}:{height},format=gray[mask];[text][clean][mask]maskedmerge[v]']
        else:
            filters += ["[0:v]subtitles=filename=captions.ass[v]"]
    output = directory/'final.mp4'
    args += ['-filter_complex',';'.join(filters),'-map','[v]' if events else '0:v:0','-map','[a]',
             '-t',str(edl['duration']),'-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-ar','48000','-movflags','+faststart',str(output)]
    import subprocess
    result = subprocess.run([binary('ffmpeg'),'-hide_banner','-loglevel','error','-nostdin','-y',*args],cwd=directory,capture_output=True,timeout=3600)
    if result.returncode: raise RuntimeError(result.stderr.decode('utf-8','replace')[-5000:])
    metadata = probe(output)
    if (metadata['width'],metadata['height']) != (width,height) or metadata['codec']!='h264' or not metadata['has_audio'] or abs(metadata['duration']-edl['duration']) > .15:
        raise ValueError(f'Render validation failed: {metadata}')
    ffmpeg('-i',output,'-f','null','-',timeout=3600)
    # Keep the final version and EDL, not redundant per-shot/concatenated encodes.
    for intermediate in [*parts,joined,concat,directory/'person-mask.mp4']: intermediate.unlink(missing_ok=True)
    return str(output), metadata
