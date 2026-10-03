import os
from pathlib import Path

# Isolated storage; tests never touch creator projects.
TEST_DIR=Path(os.environ['DATA_DIR'])

from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.media import ffmpeg,probe
from backend.schemas import Plan
from backend.planner import validate_plan
from backend.render import render,caption_events
from backend.preprocess import segment_video
from backend.db import session,Job,Project,DOCUMENTS

@pytest.fixture(scope='module')
def footage():
    path=TEST_DIR/'source.mp4'
    ffmpeg('-f','lavfi','-i','testsrc2=size=640x360:rate=30','-f','lavfi','-i','sine=frequency=300:sample_rate=48000','-t','4','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac',path)
    return path

def test_milestone_one_real_ffmpeg_render(footage):
    asset=SimpleNamespace(id='a',path=str(footage),duration=4,meta=probe(footage))
    edl={'duration':2,'timeline':[{'asset_id':'a','source_start':1,'source_end':3,'duration':2,'timeline_start':0,'speed':1,'crop':{'mode':'fit','points':[]}}]}
    output,meta=render(edl,[asset],TEST_DIR/'render',width=1080,height=1920)
    assert Path(output).stat().st_size>1000
    assert (meta['width'],meta['height'],meta['codec'])==(1080,1920,'h264')
    assert abs(meta['duration']-2)<.1

def test_music_captions_tracking_and_silent_source(footage):
    silent=TEST_DIR/'silent.mp4'
    ffmpeg('-i',footage,'-an','-c:v','copy',silent)
    asset=SimpleNamespace(id='silent',path=str(silent),duration=4,meta=probe(silent))
    music=SimpleNamespace(path=str(footage))
    edl={'duration':2,'timeline':[{'asset_id':'silent','source_start':0,'source_end':2,'duration':2,'timeline_start':0,'speed':1,'track_offset':0,'crop':{'mode':'track','points':[[0,.45],[1,.55],[2,.5]]}}]}
    transcripts=[{'asset_id':'silent','segments':[{'start':.2,'end':1.3,'text':'One more rep'}]}]
    output,meta=render(edl,[asset],TEST_DIR/'mix',transcripts,music,width=270,height=480)
    assert meta['has_audio'] and Path(output).exists()
    assert (TEST_DIR/'mix'/'captions.ass').exists()

def test_segmentation_covers_entire_source(footage):
    a=SimpleNamespace(id='a',path=str(footage),duration=4)
    segments=segment_video(a,TEST_DIR/'segments')
    assert segments[0]['source_start']==0
    assert segments[-1]['source_end']==4
    assert sum(s['duration'] for s in segments)==pytest.approx(4)
    assert all(len(s['frames'])==3 and len(s['embedding'])==128 for s in segments)

def plan(story):
    return Plan(title='Workout',style='high-energy',target_duration=2,story=story,rejected=[])

def shot(id='s',start=0,end=2):
    return {'segment_id':id,'role':'hook','source_start':start,'source_end':end,'reason':'Action'}

def test_plan_rejects_hallucinations_and_invalid_ranges():
    segments=[{'segment_id':'s','asset_id':'a','source_start':0,'source_end':4}]
    validate_plan(plan([shot()]),segments,2)
    for story in ([shot('fake')],[shot(start=-1,end=1)],[shot(end=5)],[shot(),shot()]):
        with pytest.raises(ValueError): validate_plan(plan(story),segments,2)

def test_caption_timestamps_map_to_selected_source():
    edl={'timeline':[{'asset_id':'a','source_start':10,'source_end':12,'timeline_start':5,'speed':1}]}
    result=caption_events(edl,[{'asset_id':'a','segments':[{'start':9,'end':11,'text':'Go now'},{'start':13,'end':14,'text':'Not selected'}]}])
    assert result==[{'start':5,'end':6,'text':'Go now'}]

def test_local_workspace_opens_without_login_and_rejects_remote_bootstrap():
    with TestClient(app,base_url='http://localhost',client=('127.0.0.1',5000)) as local:
        assert local.post('/api/local-session').status_code==200
        assert local.get('/api/projects').status_code==200
    with TestClient(app,base_url='http://localhost',client=('192.0.2.10',5000)) as remote:
        assert remote.post('/api/local-session').status_code==403
        assert remote.get('/api/projects').status_code==401

def test_private_upload_job_and_delete(footage,monkeypatch):
    with TestClient(app) as owner,TestClient(app) as outsider:
        assert owner.post('/api/register',json={'email':'one@example.test','password':'test-password-123'}).status_code==200
        p=owner.post('/api/projects',json={'name':'Integration test'}).json()['id']
        # Exercise the guard deterministically without depending on the host's free space.
        monkeypatch.setattr('backend.main.shutil.disk_usage',lambda path:SimpleNamespace(free=0))
        with footage.open('rb') as f:
            assert owner.post(f'/api/projects/{p}/assets',files={'file':('test.mp4',f,'video/mp4')}).status_code==507
        monkeypatch.setattr('backend.main.shutil.disk_usage',lambda path:SimpleNamespace(free=512*1024*1024))
        response=owner.put(f'/api/projects/{p}/assets?filename=test.mp4',content=footage.read_bytes(),headers={'Content-Type':'application/octet-stream'})
        assert response.status_code==200,response.text
        asset=response.json()['id']
        assert owner.get(f'/api/projects/{p}/assets/{asset}/file',headers={'Range':'bytes=0-99'}).status_code==206
        assert outsider.get(f'/api/projects/{p}').status_code==401
        outsider.post('/api/register',json={'email':'two@example.test','password':'test-password-123'})
        assert outsider.get(f'/api/projects/{p}').status_code==404
        assert outsider.get(f'/api/projects/{p}/assets/{asset}/file').status_code==404
        assert outsider.get(f'/api/projects/{p}/provenance').status_code==404
        assert owner.post(f'/api/projects/{p}/assets',files={'file':('bad.mp4',b'broken','video/mp4')}).status_code==422
        from backend.config import settings
        original=settings.gemini_api_key
        settings.gemini_api_key='configured-for-queue-test'
        try:
            options={'style':'high-energy','target_duration':0,'process_consent':True}
            assert owner.post(f'/api/projects/{p}/jobs',json=options).status_code==200
            assert owner.post(f'/api/projects/{p}/jobs',json=options).status_code==409
            assert owner.delete(f'/api/projects/{p}').status_code==409
            with session() as db:
                db.query(Job).filter_by(project_id=p).update({'status':'FAILED'})
                db.get(Project,p).status='FAILED'
            assert owner.delete(f'/api/projects/{p}').status_code==200
            assert owner.get(f'/api/projects/{p}').status_code==404
        finally: settings.gemini_api_key=original
