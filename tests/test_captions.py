from pathlib import Path
from types import SimpleNamespace
import numpy as np
import cv2
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient
from backend.captions import CaptionSettings,checked_settings,write_captions,person_mask,MODEL_PATH
from backend.media import ffmpeg,probe
from backend.render import render
from backend.main import app
from backend.db import session,Job,Project,save_doc
from backend.config import settings

def custom(style='pop'):
    return {'mode':'custom','style':style,'position':'center','color':'#DFFF52','lines':[{'start':0,'end':1,'text':'MAKE IT POP'}]}

def test_caption_validation_and_safe_animation(tmp_path):
    with pytest.raises(ValidationError): CaptionSettings.model_validate({**custom(),'lines':[{'start':1,'end':.5,'text':'Bad'}]})
    with pytest.raises(ValueError,match='within'): checked_settings(custom(),.5)
    with pytest.raises(ValidationError): CaptionSettings.model_validate({**custom(),'color':'red;bad'})
    with pytest.raises(ValidationError): CaptionSettings.model_validate({**custom(),'lines':[{'start':0,'end':2,'text':'A'},{'start':1,'end':3,'text':'B'}]})
    for style in ('pop','karaoke','comic','behind'):
        path=tmp_path/(style+'.ass')
        write_captions([{'start':0,'end':2,'text':r'Hello {\pos(1,1)} creator'}],path,{**custom(style),'lines':[],'mode':'auto'})
        text=path.read_text()
        assert r'{\pos(1,1)}' not in text
        assert 'Dialogue:' in text and '\\t(' in text
        if style=='karaoke': assert '\\kf' in text

def test_real_caption_render_and_foreground_occlusion(tmp_path,monkeypatch):
    source=tmp_path/'source.mp4'
    ffmpeg('-f','lavfi','-i','color=c=0x304050:s=270x480:r=30','-t','1','-c:v','libx264','-pix_fmt','yuv420p',source)
    asset=SimpleNamespace(id='a',path=str(source),duration=1,meta=probe(source))
    edl={'duration':1,'captions':custom(),'timeline':[{'asset_id':'a','source_start':0,'source_end':1,'duration':1,'timeline_start':0,'speed':1,'crop':{'mode':'fit','points':[]}}]}
    regular,_=render(edl,[asset],tmp_path/'pop',width=270,height=480)
    # Explicit synthetic all-foreground mask tests the compositing order independently.
    model=tmp_path/'test.onnx';model.touch();monkeypatch.setattr('backend.captions.MODEL_PATH',model)
    def foreground(video,output):
        ffmpeg('-f','lavfi','-i','color=c=white:s=270x480:r=30','-t','1','-c:v','libx264',output)
    monkeypatch.setattr('backend.render.person_mask',foreground)
    edl['captions']=custom('behind')
    hidden,_=render(edl,[asset],tmp_path/'behind',width=270,height=480)
    def frame(path):
        cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_MSEC,400);ok,image=cap.read();cap.release();assert ok;return image
    normal,occluded=frame(regular),frame(hidden)
    assert normal.std()>occluded.std()+3
    assert not (tmp_path/'behind/person-mask.mp4').exists()
    assert (tmp_path/'behind/captions.ass').exists()

@pytest.mark.skipif(not MODEL_PATH.is_file(),reason='Optional local model not installed')
def test_real_person_model_runs_on_video(tmp_path):
    source=tmp_path/'source.mp4'
    ffmpeg('-f','lavfi','-i','testsrc2=s=270x480:r=30','-t','0.2','-c:v','libx264',source)
    mask=tmp_path/'mask.mp4'
    assert person_mask(source,mask)==6
    assert probe(mask)['width']==270

def test_caption_job_ownership_and_persisted_settings():
    with TestClient(app,base_url='http://localhost',client=('127.0.0.1',5000)) as client:
        client.post('/api/local-session')
        p=client.post('/api/projects',json={'name':'Subtitle API test'}).json()['id']
        edl=save_doc('edit_decisions',p,{'duration':2,'timeline':[]})
        v=save_doc('video_versions',p,{'duration':2,'edl_id':edl,'critic':{'overall':.7}})
        route=f'/api/projects/{p}/captions'
        assert client.post(route,json={'version_id':'missing','captions':custom()}).status_code==404
        assert client.post(route,json={'version_id':v,'captions':{'mode':'auto'}}).status_code==422
        result=client.post(route,json={'version_id':v,'captions':custom()})
        assert result.status_code==200,result.text
        with session() as db:
            job=db.get(Job,result.json()['id'])
            assert job.metrics['captions']['lines'][0]['text']=='MAKE IT POP'
            assert job.metrics['kind']=='captions'
        assert client.post(route,json={'version_id':v,'captions':custom()}).status_code==409
        with TestClient(app) as outsider:
            outsider.post('/api/register',json={'email':'subtitle-outsider@example.test','password':'test-password-123'})
            assert outsider.post(route,json={'version_id':v,'captions':custom()}).status_code==404
        with session() as db:
            db.get(Job,result.json()['id']).status='FAILED'
            db.get(Project,p).status='FAILED'
        assert client.delete('/api/projects/'+p).status_code==200
