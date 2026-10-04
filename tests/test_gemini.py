import json
from types import SimpleNamespace
from unittest.mock import patch
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.gemini import generate,GeminiError,json_schema
from backend.main import app,ConnectionProbe
from backend.audio import gemini_transcribe
from backend.schemas import SpeechTranscript

def test_wire_schema_avoids_decoder_expansion_but_retains_local_limits():
    from backend.schemas import Plan
    from pydantic import ValidationError
    wire=json_schema(Plan)
    assert 'maxItems' not in wire['properties']['story']
    assert Plan.model_json_schema()['properties']['story']['maxItems']==50
    with pytest.raises(ValidationError):
        Plan.model_validate({'title':'Test','style':'cinematic','target_duration':1,'story':[],'rejected':[]})

def response(text='{"status":"ok"}',finish='STOP'):
    return {'candidates':[{'finishReason':finish,'content':{'parts':[{'text':text}]}}],'usageMetadata':{'totalTokenCount':10}}

def test_gemini_image_and_audio_request_uses_headers_and_schema(tmp_path):
    image=tmp_path/'sample.jpg';image.write_bytes(b'jpeg-test')
    audio=tmp_path/'sample.wav';audio.write_bytes(b'wav-test')
    seen=[]
    def handler(request):
        seen.append(request)
        return httpx.Response(200,json=response())
    real_client=httpx.Client
    with patch('backend.gemini.httpx.Client',side_effect=lambda **kwargs:real_client(transport=httpx.MockTransport(handler),**kwargs)):
        result=generate('test-private-key','gemini-3.8-flash',ConnectionProbe,'check',[image],audio)
    assert result.status=='ok'
    req=seen[0]
    assert 'test-private-key' not in str(req.url)
    assert req.headers['x-goog-api-key']=='test-private-key'
    body=json.loads(req.content)
    assert body['generationConfig']['responseJsonSchema']['properties']['status']['enum']==['ok']
    assert [p['inlineData']['mimeType'] for p in body['contents'][0]['parts'] if 'inlineData' in p]==['image/jpeg','audio/wav']

def test_gemini_repairs_invalid_json_and_never_accepts_blocked_output():
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(200,json=response('not JSON' if len(calls)==1 else '{"status":"ok"}'))
    real_client=httpx.Client
    with patch('backend.gemini.httpx.Client',side_effect=lambda **kwargs:real_client(transport=httpx.MockTransport(handler),**kwargs)):
        assert generate('test','gemini-test',ConnectionProbe,'check').status=='ok'
    assert len(calls)==2
    with patch('backend.gemini.httpx.Client',side_effect=lambda **kwargs:real_client(transport=httpx.MockTransport(lambda req:httpx.Response(200,json=response(finish='SAFETY'))),**kwargs)):
        with pytest.raises(ValueError,match='complete response'): generate('test','gemini-test',ConnectionProbe,'check')

def test_gemini_quota_error_does_not_echo_secrets_or_fallback():
    real_client=httpx.Client
    with patch('backend.gemini.httpx.Client',side_effect=lambda **kwargs:real_client(transport=httpx.MockTransport(lambda req:httpx.Response(429,json={'error':{'message':'private-key-in-upstream-error'}})),**kwargs)):
        with pytest.raises(GeminiError,match='usage limit') as error: generate('private-key','gemini-test',ConnectionProbe,'check')
    assert 'private-key' not in str(error.value)

def test_gemini_overload_uses_bounded_backoff_and_recovers():
    attempts=[]
    def handler(request):
        attempts.append(request)
        return httpx.Response(503,json={'error':{}}) if len(attempts)<5 else httpx.Response(200,json=response())
    real_client=httpx.Client
    with patch('backend.gemini.time.sleep') as sleep, patch('backend.gemini.httpx.Client',side_effect=lambda **kwargs:real_client(transport=httpx.MockTransport(handler),**kwargs)):
        assert generate('test','gemini-test',ConnectionProbe,'check').status=='ok'
    assert [call.args[0] for call in sleep.call_args_list]==[5,10,20,40]
    assert len(attempts)==5

def test_gemini_overload_exhaustion_preserves_actionable_error():
    real_client=httpx.Client
    with patch('backend.gemini.time.sleep'), patch('backend.gemini.httpx.Client',side_effect=lambda **kwargs:real_client(transport=httpx.MockTransport(lambda req:httpx.Response(503,json={'error':{}})),**kwargs)):
        with pytest.raises(GeminiError,match='HTTP 503'):
            generate('test','gemini-test',ConnectionProbe,'check')

def test_gemini_transcript_validates_source_timestamps_and_filters_uncertain_speech(tmp_path):
    asset=SimpleNamespace(id='audio',duration=2)
    ai=SimpleNamespace(calls=[])
    transcript=SpeechTranscript(classification='speech',segments=[{'start':.2,'end':1,'text':'One more rep','confidence':.95},{'start':1.1,'end':1.8,'text':'Uncertain words','confidence':.4}])
    with patch('backend.gemini.generate',return_value=transcript):
        result=gemini_transcribe(asset,tmp_path,ai,np.zeros(32000,dtype=np.int16),16000,.1)
    assert result['segments']==[{'start':.2,'end':1.,'text':'One more rep'}]
    transcript.segments[0].end=8
    with patch('backend.gemini.generate',return_value=transcript):
        with pytest.raises(ValueError,match='timestamps'): gemini_transcribe(asset,tmp_path,ai,np.zeros(32000,dtype=np.int16),16000,.1)
    assert not list(tmp_path.glob('*.wav'))

def test_local_key_setup_checks_connection_saves_privately_and_requires_local_access(tmp_path,monkeypatch):
    import backend.main as main
    from backend.config import settings
    monkeypatch.setattr(main,'ROOT',tmp_path)
    monkeypatch.setattr(settings,'gemini_api_key','')
    monkeypatch.setattr(settings,'ai_provider','gemini')
    with TestClient(app,base_url='http://localhost',client=('127.0.0.1',5100)) as client:
        assert client.post('/api/settings/gemini',json={'api_key':'test-gemini-private-key'}).status_code==401
        client.post('/api/local-session')
        with patch('backend.gemini.generate',side_effect=GeminiError(403)):
            denied=client.post('/api/settings/gemini',json={'api_key':'test-gemini-private-key'})
            assert denied.status_code==502
            assert not (tmp_path/'.env').exists()
        with patch('backend.gemini.generate',return_value=ConnectionProbe(status='ok')):
            result=client.post('/api/settings/gemini',json={'api_key':'test-gemini-private-key'})
        assert result.status_code==200
        assert 'test-gemini-private-key' not in result.text
        assert 'test-gemini-private-key' in (tmp_path/'.env').read_text()

@pytest.fixture(autouse=True)
def single_model(monkeypatch,request):
    import backend.gemini as g
    monkeypatch.setattr(g,'EXHAUSTED',{})
    if 'fallback' not in request.node.name: monkeypatch.setattr(g,'FALLBACK_MODELS',())

def test_gemini_daily_quota_falls_back_to_next_model(monkeypatch):
    import backend.gemini as g
    monkeypatch.setattr(g,'FALLBACK_MODELS',('backup-model',))
    seen=[]
    def handler(request):
        seen.append(request.url.path)
        if 'primary' in request.url.path:
            return httpx.Response(429,json={'error':{'details':[{'retryDelay':'11260s'}]}})
        return httpx.Response(200,json=response())
    real_client=httpx.Client
    with patch('backend.gemini.time.sleep'), patch('backend.gemini.httpx.Client',side_effect=lambda **kwargs:real_client(transport=httpx.MockTransport(handler),**kwargs)):
        assert generate('test','primary',ConnectionProbe,'check').status=='ok'
    assert any('backup-model' in p for p in seen)
