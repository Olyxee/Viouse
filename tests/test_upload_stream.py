from types import SimpleNamespace
from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest
from backend.main import app
from backend.config import settings
from backend.uploads import IncomingAsset,RESERVE

def test_raw_stream_preserves_bytes_and_preflights_without_temp_copy(monkeypatch):
    monkeypatch.setattr('backend.uploads.shutil.disk_usage',lambda path:SimpleNamespace(free=RESERVE+10))
    monkeypatch.setattr('backend.uploads.probe',lambda path:{'duration':2,'has_video':True,'has_audio':False})
    with TestClient(app,base_url='http://localhost',client=('127.0.0.1',5000)) as client:
        client.post('/api/local-session')
        project=client.post('/api/projects',json={'name':'Streaming upload test'}).json()['id']
        base=f'/api/projects/{project}/assets?filename=Walking.mp4'
        too_large=client.put(base,content=b'12345678901',headers={'Content-Type':'application/octet-stream'})
        assert too_large.status_code==507
        assert 'original file is unchanged' in too_large.json()['detail']
        folder=settings.data_dir/'projects'/project/'originals'
        assert not list(folder.iterdir())
        result=client.put(base,content=b'1234567890',headers={'Content-Type':'application/octet-stream'})
        assert result.status_code==200,result.text
        saved=list(folder.iterdir())
        assert len(saved)==1 and saved[0].read_bytes()==b'1234567890'
        client.delete('/api/projects/'+project)

def test_mid_stream_failure_removes_partial_file(monkeypatch):
    monkeypatch.setattr('backend.uploads.shutil.disk_usage',lambda path:SimpleNamespace(free=RESERVE+100))
    with TestClient(app,base_url='http://localhost',client=('127.0.0.1',5000)) as client:
        client.post('/api/local-session')
        project=client.post('/api/projects',json={'name':'Interrupted upload'}).json()['id']
        upload=IncomingAsset(project,'local-workspace','Walking.mp4','video')
        upload.write(b'first bytes')
        monkeypatch.setattr('backend.uploads.shutil.disk_usage',lambda path:SimpleNamespace(free=0))
        with pytest.raises(HTTPException) as error:
            upload.write(b'next bytes')
        assert error.value.status_code==507
        upload.close()
        assert not upload.path.exists()
        client.delete('/api/projects/'+project)
