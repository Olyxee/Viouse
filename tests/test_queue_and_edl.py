from unittest.mock import patch
from types import SimpleNamespace
import time
import pytest
from backend.db import init_db,session,User,Project,Job
from backend.worker import claim
from backend.schemas import Plan,Analysis
from backend.planner import build_edl
from backend.ai import AI

def test_edl_snaps_beats_and_preserves_exact_duration():
    segments=[{'segment_id':name,'asset_id':name,'source_start':0,'source_end':4,'duration':4,'analysis':{'subject_box':[0,0,1,1]},'track':{'mode':'fit','points':[]},'brightness':.5,'highlight_score':.7} for name in ['a','b']]
    plan=Plan(title='Test',style='high-energy',target_duration=4,story=[{'segment_id':s['segment_id'],'source_start':.5,'source_end':2.5,'role':'hook' if i==0 else 'ending','reason':'Test'} for i,s in enumerate(segments)],rejected=[])
    edl=build_edl(plan,segments,4,beats=[2.1])
    assert edl['timeline'][1]['timeline_start']==pytest.approx(2.1)
    assert sum(s['frames'] for s in edl['timeline'])==120
    assert edl['duration']==4
    assert all(s['source_end']<=4 for s in edl['timeline'])

def test_durable_claim_and_stale_lease_recovery():
    init_db()
    with session() as db:
        user=User(email=f'queue-{time.time()}@test.invalid',password='not-a-login')
        db.add(user);db.flush()
        p=Project(user_id=user.id,name='Queue test',status='QUEUED')
        db.add(p);db.flush()
        j=Job(project_id=p.id)
        db.add(j);db.flush()
        jid,pid=j.id,p.id
    assert claim()==jid
    assert claim() is None
    with session() as db: db.get(Job,jid).heartbeat=time.time()-200
    assert claim() is None
    with session() as db:
        assert db.get(Job,jid).status=='FAILED'
        assert db.get(Project,pid).status=='FAILED'

def test_structured_output_refusal_is_not_replaced_with_fake_analysis():
    ai=AI.__new__(AI)
    ai.provider='openai'
    ai.calls=[]
    response=SimpleNamespace(output_parsed=None,model='test',usage=None)
    with patch('backend.ai.OpenAI'):
        from unittest.mock import Mock
        ai.client=Mock()
        ai.client.responses.parse.return_value=response
        with pytest.raises(ValueError,match='valid Analysis'):
            ai.structured(Analysis,'test')
        assert ai.client.responses.parse.call_count==2
