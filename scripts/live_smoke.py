"""Explicit live-provider test; uses synthetic footage, never pretends it is gym footage.
Run: python -m scripts.live_smoke
Consumes API credits. Keeps full provenance in a clearly named test project.
"""
import time
from pathlib import Path
from backend.config import settings
from backend.db import init_db,session,User,Project,Asset,Job
from backend.main import password_hash
from backend.media import ffmpeg,probe
from backend.worker import run_once
import hashlib
import secrets

def main():
    init_db()
    with session() as db:
        user=db.query(User).filter_by(email='smoke-test@local.invalid').first()
        if not user:
            user=User(email='smoke-test@local.invalid',password=password_hash(secrets.token_urlsafe(32)))
            db.add(user);db.flush()
        project=Project(user_id=user.id,name='Synthetic pipeline verification',style='high-energy',target_duration=4,status='QUEUED')
        db.add(project);db.flush()
        pid=project.id
    directory=settings.data_dir/'projects'/pid/'originals'
    directory.mkdir(parents=True,exist_ok=True)
    for i,source in enumerate(['testsrc2=size=640x360:rate=30','smptebars=size=360x640:rate=30']):
        path=directory/f'fixture-{i}.mp4'
        ffmpeg('-f','lavfi','-i',source,'-t',3,'-c:v','libx264','-pix_fmt','yuv420p',path)
        meta=probe(path)
        with session() as db: db.add(Asset(project_id=pid,filename=path.name,path=str(path),kind='video',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),duration=meta['duration'],meta=meta))
    with session() as db:
        job=Job(project_id=pid)
        db.add(job);db.flush()
        jid=job.id
    run_once()
    with session() as db:
        job=db.get(Job,jid)
        print({'project_id':pid,'job_status':job.status,'stage':job.stage,'error':job.error,'metrics':job.metrics})
        if job.status!='COMPLETED': raise SystemExit(1)

if __name__=='__main__': main()
