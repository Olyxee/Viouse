"""Durable database queue. Run separately from HTTP: python -m backend.worker."""
import logging
import threading
import time
from sqlalchemy import update as sql_update
from .db import init_db, session, Job, Project
from .pipeline import process, process_caption_job
from .config import settings
from .gemini import GeminiError

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(settings.data_dir/'worker.log'),logging.StreamHandler()])
log = logging.getLogger(__name__)

def heartbeat(job_id, stop):
    while not stop.wait(10):
        try:
            with session() as db:
                job = db.get(Job,job_id)
                if job and job.status=='RUNNING': job.heartbeat=time.time()
        except Exception: log.exception('Heartbeat failed')

def claim():
    with session() as db:
        # Expired leases are recoverable, never reported as completed.
        for stale in db.query(Job).filter(Job.status=='RUNNING',Job.heartbeat < time.time()-120).all():
            stale.status,stale.error = 'FAILED','The worker stopped. Retry to continue using saved analysis.'
            db.get(Project,stale.project_id).status='FAILED'
        job = db.query(Job).filter_by(status='QUEUED').order_by(Job.created_at).first()
        if not job: return None
        claimed = db.execute(sql_update(Job).where(Job.id==job.id,Job.status=='QUEUED').values(status='RUNNING',heartbeat=time.time()))
        return job.id if claimed.rowcount==1 else None

def run_once():
    job_id=claim()
    if not job_id: return False
    stop=threading.Event()
    thread=threading.Thread(target=heartbeat,args=(job_id,stop),daemon=True)
    thread.start()
    try:
        with session() as db: caption_only=db.get(Job,job_id).metrics.get('kind')=='captions'
        (process_caption_job if caption_only else process)(job_id)
    except Exception as exc:
        log.exception('Job %s failed',job_id)
        name=type(exc).__name__
        message = 'We could not finish this edit. Your footage is saved; you can retry.'
        if name in ('AuthenticationError','PermissionDeniedError'): message='The AI service credentials need attention. Update the server API key, then retry.'
        elif name=='RateLimitError': message='The AI service is temporarily unavailable or out of quota. Check the API account, then retry.'
        elif isinstance(exc,GeminiError): message=str(exc)
        elif isinstance(exc,ValueError): message='The edit did not pass validation. Retry to request a new valid plan.'
        with session() as db:
            job=db.get(Job,job_id)
            job.status,job.error='FAILED',message
            job.detail='Processing stopped safely'
            db.get(Project,job.project_id).status='FAILED'
    finally:
        stop.set()
        thread.join(timeout=2)
    return True

if __name__=='__main__':
    init_db()
    log.info('Worker ready')
    while True:
        if not run_once(): time.sleep(2)
