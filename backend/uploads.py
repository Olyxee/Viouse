"""Write uploads once, retaining a reserve and removing incomplete files on failure."""
import hashlib
import shutil
from pathlib import Path
from fastapi import HTTPException
from sqlalchemy import update
from .config import settings
from .db import session,Project,Asset,uid
from .media import probe
from .storage import publish_private

BUSY=('QUEUED','PREPROCESSING','ANALYZING','PLANNING','EDITING','RENDERING','REVIEWING','DELETING')
RESERVE=64*1024*1024

def check_space(directory,additional=0):
    free=shutil.disk_usage(directory).free
    needed=RESERVE+additional
    if free<needed:
        shortfall=max(1,(needed-free+1048575)//1048576)
        raise HTTPException(507,f'Not enough space to save this clip. Free at least {shortfall} MB on the storage drive, then retry. Your original file is unchanged.')

class IncomingAsset:
    def __init__(self,project_id,user_id,filename,kind,expected_size=None):
        self.project_id,self.kind=project_id,kind
        self.filename=Path(filename.replace('\\','/')).name
        suffix=Path(self.filename).suffix.lower()
        if suffix not in (('.mp4','.mov','.webm') if kind in ('video','reference') else ('.mp3','.wav','.m4a','.aac','.ogg','.flac')):
            raise HTTPException(415,'Unsupported file type')
        with session() as db:
            project=db.get(Project,project_id)
            if not project or project.user_id!=user_id: raise HTTPException(404,'Project not found')
            if project.status in BUSY: raise HTTPException(409,'Wait for the current edit to finish')
            if kind=='video' and db.query(Asset).filter_by(project_id=project_id,kind='video').count()>=30:
                raise HTTPException(422,'Maximum 30 video clips per project')
        self.directory=settings.data_dir/'projects'/project_id/'originals'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.limit=settings.max_upload_mb*1024*1024
        if expected_size is not None and (expected_size<0 or expected_size>self.limit):
            raise HTTPException(413,f'Maximum file size is {settings.max_upload_mb} MB')
        check_space(self.directory,expected_size or 0)
        self.expected_size=expected_size
        self.id=uid()
        self.path=self.directory/(self.id+suffix)
        self.file=self.path.open('wb')
        self.size=0
        self.checksum=hashlib.sha256()
        self.committed=False

    def write(self,chunk):
        if not chunk: return
        if self.size+len(chunk)>self.limit: raise HTTPException(413,f'Maximum file size is {settings.max_upload_mb} MB')
        check_space(self.directory,len(chunk))
        try: self.file.write(chunk)
        except OSError as exc:
            if exc.errno==28: raise HTTPException(507,'The storage drive filled during upload. Free space and retry; the partial upload will be removed.') from None
            raise
        self.size+=len(chunk)
        self.checksum.update(chunk)

    def finish(self):
        self.file.close()
        if self.expected_size is not None and self.size!=self.expected_size:
            raise HTTPException(422,'The upload was interrupted. Please try the file again.')
        try: metadata=probe(self.path)
        except Exception: raise HTTPException(422,'This file could not be read. Try exporting it as MP4 or WAV.') from None
        if metadata['duration']<.5: raise HTTPException(422,'Clips must be at least half a second long')
        if self.kind in ('video','reference') and not metadata['has_video']: raise HTTPException(422,'The file contains no video')
        if self.kind=='music' and not metadata['has_audio']: raise HTTPException(422,'The file contains no audio')
        digest=self.checksum.hexdigest()
        with session() as db:
            lock=db.execute(update(Project).where(Project.id==self.project_id,Project.status.notin_(BUSY)).values(status='UPLOADING'))
            if lock.rowcount!=1: raise HTTPException(409,'An edit has started; upload again when it finishes')
            existing=db.query(Asset).filter_by(project_id=self.project_id).all()
            if self.kind=='video' and sum(a.kind=='video' for a in existing)>=30: raise HTTPException(422,'Maximum 30 video clips per project')
            total=sum(a.duration for a in existing if a.kind=='video')+(metadata['duration'] if self.kind=='video' else 0)
            if total>settings.max_project_minutes*60 or metadata['duration']>settings.max_project_minutes*60:
                raise HTTPException(422,f'Maximum {settings.max_project_minutes} minutes per project')
            if any(a.sha256==digest and a.kind==self.kind for a in existing): raise HTTPException(409,'This file is already in the project')
            publish_private(self.path)
            db.add(Asset(id=self.id,project_id=self.project_id,filename=self.filename,path=str(self.path),kind=self.kind,sha256=digest,duration=metadata['duration'],meta=metadata))
        self.committed=True
        return {'id':self.id,'duration':metadata['duration']}

    def close(self):
        self.file.close()
        if not self.committed: self.path.unlink(missing_ok=True)
