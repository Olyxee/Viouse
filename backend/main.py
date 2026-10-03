import hashlib
import hmac
import secrets
import shutil
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, Depends, HTTPException, Request, Response, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, SecretStr
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from .config import settings, ROOT, binary, active_key
from .db import init_db, session, User, Session, Project, Asset, Job, DOCUMENTS, docs, save_doc, uid
from .media import probe
from .storage import publish_private, delete_project_objects
from .uploads import IncomingAsset, BUSY
from .captions import CaptionSettings, checked_settings, MODEL_PATH

@asynccontextmanager
async def lifespan(app):
    init_db()
    if settings.local_mode:
        with session() as db:
            if not db.get(User,'local-workspace'):
                db.add(User(id='local-workspace',email='workspace@localhost',password=password_hash(secrets.token_urlsafe(48))))
    yield

app=FastAPI(title='Autonomous Editor',lifespan=lifespan)

@app.middleware('http')
async def browser_security(request,call_next):
    if request.url.path.endswith('/assets') and request.method in ('POST','PUT'):
        try: length=int(request.headers.get('content-length','0'))
        except ValueError: return Response(status_code=400)
        if length > settings.max_upload_mb*1024*1024+1024*1024:
            from fastapi.responses import JSONResponse
            return JSONResponse({'detail':f'Maximum file size is {settings.max_upload_mb} MB'},status_code=413)
    if request.method not in ('GET','HEAD','OPTIONS') and request.headers.get('sec-fetch-site')=='cross-site':
        return Response(status_code=403)
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Referrer-Policy']='same-origin'
    response.headers['X-Frame-Options']='DENY'
    if request.url.path.startswith('/api'): response.headers['Cache-Control']='no-store'
    return response

def current_user(request:Request):
    token=request.cookies.get('editor_session','')
    with session() as db:
        login=db.get(Session,hashlib.sha256(token.encode()).hexdigest())
        if not login or login.expires < time.time(): raise HTTPException(401,'Sign in to continue')
        user=db.get(User,login.user_id)
        if not user: raise HTTPException(401,'Sign in to continue')
        return user

def owned(project_id,user):
    with session() as db:
        p=db.get(Project,project_id)
        if not p or p.user_id!=user.id: raise HTTPException(404,'Project not found')
        return p

def password_hash(password,salt=None):
    salt=salt or secrets.token_hex(16)
    value=hashlib.scrypt(password.encode(),salt=salt.encode(),n=16384,r=8,p=1).hex()
    return salt+':'+value

class Credentials(BaseModel):
    email:str=Field(min_length=3,max_length=254)
    password:str=Field(min_length=10,max_length=200)

def sign_in(response,user_id):
    token=secrets.token_urlsafe(48)
    with session() as db:
        db.add(Session(id=hashlib.sha256(token.encode()).hexdigest(),user_id=user_id,expires=time.time()+86400*7))
    response.set_cookie('editor_session',token,httponly=True,secure=settings.cookie_secure,samesite='strict',max_age=86400*7)

@app.post('/api/register')
def register(body:Credentials,response:Response):
    if not settings.allow_registration: raise HTTPException(403,'Registration is disabled')
    if '@' not in body.email: raise HTTPException(422,'Enter a valid email address')
    try:
        with session() as db:
            user=User(email=body.email.strip().lower(),password=password_hash(body.password))
            db.add(user)
            db.flush()
            user_id=user.id
    except IntegrityError: raise HTTPException(409,'This email is already registered')
    sign_in(response,user_id)
    return {'email':body.email}

@app.post('/api/login')
def login(body:Credentials,response:Response):
    with session() as db:
        user=db.query(User).filter_by(email=body.email.strip().lower()).first()
    salt=user.password.split(':')[0] if user else '0'*32
    candidate=password_hash(body.password,salt)
    if not user or not hmac.compare_digest(candidate,user.password): raise HTTPException(401,'Email or password is incorrect')
    sign_in(response,user.id)
    return {'email':user.email}

@app.post('/api/logout')
def logout(request:Request,response:Response):
    with session() as db:
        db.query(Session).filter_by(id=hashlib.sha256(request.cookies.get('editor_session','').encode()).hexdigest()).delete()
    response.delete_cookie('editor_session')
    return {'ok':True}

@app.get('/api/me')
def me(user=Depends(current_user)): return {'email':user.email}

@app.post('/api/local-session')
def local_session(request:Request,response:Response):
    # Deliberately single-user and loopback-only; never use this mode on a public server.
    if not settings.local_mode or not request.client or request.client.host not in ('127.0.0.1','::1') or request.url.hostname not in ('127.0.0.1','localhost','::1'):
        raise HTTPException(403,'Local workspace access is only available on this computer')
    sign_in(response,'local-workspace')
    return {'email':'workspace@localhost','local':True}

@app.get('/api/health')
def health():
    try:
        binary('ffmpeg');binary('ffprobe')
        media=True
    except RuntimeError: media=False
    return {'ok':True,'media_ready':media,'ai_configured':bool(active_key()),'provider':settings.ai_provider,'depth_captions_ready':MODEL_PATH.is_file()}

class GeminiSetup(BaseModel):
    api_key:SecretStr=Field(min_length=20,max_length=256)

class ConnectionProbe(BaseModel):
    status:Literal['ok']

@app.post('/api/settings/gemini')
def configure_gemini(body:GeminiSetup,request:Request,user=Depends(current_user)):
    if not settings.local_mode or not request.client or request.client.host not in ('127.0.0.1','::1') or request.url.hostname not in ('localhost','127.0.0.1','::1'):
        raise HTTPException(403,'Keys can only be configured from this computer')
    with session() as db:
        if db.query(Job).filter(Job.status.in_(('QUEUED','RUNNING'))).first():
            raise HTTPException(409,'Wait for current edits to finish before changing the key')
    from .gemini import generate,GeminiError
    from dotenv import set_key
    key=body.api_key.get_secret_value().strip()
    try:
        generate(key,settings.gemini_model,ConnectionProbe,'Return status ok to confirm the connection. No other content.')
    except GeminiError as exc: raise HTTPException(502,str(exc)) from None
    except ValueError: raise HTTPException(502,'Gemini did not return a valid connection check. Try again.') from None
    set_key(str(ROOT/'.env'),'GEMINI_API_KEY',key)
    set_key(str(ROOT/'.env'),'AI_PROVIDER','gemini')
    settings.gemini_api_key=key
    settings.ai_provider='gemini'
    return {'ok':True,'provider':'gemini','ai_configured':True}

class NewProject(BaseModel):
    name:str=Field(default='Untitled reel',min_length=1,max_length=120)

@app.post('/api/projects')
def create_project(body:NewProject,user=Depends(current_user)):
    with session() as db:
        p=Project(user_id=user.id,name=body.name)
        db.add(p);db.flush()
        return {'id':p.id}

def project_json(p): return {'id':p.id,'name':p.name,'status':p.status,'style':p.style,'target_duration':p.target_duration,'created_at':p.created_at}

@app.get('/api/projects')
def list_projects(user=Depends(current_user)):
    with session() as db:
        return [project_json(p) for p in db.query(Project).filter_by(user_id=user.id).order_by(Project.created_at.desc()).all()]

@app.get('/api/projects/{project_id}')
def get_project(project_id:str,user=Depends(current_user)):
    p=owned(project_id,user)
    with session() as db:
        assets=db.query(Asset).filter_by(project_id=project_id).order_by(Asset.created_at).all()
        job=db.query(Job).filter_by(project_id=project_id).order_by(Job.created_at.desc()).first()
    versions=docs('video_versions',project_id)
    best=job.metrics.get('best_version_id') if job else None
    if not best and versions: best=max(versions,key=lambda v:v.payload['critic']['overall']).id
    return {**project_json(p),'assets':[{'id':a.id,'filename':a.filename,'kind':a.kind,'duration':a.duration,'metadata':a.meta} for a in assets],
            'job':{'id':job.id,'status':job.status,'stage':job.stage,'detail':job.detail,'error':job.error,'kind':job.metrics.get('kind','edit')} if job else None,
            'versions':[{'id':v.id,**{k:value for k,value in v.payload.items() if k not in ('path','critic')}} for v in versions], 'best_version_id':best,
            'captions':job.metrics.get('captions',{}) if job else {}}

@app.post('/api/projects/{project_id}/assets')
def upload(project_id:str,file:UploadFile=File(...),kind:Literal['video','music']='video',user=Depends(current_user)):
    incoming=None
    try:
        incoming=IncomingAsset(project_id,user.id,file.filename or '',kind,file.size)
        while chunk:=file.file.read(1024*1024): incoming.write(chunk)
        return incoming.finish()
    finally:
        if incoming: incoming.close()
        file.file.close()

@app.put('/api/projects/{project_id}/assets')
async def stream_upload(project_id:str,request:Request,filename:str,kind:Literal['video','music']='video',user=Depends(current_user)):
    try: expected=int(request.headers['content-length']) if 'content-length' in request.headers else None
    except ValueError: raise HTTPException(400,'Invalid upload length') from None
    incoming=await run_in_threadpool(IncomingAsset,project_id,user.id,filename,kind,expected)
    try:
        async for chunk in request.stream(): await run_in_threadpool(incoming.write,chunk)
        return await run_in_threadpool(incoming.finish)
    finally: incoming.close()

@app.delete('/api/projects/{project_id}/assets/{asset_id}')
def remove_asset(project_id:str,asset_id:str,user=Depends(current_user)):
    p=owned(project_id,user)
    if p.status in BUSY or docs('edit_plans',project_id): raise HTTPException(409,'Sources are locked after editing; create a new project to change them')
    with session() as db:
        a=db.get(Asset,asset_id)
        if not a or a.project_id!=project_id: raise HTTPException(404,'Asset not found')
        Path(a.path).unlink(missing_ok=True)
        if settings.s3_bucket:
            from .storage import client,key
            client().delete_object(Bucket=settings.s3_bucket,Key=key(a.path))
        db.delete(a)
    return {'ok':True}

class StartEdit(BaseModel):
    style:Literal['automatic','high-energy','cinematic']='automatic'
    target_duration:Literal[0,30,45,60]=45
    process_consent:bool=False
    captions:CaptionSettings=Field(default_factory=CaptionSettings)

@app.post('/api/projects/{project_id}/jobs')
def start_edit(project_id:str,body:StartEdit,user=Depends(current_user)):
    owned(project_id,user)
    if not body.process_consent: raise HTTPException(422,'Consent to AI processing is required')
    if not active_key(): raise HTTPException(503,'Connect Gemini in your workspace before creating a video' if settings.ai_provider=='gemini' else 'Set OPENAI_API_KEY on the server before creating a video')
    try: binary('ffmpeg');binary('ffprobe')
    except RuntimeError as exc: raise HTTPException(503,str(exc))
    with session() as db:
        videos=db.query(Asset).filter_by(project_id=project_id,kind='video').all()
        if not videos: raise HTTPException(422,'Upload footage first')
        if body.target_duration and sum(a.duration for a in videos) < body.target_duration:
            raise HTTPException(422,'Upload more footage for this length, or choose Auto')
        try: checked_settings(body.captions.model_dump(),body.target_duration or min(60,max(30,sum(a.duration for a in videos)*.2),sum(a.duration for a in videos)))
        except ValueError as exc: raise HTTPException(422,str(exc)) from None
        lock=db.execute(update(Project).where(Project.id==project_id,Project.status.notin_(BUSY)).values(status='QUEUED',style=body.style,target_duration=body.target_duration))
        if lock.rowcount!=1: raise HTTPException(409,'An edit is already running')
        job=Job(project_id=project_id,metrics={'captions':body.captions.model_dump()})
        db.add(job);db.flush()
        return {'id':job.id}

class RestyleCaptions(BaseModel):
    version_id:str
    captions:CaptionSettings

@app.post('/api/projects/{project_id}/captions')
def restyle_captions(project_id:str,body:RestyleCaptions,user=Depends(current_user)):
    owned(project_id,user)
    with session() as db:
        version=db.get(DOCUMENTS['video_versions'],body.version_id)
        if not version or version.project_id!=project_id: raise HTTPException(404,'Version not found')
        try: checked_settings(body.captions.model_dump(),version.payload['duration'])
        except ValueError as exc: raise HTTPException(422,str(exc)) from None
        if body.captions.mode=='auto':
            from .render import caption_events
            edl=db.get(DOCUMENTS['edit_decisions'],version.payload['edl_id'])
            transcripts=[d.payload for d in docs('transcripts',project_id)]
            if not edl or not caption_events(edl.payload,transcripts):
                raise HTTPException(422,'No detected speech in this edit. Choose Write my own to add text.')
        lock=db.execute(update(Project).where(Project.id==project_id,Project.status.notin_(BUSY)).values(status='QUEUED'))
        if lock.rowcount!=1: raise HTTPException(409,'An edit is already running')
        job=Job(project_id=project_id,metrics={'kind':'captions','source_version_id':body.version_id,'captions':body.captions.model_dump()})
        db.add(job);db.flush()
        return {'id':job.id}

@app.get('/api/projects/{project_id}/versions/{version_id}/video')
def video(project_id:str,version_id:str,download:bool=False,user=Depends(current_user)):
    owned(project_id,user)
    with session() as db: v=db.get(DOCUMENTS['video_versions'],version_id)
    if not v or v.project_id!=project_id: raise HTTPException(404,'Video not found')
    path=Path(v.payload['path'])
    if not path.exists(): raise HTTPException(404,'Video file is unavailable')
    return FileResponse(path,media_type='video/mp4',filename='reel.mp4' if download else None)

@app.get('/api/projects/{project_id}/assets/{asset_id}/file')
def asset_file(project_id:str,asset_id:str,user=Depends(current_user)):
    owned(project_id,user)
    with session() as db: a=db.get(Asset,asset_id)
    if not a or a.project_id!=project_id: raise HTTPException(404,'Asset not found')
    return FileResponse(a.path)

@app.get('/api/projects/{project_id}/provenance')
def provenance(project_id:str,user=Depends(current_user)):
    owned(project_id,user)
    with session() as db: jobs=db.query(Job).filter_by(project_id=project_id).all()
    return {**{name:[{'id':d.id,'job_id':d.job_id,'payload':d.payload} for d in docs(name,project_id)] for name in DOCUMENTS if name!='feedback'},'metrics':[j.metrics for j in jobs]}

class Feedback(BaseModel):
    version_id:str
    rating:Literal['good','needs-work']
    text:str=Field(default='',max_length=2000)
    research_consent:bool=False

@app.post('/api/projects/{project_id}/feedback')
def feedback(project_id:str,body:Feedback,user=Depends(current_user)):
    owned(project_id,user)
    with session() as db: v=db.get(DOCUMENTS['video_versions'],body.version_id)
    if not v or v.project_id!=project_id: raise HTTPException(404,'Version not found')
    save_doc('feedback',project_id,body.model_dump())
    return {'ok':True}

@app.delete('/api/projects/{project_id}')
def delete_project(project_id:str,user=Depends(current_user)):
    p=owned(project_id,user)
    if p.status in BUSY: raise HTTPException(409,'Wait for processing to finish before deleting')
    directory=(settings.data_dir/'projects'/project_id).resolve()
    if not directory.is_relative_to(settings.data_dir/'projects'): raise HTTPException(400,'Invalid path')
    with session() as db:
        locked=db.execute(update(Project).where(Project.id==project_id,Project.status.notin_(BUSY)).values(status='DELETING'))
        if locked.rowcount!=1: raise HTTPException(409,'Project is processing')
        delete_project_objects(project_id)
        if directory.exists(): shutil.rmtree(directory)
        for table in DOCUMENTS.values(): db.query(table).filter_by(project_id=project_id).delete()
        db.query(Job).filter_by(project_id=project_id).delete()
        db.query(Asset).filter_by(project_id=project_id).delete()
        db.query(Project).filter_by(id=project_id).delete()
    return {'ok':True}

dist=ROOT/'frontend'/'dist'
if dist.exists(): app.mount('/',StaticFiles(directory=dist,html=True),name='frontend')
