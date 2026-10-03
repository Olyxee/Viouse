import time
import uuid
from contextlib import contextmanager
from sqlalchemy import create_engine, String, Float, JSON, ForeignKey, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from .config import settings

def uid(): return uuid.uuid4().hex

class Base(DeclarativeBase): pass

class User(Base):
    __tablename__ = 'users'
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    email: Mapped[str] = mapped_column(String, unique=True)
    password: Mapped[str] = mapped_column(String)

class Session(Base):
    __tablename__ = 'sessions'
    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'))
    expires: Mapped[float] = mapped_column(Float)

class Project(Base):
    __tablename__ = 'projects'
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'))
    name: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default='UPLOADING')
    style: Mapped[str] = mapped_column(String, default='automatic')
    target_duration: Mapped[float] = mapped_column(Float, default=45)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)

class Asset(Base):
    __tablename__ = 'assets'
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    project_id: Mapped[str] = mapped_column(ForeignKey('projects.id'), index=True)
    filename: Mapped[str] = mapped_column(String)
    path: Mapped[str] = mapped_column(String)
    kind: Mapped[str] = mapped_column(String)
    sha256: Mapped[str] = mapped_column(String)
    duration: Mapped[float] = mapped_column(Float)
    meta: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)

class Job(Base):
    __tablename__ = 'render_jobs'
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    project_id: Mapped[str] = mapped_column(ForeignKey('projects.id'), index=True)
    status: Mapped[str] = mapped_column(String, default='QUEUED')
    stage: Mapped[str] = mapped_column(String, default='QUEUED')
    detail: Mapped[str] = mapped_column(String, default='Waiting for worker')
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    heartbeat: Mapped[float] = mapped_column(Float, default=time.time)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)

# Dedicated, queryable provenance tables share a document envelope. Payloads are
# validated by pipeline schemas before persistence; originals are never overwritten.
def document_model(name):
    return type(name.title().replace('_', ''), (Base,), {
        '__tablename__': name,
        '__annotations__': {'id': Mapped[str], 'project_id': Mapped[str], 'job_id': Mapped[str | None], 'payload': Mapped[dict], 'created_at': Mapped[float]},
        'id': mapped_column(String, primary_key=True, default=uid),
        'project_id': mapped_column(ForeignKey('projects.id'), index=True),
        'job_id': mapped_column(String, nullable=True, index=True),
        'payload': mapped_column(JSON),
        'created_at': mapped_column(Float, default=time.time),
    })

DOCUMENTS = {name: document_model(name) for name in (
    'segments', 'analysis_results', 'event_groups', 'transcripts', 'edit_plans',
    'edit_decisions', 'video_versions', 'critic_results', 'feedback', 'workouts',
    'clip_understanding', 'moments', 'concepts', 'timelines')}

class CreatorProfile(Base):
    """Learned editing preferences for one creator (from reference edits and feedback)."""
    __tablename__ = 'creator_profiles'
    id: Mapped[str] = mapped_column(String, primary_key=True, default=uid)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id'), index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)

engine = create_engine(settings.database_url, connect_args={'check_same_thread': False, 'timeout': 30} if settings.database_url.startswith('sqlite') else {}, pool_pre_ping=True)
if settings.database_url.startswith('sqlite'):
    @event.listens_for(engine, 'connect')
    def sqlite_setup(connection, _):
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('PRAGMA foreign_keys=ON')
DB = sessionmaker(engine, expire_on_commit=False)

def init_db(): Base.metadata.create_all(engine)

@contextmanager
def session():
    with DB() as db:
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise

def save_doc(table, project_id, payload, job_id=None, id=None):
    with session() as db:
        record = DOCUMENTS[table](project_id=project_id, payload=payload, job_id=job_id, **({'id': id} if id else {}))
        db.add(record)
        db.flush()
        return record.id

def docs(table, project_id):
    with session() as db:
        return db.query(DOCUMENTS[table]).filter_by(project_id=project_id).order_by(DOCUMENTS[table].created_at).all()
