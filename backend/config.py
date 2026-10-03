from pathlib import Path
from typing import Literal
import shutil
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / '.env', extra='ignore')
    openai_api_key: str = ''
    ai_provider: Literal['gemini','openai'] = 'gemini'
    gemini_api_key: str = ''
    gemini_model: str = 'gemini-2.5-flash'
    vision_model: str = 'gpt-4.1'
    planner_model: str = 'gpt-4.1'
    database_url: str = 'sqlite:///./data/editor.db'
    data_dir: Path = ROOT / 'data'
    s3_bucket: str = ''
    s3_endpoint_url: str = ''
    cookie_secure: bool = False
    allow_registration: bool = True
    local_mode: bool = True
    max_upload_mb: int = 2048
    max_project_minutes: int = 60
    ffmpeg_path: str = ''
    ffprobe_path: str = ''

# A project-local key deliberately overrides an inherited desktop-app key.
# Other runtime settings retain normal environment precedence (including tests).
local_values = dotenv_values(ROOT / '.env')
settings = Settings(**{name.lower():local_values[name] for name in ('OPENAI_API_KEY','GEMINI_API_KEY','AI_PROVIDER','GEMINI_MODEL') if local_values.get(name) is not None})
settings.data_dir = settings.data_dir.resolve()
settings.data_dir.mkdir(parents=True, exist_ok=True)

def active_key():
    return settings.gemini_api_key if settings.ai_provider=='gemini' else settings.openai_api_key

def active_vision_model():
    return settings.gemini_model if settings.ai_provider=='gemini' else settings.vision_model

def active_planner_model():
    return settings.gemini_model if settings.ai_provider=='gemini' else settings.planner_model

def refresh_ai_settings():
    # Let a long-running worker pick up a key saved through local setup.
    values=dotenv_values(ROOT/'.env')
    for name in ('OPENAI_API_KEY','GEMINI_API_KEY','AI_PROVIDER','GEMINI_MODEL'):
        if values.get(name) is not None: setattr(settings,name.lower(),values[name])

def binary(name: str) -> str:
    configured = getattr(settings, name + '_path')
    found = configured or shutil.which(name)
    if not found:
        found = next((str(p) for p in (ROOT / '.tools').glob(f'**/{name}.exe')), None)
    if not found:
        raise RuntimeError(f'{name} is not installed. Run scripts/setup.ps1.')
    return found
