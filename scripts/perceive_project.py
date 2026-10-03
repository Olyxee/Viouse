"""Run perception + clip understanding for every clip of a project (cached)."""
import sys, time
from backend.db import init_db
from backend.pipeline import perceive_all, understand_all, load_project
init_db()
project, assets, videos, music, refs, directory = load_project(sys.argv[1])
from backend.ai import AI
t = time.time()
signals = perceive_all(project, videos, directory, lambda m: print(m, round(time.time()-t), flush=True))
if '--no-ai' not in sys.argv:
    understand_all(AI(), project, videos, directory, lambda m: print(m, round(time.time()-t), flush=True))
