"""Run the full autonomous edit for a project outside the worker (development benchmark)."""
import json, sys, time
from backend.db import init_db
from backend.pipeline import process
init_db()
best_id, best, metrics = process(None, sys.argv[1])
print(json.dumps({'version': best_id, 'path': best['path'], 'duration': best['duration'], 'critic': best['critic']['overall'],
                  'renders': metrics['renders'], 'timings': metrics['timings_seconds'], 'total': metrics['total_seconds']}, indent=1))
