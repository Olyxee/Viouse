import os
from pathlib import Path
import tempfile
import shutil

# Set before any backend import, including when a single test module is selected.
TEST_DIR=Path(tempfile.mkdtemp(prefix='editor-tests-'))
os.environ['DATA_DIR']=str(TEST_DIR)
os.environ['DATABASE_URL']='sqlite:///'+(TEST_DIR/'test.db').as_posix()

def pytest_sessionfinish(session,exitstatus):
    from backend.db import engine
    engine.dispose()
    target=TEST_DIR.resolve()
    expected_parent=Path(tempfile.gettempdir()).resolve()
    if target.parent==expected_parent and target.name.startswith('editor-tests-'):
        shutil.rmtree(target,ignore_errors=True)
