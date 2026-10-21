"""Load each submission with its own sibling modules and restore host imports."""
import hashlib
import importlib.util
from pathlib import Path
import sys
from threading import RLock

_LOCK = RLock()

def load_agent(main_py):
    path = Path(main_py).resolve()
    names = {p.stem for p in path.parent.glob('*.py')}
    names.update(p.name for p in path.parent.iterdir() if p.is_dir() and (p/'__init__.py').exists())
    # The returned module and its globals retain their own imported objects.
    # This loader is called during setup, before worker threads are started.
    with _LOCK:
        old_path = sys.path[:]
        saved = {k:v for k,v in sys.modules.items() if k.split('.')[0] in names}
        try:
            for k in saved: sys.modules.pop(k, None)
            sys.path.insert(0, str(path.parent))
            key = '_submission_' + hashlib.sha256(str(path).encode()).hexdigest()
            spec = importlib.util.spec_from_file_location(key, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            try: spec.loader.exec_module(module)
            finally: sys.modules.pop(key, None)
            module._local_modules = {k:v for k,v in sys.modules.items() if k.split('.')[0] in names}
            return module
        finally:
            for k in list(sys.modules):
                if k.split('.')[0] in names: sys.modules.pop(k, None)
            sys.modules.update(saved)
            sys.path[:] = old_path
