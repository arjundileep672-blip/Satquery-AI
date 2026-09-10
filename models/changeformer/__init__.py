"""
SatQuery AI — ChangeFormer Package Proxy
Exposes ChangeFormer detection interface from backend/models/changeformer.py
to prevent root-level checkpoint folder shadowing backend/models.
"""

import importlib.util
import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_cf_file = _backend_dir / "models" / "changeformer.py"
if _cf_file.exists():
    _spec = importlib.util.spec_from_file_location("_changeformer_impl", str(_cf_file))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    detect_changes = getattr(_mod, "detect_changes", None)
    ChangeFormerDetector = getattr(_mod, "ChangeFormerDetector", None)
    _probe_changeformer = getattr(_mod, "_probe_changeformer", None)
    _get_changeformer_model = getattr(_mod, "_get_changeformer_model", None)
