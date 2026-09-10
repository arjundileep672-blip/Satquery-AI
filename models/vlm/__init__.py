"""
SatQuery AI — VLM Package Proxy
Exposes VLM bridge interface from backend/models/vlm.py
to prevent root-level checkpoint folder shadowing backend/models.
"""

import importlib.util
import sys
from pathlib import Path

_backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

_vlm_file = _backend_dir / "models" / "vlm.py"
if _vlm_file.exists():
    _spec = importlib.util.spec_from_file_location("_vlm_impl", str(_vlm_file))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    get_vlm_bridge = getattr(_mod, "get_vlm_bridge", None)
    VLMBridge = getattr(_mod, "VLMBridge", None)
