"""
SatQuery AI Backend
Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis
"""

import sys
from pathlib import Path

# Ensure backend root takes precedence on sys.path to prevent root-level ./models directory
# from shadowing backend/models as an implicit namespace package.
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir in sys.path:
    sys.path.remove(_backend_dir)
sys.path.insert(0, _backend_dir)

if "models" in sys.modules and not hasattr(sys.modules["models"], "__file__"):
    del sys.modules["models"]

__version__ = "0.1.0"
