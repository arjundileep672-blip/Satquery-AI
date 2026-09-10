"""
SatQuery AI — Model Configuration Loader
Reads config/models.yaml and exposes typed settings.
Extends existing Phase-1 config without breaking it.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict

# ── Lazy YAML import so startup never fails on missing pyyaml ────────────────
try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False


def _load_env_file():
    """Load key-value pairs from .env into os.environ if not already present."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
        Path(".env").resolve(),
    ]
    for env_path in candidates:
        if env_path.is_file():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k:
                            os.environ[k] = v
                break
            except Exception:
                pass


_load_env_file()


class Settings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    PROJECT_NAME: str = "SatQuery AI"
    VERSION: str = "0.2.0"
    API_V1_STR: str = "/api/v1"

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR.parent / "data"
    DEMO_DIR: Path = DATA_DIR / "demo"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
    CHECKPOINTS_DIR: Path = BASE_DIR.parent / "checkpoints"

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ]

    # File upload limits
    MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    ALLOWED_EXTENSIONS: List[str] = [
        ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".heic", ".heif"
    ]
    ALLOWED_MIME_TYPES: List[str] = [
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/pjpeg",
        "image/tiff",
        "image/x-tiff",
        "image/geotiff",
        "image/webp",
        "image/bmp",
        "image/x-ms-bmp",
        "image/heic",
        "image/heif",
        "application/octet-stream",
    ]

    # AI & Model settings
    DEMO_MODE: bool = os.getenv("DEMO_MODE", "true").lower() in ("true", "1", "yes")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Offline LLM / Ollama configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto")  # "auto" | "ollama" | "gemini" | "mock"
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    OLLAMA_TIMEOUT_SECONDS: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60.0"))

    # Device selection: "auto" | "cuda" | "cpu"
    DEVICE: str = os.getenv("DEVICE", "auto")

    # Models config YAML path (resolved relative to repo root)
    MODELS_CONFIG_PATH: Path = Path(__file__).resolve().parent.parent.parent.parent / "config" / "models.yaml"


settings = Settings()

# Ensure required directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.DEMO_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
settings.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def load_models_config() -> Dict[str, Any]:
    """
    Load models.yaml. Returns empty dict if file missing or YAML unavailable.
    Never crashes startup.
    """
    if not _YAML_OK:
        return {}
    cfg_path = settings.MODELS_CONFIG_PATH
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = _yaml.safe_load(fh) or {}
        return data.get("models", {})
    except Exception:
        return {}


def resolve_device() -> str:
    """
    Returns 'cuda' if CUDA is available and DEVICE != 'cpu', else 'cpu'.
    Imports torch lazily to avoid hard dependency at module load.
    """
    if settings.DEVICE == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
