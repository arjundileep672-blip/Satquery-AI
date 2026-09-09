"""
SatQuery AI - FastAPI Application Entrypoint
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

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("satquery.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(settings, "LLM_PROVIDER", "").lower() == "ollama":
        mode_str = f"PRODUCTION (Ollama: {settings.OLLAMA_MODEL})"
    elif settings.DEMO_MODE:
        mode_str = "DEMO (Mock VLM)"
    else:
        mode_str = f"PRODUCTION (Gemini: {settings.GEMINI_MODEL})"
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} in {mode_str} mode")
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Interactive Vision-Language Assistant for Remote Sensing Image Analysis Through Text Queries",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 routes
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
def root_status():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": f"{settings.API_V1_STR}/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
