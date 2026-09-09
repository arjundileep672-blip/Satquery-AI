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


# Serve built frontend if dist exists
dist_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if dist_dir.exists() and (dist_dir / "index.html").exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    dist_data_dir = dist_dir / "data"
    if dist_data_dir.exists():
        app.mount("/data", StaticFiles(directory=str(dist_data_dir)), name="static_data")

    @app.get("/", include_in_schema=False)
    async def serve_root():
        return FileResponse(dist_dir / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path in ("docs", "redoc", "openapi.json"):
            return None
        file_path = dist_dir / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(dist_dir / "index.html")
else:
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
