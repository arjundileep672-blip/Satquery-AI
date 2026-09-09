"""
API v1 Router Definition
"""

from fastapi import APIRouter
from app.api.v1.endpoints import health, analyze, metrics

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(analyze.router, tags=["Analysis"])
api_router.include_router(metrics.router, tags=["Metrics"])
