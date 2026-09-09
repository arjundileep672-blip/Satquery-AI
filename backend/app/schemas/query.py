"""
Query Request/Response and WebSocket Protocol Schemas
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from app.schemas.agent import StructuredAnalysisResult, ToolInvocation


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=2, description="Natural language query")
    session_id: Optional[str] = Field(default=None, description="Chat session ID")
    primary_asset_id: Optional[str] = Field(default=None, description="Primary satellite scene ID")
    secondary_asset_id: Optional[str] = Field(default=None, description="Secondary asset ID for change detection or cross-modal")
    aoi_geojson: Optional[Dict[str, Any]] = Field(default=None, description="User drawn Area of Interest polygon")
    confidence_threshold: float = Field(default=0.45, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    query_id: str
    session_id: str
    prompt: str
    result: StructuredAnalysisResult


class WSClientQuery(BaseModel):
    action: Literal["execute_query", "ping", "cancel"]
    prompt: Optional[str] = None
    session_id: Optional[str] = None
    primary_asset_id: Optional[str] = None
    secondary_asset_id: Optional[str] = None
    aoi_geojson: Optional[Dict[str, Any]] = None
    confidence_threshold: Optional[float] = 0.45


class WSServerEvent(BaseModel):
    event_type: Literal[
        "connected",
        "agent_status",
        "plan_generated",
        "step_started",
        "step_progress",
        "step_completed",
        "final_result",
        "error",
        "pong",
    ]
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
