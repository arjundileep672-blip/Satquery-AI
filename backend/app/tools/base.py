"""
Controlled Tool Base Interface for SatQuery AI
Every registered tool strictly defines input/output contracts, capabilities, and limitations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class ToolExecutionOutput(BaseModel):
    tool_name: str
    status: str = "success"  # "success" | "failed"
    answer: str = ""
    confidence: Any = None   # None if uncalibrated
    observations: List[str] = Field(default_factory=list)
    measurements: Dict[str, Any] = Field(default_factory=dict)
    model_name: str = ""
    error_message: Any = None
    summary: str = ""
    evidence: Any = Field(default_factory=dict)
    detected_objects: List[Any] = Field(default_factory=list)
    geojson_features: List[Any] = Field(default_factory=list)
    change_regions: List[Any] = Field(default_factory=list)
    execution_time_ms: float = 0.0


class BaseTool(ABC):
    """
    Abstract Base Class for all SatQuery AI tools.
    Enforces explicit capability declarations and structured inputs/outputs.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Domain description of tool purpose."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """JSON-schema dictionary of input arguments."""
        pass

    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """JSON-schema dictionary of output results."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """List of functional capabilities provided by this tool."""
        pass

    @property
    @abstractmethod
    def supported_data_types(self) -> List[str]:
        """List of supported raster and metadata formats."""
        pass

    @property
    @abstractmethod
    def limitations(self) -> List[str]:
        """Known domain and operational limitations."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> ToolExecutionOutput:
        """Execute the tool with validated arguments."""
        pass


class BaseRemoteSensingTool(BaseTool):
    """
    Convenience base class for specialized remote sensing analytical tools.
    Delegates execute() to run() and provides default schemas.
    """
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {}

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {}

    @property
    def capabilities(self) -> List[str]:
        return [self.description]

    @property
    def supported_data_types(self) -> List[str]:
        return ["image/png", "image/jpeg", "image/tiff"]

    @property
    def limitations(self) -> List[str]:
        return []

    def execute(self, **kwargs) -> ToolExecutionOutput:
        return self.run(**kwargs)

    @abstractmethod
    def run(self, **kwargs) -> ToolExecutionOutput:
        pass
