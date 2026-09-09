"""
Controlled Tool Registry for SatQuery AI
Ensures all tools are cataloged, schema-validated, and invoked under least-privilege principles.
No arbitrary code or dynamic execution permitted.
"""

from typing import Any, Dict, List, Optional
from app.core.logging import get_logger
from app.tools.base import BaseTool, ToolExecutionOutput

logger = get_logger("satquery.tools.registry")


class ToolNotFoundError(Exception):
    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' is not registered in the Tool Registry.")
        self.tool_name = tool_name


class ToolRegistry:
    def __init__(self):
        self._registry: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a validated tool instance."""
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Object {tool} must inherit from BaseTool.")
        self._registry[tool.name] = tool
        logger.info(f"Registered tool: '{tool.name}'")

    def get(self, name: str) -> BaseTool:
        """Retrieve a tool by canonical name."""
        if name not in self._registry:
            raise ToolNotFoundError(name)
        return self._registry[name]

    def has_tool(self, name: str) -> bool:
        return name in self._registry

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with capabilities and limitations."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "capabilities": tool.capabilities,
                "supported_data_types": tool.supported_data_types,
                "limitations": tool.limitations,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
            }
            for tool in self._registry.values()
        ]

    def execute(self, tool_name: str, **kwargs) -> ToolExecutionOutput:
        """Safely execute a registered tool."""
        tool = self.get(tool_name)
        return tool.execute(**kwargs)


registry = ToolRegistry()
