"""
Abstract base classes for MCP (Model Context Protocol) integration.
Implements Strategy pattern for different MCP tools and capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json


class ToolParameterType(Enum):
    """Enumeration for tool parameter types."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """Represents a tool parameter definition."""
    name: str
    type: ToolParameterType
    description: str
    required: bool = True
    default: Any = None
    enum_values: Optional[List[Any]] = None
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format."""
        schema = {
            "type": self.type.value,
            "description": self.description
        }
        
        if self.enum_values:
            schema["enum"] = self.enum_values
        
        if self.default is not None:
            schema["default"] = self.default
        
        return schema


@dataclass
class ToolDefinition:
    """Represents a tool definition for MCP."""
    name: str
    description: str
    parameters: List[ToolParameter]
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to MCP tool schema format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }


@dataclass
class ToolCall:
    """Represents a tool call request."""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata
        }


class MCPTool(ABC):
    """Abstract base class for MCP tools."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the MCP tool."""
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
    
    @abstractmethod
    def get_definition(self) -> ToolDefinition:
        """Get the tool definition for MCP registration."""
        pass
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate tool parameters against definition."""
        definition = self.get_definition()
        
        # Check required parameters
        for param in definition.parameters:
            if param.required and param.name not in parameters:
                return False
        
        # Basic type checking could be added here
        return True
    
    async def safe_execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Safely execute the tool with error handling."""
        try:
            # Validate parameters
            if not self.validate_parameters(parameters):
                return ToolResult(
                    success=False,
                    error="Invalid parameters provided"
                )
            
            # Execute the tool
            return await self.execute(parameters)
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )


class MCPToolRegistry:
    """Registry for managing MCP tools."""
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, MCPTool] = {}
    
    def register_tool(self, tool: MCPTool) -> None:
        """Register a tool in the registry."""
        definition = tool.get_definition()
        self._tools[definition.name] = tool
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool from the registry."""
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [name for name, tool in self._tools.items() if tool.enabled]
    
    def get_tool_definitions(self) -> List[ToolDefinition]:
        """Get all tool definitions."""
        return [tool.get_definition() for tool in self._tools.values() if tool.enabled]
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get_tool(tool_call.tool_name)
        
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_call.tool_name}' not found"
            )
        
        if not tool.enabled:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_call.tool_name}' is disabled"
            )
        
        return await tool.safe_execute(tool_call.parameters)


class MCPServer(ABC):
    """Abstract base class for MCP servers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the MCP server."""
        self.config = config
        self.tool_registry = MCPToolRegistry()
        self.running = False
    
    @abstractmethod
    async def start(self) -> None:
        """Start the MCP server."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the MCP server."""
        pass
    
    @abstractmethod
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request."""
        pass
    
    def register_tool(self, tool: MCPTool) -> None:
        """Register a tool with the server."""
        self.tool_registry.register_tool(tool)
    
    def register_tools(self, tools: List[MCPTool]) -> None:
        """Register multiple tools with the server."""
        for tool in tools:
            self.register_tool(tool)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()