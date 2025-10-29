"""
Factory for creating MCP servers and components.
Implements Factory pattern for extensible MCP system creation.
"""

from typing import Dict, Any, Type, List
from .base import MCPServer, MCPTool
from .server import SimpleMCPServer
from .tools import get_builtin_tools


class MCPServerFactory:
    """Factory class for creating MCP servers."""
    
    _servers: Dict[str, Type[MCPServer]] = {
        "simple": SimpleMCPServer,
    }
    
    @classmethod
    def register_server(cls, name: str, server_class: Type[MCPServer]) -> None:
        """Register a new MCP server type."""
        cls._servers[name] = server_class
    
    @classmethod
    def create_server(cls, server_type: str, config: Dict[str, Any]) -> MCPServer:
        """Create an MCP server instance."""
        if server_type not in cls._servers:
            available = list(cls._servers.keys())
            raise ValueError(f"Unknown server type: {server_type}. Available: {available}")
        
        server_class = cls._servers[server_type]
        return server_class(config)
    
    @classmethod
    def get_available_servers(cls) -> List[str]:
        """Get list of available server types."""
        return list(cls._servers.keys())


class MCPToolFactory:
    """Factory class for creating MCP tools."""
    
    _custom_tools: Dict[str, Type[MCPTool]] = {}
    
    @classmethod
    def register_tool(cls, name: str, tool_class: Type[MCPTool]) -> None:
        """Register a custom tool type."""
        cls._custom_tools[name] = tool_class
    
    @classmethod
    def create_tool(cls, tool_type: str, config: Dict[str, Any] = None) -> MCPTool:
        """Create a tool instance."""
        if tool_type not in cls._custom_tools:
            available = list(cls._custom_tools.keys())
            raise ValueError(f"Unknown tool type: {tool_type}. Available: {available}")
        
        tool_class = cls._custom_tools[tool_type]
        return tool_class(config or {})
    
    @classmethod
    def get_builtin_tools(cls) -> List[MCPTool]:
        """Get all built-in tools."""
        return get_builtin_tools()
    
    @classmethod
    def get_available_custom_tools(cls) -> List[str]:
        """Get list of available custom tool types."""
        return list(cls._custom_tools.keys())


# Convenience functions
def create_mcp_server(server_type: str = "simple", config: Dict[str, Any] = None) -> MCPServer:
    """Create an MCP server using the factory."""
    if config is None:
        config = {}
    return MCPServerFactory.create_server(server_type, config)


def create_basic_mcp_server(port: int = 8000, enable_builtin_tools: bool = True) -> MCPServer:
    """Create a basic MCP server with default configuration."""
    config = {
        "port": port,
        "host": "localhost",
        "enable_builtin_tools": enable_builtin_tools
    }
    return create_mcp_server("simple", config)