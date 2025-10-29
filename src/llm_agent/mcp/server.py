"""
Simple MCP server implementation.
Provides a basic HTTP-based MCP server for tool integration.
"""

import json
import asyncio
from typing import Dict, Any, Optional
from loguru import logger
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .base import MCPServer, ToolCall, ToolResult
from .tools import get_builtin_tools


class SimpleMCPServer(MCPServer):
    """Simple HTTP-based MCP server implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize simple MCP server."""
        super().__init__(config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 8001)
        self.app = FastAPI(title="Local LLM Agent MCP Server")
        self.server: Optional[uvicorn.Server] = None
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        # Register built-in tools if enabled
        if config.get("enable_builtin_tools", True):
            self.register_tools(get_builtin_tools())
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "name": "Local LLM Agent MCP Server",
                "version": "0.1.0",
                "status": "running" if self.running else "stopped"
            }
        
        @self.app.get("/tools")
        async def list_tools():
            """List available tools."""
            definitions = self.tool_registry.get_tool_definitions()
            return {
                "tools": [def_.to_schema() for def_ in definitions]
            }
        
        @self.app.post("/tools/execute")
        async def execute_tool(request: Dict[str, Any]):
            """Execute a tool."""
            try:
                tool_name = request.get("tool_name")
                parameters = request.get("parameters", {})
                call_id = request.get("call_id")
                
                if not tool_name:
                    raise HTTPException(status_code=400, detail="tool_name is required")
                
                tool_call = ToolCall(
                    tool_name=tool_name,
                    parameters=parameters,
                    call_id=call_id
                )
                
                result = await self.tool_registry.execute_tool(tool_call)
                return result.to_dict()
                
            except Exception as e:
                logger.error(f"Tool execution error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "tools": len(self.tool_registry.get_available_tools())}
    
    async def start(self) -> None:
        """Start the MCP server."""
        if self.running:
            logger.warning("MCP server is already running")
            return
        
        try:
            config = uvicorn.Config(
                app=self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
            self.server = uvicorn.Server(config)
            
            # Start server in background task
            self._server_task = asyncio.create_task(self.server.serve())
            self.running = True
            
            logger.info(f"MCP server started on http://{self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop the MCP server."""
        if not self.running:
            logger.warning("MCP server is not running")
            return
        
        try:
            if self.server:
                self.server.should_exit = True
                await self._server_task
            
            self.running = False
            logger.info("MCP server stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop MCP server: {str(e)}")
            raise
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request (for direct API usage)."""
        try:
            method = request.get("method")
            params = request.get("params", {})
            
            if method == "list_tools":
                definitions = self.tool_registry.get_tool_definitions()
                return {
                    "result": {
                        "tools": [def_.to_schema() for def_ in definitions]
                    }
                }
            
            elif method == "execute_tool":
                tool_name = params.get("tool_name")
                parameters = params.get("parameters", {})
                
                tool_call = ToolCall(
                    tool_name=tool_name,
                    parameters=parameters
                )
                
                result = await self.tool_registry.execute_tool(tool_call)
                return {"result": result.to_dict()}
            
            else:
                return {"error": f"Unknown method: {method}"}
                
        except Exception as e:
            logger.error(f"Request handling error: {str(e)}")
            return {"error": str(e)}
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        return {
            "host": self.host,
            "port": self.port,
            "running": self.running,
            "url": f"http://{self.host}:{self.port}",
            "tools": self.tool_registry.get_available_tools()
        }