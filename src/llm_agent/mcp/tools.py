"""
Built-in MCP tools for common operations.
Provides basic tools like search, calculator, weather, etc.
"""

import math
import json
import asyncio
from typing import Dict, Any, List
from datetime import datetime
import re

from .base import MCPTool, ToolDefinition, ToolParameter, ToolParameterType, ToolResult


class CalculatorTool(MCPTool):
    """Basic calculator tool for mathematical operations."""
    
    def get_definition(self) -> ToolDefinition:
        """Get calculator tool definition."""
        return ToolDefinition(
            name="calculator",
            description="Perform basic mathematical calculations",
            parameters=[
                ToolParameter(
                    name="expression",
                    type=ToolParameterType.STRING,
                    description="Mathematical expression to evaluate (e.g., '2 + 3 * 4')",
                    required=True
                )
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute mathematical calculation."""
        expression = parameters.get("expression", "").strip()
        
        if not expression:
            return ToolResult(
                success=False,
                error="Expression cannot be empty"
            )
        
        try:
            # Sanitize expression - only allow safe mathematical operations
            safe_chars = set("0123456789+-*/.() ")
            if not all(c in safe_chars for c in expression):
                return ToolResult(
                    success=False,
                    error="Expression contains invalid characters"
                )
            
            # Replace common math functions
            expression = expression.replace("^", "**")  # Power operator
            
            # Evaluate the expression safely
            result = eval(expression, {"__builtins__": {}}, {
                "abs": abs, "round": round, "min": min, "max": max,
                "pow": pow, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                "tan": math.tan, "log": math.log, "pi": math.pi, "e": math.e
            })
            
            return ToolResult(
                success=True,
                result=result,
                metadata={"expression": expression}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Calculation error: {str(e)}"
            )


class TextSearchTool(MCPTool):
    """Tool for searching text within provided content."""
    
    def get_definition(self) -> ToolDefinition:
        """Get text search tool definition."""
        return ToolDefinition(
            name="text_search",
            description="Search for text patterns in provided content",
            parameters=[
                ToolParameter(
                    name="content",
                    type=ToolParameterType.STRING,
                    description="Text content to search in",
                    required=True
                ),
                ToolParameter(
                    name="pattern",
                    type=ToolParameterType.STRING,
                    description="Search pattern (supports regex)",
                    required=True
                ),
                ToolParameter(
                    name="case_sensitive",
                    type=ToolParameterType.BOOLEAN,
                    description="Whether search should be case sensitive",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="max_results",
                    type=ToolParameterType.INTEGER,
                    description="Maximum number of results to return",
                    required=False,
                    default=10
                )
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute text search."""
        content = parameters.get("content", "")
        pattern = parameters.get("pattern", "")
        case_sensitive = parameters.get("case_sensitive", False)
        max_results = parameters.get("max_results", 10)
        
        if not content or not pattern:
            return ToolResult(
                success=False,
                error="Both content and pattern are required"
            )
        
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            matches = re.finditer(pattern, content, flags)
            
            results = []
            for i, match in enumerate(matches):
                if i >= max_results:
                    break
                
                results.append({
                    "match": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "line": content[:match.start()].count('\n') + 1
                })
            
            return ToolResult(
                success=True,
                result={
                    "matches": results,
                    "total_found": len(results),
                    "pattern": pattern
                }
            )
            
        except re.error as e:
            return ToolResult(
                success=False,
                error=f"Invalid regex pattern: {str(e)}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Search failed: {str(e)}"
            )


class DateTimeTool(MCPTool):
    """Tool for date and time operations."""
    
    def get_definition(self) -> ToolDefinition:
        """Get datetime tool definition."""
        return ToolDefinition(
            name="datetime",
            description="Get current date/time or perform date calculations",
            parameters=[
                ToolParameter(
                    name="operation",
                    type=ToolParameterType.STRING,
                    description="Operation to perform",
                    required=True,
                    enum_values=["current", "format", "parse", "add_days", "diff"]
                ),
                ToolParameter(
                    name="date_string",
                    type=ToolParameterType.STRING,
                    description="Date string for parsing or formatting",
                    required=False
                ),
                ToolParameter(
                    name="format",
                    type=ToolParameterType.STRING,
                    description="Date format string",
                    required=False,
                    default="%Y-%m-%d %H:%M:%S"
                ),
                ToolParameter(
                    name="days",
                    type=ToolParameterType.INTEGER,
                    description="Number of days for date arithmetic",
                    required=False
                )
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute datetime operation."""
        operation = parameters.get("operation")
        
        try:
            if operation == "current":
                now = datetime.now()
                return ToolResult(
                    success=True,
                    result={
                        "timestamp": now.isoformat(),
                        "formatted": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "unix": int(now.timestamp())
                    }
                )
            
            elif operation == "format":
                date_string = parameters.get("date_string")
                format_str = parameters.get("format", "%Y-%m-%d %H:%M:%S")
                
                if not date_string:
                    return ToolResult(success=False, error="date_string required for format operation")
                
                # Try to parse the date string
                dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
                formatted = dt.strftime(format_str)
                
                return ToolResult(
                    success=True,
                    result={"formatted": formatted}
                )
            
            # Add more operations as needed
            else:
                return ToolResult(
                    success=False,
                    error=f"Unsupported operation: {operation}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"DateTime operation failed: {str(e)}"
            )


class SystemInfoTool(MCPTool):
    """Tool for getting system information."""
    
    def get_definition(self) -> ToolDefinition:
        """Get system info tool definition."""
        return ToolDefinition(
            name="system_info",
            description="Get system information like OS, Python version, etc.",
            parameters=[
                ToolParameter(
                    name="info_type",
                    type=ToolParameterType.STRING,
                    description="Type of system information to retrieve",
                    required=True,
                    enum_values=["all", "os", "python", "memory", "disk"]
                )
            ]
        )
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute system info retrieval."""
        import platform
        import sys
        import psutil
        
        info_type = parameters.get("info_type", "all")
        
        try:
            info = {}
            
            if info_type in ["all", "os"]:
                info["os"] = {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor()
                }
            
            if info_type in ["all", "python"]:
                info["python"] = {
                    "version": sys.version,
                    "executable": sys.executable,
                    "platform": sys.platform
                }
            
            if info_type in ["all", "memory"]:
                memory = psutil.virtual_memory()
                info["memory"] = {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percentage": memory.percent
                }
            
            if info_type in ["all", "disk"]:
                disk = psutil.disk_usage('/')
                info["disk"] = {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percentage": (disk.used / disk.total) * 100
                }
            
            return ToolResult(
                success=True,
                result=info
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to get system info: {str(e)}"
            )


# Registry of built-in tools
def get_builtin_tools() -> List[MCPTool]:
    """Get list of all built-in MCP tools."""
    return [
        CalculatorTool(),
        TextSearchTool(),
        DateTimeTool(),
        SystemInfoTool(),
    ]