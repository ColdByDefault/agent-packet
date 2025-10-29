"""
Local LLM Agent System
A scalable, modular agent system leveraging local LLMs with RAG and MCP integration.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .core.agent import LocalLLMAgent
from .core.config import AgentConfig
from .llm.factory import LLMProviderFactory
from .rag.factory import RAGSystemFactory
from .mcp.factory import MCPServerFactory

__all__ = [
    "LocalLLMAgent",
    "AgentConfig", 
    "LLMProviderFactory",
    "RAGSystemFactory",
    "MCPServerFactory",
]