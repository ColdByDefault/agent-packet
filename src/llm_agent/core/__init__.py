"""
Core module initialization.
"""

from .config import AgentConfig
from .agent import LocalLLMAgent

__all__ = [
    "AgentConfig",
    "LocalLLMAgent",
]