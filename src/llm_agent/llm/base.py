"""
Abstract base classes for LLM providers.
Implements Strategy pattern for different LLM backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass
from enum import Enum
import asyncio


class LLMRole(Enum):
    """Enumeration for message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class LLMMessage:
    """Represents a message in the conversation."""
    role: LLMRole
    content: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        result = {
            "role": self.role.value,
            "content": self.content
        }
        if self.metadata:
            result.update(self.metadata)
        return result


@dataclass
class LLMResponse:
    """Represents a response from the LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "metadata": self.metadata,
            "finish_reason": self.finish_reason
        }


@dataclass
class LLMGenerationConfig:
    """Configuration for LLM generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format."""
        config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": self.stream
        }
        if self.max_tokens is not None:
            config["max_tokens"] = self.max_tokens
        if self.stop_sequences:
            config["stop"] = self.stop_sequences
        return config


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LLM provider with configuration."""
        self.config = config
        self.model_name = config.get("model", "default")
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM provider (e.g., setup connections, validate config)."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMGenerationConfig] = None
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMGenerationConfig] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM."""
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """Get list of available models."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM provider is healthy and accessible."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not self._initialized:
            await self.initialize()
            self._initialized = True
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def cleanup(self) -> None:
        """Cleanup resources (override if needed)."""
        pass
    
    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(model={self.model_name})"


class LLMError(Exception):
    """Base exception for LLM-related errors."""
    def __init__(self, message: str, provider: str = None, model: str = None):
        super().__init__(message)
        self.provider = provider
        self.model = model


class LLMConnectionError(LLMError):
    """Exception raised when connection to LLM provider fails."""
    pass


class LLMGenerationError(LLMError):
    """Exception raised when LLM generation fails."""
    pass


class LLMConfigurationError(LLMError):
    """Exception raised when LLM configuration is invalid."""
    pass