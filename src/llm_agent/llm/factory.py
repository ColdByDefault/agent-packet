"""
Factory for creating LLM providers.
Implements Factory pattern for extensible LLM provider creation.
"""

from typing import Dict, Any, Type
from .base import LLMProvider
from .ollama import OllamaProvider


class LLMProviderFactory:
    """Factory class for creating LLM providers."""
    
    _providers: Dict[str, Type[LLMProvider]] = {
        "ollama": OllamaProvider,
    }
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[LLMProvider]) -> None:
        """Register a new LLM provider."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create_provider(cls, provider_type: str, config: Dict[str, Any]) -> LLMProvider:
        """Create an LLM provider instance."""
        if provider_type not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(f"Unknown provider type: {provider_type}. Available: {available}")
        
        provider_class = cls._providers[provider_type]
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available provider types."""
        return list(cls._providers.keys())


# Convenience function for easy provider creation
def create_llm_provider(provider_type: str, config: Dict[str, Any]) -> LLMProvider:
    """Create an LLM provider using the factory."""
    return LLMProviderFactory.create_provider(provider_type, config)