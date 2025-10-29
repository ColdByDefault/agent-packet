"""
Ollama LLM Provider implementation.
Integrates with your local Ollama server for LLM operations.
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
import httpx
from loguru import logger

from .base import (
    LLMProvider, LLMMessage, LLMResponse, LLMGenerationConfig,
    LLMConnectionError, LLMGenerationError, LLMRole
)


class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama provider."""
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.timeout = config.get("timeout", 120)
        self.client: Optional[httpx.AsyncClient] = None
        
        # Validate model exists in config
        if not self.model_name or self.model_name == "default":
            self.model_name = "llama3.1:8b"  # Default to your installed model
        
        logger.info(f"Initialized Ollama provider with model: {self.model_name}")
    
    async def initialize(self) -> None:
        """Initialize HTTP client and validate connection."""
        try:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout)
            )
            
            # Test connection
            health_ok = await self.health_check()
            if not health_ok:
                raise LLMConnectionError(
                    f"Failed to connect to Ollama server at {self.base_url}",
                    provider="ollama"
                )
            
            # Validate model exists
            available_models = await self.get_available_models()
            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found. Available: {available_models}")
                
            logger.info(f"Ollama provider initialized successfully")
            
        except Exception as e:
            raise LLMConnectionError(
                f"Failed to initialize Ollama provider: {str(e)}",
                provider="ollama"
            )
    
    async def generate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMGenerationConfig] = None
    ) -> LLMResponse:
        """Generate response using Ollama chat API."""
        if not self.client:
            raise LLMConnectionError("Client not initialized", provider="ollama")
        
        try:
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": [msg.to_dict() for msg in messages],
                "stream": False
            }
            
            # Add generation config if provided
            if config:
                payload.update(config.to_dict())
                # Remove stream from config dict since we set it explicitly
                if "stream" in payload:
                    del payload["stream"]
                payload["stream"] = False
            
            logger.debug(f"Sending request to Ollama: {payload}")
            
            # Make API call
            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response content
            content = result.get("message", {}).get("content", "")
            usage = {
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
                "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            }
            
            return LLMResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                metadata={
                    "eval_duration": result.get("eval_duration"),
                    "load_duration": result.get("load_duration"),
                    "prompt_eval_duration": result.get("prompt_eval_duration"),
                },
                finish_reason=result.get("done_reason", "stop")
            )
            
        except httpx.HTTPStatusError as e:
            raise LLMGenerationError(
                f"Ollama API error: {e.response.status_code} - {e.response.text}",
                provider="ollama",
                model=self.model_name
            )
        except Exception as e:
            raise LLMGenerationError(
                f"Failed to generate response: {str(e)}",
                provider="ollama",
                model=self.model_name
            )
    
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMGenerationConfig] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response using Ollama chat API."""
        if not self.client:
            raise LLMConnectionError("Client not initialized", provider="ollama")
        
        try:
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": [msg.to_dict() for msg in messages],
                "stream": True
            }
            
            # Add generation config if provided
            if config:
                payload.update(config.to_dict())
                payload["stream"] = True  # Ensure streaming is enabled
            
            logger.debug(f"Sending streaming request to Ollama: {payload}")
            
            # Make streaming API call
            async with self.client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk:
                                content = chunk["message"].get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse streaming response: {line}")
                            continue
                            
        except httpx.HTTPStatusError as e:
            raise LLMGenerationError(
                f"Ollama streaming API error: {e.response.status_code}",
                provider="ollama",
                model=self.model_name
            )
        except Exception as e:
            raise LLMGenerationError(
                f"Failed to generate streaming response: {str(e)}",
                provider="ollama",
                model=self.model_name
            )
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        if not self.client:
            raise LLMConnectionError("Client not initialized", provider="ollama")
        
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            
            logger.debug(f"Available Ollama models: {models}")
            return models
            
        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            return []
    
    async def health_check(self) -> bool:
        """Check if Ollama server is accessible."""
        if not self.client:
            return False
        
        try:
            response = await self.client.get("/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {str(e)}")
            return False
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        if not self.client:
            raise LLMConnectionError("Client not initialized", provider="ollama")
        
        try:
            payload = {"name": model_name}
            response = await self.client.post("/api/pull", json=payload)
            response.raise_for_status()
            
            logger.info(f"Successfully pulled model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {str(e)}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None