"""
Ollama embedding provider implementation.
Uses Ollama's embedding API with nomic-embed-text model.
"""

import asyncio
from typing import List, Dict, Any
import httpx
from loguru import logger

from .base import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider using nomic-embed-text model."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama embedding provider."""
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.timeout = config.get("timeout", 60)
        self.client: httpx.AsyncClient = None
        
        # Use nomic-embed-text as default model
        if not self.model_name or self.model_name == "default":
            self.model_name = "nomic-embed-text:latest"
        
        # Cache for embedding dimension
        self._embedding_dimension: int = None
    
    async def initialize(self) -> None:
        """Initialize HTTP client and validate model."""
        try:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout)
            )
            
            # Test embedding to get dimension
            test_embedding = await self.embed_text("test")
            self._embedding_dimension = len(test_embedding)
            
            logger.info(f"Ollama embedding provider initialized with {self.model_name} (dim: {self._embedding_dimension})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama embedding provider: {str(e)}")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text using Ollama embeddings API."""
        if not self.client:
            raise RuntimeError("Client not initialized")
        
        try:
            payload = {
                "model": self.model_name,
                "input": text
            }
            
            response = await self.client.post("/api/embed", json=payload)
            response.raise_for_status()
            
            result = response.json()
            embeddings = result.get("embeddings", [])
            
            if not embeddings:
                raise ValueError("No embeddings returned from Ollama")
            
            # Return the first (and should be only) embedding
            return embeddings[0]
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama embeddings API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Failed to embed text: {str(e)}")
            raise
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts. Can be optimized with batch API if available."""
        embeddings = []
        
        # Process texts in batches to avoid overwhelming the API
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Create tasks for concurrent processing
            tasks = [self.embed_text(text) for text in batch]
            batch_embeddings = await asyncio.gather(*tasks)
            embeddings.extend(batch_embeddings)
        
        logger.debug(f"Generated embeddings for {len(texts)} texts")
        return embeddings
    
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self._embedding_dimension is None:
            # Initialize if not done already
            await self.initialize()
        return self._embedding_dimension
    
    async def cleanup(self) -> None:
        """Cleanup HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None