"""
Factory for creating RAG system components.
Implements Factory pattern for extensible RAG system creation.
"""

from typing import Dict, Any, Type
from .base import RAGSystem, EmbeddingProvider, VectorDatabase, DocumentProcessor
from .embeddings import OllamaEmbeddingProvider
from .chroma import ChromaVectorDatabase
from .processors import TextDocumentProcessor, MarkdownDocumentProcessor
from .system import LocalRAGSystem


class RAGSystemFactory:
    """Factory class for creating RAG system components."""
    
    _embedding_providers: Dict[str, Type[EmbeddingProvider]] = {
        "ollama": OllamaEmbeddingProvider,
    }
    
    _vector_databases: Dict[str, Type[VectorDatabase]] = {
        "chroma": ChromaVectorDatabase,
    }
    
    _document_processors: Dict[str, Type[DocumentProcessor]] = {
        "text": TextDocumentProcessor,
        "markdown": MarkdownDocumentProcessor,
    }
    
    @classmethod
    def register_embedding_provider(cls, name: str, provider_class: Type[EmbeddingProvider]) -> None:
        """Register a new embedding provider."""
        cls._embedding_providers[name] = provider_class
    
    @classmethod
    def register_vector_database(cls, name: str, db_class: Type[VectorDatabase]) -> None:
        """Register a new vector database."""
        cls._vector_databases[name] = db_class
    
    @classmethod
    def register_document_processor(cls, name: str, processor_class: Type[DocumentProcessor]) -> None:
        """Register a new document processor."""
        cls._document_processors[name] = processor_class
    
    @classmethod
    def create_embedding_provider(cls, provider_type: str, config: Dict[str, Any]) -> EmbeddingProvider:
        """Create an embedding provider instance."""
        if provider_type not in cls._embedding_providers:
            available = list(cls._embedding_providers.keys())
            raise ValueError(f"Unknown embedding provider: {provider_type}. Available: {available}")
        
        provider_class = cls._embedding_providers[provider_type]
        return provider_class(config)
    
    @classmethod
    def create_vector_database(cls, db_type: str, config: Dict[str, Any]) -> VectorDatabase:
        """Create a vector database instance."""
        if db_type not in cls._vector_databases:
            available = list(cls._vector_databases.keys())
            raise ValueError(f"Unknown vector database: {db_type}. Available: {available}")
        
        db_class = cls._vector_databases[db_type]
        return db_class(config)
    
    @classmethod
    def create_document_processor(cls, processor_type: str, config: Dict[str, Any]) -> DocumentProcessor:
        """Create a document processor instance."""
        if processor_type not in cls._document_processors:
            available = list(cls._document_processors.keys())
            raise ValueError(f"Unknown document processor: {processor_type}. Available: {available}")
        
        processor_class = cls._document_processors[processor_type]
        return processor_class(config)
    
    @classmethod
    def create_rag_system(cls, config: Dict[str, Any]) -> LocalRAGSystem:
        """Create a complete RAG system from configuration."""
        # Extract component configurations
        embedding_config = config.get("embedding", {})
        vector_db_config = config.get("vector_db", {})
        processor_config = config.get("processor", {})
        
        # Determine component types
        embedding_type = embedding_config.get("type", "ollama")
        vector_db_type = vector_db_config.get("type", "chroma")
        processor_type = processor_config.get("type", "text")
        
        # Create components
        embedding_provider = cls.create_embedding_provider(embedding_type, embedding_config)
        vector_database = cls.create_vector_database(vector_db_type, vector_db_config)
        document_processor = cls.create_document_processor(processor_type, processor_config)
        
        # Create RAG system
        return LocalRAGSystem(
            embedding_provider=embedding_provider,
            vector_database=vector_database,
            document_processor=document_processor,
            config=config
        )
    
    @classmethod
    def get_available_components(cls) -> Dict[str, list]:
        """Get available component types."""
        return {
            "embedding_providers": list(cls._embedding_providers.keys()),
            "vector_databases": list(cls._vector_databases.keys()),
            "document_processors": list(cls._document_processors.keys())
        }


# Convenience function for easy RAG system creation
def create_rag_system(config: Dict[str, Any]) -> LocalRAGSystem:
    """Create a RAG system using the factory."""
    return RAGSystemFactory.create_rag_system(config)