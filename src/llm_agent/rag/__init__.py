"""
RAG module initialization.
"""

from .base import (
    RAGSystem, EmbeddingProvider, VectorDatabase, DocumentProcessor,
    Document, DocumentChunk, RetrievalResult
)
from .embeddings import OllamaEmbeddingProvider
from .chroma import ChromaVectorDatabase
from .processors import TextDocumentProcessor, MarkdownDocumentProcessor
from .system import LocalRAGSystem
from .factory import RAGSystemFactory, create_rag_system

__all__ = [
    # Base classes
    "RAGSystem",
    "EmbeddingProvider", 
    "VectorDatabase",
    "DocumentProcessor",
    "Document",
    "DocumentChunk",
    "RetrievalResult",
    
    # Implementations
    "OllamaEmbeddingProvider",
    "ChromaVectorDatabase", 
    "TextDocumentProcessor",
    "MarkdownDocumentProcessor",
    "LocalRAGSystem",
    
    # Factory
    "RAGSystemFactory",
    "create_rag_system",
]