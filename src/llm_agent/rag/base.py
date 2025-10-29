"""
Abstract base classes for RAG (Retrieval Augmented Generation) system.
Implements Strategy pattern for different vector databases and embedding models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import hashlib


@dataclass
class Document:
    """Represents a document in the RAG system."""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate document ID if not provided."""
        if self.doc_id is None:
            # Create hash-based ID from content and metadata
            content_hash = hashlib.md5(
                (self.content + str(sorted(self.metadata.items()))).encode()
            ).hexdigest()
            self.doc_id = f"doc_{content_hash[:16]}"


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate chunk ID if not provided."""
        if self.chunk_id is None:
            content_hash = hashlib.md5(self.content.encode()).hexdigest()
            self.chunk_id = f"chunk_{content_hash[:16]}"


@dataclass
class RetrievalResult:
    """Represents a retrieval result from the vector database."""
    chunk: DocumentChunk
    score: float
    distance: Optional[float] = None
    
    def __lt__(self, other):
        """Enable sorting by score (descending)."""
        return self.score > other.score


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the embedding provider."""
        self.config = config
        self.model_name = config.get("model", "default")
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text."""
        pass
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        pass
    
    @abstractmethod
    async def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the embedding provider."""
        pass


class VectorDatabase(ABC):
    """Abstract base class for vector databases."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the vector database."""
        self.config = config
        self.db_path = config.get("db_path", "./vectordb")
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector database."""
        pass
    
    @abstractmethod
    async def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Add document chunks with their embeddings to the database."""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks of a document."""
        pass
    
    @abstractmethod
    async def get_document_count(self) -> int:
        """Get total number of document chunks."""
        pass


class DocumentProcessor(ABC):
    """Abstract base class for document processors."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the document processor."""
        self.config = config
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
    
    @abstractmethod
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split a document into chunks."""
        pass
    
    @abstractmethod
    def load_from_file(self, file_path: Union[str, Path]) -> Document:
        """Load a document from file."""
        pass
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Process raw text into a document."""
        if metadata is None:
            metadata = {}
        return Document(content=text, metadata=metadata)


class RAGSystem(ABC):
    """Abstract base class for RAG systems."""
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_database: VectorDatabase,
        document_processor: DocumentProcessor
    ):
        """Initialize the RAG system."""
        self.embedding_provider = embedding_provider
        self.vector_database = vector_database
        self.document_processor = document_processor
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all components."""
        if not self._initialized:
            await self.embedding_provider.initialize()
            await self.vector_database.initialize()
            self._initialized = True
    
    async def add_document(self, document: Document) -> None:
        """Add a document to the RAG system."""
        # Chunk the document
        chunks = self.document_processor.chunk_document(document)
        
        # Generate embeddings for chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_provider.embed_texts(chunk_texts)
        
        # Store in vector database
        await self.vector_database.add_documents(chunks, embeddings)
    
    async def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add raw text to the RAG system."""
        document = self.document_processor.process_text(text, metadata)
        await self.add_document(document)
    
    async def search(
        self, 
        query: str, 
        k: int = 5, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Search for relevant documents."""
        # Embed the query
        query_embedding = await self.embedding_provider.embed_text(query)
        
        # Search vector database
        results = await self.vector_database.search(query_embedding, k, filter_dict)
        
        return results
    
    async def get_context(
        self, 
        query: str, 
        max_tokens: int = 4000, 
        k: int = 5
    ) -> str:
        """Get context for a query, respecting token limits."""
        results = await self.search(query, k)
        
        context_parts = []
        current_tokens = 0
        
        for result in results:
            chunk_text = result.chunk.content
            # Rough token estimation (1 token ≈ 4 characters)
            chunk_tokens = len(chunk_text) // 4
            
            if current_tokens + chunk_tokens > max_tokens:
                break
            
            context_parts.append(chunk_text)
            current_tokens += chunk_tokens
        
        return "\n\n".join(context_parts)
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the RAG system."""
        return await self.vector_database.delete_document(doc_id)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        doc_count = await self.vector_database.get_document_count()
        embedding_dim = await self.embedding_provider.get_embedding_dimension()
        
        return {
            "document_chunks": doc_count,
            "embedding_dimension": embedding_dim,
            "model": self.embedding_provider.model_name
        }