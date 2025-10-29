"""
Complete RAG system implementation.
Combines embedding provider, vector database, and document processor.
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from .base import RAGSystem, Document, RetrievalResult
from .embeddings import OllamaEmbeddingProvider
from .chroma import ChromaVectorDatabase
from .processors import TextDocumentProcessor, MarkdownDocumentProcessor


class LocalRAGSystem(RAGSystem):
    """Complete local RAG system implementation."""
    
    def __init__(
        self,
        embedding_provider,
        vector_database,
        document_processor,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize local RAG system."""
        super().__init__(embedding_provider, vector_database, document_processor)
        self.config = config or {}
        self.max_context_tokens = self.config.get("max_context_tokens", 4000)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
    
    async def search_with_threshold(
        self, 
        query: str, 
        k: int = 5, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Search with similarity threshold filtering."""
        results = await self.search(query, k, filter_dict)
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results 
            if result.score >= self.similarity_threshold
        ]
        
        logger.debug(f"Filtered {len(results)} -> {len(filtered_results)} results by threshold {self.similarity_threshold}")
        return filtered_results
    
    async def get_augmented_context(
        self, 
        query: str, 
        system_context: str = "", 
        k: int = 5
    ) -> Dict[str, Any]:
        """Get augmented context for RAG generation."""
        # Search for relevant documents
        results = await self.search_with_threshold(query, k)
        
        if not results:
            logger.warning("No relevant documents found for query")
            return {
                "context": system_context,
                "sources": [],
                "query": query,
                "found_results": False
            }
        
        # Build context from results
        context_parts = [system_context] if system_context else []
        sources = []
        
        current_tokens = len(system_context) // 4  # Rough token estimation
        
        for result in results:
            chunk_text = result.chunk.content
            chunk_tokens = len(chunk_text) // 4
            
            if current_tokens + chunk_tokens > self.max_context_tokens:
                break
            
            context_parts.append(f"Source: {chunk_text}")
            sources.append({
                "content": chunk_text,
                "score": result.score,
                "metadata": result.chunk.metadata,
                "doc_id": result.chunk.doc_id
            })
            current_tokens += chunk_tokens
        
        full_context = "\n\n".join(context_parts)
        
        return {
            "context": full_context,
            "sources": sources,
            "query": query,
            "found_results": True,
            "total_results": len(results),
            "used_results": len(sources)
        }
    
    async def add_document_from_file(self, file_path: str) -> str:
        """Add document from file and return document ID."""
        document = self.document_processor.load_from_file(file_path)
        await self.add_document(document)
        logger.info(f"Added document from file: {file_path} (ID: {document.doc_id})")
        return document.doc_id
    
    async def bulk_add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add multiple texts efficiently."""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        doc_ids = []
        for i, text in enumerate(texts):
            document = self.document_processor.process_text(text, metadatas[i])
            await self.add_document(document)
            doc_ids.append(document.doc_id)
        
        logger.info(f"Added {len(texts)} texts to RAG system")
        return doc_ids