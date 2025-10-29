"""
ChromaDB vector database implementation.
Uses ChromaDB for efficient vector storage and similarity search.
"""

import asyncio
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from loguru import logger
import os

from .base import VectorDatabase, DocumentChunk, RetrievalResult


class ChromaVectorDatabase(VectorDatabase):
    """ChromaDB implementation of vector database."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ChromaDB vector database."""
        super().__init__(config)
        self.collection_name = config.get("collection_name", "documents")
        self.client = None
        self.collection = None
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure directory exists
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB client with persistent storage
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
            
            logger.info(f"ChromaDB initialized at: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    async def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Add document chunks with embeddings to ChromaDB."""
        if not self.collection:
            raise RuntimeError("ChromaDB not initialized")
        
        try:
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                metadata = chunk.metadata.copy()
                if chunk.doc_id:
                    metadata["doc_id"] = chunk.doc_id
                metadatas.append(metadata)
            
            # Add to collection
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
            )
            
            logger.debug(f"Added {len(chunks)} chunks to ChromaDB")
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {str(e)}")
            raise
    
    async def search(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Search for similar documents in ChromaDB."""
        if not self.collection:
            raise RuntimeError("ChromaDB not initialized")
        
        try:
            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": k
            }
            
            # Add filter if provided
            if filter_dict:
                query_params["where"] = filter_dict
            
            # Perform search
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.query(**query_params)
            )
            
            # Process results
            retrieval_results = []
            
            if results and results["ids"] and len(results["ids"]) > 0:
                ids = results["ids"][0]
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                
                for i in range(len(ids)):
                    chunk = DocumentChunk(
                        content=documents[i],
                        metadata=metadatas[i] or {},
                        chunk_id=ids[i],
                        doc_id=metadatas[i].get("doc_id") if metadatas[i] else None
                    )
                    
                    # Convert distance to similarity score (cosine similarity)
                    # ChromaDB returns distances, lower is more similar
                    # For cosine distance: similarity = 1 - distance
                    score = 1.0 - distances[i]
                    
                    retrieval_results.append(
                        RetrievalResult(
                            chunk=chunk,
                            score=score,
                            distance=distances[i]
                        )
                    )
            
            logger.debug(f"Retrieved {len(retrieval_results)} results from ChromaDB")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {str(e)}")
            raise
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks of a document from ChromaDB."""
        if not self.collection:
            raise RuntimeError("ChromaDB not initialized")
        
        try:
            # Find all chunks with the given doc_id
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.get(where={"doc_id": doc_id})
            )
            
            if results and results["ids"]:
                # Delete the chunks
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.collection.delete(ids=results["ids"])
                )
                
                logger.info(f"Deleted {len(results['ids'])} chunks for document: {doc_id}")
                return True
            else:
                logger.warning(f"No chunks found for document: {doc_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
            return False
    
    async def get_document_count(self) -> int:
        """Get total number of document chunks in ChromaDB."""
        if not self.collection:
            raise RuntimeError("ChromaDB not initialized")
        
        try:
            count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.collection.count()
            )
            return count
            
        except Exception as e:
            logger.error(f"Failed to get document count: {str(e)}")
            return 0
    
    async def reset_collection(self) -> None:
        """Reset (delete all data from) the collection."""
        if not self.collection:
            raise RuntimeError("ChromaDB not initialized")
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.delete_collection(name=self.collection_name)
            )
            
            # Recreate the collection
            self.collection = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            )
            
            logger.info(f"Reset ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            raise