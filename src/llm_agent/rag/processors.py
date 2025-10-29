"""
Document processing utilities.
Handles text chunking and document loading from various formats.
"""

import re
from typing import List, Dict, Any, Union
from pathlib import Path
from loguru import logger

from .base import Document, DocumentChunk, DocumentProcessor


class TextDocumentProcessor(DocumentProcessor):
    """Basic text document processor with intelligent chunking."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize text document processor."""
        super().__init__(config)
        self.separators = config.get("separators", ["\n\n", "\n", ". ", "! ", "? ", " "])
        self.keep_separator = config.get("keep_separator", True)
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Split document into chunks using hierarchical text splitting."""
        chunks = []
        text = document.content
        
        # Start with the full text
        current_chunks = [text]
        
        for separator in self.separators:
            new_chunks = []
            
            for chunk in current_chunks:
                if len(chunk) <= self.chunk_size:
                    new_chunks.append(chunk)
                else:
                    # Split this chunk further
                    split_chunks = self._split_text(chunk, separator)
                    new_chunks.extend(split_chunks)
            
            current_chunks = new_chunks
            
            # If all chunks are small enough, we're done
            if all(len(chunk) <= self.chunk_size for chunk in current_chunks):
                break
        
        # Create DocumentChunk objects with overlap
        for i, chunk_text in enumerate(current_chunks):
            if not chunk_text.strip():
                continue
            
            # Add metadata for chunk position
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(current_chunks),
                "chunk_size": len(chunk_text)
            })
            
            chunk = DocumentChunk(
                content=chunk_text.strip(),
                metadata=chunk_metadata,
                doc_id=document.doc_id
            )
            chunks.append(chunk)
        
        # Add overlapping context if configured
        if self.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)
        
        logger.debug(f"Split document into {len(chunks)} chunks")
        return chunks
    
    def _split_text(self, text: str, separator: str) -> List[str]:
        """Split text by separator while trying to maintain chunk size."""
        if separator == " ":
            # Word-level splitting
            words = text.split()
            chunks = []
            current_chunk = []
            current_size = 0
            
            for word in words:
                word_size = len(word) + 1  # +1 for space
                
                if current_size + word_size > self.chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_size = word_size
                else:
                    current_chunk.append(word)
                    current_size += word_size
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            return chunks
        else:
            # Split by separator
            parts = text.split(separator)
            if self.keep_separator and separator != " ":
                # Rejoin with separator except for last part
                for i in range(len(parts) - 1):
                    parts[i] += separator
            
            return [part for part in parts if part.strip()]
    
    def _add_overlap(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Add overlapping context between chunks."""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            content = chunk.content
            
            # Add overlap from previous chunk
            if i > 0:
                prev_content = chunks[i - 1].content
                overlap_text = prev_content[-self.chunk_overlap:]
                content = overlap_text + " ... " + content
            
            # Add overlap from next chunk
            if i < len(chunks) - 1:
                next_content = chunks[i + 1].content
                overlap_text = next_content[:self.chunk_overlap]
                content = content + " ... " + overlap_text
            
            # Create new chunk with overlapped content
            overlapped_chunk = DocumentChunk(
                content=content,
                metadata=chunk.metadata.copy(),
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id
            )
            overlapped_chunk.metadata["has_overlap"] = True
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def load_from_file(self, file_path: Union[str, Path]) -> Document:
        """Load document from text file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_type": file_path.suffix.lower()
            }
            
            document = Document(content=content, metadata=metadata)
            logger.info(f"Loaded document from: {file_path}")
            return document
            
        except Exception as e:
            logger.error(f"Failed to load document from {file_path}: {str(e)}")
            raise


class MarkdownDocumentProcessor(TextDocumentProcessor):
    """Document processor optimized for Markdown files."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Markdown document processor."""
        # Markdown-specific separators
        markdown_separators = [
            "\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",  # Headers
            "\n\n",  # Paragraphs
            "\n",     # Lines
            ". ", "! ", "? ",  # Sentences
            " "       # Words
        ]
        config["separators"] = config.get("separators", markdown_separators)
        super().__init__(config)
    
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Chunk Markdown document while preserving structure."""
        chunks = super().chunk_document(document)
        
        # Add Markdown-specific metadata
        for chunk in chunks:
            content = chunk.content
            
            # Detect headers in chunk
            headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            if headers:
                chunk.metadata["headers"] = [{"level": len(h[0]), "text": h[1]} for h in headers]
            
            # Detect code blocks
            code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL)
            if code_blocks:
                chunk.metadata["code_blocks"] = len(code_blocks)
                chunk.metadata["code_languages"] = [cb[0] for cb in code_blocks if cb[0]]
        
        return chunks