"""
Configuration management for the LLM Agent system.
Uses Pydantic for type safety and validation.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import os


class OllamaConfig(BaseModel):
    """Configuration for Ollama LLM provider."""
    base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    model: str = Field(default="llama3.1:8b", description="Default model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for generation")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    max_tokens: Optional[int] = Field(default=2048, description="Maximum tokens to generate")
    timeout: int = Field(default=120, description="Request timeout in seconds")


class RAGConfig(BaseModel):
    """Configuration for RAG (Retrieval Augmented Generation) system."""
    vector_db_path: str = Field(default="./data/vectordb", description="Path to vector database")
    embedding_model: str = Field(default="nomic-embed-text:latest", description="Embedding model name")
    chunk_size: int = Field(default=1000, ge=100, le=2000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, ge=0, le=500, description="Overlap between chunks")
    max_results: int = Field(default=5, ge=1, le=20, description="Max retrieval results")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")


class MCPConfig(BaseModel):
    """Configuration for MCP (Model Context Protocol) integration."""
    enabled: bool = Field(default=True, description="Enable MCP integration")
    server_port: int = Field(default=8000, ge=1024, le=65535, description="MCP server port")
    max_connections: int = Field(default=10, ge=1, le=100, description="Max concurrent connections")
    tools_enabled: List[str] = Field(default_factory=lambda: ["search", "calculator", "weather"], 
                                   description="List of enabled tools")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="{time} | {level} | {name} | {message}", description="Log format")
    file_path: Optional[str] = Field(default="./logs/agent.log", description="Log file path")
    rotation: str = Field(default="10 MB", description="Log rotation size")


class AgentConfig(BaseSettings):
    """Main configuration class for the LLM Agent system."""
    
    # Environment
    environment: str = Field(default="development", description="Environment (development/production)")
    debug: bool = Field(default=True, description="Enable debug mode")
    
    # Component configurations
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Agent behavior
    agent_name: str = Field(default="LocalLLMAgent", description="Agent instance name")
    max_conversation_length: int = Field(default=20, ge=1, le=100, description="Max conversation history length")
    system_prompt: str = Field(
        default="You are a helpful AI assistant with access to local knowledge and tools.",
        description="System prompt for the agent"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False
        
    @classmethod
    def load_from_file(cls, config_path: str = None) -> "AgentConfig":
        """Load configuration from file with environment variable overrides."""
        if config_path and os.path.exists(config_path):
            # Could add YAML/JSON loading here if needed
            pass
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
    
    def validate_paths(self) -> bool:
        """Validate and create necessary directories."""
        paths_to_create = [
            os.path.dirname(self.rag.vector_db_path),
            os.path.dirname(self.logging.file_path) if self.logging.file_path else None
        ]
        
        for path in paths_to_create:
            if path and not os.path.exists(path):
                try:
                    os.makedirs(path, exist_ok=True)
                except Exception as e:
                    print(f"Warning: Could not create directory {path}: {e}")
                    return False
        return True