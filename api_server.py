"""
FastAPI server for the Local LLM Agent.
Provides REST API endpoints to interact with the agent.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add src to path to import the agent
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_agent import LocalLLMAgent, AgentConfig
from llm_agent.core.config import OllamaConfig, RAGConfig, MCPConfig

# Create FastAPI app
app = FastAPI(
    title="Local LLM Agent API",
    description="REST API for interacting with the Local LLM Agent",
    version="1.0.0"
)

# Configure CORS to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent: LocalLLMAgent | None = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    metadata: Dict[str, Any] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup."""
    global agent
    
    print("🚀 Starting Local LLM Agent API Server...")
    
    # Create configuration
    config = AgentConfig(
        agent_name="LocalAgent",
        system_prompt="""You are a helpful AI assistant running locally. You have access to:
        - Your local knowledge base for retrieving relevant information
        - Various tools for calculations, text search, and system information
        - The ability to remember our conversation
        
        Always be helpful, accurate, and mention when you're using your knowledge base or tools.""",
        
        ollama=OllamaConfig(
            base_url="http://localhost:11434",
            model="llama3.1:8b",
            temperature=0.7
        ),
        
        rag=RAGConfig(
            vector_db_path="./data/vectordb",
            embedding_model="nomic-embed-text:latest",
            chunk_size=800,
            max_results=5
        ),
        
        mcp=MCPConfig(
            enabled=True,
            server_port=8002  # MCP internal server on different port
        )
    )
    
    # Create and initialize agent
    agent = LocalLLMAgent(config)
    
    try:
        await agent.initialize()
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not fully initialize agent: {e}")
        print("    The health check will still work, but chat functionality may be limited.")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global agent
    if agent:
        await agent.cleanup()
        print("👋 Agent cleaned up")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Local LLM Agent API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint to verify backend is running."""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None and agent._initialized,
        "agent_name": agent.config.agent_name if agent else None,
        "timestamp": asyncio.get_event_loop().time()
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the agent.
    
    Args:
        request: Chat request with user message
        
    Returns:
        Agent's response
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not agent._initialized:
        raise HTTPException(status_code=503, detail="Agent still initializing")
    
    try:
        # Query the agent
        response = await agent.query(request.message)
        
        return ChatResponse(
            response=response.content,
            metadata={
                "model": agent.config.ollama.model,
                "tokens_used": response.metadata.get("tokens_used", 0)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.get("/status")
async def get_status():
    """Get detailed agent status."""
    if not agent:
        return {"status": "not_initialized"}
    
    return {
        "status": "ready" if agent._initialized else "initializing",
        "agent_name": agent.config.agent_name,
        "llm_model": agent.config.ollama.model,
        "ollama_url": agent.config.ollama.base_url,
        "rag_enabled": agent.rag_system is not None,
        "mcp_enabled": agent.config.mcp.enabled,
        "conversation_length": len(agent.conversation.messages)
    }


def main():
    """Run the API server."""
    print("=" * 60)
    print("🤖 Local LLM Agent API Server")
    print("=" * 60)
    print()
    print("📡 Starting server on http://localhost:8001")
    print("📖 API docs available at http://localhost:8001/docs")
    print()
    print("⚠️  Make sure Ollama is running on http://localhost:11434")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )


if __name__ == "__main__":
    main()
