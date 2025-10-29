"""
Simple FastAPI server for the Local LLM Agent (without MCP tools).
Provides REST API endpoints for chat functionality only.
"""

import asyncio
import sys
import threading
from pathlib import Path
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add src to path to import the agent
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_agent import LocalLLMAgent, AgentConfig
from llm_agent.core.config import OllamaConfig, RAGConfig, MCPConfig

# Global agent instance
agent: LocalLLMAgent | None = None
restart_flag = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global agent
    
    # Startup
    print("🚀 Starting Simple LLM Agent API Server (No MCP)...")
    
    # Create configuration without MCP
    config = AgentConfig(
        agent_name="SimpleAgent",
        system_prompt="""You are a helpful AI assistant running locally. 
        You can chat and answer questions using your training data.
        You have access to your local knowledge base for retrieving relevant information.
        
        Always be helpful, accurate, and conversational.""",
        
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
            enabled=False,  # MCP disabled - no tools, no extra port
            server_port=8002
        )
    )
    
    # Create and initialize agent
    agent = LocalLLMAgent(config)
    
    try:
        await agent.initialize()
        print("✅ Agent initialized successfully!")
        print("📌 MCP Tools: DISABLED")
    except Exception as e:
        print(f"⚠️  Warning: Could not fully initialize agent: {e}")
        print("    The health check will still work, but chat functionality may be limited.")
    
    yield
    
    # Shutdown
    if agent:
        await agent.cleanup()
        print("👋 Agent cleaned up")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Simple LLM Agent API (No MCP)",
    description="REST API for chatting with Local LLM - No tool execution",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

restart_flag = False


def listen_for_restart():
    """Listen for 'rs' command to restart the server."""
    global restart_flag
    print("\n💡 Tip: Type 'rs' and press Enter to restart the server\n")
    
    while not restart_flag:
        try:
            user_input = input().strip().lower()
            if user_input == "rs":
                print("\n🔄 Restart requested! Restarting server...\n")
                restart_flag = True
                import os
                os._exit(0)
        except (EOFError, KeyboardInterrupt):
            break


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    metadata: Dict[str, Any] = {}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Simple LLM Agent API (No MCP)",
        "status": "running",
        "version": "1.0.0",
        "mcp_enabled": False
    }


@app.get("/health")
async def health_check():
    """Health check endpoint to verify backend is running."""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None and agent._initialized,
        "agent_name": agent.config.agent_name if agent else None,
        "mcp_enabled": False,
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
        "mcp_enabled": False,
        "conversation_length": len(agent.conversation.messages)
    }


def main():
    """Run the API server."""
    print("=" * 60)
    print("🤖 Simple LLM Agent API Server (No MCP)")
    print("=" * 60)
    print()
    print("📡 Starting server on http://localhost:8001")
    print("📖 API docs available at http://localhost:8001/docs")
    print()
    print("⚠️  Make sure Ollama is running on http://localhost:11434")
    print("🔧 MCP Tools: DISABLED (no extra port needed)")
    print()
    
    # Start input listener in a separate thread
    input_thread = threading.Thread(target=listen_for_restart, daemon=True)
    input_thread.start()
    
    uvicorn.run(
        "api_server_simple:app",  # Use import string for reload to work
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=True,
        reload_dirs=["./src", "."]
    )


if __name__ == "__main__":
    main()
